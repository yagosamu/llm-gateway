import asyncio

import fakeredis
import pytest
from fastapi.testclient import TestClient

from llm_gateway.api import create_app
from llm_gateway.breaker import CircuitBreaker
from llm_gateway.health import HealthTracker
from llm_gateway.providers.base import ProviderError, ProviderResult
from llm_gateway.registry import load_registry
from llm_gateway.request_log import RequestLog
from llm_gateway.routing import DEFAULT_CONFIG_PATH, BreakerConfig, Router, RoutingConfigError, load_config

REGISTRY = load_registry()


def write_config(directory, fallbacks="    low: [gpt-oss-20b]\n", deadline=90):
    path = directory / "routing.yaml"
    path.write_text('version: "t"\npolicy: always_low\ntier_map:\n  low: gpt-6-luna\n  medium: claude-haiku-4-5\n'
                    f"  high: gpt-6.1-sol\nfailover:\n  deadline_seconds: {deadline}\n  fallbacks:\n{fallbacks}",
                    encoding="utf-8")
    return path


class Providers:
    """One fake per provider name; `behaviour` maps a provider to an error kind, "hang" or None (answers)."""

    def __init__(self, behaviour):
        self.behaviour, self.calls = behaviour, []

    def for_provider(self, name):
        outer = self

        class One:
            async def complete(self, model, request):
                outer.calls.append(model.id)
                what = outer.behaviour.get(name)
                if what == "hang":
                    await asyncio.sleep(30)
                if what:
                    raise ProviderError(what, name, "boom")
                return ProviderResult(f"from {model.id}", 10, 5, "stop")
        return One()


def gateway(tmp_path, behaviour, breaker=False, **config):
    providers = Providers(behaviour)
    log = RequestLog(":memory:")
    health = cb = None
    if breaker:
        store = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())
        health = HealthTracker(store)
        cb = CircuitBreaker(store, health, config=lambda: BreakerConfig(min_calls=2, open_seconds=600))
    app = create_app(REGISTRY, {p: providers.for_provider(p) for p in ("openai", "anthropic", "groq")}, log,
                     router=Router(write_config(tmp_path, **config), set(REGISTRY)), health=health, breaker=cb)
    return TestClient(app), providers, log


def ask(client, model="auto", i=0):
    return client.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
                       headers={"X-Tenant-Id": "t", "X-Feature": "open_qa", "X-Request-Id": f"r{i}"})


def test_a_retryable_error_fails_over_to_the_next_model_and_says_so(tmp_path):
    client, providers, log = gateway(tmp_path, {"openai": "server_error"})
    response = ask(client)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "gpt-oss-20b" and data["gateway"]["provider"] == "groq"
    assert data["gateway"]["attempts"] == [{"model": "gpt-6-luna", "outcome": "server_error"},
                                           {"model": "gpt-oss-20b", "outcome": "ok"}]
    [row] = log.rows()
    assert (row["model"], row["failover_from"], row["attempts"]) == ("gpt-oss-20b", "gpt-6-luna", 2)
    assert 'llm_gateway_failovers_total{from_model="gpt-6-luna",reason="server_error",to_model="gpt-oss-20b"} 1.0' \
        in client.get("/metrics").text


def test_cost_is_attributed_to_the_model_that_answered(tmp_path):
    from llm_gateway.registry import cost_usd
    client, _, log = gateway(tmp_path, {"openai": "timeout"})
    ask(client)
    assert log.rows()[0]["cost_usd"] == pytest.approx(cost_usd(REGISTRY["gpt-oss-20b"], 10, 5))


def test_a_healthy_primary_answers_with_no_attempts_block(tmp_path):
    client, providers, _ = gateway(tmp_path, {})
    data = ask(client).json()
    assert data["model"] == "gpt-6-luna" and data["gateway"]["attempts"] is None
    assert providers.calls == ["gpt-6-luna"]


def test_a_bad_request_does_not_fail_over(tmp_path):
    client, providers, _ = gateway(tmp_path, {"openai": "bad_request"})
    response = ask(client)
    assert response.status_code == 400 and response.json()["error"]["code"] == "upstream_bad_request"
    assert providers.calls == ["gpt-6-luna"]


def test_a_named_model_never_fails_over(tmp_path):
    client, providers, _ = gateway(tmp_path, {"openai": "server_error"})
    response = ask(client, model="gpt-6-luna")
    assert response.status_code == 502 and providers.calls == ["gpt-6-luna"]


def test_when_every_model_fails_the_client_gets_the_real_error_and_the_trail(tmp_path):
    client, _, _ = gateway(tmp_path, {"openai": "server_error", "groq": "rate_limit"})
    response = ask(client)
    assert response.status_code == 429 and response.json()["error"]["code"] == "upstream_rate_limit"
    assert "gpt-6-luna: server_error, gpt-oss-20b: rate_limit" in response.json()["error"]["message"]


def test_an_open_circuit_is_skipped_without_a_call(tmp_path):
    client, providers, _ = gateway(tmp_path, {"openai": "server_error"}, breaker=True)
    for i in range(2):
        ask(client, i=i)  # two failures open openai's circuit (min_calls=2)
    providers.calls.clear()
    data = ask(client, i=9).json()
    assert data["model"] == "gpt-oss-20b" and providers.calls == ["gpt-oss-20b"]
    assert data["gateway"]["attempts"][0] == {"model": "gpt-6-luna", "outcome": "circuit_open"}


def test_every_circuit_open_is_a_503_without_any_call(tmp_path):
    client, providers, _ = gateway(tmp_path, {"openai": "server_error", "groq": "server_error"}, breaker=True)
    for i in range(2):
        ask(client, i=i)
    providers.calls.clear()
    response = ask(client, i=9)
    assert response.status_code == 503 and response.json()["error"]["code"] == "upstream_circuit_open"
    assert providers.calls == []


def test_the_deadline_bounds_the_whole_request(tmp_path):
    client, providers, _ = gateway(tmp_path, {"openai": "hang"}, deadline=0.3)
    response = ask(client)
    assert response.status_code == 504 and response.json()["error"]["code"] == "upstream_timeout"
    assert providers.calls == ["gpt-6-luna"]  # no time was left for the fallback


@pytest.mark.parametrize("fallbacks, message", [
    ("    low: [gpt-9]\n", "outside the registry"),
    ("    low: [gpt-6-luna]\n", "repeats the tier's own model"),
    ("    lowest: [gpt-oss-20b]\n", "unknown tier"),
])
def test_invalid_fallbacks_are_refused(tmp_path, fallbacks, message):
    with pytest.raises(RoutingConfigError, match=message):
        load_config(write_config(tmp_path, fallbacks=fallbacks), set(REGISTRY))


def test_every_committed_fallback_is_on_another_provider():
    config = load_config(DEFAULT_CONFIG_PATH, set(REGISTRY))
    assert set(config.failover.fallbacks) == {"low", "medium", "high"}
    for tier, models in config.failover.fallbacks.items():
        primary = REGISTRY[config.tier_map[tier]].provider
        assert all(REGISTRY[m].provider != primary for m in models), tier


def test_fallback_quality_scores_every_candidate_the_same_way():
    from harness.routing import prereg
    from harness.routing.fallback_quality import CHOSEN, PRIMARY, build_results
    rows = [{"id": f"p{i}"} for i in range(4)]
    matrix = {}
    for i in range(4):
        for model in (prereg.REFERENCE_MODEL, prereg.FALLBACK_REFERENCE_MODEL, PRIMARY, CHOSEN):
            matrix[(f"p{i}", model)] = {"model": model, "text": "a", "finish_reason": "stop", "cost_usd": 0.001}
        matrix[(f"p{i}", "gpt-oss-20b")] = {"model": "gpt-oss-20b", "text": "a", "cost_usd": 0.0001,
                                            "finish_reason": "length" if i == 0 else "stop"}
    verdict = lambda pid, tier, ok: {"prompt_id": pid, "tier": tier, "judge": "gpt-4.1-mini", "acceptable": ok, "status": "ok"}
    slice2 = [verdict(f"p{i}", "low", True) for i in range(4)] + [verdict(f"p{i}", "medium", True) for i in range(4)]
    fallback = [verdict(f"p{i}", "fallback-gpt-oss-20b", i != 1) for i in range(1, 4)]
    r = build_results(rows, matrix, slice2, fallback)
    haiku, oss = r["candidates"]
    assert r["primary"]["acceptance"][0] == 1.0 and haiku["acceptance"][0] == 1.0
    assert oss["acceptance"][0] == 0.5 and oss["cut_by_limit"] == 1  # p0 cut, p1 rejected
    assert (oss["paired_vs_primary"]["b"], oss["paired_vs_primary"]["c"]) == (2, 0)
    assert oss["cost_per_1k_usd"] == pytest.approx(0.1)


@pytest.mark.gate
def test_the_committed_fallback_quality_matches_a_fresh_computation():
    import json
    from harness.record import RECORDINGS_PATH, load_records, load_rows
    from harness.replay import current_records
    from harness.routing.fallback_quality import JSON_PATH, build_results
    from harness.routing.judge import FALLBACK_VERDICTS_PATH, VERDICTS_PATH, load_verdicts
    matrix = current_records(load_records(RECORDINGS_PATH), REGISTRY)
    fresh = build_results(load_rows(), matrix, load_verdicts(VERDICTS_PATH), load_verdicts(FALLBACK_VERDICTS_PATH))
    assert json.loads(json.dumps(fresh)) == json.loads(JSON_PATH.read_text(encoding="utf-8"))
