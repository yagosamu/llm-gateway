import asyncio

import fakeredis
import pytest
from fastapi.testclient import TestClient

from llm_gateway.api import create_app
from llm_gateway.breaker import Admission, CircuitBreaker, trip_reason
from llm_gateway.health import BUCKET_SECONDS, HealthSnapshot, HealthTracker
from llm_gateway.providers.base import ProviderError, ProviderResult
from llm_gateway.registry import load_registry
from llm_gateway.routing import BreakerConfig, RoutingConfigError, load_config

REGISTRY = load_registry()
CONFIG = BreakerConfig(error_rate_threshold=0.5, min_calls=4, p95_budget_ms=5_000, open_seconds=15,
                       probe_lease_seconds=70)


class Clock:
    def __init__(self, now=2_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def instance(server, clock):
    store = fakeredis.FakeAsyncRedis(server=server)
    health = HealthTracker(store, clock)
    return health, CircuitBreaker(store, health, config=lambda: CONFIG, clock=clock)


def world():
    server, clock = fakeredis.FakeServer(), Clock()
    return clock, instance(server, clock), instance(server, clock)


async def call(health, breaker, provider, outcome, latency=1.0):
    admission = await breaker.allow(provider)
    if admission.allowed:
        await health.record(provider, outcome, latency if outcome == "ok" else None)
        await breaker.on_result(provider, outcome, admission)
    return admission


def run(coro):
    return asyncio.run(coro)


def snap(**outcomes):
    return HealthSnapshot("p", 60, sum(outcomes.values()), {"ok": 0, **outcomes})


def test_trip_reason_needs_enough_calls_and_ignores_bad_requests():
    assert trip_reason(snap(ok=1, timeout=2), CONFIG) is None  # 3 calls, under min_calls
    assert trip_reason(snap(ok=2, timeout=2), CONFIG) == "2 of 4 calls failed"
    assert trip_reason(snap(ok=4, bad_request=40), CONFIG) is None
    slow = HealthSnapshot("p", 60, 5, {"ok": 5}, 1.0, 1_000, 6_000, 7_000)
    assert "p95 6000 ms over" in trip_reason(slow, CONFIG)


def test_the_circuit_opens_on_failures_and_turns_calls_away():
    clock, (health, breaker), _ = world()

    async def scenario():
        for _ in range(2):
            await call(health, breaker, "openai", "ok")
        for _ in range(2):
            await call(health, breaker, "openai", "timeout")
        return await breaker.state("openai"), await breaker.allow("openai")

    (state, _), admission = run(scenario())
    assert state == "open" and admission == Admission(False, "open")


def test_after_the_open_period_exactly_one_instance_gets_the_probe():
    clock, (h1, b1), (h2, b2) = world()

    async def scenario():
        for _ in range(4):
            await call(h1, b1, "openai", "server_error")
        clock.now += CONFIG.open_seconds
        first, second = await b1.allow("openai"), await b2.allow("openai")
        return first, second, (await b2.state("openai"))[0]

    first, second, state = run(scenario())
    assert first == Admission(True, "half_open", probe=True)
    assert second == Admission(False, "half_open")
    assert state == "half_open"


def test_a_successful_probe_closes_and_old_errors_do_not_reopen_it():
    clock, (health, breaker), _ = world()

    async def scenario():
        for _ in range(4):
            await call(health, breaker, "openai", "server_error")
        clock.now += CONFIG.open_seconds
        probe = await call(health, breaker, "openai", "ok")
        clock.now += BUCKET_SECONDS  # the window still holds the four errors
        await call(health, breaker, "openai", "ok")
        await call(health, breaker, "openai", "ok")
        await call(health, breaker, "openai", "server_error")
        whole_window = await health.snapshot("openai")
        return probe, (await breaker.state("openai"))[0], whole_window

    probe, state, whole_window = run(scenario())
    assert probe.probe and state == "closed"
    # Counted over the whole window (5 errors in 8 calls) the circuit would have reopened at once.
    assert trip_reason(whole_window, CONFIG) == "5 of 8 calls failed"


def test_a_failed_probe_reopens_and_restarts_the_open_period():
    clock, (health, breaker), _ = world()

    async def scenario():
        for _ in range(4):
            await call(health, breaker, "groq", "timeout")
        clock.now += CONFIG.open_seconds
        await call(health, breaker, "groq", "timeout")
        state = (await breaker.state("groq"))[0]
        clock.now += CONFIG.open_seconds - 1
        still_open = await breaker.allow("groq")
        return state, still_open

    state, still_open = run(scenario())
    assert state == "open" and still_open == Admission(False, "open")


def test_once_an_abandoned_probe_lease_is_gone_another_instance_probes():
    """fakeredis does not advance TTLs with a fake clock, so the lease's expiry is simulated by deleting
    it; that the lease carries a TTL at all is checked against the real Redis."""
    clock, (h1, b1), (h2, b2) = world()

    async def scenario():
        for _ in range(4):
            await call(h1, b1, "anthropic", "server_error")
        clock.now += CONFIG.open_seconds
        await b1.allow("anthropic")  # takes the probe, then this instance "dies"
        store = b2.redis
        await store.delete("llmgw:breaker:anthropic:probe")  # fakeredis does not advance real TTLs
        return await b2.allow("anthropic")

    assert run(scenario()).probe


def test_providers_have_independent_circuits():
    clock, (health, breaker), _ = world()

    async def scenario():
        for _ in range(4):
            await call(health, breaker, "openai", "timeout")
        return await breaker.allow("groq")

    assert run(scenario()) == Admission(True, "closed")


class Provider:
    def __init__(self, fail_kind=None):
        self.fail_kind, self.calls = fail_kind, 0

    async def complete(self, model, request):
        self.calls += 1
        if self.fail_kind:
            raise ProviderError(self.fail_kind, model.provider, "boom")
        return ProviderResult("ok", 10, 5, "stop")


def gateway(provider):
    clock = Clock()
    health, breaker = instance(fakeredis.FakeServer(), clock)
    app = create_app(REGISTRY, {p: provider for p in ("openai", "anthropic", "groq")}, health=health, breaker=breaker)
    return TestClient(app)


def ask(client):
    return client.post("/v1/chat/completions", json={"model": "gpt-6-luna", "messages": [{"role": "user", "content": "hi"}]},
                       headers={"X-Tenant-Id": "t", "X-Feature": "f", "X-Request-Id": "r"})


def test_an_open_circuit_fails_fast_without_calling_the_provider():
    provider = Provider(fail_kind="server_error")
    client = gateway(provider)
    codes = [ask(client).json()["error"]["code"] for _ in range(6)]
    assert codes[:4] == ["upstream_server_error"] * 4
    assert codes[4:] == ["upstream_circuit_open"] * 2
    assert provider.calls == 4
    body = client.get("/v1/health").json()
    assert body["providers"]["openai"]["circuit"] == "open" and body["providers"]["groq"]["circuit"] == "closed"
    text = client.get("/metrics").text
    assert 'llm_gateway_breaker_state{provider="openai"} 2.0' in text
    assert 'llm_gateway_breaker_transitions_total{from_state="closed",provider="openai",to_state="open"} 1.0' in text


def test_breaker_settings_are_validated(tmp_path):
    path = tmp_path / "routing.yaml"
    path.write_text('version: "t"\npolicy: always_low\ntier_map:\n  low: gpt-6-luna\n  medium: claude-haiku-4-5\n'
                    "  high: gpt-6.1-sol\nbreaker:\n  error_rate: 0.5\n", encoding="utf-8")
    with pytest.raises(RoutingConfigError, match="unknown keys"):
        load_config(path, set(REGISTRY))


def test_the_committed_breaker_config_sits_above_the_recorded_p95():
    from llm_gateway.routing import DEFAULT_CONFIG_PATH
    breaker = load_config(DEFAULT_CONFIG_PATH, set(REGISTRY)).breaker
    assert breaker.p95_budget_ms > 16_500 and breaker.probe_lease_seconds > 60
