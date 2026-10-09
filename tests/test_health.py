import asyncio

import fakeredis
import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from llm_gateway.api import create_app
from llm_gateway.health import BUCKET_SECONDS, WINDOW_SECONDS, HealthTracker, latency_field, quantile
from llm_gateway.metrics import GatewayMetrics
from llm_gateway.providers.base import ProviderError, ProviderResult
from llm_gateway.registry import load_registry

REGISTRY = load_registry()


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def tracker(server=None, clock=None):
    server = server or fakeredis.FakeServer()
    return HealthTracker(fakeredis.FakeAsyncRedis(server=server), clock or Clock()), server


def run(coro):
    return asyncio.run(coro)


def test_a_snapshot_sums_outcomes_and_the_success_rate():
    health, _ = tracker()

    async def scenario():
        for _ in range(8):
            await health.record("openai", "ok", 1.2)
        await health.record("openai", "timeout")
        await health.record("openai", "server_error")
        return await health.snapshot("openai")

    snap = run(scenario())
    assert (snap.calls, snap.outcomes["ok"], snap.outcomes["timeout"], snap.outcomes["server_error"]) == (10, 8, 1, 1)
    assert snap.success_rate == pytest.approx(0.8)
    assert 1000 < snap.p50_ms <= 1500


def test_events_older_than_the_window_drop_out():
    clock = Clock()
    health, _ = tracker(clock=clock)

    async def scenario():
        await health.record("groq", "server_error")
        clock.now += WINDOW_SECONDS + BUCKET_SECONDS
        await health.record("groq", "ok", 0.4)
        return await health.snapshot("groq")

    snap = run(scenario())
    assert (snap.calls, snap.success_rate) == (1, 1.0)


def test_an_event_stays_visible_for_at_least_window_minus_one_bucket():
    clock = Clock(1_000_005.0)
    health, _ = tracker(clock=clock)

    async def scenario():
        await health.record("groq", "timeout")
        clock.now += WINDOW_SECONDS - BUCKET_SECONDS
        return await health.snapshot("groq")

    assert run(scenario()).calls == 1


def test_two_gateway_instances_share_one_view():
    server, clock = fakeredis.FakeServer(), Clock()
    first, _ = tracker(server, clock)
    second, _ = tracker(server, clock)

    async def scenario():
        await first.record("anthropic", "ok", 2.0)
        await second.record("anthropic", "rate_limit")
        return await first.snapshot("anthropic"), await second.snapshot("anthropic")

    a, b = run(scenario())
    assert a == b and a.calls == 2 and a.outcomes["rate_limit"] == 1


def test_an_empty_window_reports_no_rate_and_no_latency():
    health, _ = tracker()
    snap = run(health.snapshot("openai"))
    assert (snap.calls, snap.success_rate, snap.p95_ms) == (0, None, None)


def test_quantiles_interpolate_inside_the_histogram_bucket():
    counts = {latency_field(0.6): 10}  # all in (0.5, 0.75]
    assert quantile(counts, 0.5) == pytest.approx(625.0)
    assert quantile({latency_field(100.0): 3}, 0.99) == pytest.approx(64_000.0)  # open bucket: lower edge
    assert quantile({}, 0.5) is None


def test_an_unknown_outcome_is_refused():
    health, _ = tracker()
    with pytest.raises(ValueError):
        run(health.record("openai", "teapot"))


class BrokenRedis:
    def pipeline(self, transaction=False):
        raise RedisConnectionError("redis is down")


def test_a_write_to_an_unreachable_redis_is_counted_not_raised():
    metrics = GatewayMetrics()
    health = HealthTracker(BrokenRedis(), Clock(), metrics)
    run(health.record("openai", "ok", 1.0))
    assert metrics.health_write_errors._value.get() == 1


class Provider:
    def __init__(self, fail_kind=None):
        self.fail_kind = fail_kind

    async def complete(self, model, request):
        if self.fail_kind:
            raise ProviderError(self.fail_kind, model.provider, "boom")
        return ProviderResult("ok", 10, 5, "stop")


def gateway(fail_kind=None, health=None):
    health = health or tracker()[0]
    app = create_app(REGISTRY, {p: Provider(fail_kind) for p in ("openai", "anthropic", "groq")}, health=health)
    return TestClient(app), health


def ask(client, model="gpt-6-luna"):
    return client.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
                       headers={"X-Tenant-Id": "t", "X-Feature": "f", "X-Request-Id": "r"})


def test_the_gateway_records_successes_and_failures_per_provider():
    client, _ = gateway()
    ask(client)
    ask(client, "gpt-oss-20b")
    body = client.get("/v1/health").json()
    assert body["window_seconds"] == WINDOW_SECONDS
    assert body["providers"]["openai"]["outcomes"]["ok"] == 1 and body["providers"]["groq"]["calls"] == 1
    assert body["providers"]["anthropic"]["calls"] == 0

    failing, _ = gateway(fail_kind="timeout")
    ask(failing)
    assert failing.get("/v1/health").json()["providers"]["openai"]["outcomes"]["timeout"] == 1


def test_a_request_is_served_when_redis_is_down():
    client, _ = gateway(health=HealthTracker(BrokenRedis(), Clock()))
    assert ask(client).status_code == 200
    assert client.get("/v1/health").status_code == 503
    assert "llm_gateway_health_write_errors_total 1.0" in client.get("/metrics").text


def test_metrics_expose_the_shared_window():
    client, _ = gateway()
    ask(client)
    text = client.get("/metrics").text
    assert 'llm_gateway_provider_success_rate{provider="openai"} 1.0' in text
    assert 'llm_gateway_provider_window_calls{outcome="ok",provider="openai"} 1.0' in text
    assert 'llm_gateway_provider_window_latency_ms{provider="openai",quantile="0.95"}' in text


def test_without_a_health_store_the_endpoint_says_so():
    app = create_app(REGISTRY, {p: Provider() for p in ("openai", "anthropic", "groq")})
    assert TestClient(app).get("/v1/health").json()["error"]["code"] == "health_disabled"
