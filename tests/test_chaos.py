import asyncio
import time

import fakeredis
import httpx2
import pytest
from fastapi.testclient import TestClient

from harness.chaos.load import generate
from harness.chaos.simulated import SimulatedProvider
from harness.record import RECORDINGS_PATH, load_records, load_rows
from harness.replay import ReplayMiss, current_records
from llm_gateway.api import create_app
from llm_gateway.breaker import Admission, CircuitBreaker
from llm_gateway.chaos import ChaosConfigError, ChaosControl, ChaosProvider, validate
from llm_gateway.health import HealthTracker
from llm_gateway.providers.base import ProviderError, ProviderResult
from llm_gateway.registry import load_registry
from llm_gateway.routing import DEFAULT_CONFIG_PATH, BreakerConfig, Router
from llm_gateway.schemas import ChatCompletionRequest

REGISTRY = load_registry()
ROWS = load_rows()
MATRIX = current_records(load_records(RECORDINGS_PATH), REGISTRY)
PROVIDERS = sorted({m.provider for m in REGISTRY.values()})


def chat(row):
    return ChatCompletionRequest.model_validate({"model": "auto", "messages": row["messages"]})


class Answer:
    async def complete(self, model, request):
        return ProviderResult("fine", 1, 1, "stop")


def test_a_fault_is_validated():
    assert validate({"error_rate": 1, "kind": "timeout"})["timeout_seconds"] == 10.0
    for bad in ({"error_rate": 2}, {"kind": "meteor"}, {"extra_latency_ms": -1}, {"color": "red"}):
        with pytest.raises(ChaosConfigError):
            validate(bad)


def test_faults_are_shared_through_redis_and_cleared():
    server = fakeredis.FakeServer()
    a, b = ChaosControl(fakeredis.FakeAsyncRedis(server=server)), ChaosControl(fakeredis.FakeAsyncRedis(server=server))

    async def scenario():
        await a.set("openai", {"error_rate": 0.5, "kind": "server_error"})
        seen = await b.get("openai")
        await b.clear()
        return seen, await a.get("openai")

    seen, after = asyncio.run(scenario())
    assert seen["error_rate"] == 0.5 and seen["kind"] == "server_error" and after is None


def test_the_wrapper_injects_errors_timeouts_and_latency_only_when_told():
    control = ChaosControl(fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer()))
    provider = ChaosProvider(Answer(), "openai", control)
    model = REGISTRY["gpt-6-luna"]

    async def scenario():
        clean = await provider.complete(model, None)
        await control.set("openai", {"error_rate": 1, "kind": "rate_limit"})
        with pytest.raises(ProviderError) as limited:
            await provider.complete(model, None)
        await control.set("openai", {"error_rate": 1, "kind": "timeout", "timeout_seconds": 0.05})
        started = time.perf_counter()
        with pytest.raises(ProviderError) as timed_out:
            await provider.complete(model, None)
        hung = time.perf_counter() - started
        await control.set("openai", {"error_rate": 0, "extra_latency_ms": 50})
        started = time.perf_counter()
        slow = await provider.complete(model, None)
        return clean, limited.value.kind, timed_out.value.kind, hung, slow, time.perf_counter() - started

    clean, limited, timed_out, hung, slow, slow_time = asyncio.run(scenario())
    assert clean.text == "fine" and slow.text == "fine"
    assert (limited, timed_out) == ("rate_limit", "timeout")
    assert hung >= 0.05 and slow_time >= 0.05


def test_the_simulated_provider_answers_from_the_recording():
    provider = SimulatedProvider(MATRIX, ROWS, latency_scale=0)
    row = ROWS[0]
    result = asyncio.run(provider.complete(REGISTRY["gpt-6-luna"], chat(row)))
    assert result.text == MATRIX[(row["id"], "gpt-6-luna")]["text"]
    unknown = ChatCompletionRequest.model_validate({"model": "auto", "messages": [{"role": "user", "content": "new"}]})
    with pytest.raises(ReplayMiss):
        asyncio.run(provider.complete(REGISTRY["gpt-6-luna"], unknown))


def chaos_app(token="secret", breaker_config=None):
    server = fakeredis.FakeServer()
    store = fakeredis.FakeAsyncRedis(server=server)
    control = ChaosControl(store)
    health = HealthTracker(store)
    router = Router(DEFAULT_CONFIG_PATH, set(REGISTRY))
    breaker = CircuitBreaker(store, health, config=lambda: breaker_config or router.config.breaker)
    simulated = SimulatedProvider(MATRIX, ROWS, latency_scale=0)
    app = create_app(REGISTRY, {p: ChaosProvider(simulated, p, control) for p in PROVIDERS}, router=router,
                     health=health, breaker=breaker, chaos=control, admin_token=token)
    return app, control


def test_admin_endpoints_need_the_token_and_exist_only_with_chaos():
    plain = create_app(REGISTRY, {p: Answer() for p in PROVIDERS})
    assert TestClient(plain).get("/admin/chaos").status_code == 404
    app, _ = chaos_app()
    client = TestClient(app)
    assert client.get("/admin/chaos").status_code == 403
    assert client.get("/admin/chaos", headers={"X-Admin-Token": "wrong"}).status_code == 403
    ok = {"X-Admin-Token": "secret"}
    assert client.post("/admin/chaos/openai", json={"error_rate": 1}, headers=ok).status_code == 200
    assert client.get("/admin/chaos", headers=ok).json()["openai"]["error_rate"] == 1.0
    assert client.post("/admin/chaos/openai", json={"error_rate": 7}, headers=ok).status_code == 400
    assert client.post("/admin/chaos/nobody", json={}, headers=ok).status_code == 404
    client.delete("/admin/chaos", headers=ok)
    assert client.get("/admin/chaos", headers=ok).json()["openai"] is None


def test_a_disabled_breaker_lets_everything_through_and_never_moves():
    store = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())
    health = HealthTracker(store)
    breaker = CircuitBreaker(store, health, config=lambda: BreakerConfig(enabled=False, min_calls=1))

    async def scenario():
        for _ in range(5):
            admission = await breaker.allow("openai")
            await health.record("openai", "server_error")
            await breaker.on_result("openai", "server_error", admission)
        return admission, (await breaker.state("openai"))[0]

    assert asyncio.run(scenario()) == (Admission(True, "disabled"), "closed")


def test_the_load_generator_reports_failover_under_an_injected_outage():
    app, control = chaos_app()
    asyncio.run(control.set("openai", {"error_rate": 1, "kind": "server_error"}))
    transport = httpx2.ASGITransport(app=app)
    results = asyncio.run(generate(["http://gateway"], rate=100, duration=0.1, rows=ROWS[:5], transport=transport))
    assert len(results) == 10 and all(r["status"] == 200 for r in results)
    assert {r["model"] for r in results} == {"claude-haiku-4-5"}  # always_low's fallback
    assert results[0]["attempts"] == ["server_error", "ok"]
    assert all(r["sent"] <= r["done"] for r in results)
