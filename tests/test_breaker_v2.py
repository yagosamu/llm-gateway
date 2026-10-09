"""Deterministic checks of each v2 mechanism, several against v1 on the same call sequence."""
import asyncio

import fakeredis
import pytest

from llm_gateway.breaker import Admission, CircuitBreaker, count_trip_reason
from llm_gateway.health import HealthTracker
from llm_gateway.routing import BreakerConfig, RoutingConfigError, load_config

V1 = BreakerConfig(mode="time_window", min_calls=10, error_rate_threshold=0.5, open_seconds=15)
V2 = BreakerConfig(mode="count_window", min_calls=10, error_rate_threshold=0.5, open_seconds=15, window_calls=20,
                   slow_call_seconds=30)


class Clock:
    def __init__(self, now=3_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def make(config, server=None, clock=None):
    server, clock = server or fakeredis.FakeServer(), clock or Clock()
    store = fakeredis.FakeAsyncRedis(server=server)
    health = HealthTracker(store, clock)
    return health, CircuitBreaker(store, health, config=lambda: config, clock=clock), clock


async def call(health, breaker, outcome, provider="openai", duration=1.0):
    admission = await breaker.allow(provider)
    if admission.allowed:
        watch = breaker.watch(provider, admission)
        await health.record(provider, outcome, duration if outcome == "ok" else None)
        await breaker.on_result(provider, outcome, admission, duration, watch)
    return admission


def run(coro):
    return asyncio.run(coro)


def test_v2_detects_an_outage_after_healthy_traffic_where_v1_waits():
    """80 healthy calls, then failures: v2 opens on the 10th failure, v1 needs as many failures as
    the window holds successes."""
    async def failures_until_open(config):
        health, breaker, _ = make(config)
        for _ in range(80):
            await call(health, breaker, "ok")
        for n in range(1, 200):
            await call(health, breaker, "server_error")
            if (await breaker.state("openai"))[0] == "open":
                return n
        return None

    assert run(failures_until_open(V2)) == 10
    assert run(failures_until_open(V1)) == 80


def test_v2_opens_at_a_50_percent_error_rate_and_v1_does_not():
    async def scenario(config):
        health, breaker, _ = make(config)
        for _ in range(40):
            await call(health, breaker, "ok")
        for i in range(40):
            await call(health, breaker, "server_error" if i % 2 else "ok")
        return (await breaker.state("openai"))[0]

    assert run(scenario(V2)) == "open"
    assert run(scenario(V1)) == "closed"


def test_v2_never_opens_at_a_10_percent_error_rate():
    async def scenario():
        health, breaker, _ = make(V2)
        for i in range(300):
            await call(health, breaker, "server_error" if i % 10 == 0 else "ok")
        return (await breaker.state("openai"))[0]

    assert run(scenario()) == "closed"


def test_bad_requests_never_count():
    async def scenario():
        health, breaker, _ = make(V2)
        for _ in range(30):
            await call(health, breaker, "bad_request")
        return (await breaker.state("openai"))[0]

    assert run(scenario()) == "closed"


def test_a_straggler_from_before_the_close_cannot_reopen_the_circuit():
    """The latency runs' defect: calls admitted during the fault finish after the circuit closed. In
    v2 their outcome belongs to an older epoch and is ignored."""
    async def scenario():
        health, breaker, clock = make(V2)
        stragglers = [await breaker.allow("openai") for _ in range(15)]  # admitted while closed
        for _ in range(10):
            await call(health, breaker, "timeout")  # the circuit opens
        clock.now += V2.open_seconds
        await call(health, breaker, "ok")  # the probe closes it
        for admission in stragglers:  # the slow calls admitted before all that finally fail
            await breaker.on_result("openai", "timeout", admission, 40.0, None)
        return (await breaker.state("openai"))[0]

    assert run(scenario()) == "closed"


def test_the_same_stragglers_do_reopen_v1():
    async def scenario():
        health, breaker, clock = make(V1)
        stragglers = [await breaker.allow("openai") for _ in range(15)]
        for _ in range(10):
            await call(health, breaker, "timeout")
        clock.now += V1.open_seconds
        await call(health, breaker, "ok")
        clock.now += 10  # the next health bucket, which v1 counts after the close
        for admission in stragglers:
            await health.record("openai", "timeout")
            await breaker.on_result("openai", "timeout", admission)
        return (await breaker.state("openai"))[0]

    assert run(scenario()) == "open"


def test_a_call_still_running_past_the_slow_threshold_is_listed_while_it_runs():
    config = BreakerConfig(mode="count_window", min_calls=2, window_calls=4, slow_call_seconds=0.05)

    async def scenario():
        health, breaker, _ = make(config)
        admissions = [await breaker.allow("openai") for _ in range(2)]
        watches = [breaker.watch("openai", a) for a in admissions]
        await asyncio.sleep(0.15)  # both calls are still "running"
        opened_while_running = (await breaker.state("openai"))[0]
        for a, w in zip(admissions, watches):  # they finally return, slowly but successfully
            await breaker.on_result("openai", "ok", a, 0.2, w)
        entries = await breaker.redis.lrange("llmgw:breaker:openai:calls", 0, -1)
        return opened_while_running, entries

    state, entries = run(scenario())
    assert state == "open"
    assert entries == []  # the transition emptied the list, and the late returns were not listed twice


def test_a_slow_success_that_beat_its_timer_still_counts_as_slow():
    assert count_trip_reason(["x"] * 10, V2) == "10 of the last 10 calls ran past 30 s"


def test_a_probe_that_runs_too_long_reopens_the_circuit():
    config = BreakerConfig(mode="count_window", min_calls=2, window_calls=4, slow_call_seconds=0.05, open_seconds=15)

    async def scenario():
        health, breaker, clock = make(config)
        for _ in range(2):
            await call(health, breaker, "server_error")
        clock.now += config.open_seconds
        probe = await breaker.allow("openai")
        watch = breaker.watch("openai", probe)
        await asyncio.sleep(0.15)
        state = (await breaker.state("openai"))[0]
        await breaker.on_result("openai", "ok", probe, 0.2, watch)  # returns long after
        return probe.probe, state, (await breaker.state("openai"))[0]

    is_probe, during, after = run(scenario())
    assert is_probe and during == "open" and after == "open"


def test_two_instances_share_the_outcome_list_and_the_epoch():
    async def scenario():
        server, clock = fakeredis.FakeServer(), Clock()
        h1, b1, _ = make(V2, server, clock)
        h2, b2, _ = make(V2, server, clock)
        for i in range(10):
            health, breaker = (h1, b1) if i % 2 else (h2, b2)
            await call(health, breaker, "connection")
        return (await b1.snapshot("openai")), (await b2.allow("openai"))

    snap, admission = run(scenario())
    assert snap["state"] == "open" and snap["epoch"] == 1
    assert admission == Admission(False, "open") and admission.epoch == 1


def test_every_transition_starts_a_new_epoch_and_empties_the_list():
    async def scenario():
        health, breaker, clock = make(V2)
        for _ in range(10):
            await call(health, breaker, "server_error")  # closed -> open: epoch 1
        clock.now += V2.open_seconds
        await call(health, breaker, "ok")  # open -> half_open (2) -> closed (3)
        await call(health, breaker, "ok")
        return await breaker.snapshot("openai"), await breaker.redis.lrange("llmgw:breaker:openai:calls", 0, -1)

    snap, entries = run(scenario())
    assert snap["state"] == "closed" and snap["epoch"] == 3
    assert entries == [b"s"]  # only the call admitted after the close


def test_v2_settings_are_validated(tmp_path):
    path = tmp_path / "routing.yaml"
    base = 'version: "t"\npolicy: always_low\ntier_map:\n  low: gpt-6-luna\n  medium: claude-haiku-4-5\n  high: gpt-6.1-sol\n'
    from llm_gateway.registry import load_registry
    ids = set(load_registry())
    path.write_text(base + "breaker:\n  mode: sliding\n", encoding="utf-8")
    with pytest.raises(RoutingConfigError, match="mode"):
        load_config(path, ids)
    path.write_text(base + "breaker:\n  mode: count_window\n  window_calls: 5\n  min_calls: 10\n", encoding="utf-8")
    with pytest.raises(RoutingConfigError, match="window_calls"):
        load_config(path, ids)
