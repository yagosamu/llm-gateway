import json

import pytest

from harness.chaos import prereg
from harness.chaos.analyze import JSON_PATH, build_results, load_runs, run_metrics, summarize
from harness.chaos.scenarios import routing_for
from llm_gateway.registry import load_registry


def req(sent, done, status=200, model=prereg.PRIMARY_MODEL, attempts=("ok",), error=None):
    r = {"sent": sent, "done": done, "status": status}
    if status == 200:
        r |= {"model": model, "attempts": list(attempts)}
    else:
        r["error"] = error
    return r


def test_metrics_count_the_fault_window_only():
    requests = [
        req(5, 6),  # warmup, ignored
        req(21, 23, model="claude-haiku-4-5", attempts=("server_error", "ok")),  # wasted, served by fallback
        req(22, 23, model="claude-haiku-4-5", attempts=("circuit_open", "ok")),  # skipped openai: not wasted
        req(30, 40, status=502, error="upstream_server_error"),  # client error, wasted
        req(31, 31.5, status=503, error="upstream_circuit_open"),  # client error, not wasted
        req(61, 62), req(62, 63, status=502, error="upstream_server_error"),  # after recovery
    ]
    timeline = [{"t": 10, "state": "closed"}, {"t": 24.5, "state": "open"}, {"t": 70, "state": "half_open"},
                {"t": 75.5, "state": "closed"}]
    m = run_metrics(requests, timeline)
    assert m["requests_in_fault"] == 4 and m["error_rate"] == 0.5
    assert m["wasted_calls"] == 2 and m["fallback_share"] == 1.0
    assert m["latency_p50_s"] == pytest.approx(1.5)
    assert (m["time_to_open_s"], m["time_to_close_s"]) == (4.5, 15.5)
    assert m["error_rate_after"] == 0.5


def test_a_circuit_that_never_opened_has_no_open_or_close_time():
    m = run_metrics([req(25, 26)], [{"t": 30, "state": "closed"}, {"t": 70, "state": "closed"}])
    assert (m["time_to_open_s"], m["time_to_close_s"]) == (None, None)


def test_summaries_ignore_missing_values_and_say_how_many():
    assert summarize([3.0, None, 1.0]) == {"median": 2.0, "min": 1.0, "max": 3.0, "missing": 1}
    assert summarize([None, None])["median"] is None


def test_results_group_runs_by_scenario_and_configuration():
    metrics = run_metrics([req(25, 26)], [])
    runs = [{"scenario": "outage", "config": c, "rep": r, "metrics": metrics} for c in ("none", "failover") for r in (1, 2)]
    results = build_results(runs)
    assert set(results["results"]["outage"]) == {"none", "failover"}
    assert results["results"]["outage"]["none"]["reps"] == 2


def test_configurations_switch_only_failover_and_the_breaker():
    registry = load_registry()
    for config in prereg.CONFIGS:
        data = routing_for(config)
        assert data["policy"] == "always_low"
        assert data["breaker"]["enabled"] is (config == "failover_breaker")
        assert bool(data["failover"]["fallbacks"]) is (config != "none")
    path_free = routing_for("failover_breaker")
    assert path_free["failover"]["fallbacks"]["low"] == ["claude-haiku-4-5"]
    assert set(registry) >= set(path_free["tier_map"].values())


def test_every_scenario_fault_is_valid():
    from llm_gateway.chaos import validate
    for fault in prereg.SCENARIOS.values():
        validate(fault)


@pytest.mark.gate
def test_the_committed_slice3_results_match_the_committed_runs():
    if not JSON_PATH.exists():
        pytest.skip("no chaos results committed yet")
    from harness.chaos.analyze import load_notes
    fresh = json.loads(json.dumps(build_results(load_runs(), load_notes())))
    assert fresh == json.loads(JSON_PATH.read_text(encoding="utf-8"))


def test_a_circuit_that_opens_again_after_closing_is_counted():
    timeline = [{"t": 30, "state": "open"}, {"t": 70, "state": "half_open"}, {"t": 72, "state": "closed"},
                {"t": 78, "state": "open"}, {"t": 79, "state": "open"}, {"t": 93, "state": "half_open"},
                {"t": 95, "state": "closed"}]
    m = run_metrics([req(25, 26)], timeline)
    assert (m["time_to_close_s"], m["reopens_after_close"]) == (12.0, 1)
