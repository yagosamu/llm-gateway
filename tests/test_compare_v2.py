import json

import pytest

from harness.chaos import prereg_v2
from harness.chaos.compare_v2 import JSON_PATH, build, criteria, load_study, opens_total


def metrics(open_s=None, close_s=None, reopens=None, error=0.0, p95=5.0, wasted=10, after=0.0):
    return {"time_to_open_s": open_s, "time_to_close_s": close_s, "reopens_after_close": reopens,
            "error_rate": error, "latency_p95_s": p95, "wasted_calls": wasted, "error_rate_after": after,
            "latency_p50_s": 2.0, "fallback_share": 1.0, "requests_in_fault": 160}


def world(**overrides):
    """Three repetitions of every scenario for both versions, where v2 passes everything; overrides
    replace v2's metrics or openings in one scenario: {scenario: (metrics, opens_total)}."""
    good_v1 = {"outage": (metrics(20, 13, 0, wasted=80), 1), "timeouts": (metrics(30, 9, 0, wasted=120), 1),
               "latency": (metrics(37.5, 14.5, 1, wasted=0), 2), "partial": (metrics(wasted=82), 0),
               "healthy": (metrics(wasted=0), 0), "mild": (metrics(wasted=16), 0)}
    good_v2 = {"outage": (metrics(2.5, 13, 0, wasted=10), 1), "timeouts": (metrics(12.5, 9, 0, wasted=50), 1),
               "latency": (metrics(32.5, 14, 0, wasted=0), 1), "partial": (metrics(5, 3, 0, wasted=30), 3),
               "healthy": (metrics(wasted=0), 0), "mild": (metrics(wasted=16), 0)}
    good_v2 |= overrides
    runs = []
    for scenario in prereg_v2.SCENARIOS:
        for config, table in (("breaker_v1", good_v1), ("breaker_v2", good_v2)):
            m, opens = table[scenario]
            runs += [{"scenario": scenario, "config": config, "rep": rep, "metrics": m, "opens_total": opens}
                     for rep in (1, 2, 3)]
    return runs


def failing(runs):
    return {(c["id"], c["scenario"]) for c in criteria(runs) if not c["pass"]}


def test_a_v2_that_passes_everything_is_called_better():
    assert failing(world()) == set()
    assert build(world())["v2_better"] is True


@pytest.mark.parametrize("override, expected", [
    ({"outage": (metrics(25, 13, 0, wasted=10), 1)}, ("C1", "outage")),  # slower than v1
    ({"partial": (metrics(wasted=82), 0)}, ("C2", "partial")),  # never opens at 50%
    ({"latency": (metrics(32.5, 14, 1, wasted=0), 2)}, ("C3", "latency")),  # reopens after closing
    ({"timeouts": (metrics(12.5, None, None, wasted=50), 1)}, ("C3", "timeouts")),  # never closes again
    ({"mild": (metrics(wasted=16), 1)}, ("C4", "mild")),  # a false trip
    ({"outage": (metrics(2.5, 13, 0, error=0.02, wasted=10), 1)}, ("C5", "outage")),  # more client errors
])
def test_each_criterion_can_fail(override, expected):
    assert expected in failing(world(**override))
    assert build(world(**override))["v2_better"] is False


def test_openings_are_counted_over_the_whole_run():
    timeline = [{"t": 1, "state": "closed"}, {"t": 2, "state": "open"}, {"t": 3, "state": "open"},
                {"t": 4, "state": "half_open"}, {"t": 5, "state": "open"}, {"t": 6, "state": "closed"}]
    assert opens_total(timeline) == 2


@pytest.mark.gate
def test_the_committed_v2_verdict_matches_the_committed_runs():
    if not JSON_PATH.exists():
        pytest.skip("no v2 study committed yet")
    assert json.loads(json.dumps(build(load_study()))) == json.loads(JSON_PATH.read_text(encoding="utf-8"))
