import pytest

from harness.chaos import live
from llm_gateway.registry import call_ceiling_usd, load_registry


def test_the_worst_case_bills_every_request_on_every_model_it_could_reach():
    registry, rows = load_registry(), live.planned_rows()
    expected = sum(call_ceiling_usd(registry[m], r["messages"], live.MAX_TOKENS)
                   for r in rows for m in ("gpt-6-luna", "claude-haiku-4-5"))
    assert len(rows) == 90
    assert live.worst_case_usd(rows) == pytest.approx(expected)


def test_the_planned_run_fits_its_cap():
    assert live.worst_case_usd(live.planned_rows()) <= live.CAP_USD


def test_the_summary_splits_requests_around_the_fault():
    results = [{"sent": 5, "status": 200, "model": "gpt-6-luna"},
               {"sent": 30, "status": 200, "model": "claude-haiku-4-5"},
               {"sent": 40, "status": 503, "error": "upstream_circuit_open"},
               {"sent": 70, "status": 200, "model": "gpt-6-luna"}]
    log = [{"failover_from": None, "finish_reason": "stop"}, {"failover_from": "gpt-6-luna", "finish_reason": "length"}]
    summary = live.summarize(results, log, 0.01)
    assert summary["by_window"]["fault"] == {"requests": 2, "answered": 1, "answered_by": {"claude-haiku-4-5": 1}}
    assert summary["errors"] == {"upstream_circuit_open": 1}
    assert (summary["failovers"], summary["truncated"], summary["answered"]) == (1, 1, 3)
