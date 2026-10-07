import asyncio
import json

import pytest

from harness.record import OUTPUT_LIMIT, params_for
from harness.replay import ReplayMiss, current_records, replay, require_complete
from harness.report import JSON_PATH, build_results, render_markdown
from llm_gateway.registry import cost_usd, load_registry

REGISTRY = load_registry()
CATEGORIES = ["closed_qa", "open_qa"]


def make_rows(n_per_category=4):
    return [{"id": f"dolly-{c}{i:03d}".replace("_", ""), "category": c, "tenant": f"tenant-{'abc'[i % 3]}",
             "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": f"{c} question {i}"}]}
            for c in CATEGORIES for i in range(n_per_category)]


def make_record(row, model, i, status="ok", params=None, finish="stop"):
    input_tokens, output_tokens = 20 + i, 100 + 10 * i
    return {"prompt_id": row["id"], "model": model.id, "status": status, "params": params or params_for(model),
            "text": f"{model.id} on {row['id']}", "input_tokens": input_tokens, "output_tokens": output_tokens,
            "finish_reason": finish, "latency_ms": 500.0 + 37 * i,
            "cost_usd": cost_usd(model, input_tokens, output_tokens)}


def make_matrix(rows):
    records = [make_record(row, m, i, finish="length" if i == 0 and m.id == "gpt-oss-20b" else "stop")
               for i, row in enumerate(rows) for m in REGISTRY.values()]
    return current_records(records, REGISTRY)


def test_current_records_keeps_only_successes_made_with_the_current_parameters():
    rows = make_rows(1)
    luna = REGISTRY["gpt-6-luna"]
    old = make_record(rows[0], luna, 0, params={"reasoning_effort": "low", "max_completion_tokens": 512})
    failed = make_record(rows[0], luna, 0, status="error")
    assert current_records([old, failed], REGISTRY) == {}
    good = make_record(rows[0], luna, 1)
    assert current_records([good, failed], REGISTRY) == {(rows[0]["id"], "gpt-6-luna"): good}


def test_an_incomplete_recording_is_refused_before_replay():
    rows = make_rows(1)
    matrix = make_matrix(rows)
    del matrix[(rows[0]["id"], "gpt-6-luna")]
    with pytest.raises(ReplayMiss, match="lacks 1 prompt-model pairs"):
        require_complete(matrix, rows, REGISTRY)


def test_replay_returns_recorded_answers_and_logs_every_request_with_metadata():
    rows = make_rows()
    log_rows, overhead = asyncio.run(replay(rows, make_matrix(rows), REGISTRY))
    assert len(log_rows) == len(overhead) == len(rows) * len(REGISTRY)
    assert all(r["outcome"] == "ok" and r["tenant"] and r["feature"] and r["request_id"] for r in log_rows)


def test_a_prompt_the_recording_does_not_hold_is_a_loud_miss():
    rows = make_rows(1)
    matrix = make_matrix(rows)
    unknown = [{**rows[0], "id": "dolly-99999", "messages": [{"role": "user", "content": "never recorded"}]}]
    with pytest.raises(ReplayMiss):
        asyncio.run(replay(unknown, matrix, REGISTRY))


def test_results_are_deterministic_and_report_cost_truncation_and_attribution():
    rows = make_rows()
    matrix = make_matrix(rows)
    log_rows, overhead = asyncio.run(replay(rows, matrix, REGISTRY))
    first, second = build_results(rows, matrix, log_rows, REGISTRY), build_results(rows, matrix, log_rows, REGISTRY)
    assert first == second
    luna = REGISTRY["gpt-6-luna"]
    expected = 1000 * sum(cost_usd(luna, 20 + i, 100 + 10 * i) for i in range(len(rows))) / len(rows)
    assert first["models"]["gpt-6-luna"]["cost_per_1k_usd"][0] == pytest.approx(expected, abs=1e-5)
    low, high = first["models"]["gpt-6-luna"]["cost_per_1k_usd"][1:]
    assert low <= expected <= high
    assert first["models"]["gpt-oss-20b"]["truncated"] == 1
    assert first["truncated_by_category"]["gpt-oss-20b"] == {"closed_qa": 1}
    assert first["attribution"]["with_tenant_feature_and_request_id"] == len(rows) * len(REGISTRY)
    text = render_markdown(first, overhead, spent_usd=1.23)
    assert "per 1,000" in text and "US$ 1.23 of the US$ 10" in text


def test_a_replayed_cost_that_differs_from_the_recorded_one_is_refused():
    rows = make_rows(1)
    matrix = make_matrix(rows)
    log_rows, _ = asyncio.run(replay(rows, matrix, REGISTRY))
    log_rows[0]["cost_usd"] += 0.01
    with pytest.raises(ValueError, match="differs"):
        build_results(rows, matrix, log_rows, REGISTRY)


@pytest.mark.gate
def test_the_committed_results_match_a_fresh_offline_replay():
    """The published numbers are regenerated from the committed traffic and recording, offline. A
    price change, a code change that alters cost or a recording edit fails here until
    `python -m harness.report` is run and its output committed."""
    from harness.replay import run_replay
    rows, matrix, log_rows, _ = run_replay()
    fresh = json.loads(json.dumps(build_results(rows, matrix, log_rows, REGISTRY)))
    assert fresh == json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert fresh["models"]["gpt-6-luna"]["n"] == 240 and len(fresh["models"]) == 6
    assert OUTPUT_LIMIT == fresh["output_limit"]
