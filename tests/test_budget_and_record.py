import asyncio
import json

import pytest

from harness import budget
from harness.record import (
    OUTPUT_LIMIT,
    call_key,
    interleave_by_category,
    load_records,
    params_for,
    plan_calls,
    plan_ceiling_usd,
    run,
    summarize,
)
from llm_gateway.api import create_app
from llm_gateway.providers.base import ProviderError, ProviderResult
from llm_gateway.registry import cost_usd, load_registry

REGISTRY = load_registry()


def row(i, category):
    return {"id": f"dolly-{i:05d}", "category": category, "tenant": "tenant-a",
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": f"Question {i}?"}]}


ROWS = [row(i, c) for i, c in enumerate(["a", "a", "a", "b", "b", "c"])]


def test_interleave_covers_every_category_before_repeating_one():
    assert [r["category"] for r in interleave_by_category(ROWS)] == ["a", "b", "c", "a", "b", "a"]


def test_input_bound_is_at_least_the_utf8_bytes_and_doubled_for_anthropic():
    messages = [{"role": "user", "content": "ação"}]  # 4 characters, 6 bytes
    assert budget.input_token_bound(messages, "openai") == 6 + budget.TOKENS_PER_MESSAGE
    assert budget.input_token_bound(messages, "anthropic") == 2 * (6 + budget.TOKENS_PER_MESSAGE)


def test_call_ceiling_prices_the_full_output_limit():
    model = REGISTRY["gpt-6-luna"]
    bound = budget.input_token_bound(ROWS[0]["messages"], "openai")
    assert budget.call_ceiling_usd(model, ROWS[0]["messages"], 1024) == pytest.approx(cost_usd(model, bound, 1024))


def test_preflight_refuses_a_run_that_could_pass_the_budget(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    budget.append(budget.LedgerEntry("r", "q", "m", 1, 1, 9.5, "t"), ledger)
    assert budget.spent_usd(ledger) == pytest.approx(9.5)
    assert budget.preflight("small", 0.4, ledger) == pytest.approx(0.1)
    with pytest.raises(budget.BudgetExceeded, match="US\\$ 9.50 already spent"):
        budget.preflight("big", 0.6, ledger)


def test_plan_skips_calls_already_recorded_for_the_same_parameters():
    luna = REGISTRY["gpt-6-luna"]
    done = {call_key("dolly-00000", luna.id, params_for(luna))}
    planned = plan_calls(ROWS, [luna], limit=2, done=done)
    assert [(r["id"], m.id) for r, m in planned] == [("dolly-00003", "gpt-6-luna")]


def test_a_parameter_change_makes_a_new_key():
    luna = REGISTRY["gpt-6-luna"]
    assert call_key("p", luna.id, params_for(luna)) != call_key("p", luna.id, {**params_for(luna), "reasoning_effort": "high"})


class ScriptedProvider:
    async def complete(self, model, request):
        if model.id == "gpt-oss-20b":
            raise ProviderError("rate_limit", "groq", "slow down", 429)
        return ProviderResult(text=f"answer from {model.id}", input_tokens=50, output_tokens=200, finish_reason="stop")


def test_run_records_successes_and_failures_and_charges_only_successes(tmp_path):
    models = [REGISTRY["gpt-6-luna"], REGISTRY["gpt-oss-20b"]]
    app = create_app(REGISTRY, {p: ScriptedProvider() for p in ("openai", "anthropic", "groq")})
    calls = plan_calls(ROWS, models, limit=2, done=set())
    recordings, ledger = tmp_path / "matrix.jsonl", tmp_path / "ledger.jsonl"
    records = asyncio.run(run(calls, app, "test", recordings, ledger))

    assert [(r["model"], r["status"]) for r in records] == [
        ("gpt-6-luna", "ok"), ("gpt-oss-20b", "error"), ("gpt-6-luna", "ok"), ("gpt-oss-20b", "error")]
    assert load_records(recordings) == records
    ok = records[0]
    assert (ok["text"], ok["input_tokens"], ok["output_tokens"]) == ("answer from gpt-6-luna", 50, 200)
    assert ok["params"] == {"reasoning_effort": "low", "max_completion_tokens": OUTPUT_LIMIT}
    assert records[1]["error_code"] == "upstream_rate_limit" and records[1]["http_status"] == 429
    assert budget.spent_usd(ledger) == pytest.approx(2 * cost_usd(REGISTRY["gpt-6-luna"], 50, 200))
    table = summarize(records, n_prompts=240)
    assert "| gpt-6-luna | 2 | 2 |" in table and "upstream_rate_limit" in table


def test_run_refuses_to_start_when_the_worst_case_passes_the_budget(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    budget.append(budget.LedgerEntry("r", "q", "m", 1, 1, budget.BUDGET_USD, "t"), ledger)
    calls = plan_calls(ROWS, [REGISTRY["gpt-6-luna"]], limit=1, done=set())
    assert plan_ceiling_usd(calls) > 0
    app = create_app(REGISTRY, {p: ScriptedProvider() for p in ("openai", "anthropic", "groq")})
    with pytest.raises(budget.BudgetExceeded):
        asyncio.run(run(calls, app, "test", tmp_path / "matrix.jsonl", ledger))
    assert not (tmp_path / "matrix.jsonl").exists()
