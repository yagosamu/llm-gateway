import asyncio
import json
from types import SimpleNamespace

import httpx2
import openai
import pytest

from harness import budget
from harness.routing import prereg
from harness.routing.judge import (
    VERDICT_SCHEMA,
    Judges,
    judge_models,
    plan,
    plan_ceiling_usd,
    run,
    user_message,
    verdict_key,
)
from llm_gateway.registry import cost_usd

MODELS = judge_models()


def pair(i, category="open_qa", tier="low", auto=None):
    return {"prompt_id": f"dolly-{i:05d}", "category": category, "tier": tier, "model": prereg.TIER_MAP[tier],
            "request": f"Question {i}?", "reference_model": prereg.REFERENCE_MODEL,
            "reference_text": "Reference.", "candidate_text": "Candidate.", "auto_unacceptable": auto}


class Recorder:
    def __init__(self, result):
        self.result, self.kwargs = result, None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def fake_judges(openai_text='{"reason": "fine", "acceptable": true}',
                anthropic_text='{"reason": "wrong fact", "acceptable": false}', openai_error=None):
    openai_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=openai_text))],
                                      usage=SimpleNamespace(prompt_tokens=300, completion_tokens=30))
    anthropic_response = SimpleNamespace(content=[SimpleNamespace(type="text", text=anthropic_text)],
                                         usage=SimpleNamespace(input_tokens=400, output_tokens=40))
    oa, an = Recorder(openai_error or openai_response), Recorder(anthropic_response)
    judges = Judges(SimpleNamespace(chat=SimpleNamespace(completions=oa)), SimpleNamespace(messages=an))
    return judges, oa, an


def test_the_judges_are_the_pre_registered_ones():
    assert tuple(MODELS) == prereg.JUDGES
    assert MODELS["gpt-4.1-mini"].input_per_mtok == 0.40 and MODELS["gpt-4.1-mini"].output_per_mtok == 1.60


def test_both_judges_get_the_rubric_and_a_strict_schema():
    judges, oa, an = fake_judges()
    p = pair(1)
    assert asyncio.run(judges.grade("gpt-4.1-mini", p)) == ({"reason": "fine", "acceptable": True}, 300, 30)
    assert oa.kwargs["messages"][0] == {"role": "system", "content": prereg.RUBRIC}
    assert oa.kwargs["messages"][1]["content"] == user_message(p)
    assert oa.kwargs["temperature"] == 0
    assert oa.kwargs["response_format"]["json_schema"]["strict"] is True
    assert asyncio.run(judges.grade("claude-sonnet-5-5", p)) == ({"reason": "wrong fact", "acceptable": False}, 400, 40)
    assert an.kwargs["system"] == prereg.RUBRIC
    assert an.kwargs["thinking"] == {"type": "between_tools"}
    assert an.kwargs["output_config"] == {"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}}
    assert "temperature" not in an.kwargs


def test_the_user_message_keeps_the_three_texts_apart():
    text = user_message(pair(1))
    assert text.index("<request>") < text.index("<reference_answer>") < text.index("<candidate_answer>")


def test_plan_skips_rule_rejected_pairs_and_verdicts_already_made():
    pairs = [pair(1), pair(2, auto="cut by the output limit"), pair(3)]
    done = {verdict_key(pairs[0], "gpt-4.1-mini")}
    calls = plan(pairs, done)
    assert [(p["prompt_id"], j) for p, j in calls] == [
        ("dolly-00001", "claude-sonnet-5-5"), ("dolly-00003", "gpt-4.1-mini"), ("dolly-00003", "claude-sonnet-5-5")]


def test_a_plan_can_be_restricted_to_one_judge():
    assert {j for _, j in plan([pair(1), pair(2)], set(), judges=("gpt-4.1-mini",))} == {"gpt-4.1-mini"}


def test_a_limited_plan_covers_categories_before_repeating_one():
    pairs = [pair(1, "a"), pair(2, "a"), pair(3, "b")]
    assert [p["category"] for p, _ in plan(pairs, set(), limit=2)][::2] == ["a", "b"]


def test_run_records_verdicts_and_charges_only_successful_calls(tmp_path):
    judges, _, _ = fake_judges()
    calls = plan([pair(1)], set())
    verdicts, ledger = tmp_path / "verdicts.jsonl", tmp_path / "ledger.jsonl"
    records = asyncio.run(run(calls, judges, MODELS, "test", verdicts, ledger))
    by_judge = {r["judge"]: r for r in records}
    assert by_judge["gpt-4.1-mini"]["acceptable"] is True and by_judge["claude-sonnet-5-5"]["acceptable"] is False
    assert by_judge["claude-sonnet-5-5"]["rubric_version"] == prereg.RUBRIC_VERSION
    expected = cost_usd(MODELS["gpt-4.1-mini"], 300, 30) + cost_usd(MODELS["claude-sonnet-5-5"], 400, 40)
    assert budget.spent_usd(ledger) == pytest.approx(expected)
    assert len(verdicts.read_text(encoding="utf-8").splitlines()) == 2


def test_a_provider_error_or_a_malformed_verdict_is_recorded_not_charged(tmp_path):
    request = httpx2.Request("POST", "https://provider.test")
    error = openai.RateLimitError("slow", response=httpx2.Response(429, request=request), body=None)
    judges, _, _ = fake_judges(anthropic_text="not json", openai_error=error)
    records = asyncio.run(run(plan([pair(1)], set()), judges, MODELS, "test",
                              tmp_path / "v.jsonl", tmp_path / "l.jsonl"))
    codes = {r["judge"]: r["error_code"] for r in records}
    assert codes == {"gpt-4.1-mini": "upstream_rate_limit", "claude-sonnet-5-5": "invalid_verdict: JSONDecodeError"}
    assert budget.spent_usd(tmp_path / "l.jsonl") == 0


def test_the_worst_case_prices_the_full_output_limit_for_each_call():
    calls = plan([pair(1)], set())
    assert plan_ceiling_usd(calls, MODELS) > cost_usd(MODELS["claude-sonnet-5-5"], 0, 400)
