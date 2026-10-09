"""Grade every (prompt, candidate) pair with the pre-registered judges and rubric.

Each judge sees the rubric as its system prompt and the request, the reference answer and the
candidate answer in tagged blocks, and must answer with JSON {reason, acceptable}, enforced by each
provider's structured output. Pairs the rules already reject (cut or empty answers) are never sent.
Verdicts go to data/judgments/verdicts.jsonl keyed by (prompt, tier, judge, rubric version), so a run
resumes where it stopped, and every call is charged to the project ledger after a budget preflight.

The judges are called directly with the provider SDKs, not through the gateway: they grade the
gateway's answers and are not gateway traffic. SDK retries are off, as in the gateway; a failed call
is recorded and retried on the next run.

Usage: uv run --env-file .env python -m harness.routing.judge [--limit N] [--dry-run]"""
import argparse
import asyncio
import datetime
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import anthropic
import openai

from harness import budget
from harness.record import RECORDINGS_PATH, load_records, load_rows
from harness.replay import current_records
from harness.routing import prereg
from harness.routing.pairs import build_pairs
from llm_gateway.providers.base import ProviderError, classify
from llm_gateway.registry import ModelConfig, cost_usd, load_registry

VERDICTS_PATH = Path("data/judgments/verdicts.jsonl")
FALLBACK_VERDICTS_PATH = Path("data/judgments/fallback_verdicts.jsonl")
MAX_OUTPUT_TOKENS = 400
TIMEOUT_SECONDS = 60.0
CONCURRENCY = 4
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "acceptable": {"type": "boolean"}},
    "required": ["reason", "acceptable"],
    "additionalProperties": False,
}
# gpt-4.1-mini is not a routing target, so it is not in the gateway registry; its price is kept here,
# checked on the same official page and day as the registry's.
GPT_41_MINI = ModelConfig(id="gpt-4.1-mini", provider="openai", provider_model="gpt-4.1-mini", tier="low",
                          input_per_mtok=0.40, output_per_mtok=1.60, reasoning_effort="none",
                          price_source="https://developers.openai.com/api/docs/pricing",
                          price_checked=datetime.date(2026, 10, 8))


def judge_models() -> dict[str, ModelConfig]:
    registry = load_registry()
    models = {"gpt-4.1-mini": GPT_41_MINI, "claude-sonnet-5-5": registry["claude-sonnet-5-5"]}
    assert tuple(models) == prereg.JUDGES
    return models


def user_message(pair: dict) -> str:
    return (f"<request>\n{pair['request']}\n</request>\n\n"
            f"<reference_answer>\n{pair['reference_text']}\n</reference_answer>\n\n"
            f"<candidate_answer>\n{pair['candidate_text']}\n</candidate_answer>")


def verdict_key(pair: dict, judge: str) -> str:
    return f"{pair['prompt_id']}|{pair['tier']}|{judge}|{prereg.RUBRIC_VERSION}"


def call_ceiling_usd(pair: dict, judge: ModelConfig) -> float:
    messages = [{"content": prereg.RUBRIC}, {"content": user_message(pair)}]
    return budget.call_ceiling_usd(judge, messages, MAX_OUTPUT_TOKENS)


class Judges:
    """The two judges behind one call: grade(judge_id, pair) -> (verdict dict, input tokens, output tokens)."""

    def __init__(self, openai_client=None, anthropic_client=None, env=os.environ):
        self.openai = openai_client or openai.AsyncOpenAI(
            api_key=env.get("OPENAI_API_KEY"), timeout=TIMEOUT_SECONDS, max_retries=0)
        self.anthropic = anthropic_client or anthropic.AsyncAnthropic(
            api_key=env.get("ANTHROPIC_API_KEY"), timeout=TIMEOUT_SECONDS, max_retries=0)

    async def grade(self, judge: str, pair: dict) -> tuple[dict, int, int]:
        if judge == "gpt-4.1-mini":
            try:
                response = await self.openai.chat.completions.create(
                    model="gpt-4.1-mini", temperature=0, max_completion_tokens=MAX_OUTPUT_TOKENS,
                    messages=[{"role": "system", "content": prereg.RUBRIC},
                              {"role": "user", "content": user_message(pair)}],
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "verdict", "strict": True, "schema": VERDICT_SCHEMA}})
            except openai.APIError as exc:
                raise classify(exc, openai, "openai") from exc
            text = response.choices[0].message.content or ""
            usage = (response.usage.prompt_tokens, response.usage.completion_tokens)
        elif judge == "claude-sonnet-5-5":
            try:
                # between_tools is Sonnet 5.5's lowest thinking setting: no extended thinking.
                response = await self.anthropic.messages.create(
                    model="claude-sonnet-5-5", max_tokens=MAX_OUTPUT_TOKENS, system=prereg.RUBRIC,
                    thinking={"type": "between_tools"},
                    messages=[{"role": "user", "content": user_message(pair)}],
                    output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}})
            except anthropic.APIError as exc:
                raise classify(exc, anthropic, "anthropic") from exc
            text = "".join(b.text for b in response.content if b.type == "text")
            usage = (response.usage.input_tokens, response.usage.output_tokens)
        else:
            raise ValueError(f"unknown judge {judge!r}")
        return json.loads(text), *usage


class VerifierJudge:
    """The gateway verifier's judge: gpt-4.1-mini with the pre-registered rubric. One judge, not the
    evaluation's pair: a Sonnet verdict costs about nine times a gpt-4.1-mini one, and adding it would
    make each verification about 2.4 times as expensive. results/slice2_verifier.md measures both the
    cost and what the single judge misses."""
    model = GPT_41_MINI
    rubric = prereg.RUBRIC
    max_output_tokens = MAX_OUTPUT_TOKENS

    def __init__(self, judges: Judges | None = None):
        self.judges = judges or Judges()

    async def grade(self, request: str, reference: str, candidate: str) -> tuple[bool, str, int, int]:
        pair = {"request": request, "reference_text": reference, "candidate_text": candidate}
        verdict, input_tokens, output_tokens = await self.judges.grade("gpt-4.1-mini", pair)
        return bool(verdict["acceptable"]), str(verdict["reason"]), input_tokens, output_tokens


def plan(pairs: list[dict], done: set[str], limit: int | None = None,
         judges: tuple[str, ...] = prereg.JUDGES) -> list[tuple[dict, str]]:
    """(pair, judge) calls still to make, for pairs the rules do not already reject. With a limit, the
    first `limit` judged pairs interleaved by category, so a smoke run covers every category. Running
    one judge at a time keeps each run's worst case under what the budget has left."""
    judged = [p for p in pairs if not p["auto_unacceptable"]]
    if limit is not None:
        by_category = defaultdict(list)
        for p in judged:
            by_category[p["category"]].append(p)
        queues, judged = [by_category[c] for c in sorted(by_category)], []
        for i in range(max(len(q) for q in queues)):
            judged += [q[i] for q in queues if i < len(q)]
        judged = judged[:limit]
    return [(p, j) for p in judged for j in judges if verdict_key(p, j) not in done]


def plan_ceiling_usd(calls: list[tuple[dict, str]], models: dict[str, ModelConfig]) -> float:
    return sum(call_ceiling_usd(p, models[j]) for p, j in calls)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


async def run(calls, judges: Judges, models: dict[str, ModelConfig], run_name: str,
              verdicts_path: Path = VERDICTS_PATH, ledger_path: Path = budget.LEDGER_PATH) -> list[dict]:
    budget.preflight(run_name, plan_ceiling_usd(calls, models), ledger_path)
    verdicts_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore, records = asyncio.Semaphore(CONCURRENCY), []

    async def one(pair: dict, judge: str) -> None:
        async with semaphore:
            record = {"key": verdict_key(pair, judge), "run": run_name, "prompt_id": pair["prompt_id"],
                      "category": pair["category"], "tier": pair["tier"], "candidate_model": pair["model"],
                      "reference_model": pair["reference_model"], "judge": judge,
                      "rubric_version": prereg.RUBRIC_VERSION}
            started = time.perf_counter()
            try:
                verdict, input_tokens, output_tokens = await judges.grade(judge, pair)
                cost = cost_usd(models[judge], input_tokens, output_tokens)
                record |= {"status": "ok", "acceptable": bool(verdict["acceptable"]), "reason": verdict["reason"],
                           "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost}
                budget.append(budget.LedgerEntry(run_name, record["key"], judge, input_tokens, output_tokens,
                                                 cost, _now()), ledger_path)
            except ProviderError as exc:
                record |= {"status": "error", "error_code": f"upstream_{exc.kind}"}
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                record |= {"status": "error", "error_code": f"invalid_verdict: {type(exc).__name__}"}
            record |= {"latency_ms": (time.perf_counter() - started) * 1000, "recorded_at": _now()}
            with verdicts_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    await asyncio.gather(*(one(p, j) for p, j in calls))
    return records


def summarize(records: list[dict]) -> str:
    lines = ["| judge | calls | ok | errors | acceptable | cost |", "|---|---|---|---|---|---|"]
    by_judge = defaultdict(list)
    for r in records:
        by_judge[r["judge"]].append(r)
    for judge, rs in by_judge.items():
        ok = [r for r in rs if r["status"] == "ok"]
        errors = Counter(r["error_code"] for r in rs if r["status"] == "error")
        lines.append(f"| {judge} | {len(rs)} | {len(ok)} | {dict(errors) or '-'} | "
                     f"{sum(r['acceptable'] for r in ok)} of {len(ok)} | ${sum(r['cost_usd'] for r in ok):.4f} |")
    return "\n".join(lines)


def load_verdicts(path: Path = VERDICTS_PATH) -> list[dict]:
    return load_records(path)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="first N judged pairs, interleaved by category")
    parser.add_argument("--judge", choices=prereg.JUDGES, default=None, help="run only this judge")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fallback", default=None, metavar="MODEL",
                        help="judge a failover model instead of the pre-registered tiers; verdicts go to "
                             "data/judgments/fallback_verdicts.jsonl, apart from the slice 2 evaluation")
    args = parser.parse_args()

    models = judge_models()
    candidates, verdicts_path = None, VERDICTS_PATH
    if args.fallback:
        candidates, verdicts_path = {f"fallback-{args.fallback}": args.fallback}, FALLBACK_VERDICTS_PATH
    pairs = build_pairs(load_rows(), current_records(load_records(RECORDINGS_PATH), load_registry()), candidates)
    done = {r["key"] for r in load_verdicts(verdicts_path) if r["status"] == "ok"}
    calls = plan(pairs, done, args.limit, (args.judge,) if args.judge else prereg.JUDGES)
    print(f"{len(calls)} judge calls planned, worst case US$ {plan_ceiling_usd(calls, models):.2f}; "
          f"already spent US$ {budget.spent_usd():.2f} of US$ {budget.BUDGET_USD:.2f}")
    if args.dry_run or not calls:
        return
    run_name = args.run_name or f"judge-{_now()}"
    records = asyncio.run(run(calls, Judges(), models, run_name, verdicts_path))
    print(summarize(records))
    print(f"\nproject total US$ {budget.spent_usd():.4f} of US$ {budget.BUDGET_USD:.2f}")


if __name__ == "__main__":
    main()
