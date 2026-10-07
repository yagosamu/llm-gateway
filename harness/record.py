"""Record each traffic prompt answered by each registry model, through the gateway, once.

The recording is the asset every later slice reads offline: routing policies are scored by looking
up recorded answers, the judge grades recorded answers, the cache is replayed over them. Each call
is keyed by (prompt, model, parameters); a key already recorded with status ok is never called
again, so an interrupted run resumes where it stopped and the smoke run's calls are reused by the
full run. Failed calls are recorded too, and are retried on the next run.

Usage: uv run --env-file .env python -m harness.record --limit 10 [--models gpt-6-luna gpt-oss-20b]"""
import argparse
import asyncio
import datetime
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx2

from harness import budget
from harness.traffic.build import TRAFFIC_PATH
from llm_gateway.api import create_app
from llm_gateway.registry import ModelConfig, load_registry

OUTPUT_LIMIT = 1024
RECORDINGS_PATH = Path("data/recordings/matrix.jsonl")


def load_rows(path: Path = TRAFFIC_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def interleave_by_category(rows: list[dict]) -> list[dict]:
    """Round robin over categories (sorted), keeping file order inside each, so the first N rows of
    the result cover as many categories as N allows."""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    queues = [by_category[c] for c in sorted(by_category)]
    ordered = []
    for i in range(max(len(q) for q in queues)):
        ordered += [q[i] for q in queues if i < len(q)]
    return ordered


def params_for(model: ModelConfig) -> dict:
    return {"reasoning_effort": model.reasoning_effort, "max_completion_tokens": OUTPUT_LIMIT}


def call_key(prompt_id: str, model_id: str, params: dict) -> str:
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    return f"{prompt_id}|{model_id}|{digest}"


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def plan_calls(rows: list[dict], models: list[ModelConfig], limit: int | None, done: set[str]) -> list[tuple[dict, ModelConfig]]:
    selected = interleave_by_category(rows)[:limit]
    return [(row, m) for row in selected for m in models if call_key(row["id"], m.id, params_for(m)) not in done]


def plan_ceiling_usd(calls: list[tuple[dict, ModelConfig]]) -> float:
    return sum(budget.call_ceiling_usd(m, row["messages"], OUTPUT_LIMIT) for row, m in calls)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


async def run(calls, app, run_name: str, recordings_path: Path = RECORDINGS_PATH,
              ledger_path: Path = budget.LEDGER_PATH) -> list[dict]:
    """Send every planned call through the gateway app, one at a time so latencies do not contend.
    Refuses to start when the plan's worst case would pass the budget."""
    budget.preflight(run_name, plan_ceiling_usd(calls), ledger_path)
    recordings_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://gateway", timeout=None) as client:
        for row, model in calls:
            params = params_for(model)
            request_id = f"{row['id']}:{model.id}"
            response = await client.post("/v1/chat/completions", headers={
                "X-Tenant-Id": row["tenant"], "X-Feature": row["category"], "X-Request-Id": request_id,
            }, json={"model": model.id, "messages": row["messages"], "max_completion_tokens": OUTPUT_LIMIT})
            body = response.json()
            record = {"key": call_key(row["id"], model.id, params), "run": run_name, "prompt_id": row["id"],
                      "category": row["category"], "tenant": row["tenant"], "model": model.id,
                      "provider": model.provider, "tier": model.tier, "params": params,
                      "http_status": response.status_code, "recorded_at": _now()}
            if response.status_code == 200:
                usage, gateway = body["usage"], body["gateway"]
                record |= {"status": "ok", "text": body["choices"][0]["message"]["content"],
                           "finish_reason": body["choices"][0]["finish_reason"],
                           "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
                           "cost_usd": gateway["cost_usd"], "latency_ms": gateway["latency_ms"]}
                budget.append(budget.LedgerEntry(run_name, request_id, model.id, usage["prompt_tokens"],
                                                 usage["completion_tokens"], gateway["cost_usd"], record["recorded_at"]),
                              ledger_path)
            else:
                record |= {"status": "error", "error_code": body["error"]["code"]}
            with recordings_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            print(f"{request_id:40} {record['status']:5} {record.get('finish_reason') or record.get('error_code')}",
                  flush=True)
    return records


def summarize(records: list[dict], n_prompts: int) -> str:
    """One line per model: calls, errors, finish reasons, token means, cost, latency, and the cost of
    all n_prompts projected from the mean cost of the successful calls."""
    lines = ["| model | calls | ok | errors | finish reasons | mean in | mean out | max out | cost | "
             f"p50 ms | p95 ms | projected for {n_prompts} |", "|" + "---|" * 12]
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)
    for model, rs in by_model.items():
        ok = [r for r in rs if r["status"] == "ok"]
        errors = Counter(r["error_code"] for r in rs if r["status"] == "error")
        finishes = Counter(r["finish_reason"] for r in ok)
        if ok:
            latencies = sorted(r["latency_ms"] for r in ok)
            p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
            cost = sum(r["cost_usd"] for r in ok)
            cells = [f"{statistics.mean(r['input_tokens'] for r in ok):.0f}",
                     f"{statistics.mean(r['output_tokens'] for r in ok):.0f}",
                     f"{max(r['output_tokens'] for r in ok)}", f"${cost:.4f}",
                     f"{statistics.median(latencies):.0f}", f"{p95:.0f}", f"${cost / len(ok) * n_prompts:.2f}"]
        else:
            cells = ["-"] * 7
        lines.append(f"| {model} | {len(rs)} | {len(ok)} | {dict(errors) or '-'} | {dict(finishes) or '-'} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from llm_gateway.providers.factory import build_providers

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="first N prompts, interleaved by category")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and its worst case, call nothing")
    args = parser.parse_args()

    registry = load_registry()
    models = [registry[m] for m in args.models] if args.models else list(registry.values())
    rows = load_rows()
    done = {r["key"] for r in load_records(RECORDINGS_PATH) if r["status"] == "ok"}
    calls = plan_calls(rows, models, args.limit, done)
    ceiling = plan_ceiling_usd(calls)
    spent = budget.spent_usd()
    print(f"{len(calls)} calls planned, worst case US$ {ceiling:.2f}; already spent US$ {spent:.2f} "
          f"of US$ {budget.BUDGET_USD:.2f}")
    if args.dry_run or not calls:
        return
    run_name = args.run_name or f"record-{_now()}"
    app = create_app(registry, build_providers({m.provider for m in models}))
    records = asyncio.run(run(calls, app, run_name))
    print()
    print(summarize(records, len(rows)))
    print(f"\nspent in this run US$ {sum(r.get('cost_usd', 0) for r in records):.4f}; "
          f"project total US$ {budget.spent_usd():.4f} of US$ {budget.BUDGET_USD:.2f}")


if __name__ == "__main__":
    main()
