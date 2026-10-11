"""The live demo run for the slice 3 video: real models, a fault put on openai for 40 s, the dashboard
showing the gateway's answer to it.

Design, fixed before the run: two instances of harness.chaos.live_app on ports 8000 and 8001 serving
the committed routing (always_low: gpt-6-luna, failing over to claude-haiku-4-5); Dolly traffic at
RATE requests per second for DURATION seconds, each answer capped at MAX_TOKENS; every openai call fails
with a server error from FAULT_START to FAULT_END seconds. Injected failures never reach openai, so
they cost nothing.

Cost bound: before anything starts, the worst case is computed as if every request were billed by
every model it could reach (the tier's model and its fallbacks), at its own input bound and
MAX_TOKENS; the run refuses to start above CAP_USD or past the project budget. Afterwards every
answered request is written to the spend ledger from the instances' request logs.

Usage: uv run --env-file .env python -m harness.chaos.live [run name]   (default slice3_live_video)
Writes data/chaos/live/<run name>/: load.jsonl (one line per request) and summary.json."""
import asyncio
import itertools
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx2
import redis.asyncio as redis

from harness import budget
from harness.chaos import prereg, scenarios
from harness.chaos.load import generate
from harness.record import load_rows
from llm_gateway.registry import call_ceiling_usd, load_registry
from llm_gateway.request_log import RequestLog
from llm_gateway.routing import DEFAULT_CONFIG_PATH, load_config

RATE, DURATION, MAX_TOKENS = 1.0, 90.0, 256
FAULT_START, FAULT_END, FAULT = 20.0, 60.0, {"error_rate": 1.0, "kind": "server_error"}
CAP_USD = 0.25
LEAD_SECONDS = 10  # time to start recording before traffic begins
OUT_DIR = Path("data/chaos/live")
ADMIN = {"X-Admin-Token": os.environ.get("LLM_GATEWAY_ADMIN_TOKEN", "chaos-local")}


def worst_case_usd(rows: list[dict]) -> float:
    registry = load_registry()
    config = load_config(DEFAULT_CONFIG_PATH, set(registry))
    if getattr(config.policy, "tier", None) != "low":
        raise RuntimeError("the bound assumes routing.yaml serves always_low")
    models = [config.tier_map["low"], *config.failover.fallbacks.get("low", ())]
    return sum(call_ceiling_usd(registry[m], r["messages"], MAX_TOKENS) for r in rows for m in models)


def start_instances(log_dir: Path) -> list[subprocess.Popen]:
    procs = []
    for port in prereg.PORTS:
        env = os.environ | {"LIVE_LOG": str(log_dir / f"requests_{port}.sqlite3")}
        procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "harness.chaos.live_app:app", "--host",
                                       "127.0.0.1", "--port", str(port), "--log-level", "warning"], env=env))
    return procs


def settle(run: str, log_dir: Path) -> tuple[float, list[dict]]:
    """Write every billed request to the spend ledger; returns the run's cost and the log rows."""
    rows = [r for p in sorted(log_dir.glob("requests_*.sqlite3")) for r in RequestLog(p).rows()]
    spent, now = 0.0, datetime.now(timezone.utc).isoformat(timespec="seconds")
    for r in rows:
        if r["outcome"] == "ok" and r["cost_usd"]:
            budget.append(budget.LedgerEntry(run, r["request_id"], r["model"], r["input_tokens"], r["output_tokens"],
                                             r["cost_usd"], now))
            spent += r["cost_usd"]
    return spent, rows


def summarize(results: list[dict], log_rows: list[dict], cost: float) -> dict:
    def window(r):
        return "before" if r["sent"] < FAULT_START else "fault" if r["sent"] < FAULT_END else "after"
    by_window = {}
    for name in ("before", "fault", "after"):
        rs = [r for r in results if window(r) == name]
        by_window[name] = {"requests": len(rs), "answered": sum(r["status"] == 200 for r in rs),
                           "answered_by": dict(Counter(r.get("model") for r in rs if r["status"] == 200))}
    return {"design": {"rate": RATE, "duration_s": DURATION, "max_tokens": MAX_TOKENS, "fault_start_s": FAULT_START,
                       "fault_end_s": FAULT_END, "fault": FAULT, "cap_usd": CAP_USD},
            "requests": len(results), "answered": sum(r["status"] == 200 for r in results),
            "errors": dict(Counter(r.get("error") for r in results if r["status"] != 200)),
            "failovers": sum(1 for r in log_rows if r.get("failover_from")),
            "truncated": sum(1 for r in log_rows if r.get("finish_reason") == "length"),
            "by_window": by_window, "cost_usd": round(cost, 6)}


async def run(name: str) -> None:
    out = OUT_DIR / name
    if out.exists():
        raise SystemExit(f"{out} exists; choose another run name")
    rows = planned_rows()
    ceiling = worst_case_usd(rows)
    if ceiling > CAP_USD:
        raise SystemExit(f"worst case US$ {ceiling:.3f} is above the run's cap of US$ {CAP_USD:.2f}")
    left = budget.preflight(name, ceiling)
    print(f"worst case US$ {ceiling:.3f}; the project budget keeps at least US$ {left:.2f} after it", flush=True)
    await scenarios.require_redis()
    store = redis.Redis.from_url(scenarios.REDIS_URL)
    keys = [k async for k in store.scan_iter("llmgw:*")]
    if keys:
        await store.delete(*keys)
    await store.aclose()
    log_dir = Path("var/live") / name
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = start_instances(log_dir)
    results = []
    try:
        async with httpx2.AsyncClient(timeout=10) as client:
            await scenarios.wait_ready(client)
            for left_s in range(LEAD_SECONDS, 0, -1):
                print(f"traffic starts in {left_s} s", flush=True)
                await asyncio.sleep(1)
            bases = [f"http://127.0.0.1:{p}" for p in prereg.PORTS]
            started = time.perf_counter()

            async def inject():
                await asyncio.sleep(FAULT_START - (time.perf_counter() - started))
                await client.post(f"{bases[0]}/admin/chaos/openai", json=FAULT, headers=ADMIN)
                print(f"{FAULT_START:.0f} s: openai fails every call", flush=True)
                await asyncio.sleep(FAULT_END - (time.perf_counter() - started))
                await client.delete(f"{bases[1]}/admin/chaos", headers=ADMIN)
                print(f"{FAULT_END:.0f} s: fault removed", flush=True)

            injector = asyncio.create_task(inject())
            results = await generate(bases, RATE, DURATION, rows=rows, max_tokens=MAX_TOKENS)
            await injector
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=30)
        cost, log_rows = settle(name, log_dir)
        print(f"spent US$ {cost:.4f} (written to the ledger)", flush=True)
    out.mkdir(parents=True)
    (out / "load.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8", newline="\n")
    summary = summarize(results, log_rows, cost)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


def planned_rows() -> list[dict]:
    return list(itertools.islice(itertools.cycle(load_rows()), int(RATE * DURATION)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:] == ["--ceiling"]:  # the preflight alone: no instance, no call
        print(f"worst case US$ {worst_case_usd(planned_rows()):.3f} (cap US$ {CAP_USD:.2f}); "
              f"spent so far US$ {budget.spent_usd():.2f} of US$ {budget.BUDGET_USD:.2f}")
    else:
        asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else "slice3_live_video"))
