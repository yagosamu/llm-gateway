"""Pre-registered measurement of the deferrable queue, its runner and its analysis.

Design, fixed before any run:
- Two gateway instances (the committed routing: always_low, low tier falls back to claude-haiku-4-5,
  breaker v2) and two worker processes, one per instance, over simulated providers.
- From FAULT_START to FAULT_END both openai and anthropic fail every call with a server error, so the
  low tier and its fallback are down together and failover cannot save an interactive request.
- For SUBMIT_SECONDS, deferrable jobs arrive at DEFERRED_RATE per second, each sent twice with the same
  Idempotency-Key RESUBMIT_AFTER seconds apart, as a client retrying after a timeout would. Interactive
  requests arrive at INTERACTIVE_RATE per second over the same span, for comparison.
- At KILL_AT one worker process is killed outright; workers reclaim entries idle for RECLAIM_IDLE
  seconds (shortened from the 120 s default so the reclaim happens inside the run).
- Workers use the committed Backoff (base 1 s, cap 30 s, 8 attempts), not a budget sized for this test.
- The run ends when every job is done or failed, or at DRAIN_LIMIT seconds.
- REPS repetitions.

Metrics per run: submissions and the jobs they created; duplicates absorbed by the idempotency key;
jobs done, failed and unfinished; successful model calls per job (from the gateway's request log, by
the job:<id>:<attempt> request id) and jobs answered more than once; attempts per job; time from
FAULT_END until the last job finished; interactive error rate during the fault.

Usage: uv run python -m harness.chaos.deferrable [run | analyze]"""
import asyncio
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx2
import redis.asyncio as redis

from harness.chaos.scenarios import ADMIN, REDIS_URL, start_instances, wait_ready, write_routing
from harness.record import load_rows

FAULT_START, FAULT_END = 20.0, 60.0
SUBMIT_SECONDS = 80.0
DEFERRED_RATE = 2.0
INTERACTIVE_RATE = 2.0
RESUBMIT_AFTER = 0.5
KILL_AT = 65.0
RECLAIM_IDLE = 15.0
DRAIN_LIMIT = 300.0
REPS = 3
FAULTED_PROVIDERS = ("openai", "anthropic")
PORTS = (8000, 8001)
DATA_DIR = Path("data/chaos/deferrable")
JSON_PATH = Path("results/slice3_deferrable.json")
MD_PATH = Path("results/slice3_deferrable.md")


def start_workers() -> list[subprocess.Popen]:
    return [subprocess.Popen([sys.executable, "-m", "llm_gateway.worker", "--gateway", f"http://127.0.0.1:{port}",
                              "--name", f"worker-{i}", "--reclaim-idle", str(RECLAIM_IDLE)],
                             stderr=subprocess.DEVNULL)
            for i, port in enumerate(PORTS, start=1)]


async def run_once(rep: int, client: httpx2.AsyncClient, store) -> dict:
    keys = [k async for k in store.scan_iter("llmgw:*")]
    if keys:
        await store.delete(*keys)
    workers = start_workers()
    rows = load_rows()
    bases = [f"http://127.0.0.1:{p}" for p in PORTS]
    started = time.perf_counter()
    now = lambda: time.perf_counter() - started
    # Job timestamps in Redis are wall-clock; this start event ties them to the run's own clock.
    submissions, interactive, events = [], [], [{"t": 0.0, "event": "start", "wall": time.time()}]

    async def submit(i: int):
        row = rows[i % len(rows)]
        headers = {"X-Tenant-Id": row["tenant"], "X-Feature": row["category"], "X-Request-Id": f"defer-{rep}-{i}",
                   "X-Deferrable": "true", "Idempotency-Key": f"rep{rep}-job{i}"}
        body = {"model": "auto", "messages": row["messages"]}
        for copy in (0, 1):
            if copy:
                await asyncio.sleep(RESUBMIT_AFTER)
            sent = now()
            try:
                r = await client.post(f"{bases[(i + copy) % 2]}/v1/chat/completions", json=body, headers=headers)
                data = r.json()
                submissions.append({"i": i, "copy": copy, "sent": round(sent, 3), "status": r.status_code,
                                    "job": data.get("id"), "deduplicated": data.get("deduplicated"),
                                    "tenant": row["tenant"]})
            except httpx2.HTTPError as exc:
                submissions.append({"i": i, "copy": copy, "sent": round(sent, 3), "status": 0, "error": type(exc).__name__})

    async def ask(i: int):
        row = rows[(i + 7) % len(rows)]
        sent = now()
        try:
            r = await client.post(f"{bases[i % 2]}/v1/chat/completions", json={"model": "auto", "messages": row["messages"]},
                                  headers={"X-Tenant-Id": row["tenant"], "X-Feature": row["category"],
                                           "X-Request-Id": f"live-{rep}-{i}"})
            interactive.append({"i": i, "sent": round(sent, 3), "status": r.status_code})
        except httpx2.HTTPError:
            interactive.append({"i": i, "sent": round(sent, 3), "status": 0})

    async def schedule(rate: float, fn):
        tasks = []
        for i in range(int(rate * SUBMIT_SECONDS)):
            delay = i / rate - now()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(fn(i)))
        await asyncio.gather(*tasks)

    async def chaos():
        await asyncio.sleep(FAULT_START - now())
        for provider in FAULTED_PROVIDERS:
            await client.post(f"{bases[0]}/admin/chaos/{provider}", json={"error_rate": 1.0, "kind": "server_error"}, headers=ADMIN)
        events.append({"t": round(now(), 3), "event": "fault on"})
        await asyncio.sleep(FAULT_END - now())
        await client.delete(f"{bases[1]}/admin/chaos", headers=ADMIN)
        events.append({"t": round(now(), 3), "event": "fault off"})
        await asyncio.sleep(KILL_AT - now())
        workers[0].kill()
        events.append({"t": round(now(), 3), "event": "worker-1 killed"})

    async def until_drained():
        while now() < DRAIN_LIMIT:
            await asyncio.sleep(1.0)
            job_ids = {s["job"] for s in submissions if s.get("job")}
            if now() > SUBMIT_SECONDS and job_ids:
                statuses = [(await store.hget(f"llmgw:job:{j}", "status")) for j in job_ids]
                if all(s in (b"done", b"failed") for s in statuses):
                    events.append({"t": round(now(), 3), "event": "drained"})
                    return

    try:
        await asyncio.gather(schedule(DEFERRED_RATE, submit), schedule(INTERACTIVE_RATE, ask), chaos())
        await until_drained()
    finally:
        for w in workers:
            w.kill()
    jobs = []
    for job_id in sorted({s["job"] for s in submissions if s.get("job")}):
        data = {k.decode(): v.decode() for k, v in (await store.hgetall(f"llmgw:job:{job_id}")).items()}
        jobs.append({"id": job_id, "status": data.get("status"), "attempts": int(data.get("attempts", 0)),
                     "updated_at": float(data.get("updated_at", 0)), "created_at": float(data.get("created_at", 0)),
                     "last_error": data.get("last_error") or None})
    return {"submissions": submissions, "interactive": interactive, "events": events, "jobs": jobs}


def executions(job_ids: set[str]) -> list[dict]:
    """Every gateway request a job caused, from both instances' request logs."""
    out = []
    for port in PORTS:
        path = Path(f"var/chaos/requests_{port}.sqlite3")
        if not path.exists():
            continue
        conn = sqlite3.connect(path)
        for request_id, outcome, model, received_at in conn.execute(
                "SELECT request_id, outcome, model, received_at FROM requests WHERE request_id LIKE 'job:%'"):
            job_id = request_id.split(":")[1]
            if job_id in job_ids:
                out.append({"job": job_id, "request_id": request_id, "outcome": outcome, "model": model,
                            "received_at": received_at, "instance": port})
        conn.close()
    return out


async def run_all() -> None:
    store = redis.Redis.from_url(REDIS_URL)
    write_routing("served")
    procs = start_instances()
    try:
        async with httpx2.AsyncClient(timeout=120, limits=httpx2.Limits(max_connections=None,
                                                                         max_keepalive_connections=None)) as client:
            await wait_ready(client)
            for rep in range(1, REPS + 1):
                result = await run_once(rep, client, store)
                result["executions"] = executions({j["id"] for j in result["jobs"]})
                folder = DATA_DIR / f"rep{rep}"
                folder.mkdir(parents=True, exist_ok=True)
                for name in ("submissions", "interactive", "events", "jobs", "executions"):
                    (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in result[name]),
                                                          encoding="utf-8", newline="\n")
                print(f"rep {rep} done: {len(result['jobs'])} jobs", flush=True)
    finally:
        for proc in procs:
            proc.terminate()
        await store.aclose()


def read(folder: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (folder / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line]


def rep_metrics(folder: Path) -> dict:
    subs, live, events, jobs, execs = (read(folder, n) for n in ("submissions", "interactive", "events", "jobs", "executions"))
    fault_off = next(e["t"] for e in events if e["event"] == "fault off")
    ok_calls: dict[str, int] = {}
    for e in execs:
        if e["outcome"] == "ok":
            ok_calls[e["job"]] = ok_calls.get(e["job"], 0) + 1
    done = [j for j in jobs if j["status"] == "done"]
    in_fault = [x for x in live if FAULT_START <= x["sent"] < FAULT_END]
    run_start = next(e["wall"] for e in events if e["event"] == "start")
    finished_at = [j["updated_at"] - run_start for j in jobs if j["status"] in ("done", "failed")]
    return {
        "submissions": len(subs), "accepted": sum(s["status"] == 202 for s in subs),
        "deduplicated": sum(bool(s.get("deduplicated")) for s in subs),
        "jobs": len(jobs), "done": len(done), "failed": sum(j["status"] == "failed" for j in jobs),
        "unfinished": sum(j["status"] not in ("done", "failed") for j in jobs),
        "jobs_answered_more_than_once": sum(1 for j in done if ok_calls.get(j["id"], 0) > 1),
        "jobs_done_without_a_logged_answer": sum(1 for j in done if ok_calls.get(j["id"], 0) == 0),
        "attempts_median": statistics.median(j["attempts"] for j in jobs),
        "attempts_max": max(j["attempts"] for j in jobs),
        "drained_after_fault_s": round(max(finished_at) - fault_off, 3) if finished_at else None,
        "interactive_error_rate_in_fault": round(1 - sum(x["status"] == 200 for x in in_fault) / len(in_fault), 4),
        "worker_killed_at_s": next((e["t"] for e in events if e["event"] == "worker-1 killed"), None),
    }


def analyze() -> dict:
    reps = {f.name: rep_metrics(f) for f in sorted(DATA_DIR.glob("rep*"))}
    return {"design": "harness/chaos/deferrable.py", "fault_window_s": [FAULT_START, FAULT_END],
            "faulted_providers": list(FAULTED_PROVIDERS), "reps": reps}


def render(r: dict) -> str:
    cols = ["submissions", "deduplicated", "jobs", "done", "failed", "unfinished", "jobs_answered_more_than_once",
            "attempts_median", "attempts_max", "drained_after_fault_s", "interactive_error_rate_in_fault"]
    lines = ["# Slice 3: deferrable work through a double outage", "",
             f"openai and anthropic both fail every call from {r['fault_window_s'][0]:.0f} s to {r['fault_window_s'][1]:.0f} s, "
             "so the low tier and its fallback are down together. Deferrable jobs arrive at 2 per second for 80 s, each "
             "submitted twice with the same Idempotency-Key; one of the two workers is killed at 65 s. Design: "
             "harness/chaos/deferrable.py.", "",
             "| run | " + " | ".join(c.replace("_", " ") for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for name, m in r["reps"].items():
        lines.append(f"| {name} | " + " | ".join(str(m[c]) for c in cols) + " |")
    lines += ["", "Jobs answered more than once count jobs whose request id shows two or more successful model calls in "
              "the gateway's request log, the at-least-once duplicate window. Drained after fault: seconds from the "
              "fault ending to the last job finishing.", ""]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if (sys.argv[1:] or ["analyze"])[0] == "run":
        asyncio.run(run_all())
    result = analyze()
    JSON_PATH.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    text = render(result)
    MD_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
