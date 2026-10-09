"""Open-loop load generator: sends the replay traffic to one or more gateway instances at a fixed
rate, whatever the gateway's latency, and records what happened to each request.

Open loop matters for chaos: a closed-loop client slows down when the gateway does, which hides an
outage's effect. Here request i is sent at start + i / rate, round-robin across the instances, with
model "auto". Each result records when it was sent and finished (seconds since the start), the
instance, the status, the model that answered, the attempts and the error code.

Usage: uv run python -m harness.chaos.load --rate 4 --duration 60 --ports 8000 8001 --out var/load.jsonl"""
import argparse
import asyncio
import itertools
import json
import sys
import time
from pathlib import Path

import httpx2

from harness.record import load_rows


async def one(client: httpx2.AsyncClient, base: str, row: dict, i: int, started: float) -> dict:
    sent = time.perf_counter() - started
    result = {"i": i, "instance": base, "prompt_id": row["id"], "sent": sent}
    try:
        response = await client.post(f"{base}/v1/chat/completions", headers={
            "X-Tenant-Id": row["tenant"], "X-Feature": row["category"], "X-Request-Id": f"load-{i}-{row['id']}",
        }, json={"model": "auto", "messages": row["messages"]})
        body = response.json()
        result |= {"status": response.status_code}
        if response.status_code == 200:
            attempts = body["gateway"].get("attempts") or []
            result |= {"model": body["model"], "attempts": [a["outcome"] for a in attempts] or ["ok"]}
        else:
            result |= {"error": body.get("error", {}).get("code")}
    except httpx2.HTTPError as exc:
        result |= {"status": 0, "error": f"client: {type(exc).__name__}"}
    result["done"] = time.perf_counter() - started
    return result


async def generate(bases: list[str], rate: float, duration: float, rows: list[dict] | None = None,
                   transport=None, on_result=None) -> list[dict]:
    """Send rate * duration requests at fixed intervals and return their results in send order."""
    rows = rows or load_rows()
    total = int(rate * duration)
    started = time.perf_counter()
    # No connection limit: the client's default pool (100 connections) would hold requests back once
    # that many are in flight, which turns the open loop into a closed one under a latency fault.
    limits = httpx2.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx2.AsyncClient(timeout=120, transport=transport, limits=limits) as client:
        tasks = []
        for i, (row, base) in enumerate(zip(itertools.cycle(rows), itertools.cycle(bases))):
            if i >= total:
                break
            delay = started + i / rate - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            task = asyncio.create_task(one(client, base, row, i, started))
            if on_result:
                task.add_done_callback(lambda t: on_result(t.result()))
            tasks.append(task)
        return list(await asyncio.gather(*tasks))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=4.0, help="requests per second")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds")
    parser.add_argument("--ports", type=int, nargs="+", default=[8000, 8001])
    parser.add_argument("--out", type=Path, default=Path("var/load.jsonl"))
    args = parser.parse_args()
    results = asyncio.run(generate([f"http://127.0.0.1:{p}" for p in args.ports], args.rate, args.duration))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8", newline="\n")
    ok = sum(r["status"] == 200 for r in results)
    print(f"{len(results)} requests, {ok} answered -> {args.out}")


if __name__ == "__main__":
    main()
