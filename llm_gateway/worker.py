"""The worker that drains the deferrable queue.

It is a client of the gateway: each job's request goes through POST /v1/chat/completions like any
other, so routing, failover, the breaker, the request log and the metrics apply to deferred work
unchanged. A 200 stores the answer. A retryable failure (429, 502, 503, 504, or the gateway not
answering) schedules the next attempt after an exponential backoff with full jitter, until
Backoff.max_attempts; anything else, or running out of attempts, fails the job for good and lists it
as dead. Each attempt carries X-Request-Id job:<id>:<attempt>, so the request log shows every call a
job caused, including any duplicate.

Usage: uv run python -m llm_gateway.worker --gateway http://127.0.0.1:8000 --name worker-1"""
import argparse
import asyncio
import logging
import os
import random

import httpx2

from llm_gateway.jobs import Backoff, JobQueue

RETRYABLE_STATUS = (429, 502, 503, 504)
RECLAIM_IDLE_SECONDS = 120.0  # longer than any single attempt (the gateway's 90 s deadline)
log = logging.getLogger(__name__)


class Worker:
    def __init__(self, queue: JobQueue, client: httpx2.AsyncClient, name: str, backoff: Backoff = Backoff(),
                 rng: random.Random | None = None, reclaim_idle_seconds: float = RECLAIM_IDLE_SECONDS,
                 sleep=asyncio.sleep):
        self.queue, self.client, self.name, self.backoff = queue, client, name, backoff
        self.rng, self.reclaim_idle_seconds, self.sleep = rng or random.Random(), reclaim_idle_seconds, sleep
        self.stopped = False

    async def process(self, message_id, job_id: str) -> str:
        """Run one delivery of a job; returns what happened to it."""
        job = await self.queue.get(job_id)
        if job is None or job["status"] in ("done", "failed"):
            await self.queue.acknowledge(message_id)  # a redelivery of finished work
            return "skipped"
        attempt = await self.queue.start_attempt(job_id)
        headers = {"X-Tenant-Id": job["tenant"], "X-Feature": job["feature"], "X-Request-Id": f"job:{job_id}:{attempt}"}
        try:
            response = await self.client.post("/v1/chat/completions", json=job["body"], headers=headers)
            status, body = response.status_code, response.json()
        except (httpx2.HTTPError, ValueError) as exc:
            status, body = None, {"error": {"code": f"gateway_unreachable: {type(exc).__name__}"}}
        if status == 200:
            await self.queue.finish(job_id, message_id, "done", result=body)
            return "done"
        error = (body.get("error") or {}).get("code", f"http_{status}")
        if (status is None or status in RETRYABLE_STATUS) and attempt < self.backoff.max_attempts:
            await self.queue.retry_later(job_id, message_id, self.backoff.delay(attempt, self.rng), error)
            return "retrying"
        await self.queue.finish(job_id, message_id, "failed", error=error)
        return "failed"

    async def run_once(self, block_ms: int = 1000) -> int:
        """One pass: move due retries, take new or abandoned entries, process them. Returns how many."""
        await self.queue.move_due()
        entries = await self.queue.read(self.name, count=4, block_ms=block_ms)
        if not entries:
            entries = await self.queue.reclaim(self.name, self.reclaim_idle_seconds)
        for message_id, job_id in entries:
            outcome = await self.process(message_id, job_id)
            log.info("job %s: %s", job_id, outcome)
        return len(entries)

    async def run(self) -> None:
        await self.queue.ensure_group()
        while not self.stopped:
            try:
                await self.run_once()
            except Exception as exc:  # keep draining; a Redis or gateway hiccup must not kill the worker
                log.warning("worker %s pass failed: %s", self.name, exc)
                await self.sleep(1.0)


def main() -> None:
    import redis.asyncio as redis
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="http://127.0.0.1:8000")
    parser.add_argument("--name", default="worker-1")
    parser.add_argument("--reclaim-idle", type=float, default=RECLAIM_IDLE_SECONDS)
    args = parser.parse_args()

    async def go():
        store = redis.Redis.from_url(os.environ.get("LLM_GATEWAY_REDIS_URL", "redis://127.0.0.1:6379/0"))
        async with httpx2.AsyncClient(base_url=args.gateway, timeout=120) as client:
            await Worker(JobQueue(store), client, args.name, reclaim_idle_seconds=args.reclaim_idle).run()

    asyncio.run(go())


if __name__ == "__main__":
    main()
