"""The worker that drains the deferrable queue.

It is a client of the gateway: each job's request goes through POST /v1/chat/completions like any
other, so routing, failover, the breaker, the request log and the metrics apply to deferred work
unchanged. A 200 stores the answer. A retryable failure (429, 502, 503, 504, or the gateway not
answering) schedules the next attempt after an exponential backoff with full jitter, while the next
attempt still falls within the job's age budget (Backoff.max_age_seconds, 15 minutes); anything else,
or a spent budget, fails the job for good and lists it as dead. Each attempt carries X-Request-Id
job:<id>:<attempt>, so the request log shows every call a job caused, including any duplicate.

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
CONCURRENCY = 8
log = logging.getLogger(__name__)


class Worker:
    """Processes up to `concurrency` jobs at a time. A job spends almost all its time waiting on the
    network, so one job at a time would cap a worker at roughly one job per model latency (about 0.4
    per second here); the first queue measurement showed exactly that backlog."""

    def __init__(self, queue: JobQueue, client: httpx2.AsyncClient, name: str, backoff: Backoff = Backoff(),
                 rng: random.Random | None = None, reclaim_idle_seconds: float = RECLAIM_IDLE_SECONDS,
                 sleep=asyncio.sleep, concurrency: int = CONCURRENCY):
        self.queue, self.client, self.name, self.backoff = queue, client, name, backoff
        self.rng, self.reclaim_idle_seconds, self.sleep = rng or random.Random(), reclaim_idle_seconds, sleep
        self.concurrency = concurrency
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
        if status is None or status in RETRYABLE_STATUS:
            delay = self.backoff.delay(attempt, self.rng)
            age = self.queue.clock() - float(job["created_at"])
            if self.backoff.allows_retry(attempt, age, delay):
                await self.queue.retry_later(job_id, message_id, delay, error)
                return "retrying"
            error = f"retry budget spent: {error}"
        await self.queue.finish(job_id, message_id, "failed", error=error)
        return "failed"

    async def _take(self, count: int, block_ms: int) -> list[tuple]:
        """Move due retries, then take up to `count` new entries, or abandoned ones if there are none."""
        await self.queue.move_due()
        entries = await self.queue.read(self.name, count=count, block_ms=block_ms)
        if not entries:
            entries = await self.queue.reclaim(self.name, self.reclaim_idle_seconds, count=count)
        return entries

    async def _process_logged(self, message_id, job_id: str) -> None:
        outcome = await self.process(message_id, job_id)
        log.info("job %s: %s", job_id, outcome)

    async def run_once(self, block_ms: int = 1000) -> int:
        """One pass, for tests and single steps: take up to `concurrency` entries and process them
        together. Returns how many."""
        entries = await self._take(self.concurrency, block_ms)
        await asyncio.gather(*(self._process_logged(m, j) for m, j in entries))
        return len(entries)

    async def run(self) -> None:
        """Keep up to `concurrency` jobs in flight, taking more as soon as one finishes."""
        await self.queue.ensure_group()
        in_flight: set[asyncio.Task] = set()
        while not self.stopped:
            try:
                free = self.concurrency - len(in_flight)
                if free > 0:
                    entries = await self._take(free, block_ms=100 if in_flight else 1000)
                    in_flight |= {asyncio.create_task(self._process_logged(m, j)) for m, j in entries}
                if in_flight:
                    done, in_flight = await asyncio.wait(in_flight, timeout=None if free <= 0 else 0.1,
                                                         return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        if task.exception() is not None:
                            log.warning("worker %s job failed: %s", self.name, task.exception())
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
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    async def go():
        store = redis.Redis.from_url(os.environ.get("LLM_GATEWAY_REDIS_URL", "redis://127.0.0.1:6379/0"))
        async with httpx2.AsyncClient(base_url=args.gateway, timeout=120) as client:
            await Worker(JobQueue(store), client, args.name, reclaim_idle_seconds=args.reclaim_idle,
                         concurrency=args.concurrency).run()

    asyncio.run(go())


if __name__ == "__main__":
    main()
