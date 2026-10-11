"""Deferrable work: a Redis Streams queue of chat requests that may wait, with idempotency keys and
retries spaced by exponential backoff with full jitter.

A client marks a request deferrable (X-Deferrable: true) and gets 202 with a job id instead of an
answer; a worker later sends the request through the gateway's normal path (routing, failover,
breaker) and stores the answer, which GET /v1/jobs/<id> returns to the same tenant.

Keys
- llmgw:job:<id>                  hash: status, tenant, feature, body, attempts, result, last_error...
- llmgw:jobs                      stream of job ids, read by the consumer group "workers"
- llmgw:jobs:delayed              sorted set of job ids waiting for their next attempt (score: due time)
- llmgw:jobs:dead                 list of job ids that failed for good
- llmgw:idem:<tenant>:<key>       job id for an Idempotency-Key, kept IDEMPOTENCY_TTL seconds

Guarantees, and their limits
- Submitting is atomic (WATCH/MULTI): the idempotency key, the job and its stream entry are written
  together or not at all, so the same key never creates two jobs. Reusing a key with a different
  request is refused.
- Finishing is atomic: the result and the stream acknowledgement are written in one MULTI, so a stored
  result is never redone, and a redelivered job that is already done is only acknowledged.
- Delivery is at least once: a job whose worker died stays pending and is reclaimed by another worker
  after RECLAIM_IDLE seconds. If the worker died after the model answered but before the result was
  stored, the model is called again. Exactly once end to end is not possible without the provider's
  cooperation; the duplicate window is that gap, and the chaos measurement counts it.
- Moving due retries back to the stream is atomic across workers, so no retry is enqueued twice."""
import hashlib
import json
import random
import time
import uuid
from dataclasses import dataclass
from typing import Callable

from redis.exceptions import ResponseError, WatchError

PREFIX = "llmgw"
STREAM, GROUP = f"{PREFIX}:jobs", "workers"
DELAYED, DEAD = f"{PREFIX}:jobs:delayed", f"{PREFIX}:jobs:dead"
IDEMPOTENCY_TTL = 24 * 3600
STATUSES = ("queued", "running", "retrying", "done", "failed")


class IdempotencyConflict(ValueError):
    """The Idempotency-Key was already used for a different request."""


@dataclass(frozen=True)
class Backoff:
    """Exponential backoff with full jitter: attempt n waits uniform(0, min(cap, base * 2 ** (n - 1))).

    The retry budget is the job's age, not a count of attempts: a job is retried while its next attempt
    would still start within max_age_seconds of its creation, so a deferrable job survives an outage of
    up to about that long. Counting attempts measured the wrong thing: an open circuit fails an attempt
    at once, without calling a model, so in the first queue measurement jobs spent their 8 attempts
    inside a 40 s outage. max_attempts remains only as a guard against a runaway loop."""
    base_seconds: float = 1.0
    cap_seconds: float = 30.0
    max_age_seconds: float = 900.0
    max_attempts: int = 50

    def delay(self, attempt: int, rng: random.Random) -> float:
        return rng.uniform(0, min(self.cap_seconds, self.base_seconds * 2 ** (attempt - 1)))

    def allows_retry(self, attempts_made: int, age_seconds: float, delay_seconds: float) -> bool:
        return attempts_made < self.max_attempts and age_seconds + delay_seconds <= self.max_age_seconds


def fingerprint(body: dict, feature: str) -> str:
    canonical = json.dumps({"body": body, "feature": feature}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _text(value):
    return value.decode() if isinstance(value, bytes) else value


class JobQueue:
    def __init__(self, redis, clock: Callable[[], float] = time.time):
        self.redis, self.clock = redis, clock

    def _job_key(self, job_id: str) -> str:
        return f"{PREFIX}:job:{job_id}"

    async def ensure_group(self) -> None:
        try:
            await self.redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def submit(self, body: dict, tenant: str, feature: str, request_id: str,
                     idempotency_key: str | None = None) -> tuple[str, bool]:
        """(job id, created). With an idempotency key already used for the same request, returns the
        existing job and created=False; with a different request, raises IdempotencyConflict."""
        request_print = fingerprint(body, feature)
        job_id = uuid.uuid4().hex
        record = {"status": "queued", "tenant": tenant, "feature": feature, "request_id": request_id,
                  "body": json.dumps(body, ensure_ascii=False), "fingerprint": request_print, "attempts": 0,
                  "created_at": self.clock(), "updated_at": self.clock(), "idempotency_key": idempotency_key or ""}
        idem = f"{PREFIX}:idem:{tenant}:{idempotency_key}" if idempotency_key else None
        async with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    if idem:
                        await pipe.watch(idem)
                        existing = _text(await pipe.get(idem))
                        if existing:
                            await pipe.unwatch()
                            known = _text(await self.redis.hget(self._job_key(existing), "fingerprint"))
                            if known is not None and known != request_print:
                                raise IdempotencyConflict(f"Idempotency-Key {idempotency_key!r} was used for another request")
                            return existing, False
                    pipe.multi()
                    if idem:
                        pipe.set(idem, job_id, ex=IDEMPOTENCY_TTL)
                    pipe.hset(self._job_key(job_id), mapping=record)
                    pipe.xadd(STREAM, {"job": job_id})
                    await pipe.execute()
                    return job_id, True
                except WatchError:
                    continue

    async def get(self, job_id: str) -> dict | None:
        data = await self.redis.hgetall(self._job_key(job_id))
        if not data:
            return None
        job = {_text(k): _text(v) for k, v in data.items()}
        job["id"], job["attempts"] = job_id, int(job.get("attempts", 0))
        job["body"] = json.loads(job["body"])
        job["result"] = json.loads(job["result"]) if job.get("result") else None
        return job

    async def start_attempt(self, job_id: str) -> int:
        key = self._job_key(job_id)
        pipe = self.redis.pipeline(transaction=True)
        pipe.hincrby(key, "attempts", 1)
        pipe.hset(key, mapping={"status": "running", "updated_at": self.clock()})
        attempts, _ = await pipe.execute()
        return int(attempts)

    async def finish(self, job_id: str, message_id, status: str, result: dict | None = None, error: str = "") -> None:
        """Store the outcome and acknowledge the stream entry in one transaction."""
        fields = {"status": status, "updated_at": self.clock(), "last_error": error}
        if result is not None:
            fields["result"] = json.dumps(result, ensure_ascii=False)
        pipe = self.redis.pipeline(transaction=True)
        pipe.hset(self._job_key(job_id), mapping=fields)
        if status == "failed":
            pipe.lpush(DEAD, job_id)
        pipe.xack(STREAM, GROUP, message_id)
        pipe.xdel(STREAM, message_id)
        await pipe.execute()

    async def retry_later(self, job_id: str, message_id, delay: float, error: str) -> None:
        due = self.clock() + delay
        pipe = self.redis.pipeline(transaction=True)
        pipe.hset(self._job_key(job_id), mapping={"status": "retrying", "updated_at": self.clock(),
                                                  "next_attempt_at": due, "last_error": error})
        pipe.zadd(DELAYED, {job_id: due})
        pipe.xack(STREAM, GROUP, message_id)
        pipe.xdel(STREAM, message_id)
        await pipe.execute()

    async def acknowledge(self, message_id) -> None:
        pipe = self.redis.pipeline(transaction=True)
        pipe.xack(STREAM, GROUP, message_id)
        pipe.xdel(STREAM, message_id)
        await pipe.execute()

    async def move_due(self) -> int:
        """Put retries whose time has come back on the stream; atomic across workers."""
        async with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(DELAYED)
                    due = [_text(j) for j in await pipe.zrangebyscore(DELAYED, 0, self.clock())]
                    if not due:
                        await pipe.unwatch()
                        return 0
                    pipe.multi()
                    pipe.zrem(DELAYED, *due)
                    for job_id in due:
                        pipe.hset(self._job_key(job_id), "status", "queued")
                        pipe.xadd(STREAM, {"job": job_id})
                    await pipe.execute()
                    return len(due)
                except WatchError:
                    continue

    async def read(self, consumer: str, count: int = 1, block_ms: int = 1000) -> list[tuple]:
        """New stream entries for this consumer: [(message id, job id)]."""
        found = await self.redis.xreadgroup(GROUP, consumer, {STREAM: ">"}, count=count, block=block_ms)
        return [(mid, _text(fields[b"job"] if b"job" in fields else fields["job"])) for _, entries in found or [] for mid, fields in entries]

    async def reclaim(self, consumer: str, min_idle_seconds: float, count: int = 10) -> list[tuple]:
        """Entries delivered to a worker that never acknowledged them (it died), idle long enough."""
        reply = await self.redis.xautoclaim(STREAM, GROUP, consumer, min_idle_time=int(min_idle_seconds * 1000),
                                            start_id="0-0", count=count)
        entries = reply[1] if len(reply) > 1 else []
        return [(mid, _text(fields.get(b"job", fields.get("job")))) for mid, fields in entries if fields]

    async def depth(self) -> dict[str, int]:
        """Jobs waiting in the stream (new or held by a worker), waiting for a retry, and dead."""
        pipe = self.redis.pipeline(transaction=False)
        pipe.xlen(STREAM)
        pipe.zcard(DELAYED)
        pipe.llen(DEAD)
        streamed, delayed, dead = await pipe.execute()
        return {"stream": int(streamed), "delayed": int(delayed), "dead": int(dead)}
