"""Per-provider health over a sliding window, kept in Redis so every gateway instance shares one
view and a restart does not reset it.

Each provider call adds to a bucket of BUCKET_SECONDS: one counter per outcome (ok or an error kind
of the gateway's taxonomy) and, for successful calls, a latency histogram. A snapshot sums the
buckets that overlap the last WINDOW_SECONDS, so the window slides in steps of one bucket and covers
between WINDOW_SECONDS - BUCKET_SECONDS and WINDOW_SECONDS of history. Each write costs one pipeline
of three or four commands whatever the traffic, and memory is bounded by the number of buckets;
the price is that percentiles are interpolated inside histogram buckets, not exact.

Writes fail open: if Redis is unreachable the request is still served and the failure is counted,
because a gateway that stops serving when its monitoring is down is worse than one flying blind."""
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from redis.exceptions import RedisError

from llm_gateway.providers.base import ERROR_KINDS

BUCKET_SECONDS = 10
WINDOW_SECONDS = 60
OUTCOMES = ("ok",) + ERROR_KINDS
# Upper edges, in seconds, of the latency histogram; the last bucket is open-ended.
LATENCY_EDGES = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)
KEY_PREFIX = "llmgw:health"
log = logging.getLogger(__name__)


@dataclass
class HealthSnapshot:
    provider: str
    window_seconds: int
    calls: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)
    success_rate: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def latency_field(seconds: float) -> str:
    for edge in LATENCY_EDGES:
        if seconds <= edge:
            return f"le:{edge}"
    return "le:inf"


def quantile(counts: dict[str, int], q: float) -> float | None:
    """The q-quantile in milliseconds, interpolated linearly inside the histogram bucket that holds
    it, as Prometheus's histogram_quantile does. The open last bucket reports its lower edge."""
    edges = [(edge, counts.get(f"le:{edge}", 0)) for edge in LATENCY_EDGES] + [(None, counts.get("le:inf", 0))]
    total = sum(c for _, c in edges)
    if total == 0:
        return None
    rank, seen, lower = q * total, 0, 0.0
    for edge, count in edges:
        if count and seen + count >= rank:
            if edge is None:
                return lower * 1000
            return (lower + (edge - lower) * (rank - seen) / count) * 1000
        seen += count
        lower = edge if edge is not None else lower
    return lower * 1000


class HealthTracker:
    def __init__(self, redis, clock: Callable[[], float] = time.time, metrics=None):
        self.redis, self.clock, self.metrics = redis, clock, metrics

    def _key(self, provider: str, bucket: int) -> str:
        return f"{KEY_PREFIX}:{provider}:{bucket}"

    def _bucket(self, at: float) -> int:
        return int(at // BUCKET_SECONDS) * BUCKET_SECONDS

    async def record(self, provider: str, outcome: str, latency_seconds: float | None = None) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        key = self._key(provider, self._bucket(self.clock()))
        try:
            pipe = self.redis.pipeline(transaction=False)
            pipe.hincrby(key, outcome, 1)
            if outcome == "ok" and latency_seconds is not None:
                pipe.hincrby(key, latency_field(latency_seconds), 1)
            pipe.expire(key, WINDOW_SECONDS + 2 * BUCKET_SECONDS)
            await pipe.execute()
        except (RedisError, OSError) as exc:
            log.warning("health write failed for %s: %s", provider, exc)
            if self.metrics is not None:
                self.metrics.health_write_errors.inc()

    async def snapshot(self, provider: str, since: float | None = None) -> HealthSnapshot:
        """The window's totals. With `since`, only buckets that start after the bucket holding `since`
        count, so events from before a moment (a breaker closing) never weigh on what follows it; the
        price is ignoring up to one bucket of events after it."""
        now = self.clock()
        first = self._bucket(now - WINDOW_SECONDS + BUCKET_SECONDS)
        if since is not None:
            first = max(first, self._bucket(since) + BUCKET_SECONDS)
        buckets = range(first, self._bucket(now) + BUCKET_SECONDS, BUCKET_SECONDS)
        totals: dict[str, int] = {}
        pipe = self.redis.pipeline(transaction=False)
        for bucket in buckets:
            pipe.hgetall(self._key(provider, bucket))
        for fields in (await pipe.execute() if len(buckets) else []):
            for name, value in fields.items():
                name = name.decode() if isinstance(name, bytes) else name
                totals[name] = totals.get(name, 0) + int(value)
        outcomes = {o: totals.get(o, 0) for o in OUTCOMES}
        calls = sum(outcomes.values())
        histogram = {k: v for k, v in totals.items() if k.startswith("le:")}
        return HealthSnapshot(
            provider=provider, window_seconds=WINDOW_SECONDS, calls=calls, outcomes=outcomes,
            success_rate=outcomes["ok"] / calls if calls else None,
            p50_ms=quantile(histogram, 0.50), p95_ms=quantile(histogram, 0.95), p99_ms=quantile(histogram, 0.99))
