"""Per-provider circuit breaker with closed, open and half-open states, kept in Redis so every gateway
instance sees and moves the same circuit.

closed -> open: after a call, the shared health window (only events after the circuit last closed)
    holds at least min_calls calls and either failed calls reach error_rate_threshold or the p95
    latency of successes passes p95_budget_ms. Failed means rate limit, timeout, connection, server
    error or auth; a bad request is the client's fault and does not count against the provider.
open -> half-open: once open_seconds have passed, at the next request for that provider.
half-open: one probe at a time, leased with SET NX across instances; every other request is turned
    away as if the circuit were open. The probe's success closes the circuit, its failure reopens it.
    A lease that is never released (the instance died mid-probe) expires after probe_lease_seconds.

Transitions are compare-and-set under WATCH/MULTI: when two instances try to move the same circuit,
one wins and the other sees the new state. If Redis is unreachable the breaker fails open and lets
the call through, for the same reason health writes fail open."""
import logging
import time
from dataclasses import dataclass
from typing import Callable

from redis.exceptions import RedisError, WatchError

from llm_gateway.health import HealthSnapshot, HealthTracker
from llm_gateway.routing import BreakerConfig

KEY_PREFIX = "llmgw:breaker"
STATES = ("closed", "half_open", "open")
FAILURE_KINDS = ("rate_limit", "timeout", "connection", "server_error", "auth")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Admission:
    allowed: bool
    state: str
    probe: bool = False


def trip_reason(snapshot: HealthSnapshot, config: BreakerConfig) -> str | None:
    """Why the circuit should open on this window, or None."""
    failures = sum(snapshot.outcomes.get(k, 0) for k in FAILURE_KINDS)
    counted = snapshot.outcomes.get("ok", 0) + failures
    if counted < config.min_calls:
        return None
    if failures / counted >= config.error_rate_threshold:
        return f"{failures} of {counted} calls failed"
    if snapshot.p95_ms is not None and snapshot.p95_ms > config.p95_budget_ms:
        return f"p95 {snapshot.p95_ms:.0f} ms over the {config.p95_budget_ms:.0f} ms budget"
    return None


def _text(value) -> str | None:
    return value.decode() if isinstance(value, bytes) else value


class CircuitBreaker:
    def __init__(self, redis, health: HealthTracker, config: Callable[[], BreakerConfig] = BreakerConfig,
                 clock: Callable[[], float] = time.time, metrics=None):
        self.redis, self.health, self.config, self.clock, self.metrics = redis, health, config, clock, metrics

    def _key(self, provider: str) -> str:
        return f"{KEY_PREFIX}:{provider}"

    async def state(self, provider: str) -> tuple[str, float]:
        """(state, time of the last transition); a provider never seen is closed since time 0."""
        data = await self.redis.hgetall(self._key(provider))
        data = {_text(k): _text(v) for k, v in data.items()}
        return data.get("state", "closed"), float(data.get("changed_at", 0.0))

    async def _transition(self, provider: str, expected: tuple[str, ...], new: str, reason: str) -> bool:
        key = self._key(provider)
        async with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(key)
                    current = _text(await pipe.hget(key, "state")) or "closed"
                    if current not in expected:
                        await pipe.unwatch()
                        return False
                    pipe.multi()
                    pipe.hset(key, mapping={"state": new, "changed_at": self.clock(), "reason": reason})
                    await pipe.execute()
                    break
                except WatchError:
                    continue
        log.warning("circuit %s: %s -> %s (%s)", provider, current, new, reason)
        if self.metrics is not None:
            self.metrics.breaker_transitions.labels(provider, current, new).inc()
        return True

    async def allow(self, provider: str) -> Admission:
        try:
            state, changed_at = await self.state(provider)
            if state == "open" and self.clock() - changed_at >= self.config().open_seconds:
                await self._transition(provider, ("open",), "half_open", "open period elapsed")
                state = "half_open"
            if state == "closed":
                return Admission(True, "closed")
            if state == "half_open":
                leased = await self.redis.set(f"{self._key(provider)}:probe", self.clock(), nx=True,
                                              ex=int(self.config().probe_lease_seconds))
                return Admission(bool(leased), "half_open", probe=bool(leased))
            return Admission(False, "open")
        except (RedisError, OSError) as exc:
            log.warning("breaker state unavailable for %s, letting the call through: %s", provider, exc)
            return Admission(True, "unknown")

    async def on_result(self, provider: str, outcome: str, admission: Admission) -> None:
        """Called after the call's outcome was recorded in the health window."""
        try:
            if admission.probe:
                await self.redis.delete(f"{self._key(provider)}:probe")
                if outcome == "ok":
                    await self._transition(provider, ("half_open",), "closed", "probe succeeded")
                elif outcome in FAILURE_KINDS:
                    await self._transition(provider, ("half_open",), "open", f"probe failed: {outcome}")
                return
            state, changed_at = await self.state(provider)
            if state != "closed":
                return
            reason = trip_reason(await self.health.snapshot(provider, since=changed_at or None), self.config())
            if reason:
                await self._transition(provider, ("closed",), "open", reason)
        except (RedisError, OSError) as exc:
            log.warning("breaker update failed for %s: %s", provider, exc)
