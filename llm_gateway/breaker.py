"""Per-provider circuit breaker with closed, open and half-open states, kept in Redis so every gateway
instance sees and moves the same circuit.

Two ways to decide when a closed circuit opens, chosen by breaker.mode in routing.yaml. Both use the
same thresholds; they differ in what they count.

time_window (v1): after each call, the shared 60 s health window (only buckets after the circuit last
    closed) must hold at least min_calls calls, and either failures reach error_rate_threshold or the
    p95 of successes passes p95_budget_ms. Measured in slice 3, this detects slowly when healthy
    traffic preceded the fault (the window must hold as many failures as successes), never opens at a
    50% error rate, and can reopen after closing when slow calls admitted during the fault finish
    after the close and land in the post-close window.

count_window (v2): the circuit keeps a list of the outcomes of the last window_calls calls admitted
    since it last moved, and opens when at least min_calls are listed and either failures or slow
    calls reach error_rate_threshold of them. Three mechanisms address the three v1 defects:
    - counting calls instead of seconds, so detection does not depend on traffic before the fault;
    - epochs: every transition increments the circuit's epoch and empties the list, and the outcome of
      a call admitted in an earlier epoch never enters the list, so stragglers cannot reopen it;
    - in-flight slowness: a call still running after slow_call_seconds is listed as slow at that
      moment, instead of only when it finally returns.

Failed means rate limit, timeout, connection, server error or auth; a bad request is the client's
fault and never counts against the provider.

open -> half-open: once open_seconds have passed, at the next request for that provider.
half-open: one probe at a time, leased with SET NX across instances; every other request is turned
    away as if the circuit were open. The probe's success closes the circuit; its failure, or in v2 its
    running past slow_call_seconds, reopens it. An abandoned lease expires after probe_lease_seconds.

Transitions and v2 list writes are compare-and-set under WATCH/MULTI on the circuit's hash, so two
instances never move the same circuit twice or write into a list of the wrong epoch. If Redis is
unreachable the breaker fails open and lets the call through."""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from redis.exceptions import RedisError, WatchError

from llm_gateway.health import HealthSnapshot, HealthTracker
from llm_gateway.routing import BreakerConfig

KEY_PREFIX = "llmgw:breaker"
STATES = ("closed", "half_open", "open")
FAILURE_KINDS = ("rate_limit", "timeout", "connection", "server_error", "auth")
# Entries of the v2 outcome list.
SUCCESS, FAILURE, SLOW = "s", "f", "x"
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Admission:
    allowed: bool
    state: str
    probe: bool = False
    epoch: int = field(default=0, compare=False)  # the circuit's epoch when the call was admitted


@dataclass
class CallWatch:
    """v2: a timer that lists a call as slow once it has run slow_call_seconds."""
    task: asyncio.Task | None = None
    flagged: bool = False


def trip_reason(snapshot: HealthSnapshot, config: BreakerConfig) -> str | None:
    """v1: why the circuit should open on this health window, or None."""
    failures = sum(snapshot.outcomes.get(k, 0) for k in FAILURE_KINDS)
    counted = snapshot.outcomes.get("ok", 0) + failures
    if counted < config.min_calls:
        return None
    if failures / counted >= config.error_rate_threshold:
        return f"{failures} of {counted} calls failed"
    if snapshot.p95_ms is not None and snapshot.p95_ms > config.p95_budget_ms:
        return f"p95 {snapshot.p95_ms:.0f} ms over the {config.p95_budget_ms:.0f} ms budget"
    return None


def count_trip_reason(entries: list[str], config: BreakerConfig) -> str | None:
    """v2: why the circuit should open on the listed outcomes of its recent calls, or None."""
    if len(entries) < config.min_calls:
        return None
    failures, slow = entries.count(FAILURE), entries.count(SLOW)
    if failures / len(entries) >= config.error_rate_threshold:
        return f"{failures} of the last {len(entries)} calls failed"
    if slow / len(entries) >= config.error_rate_threshold:
        return f"{slow} of the last {len(entries)} calls ran past {config.slow_call_seconds:.0f} s"
    return None


def _text(value) -> str | None:
    return value.decode() if isinstance(value, bytes) else value


class CircuitBreaker:
    def __init__(self, redis, health: HealthTracker, config: Callable[[], BreakerConfig] = BreakerConfig,
                 clock: Callable[[], float] = time.time, metrics=None):
        self.redis, self.health, self.config, self.clock, self.metrics = redis, health, config, clock, metrics

    def _key(self, provider: str) -> str:
        return f"{KEY_PREFIX}:{provider}"

    def _calls_key(self, provider: str) -> str:
        return f"{KEY_PREFIX}:{provider}:calls"

    async def snapshot(self, provider: str) -> dict:
        data = await self.redis.hgetall(self._key(provider))
        data = {_text(k): _text(v) for k, v in data.items()}
        return {"state": data.get("state", "closed"), "changed_at": float(data.get("changed_at", 0.0)),
                "epoch": int(data.get("epoch", 0))}

    async def state(self, provider: str) -> tuple[str, float]:
        """(state, time of the last transition); a provider never seen is closed since time 0."""
        s = await self.snapshot(provider)
        return s["state"], s["changed_at"]

    async def _transition(self, provider: str, expected: tuple[str, ...], new: str, reason: str,
                          epoch: int | None = None) -> bool:
        """Move the circuit to `new` if it is in one of `expected` (and, when given, still in `epoch`).
        Every transition starts a new epoch and empties the v2 outcome list."""
        key = self._key(provider)
        async with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(key)
                    data = {_text(k): _text(v) for k, v in (await pipe.hgetall(key)).items()}
                    current = data.get("state", "closed")
                    if current not in expected or (epoch is not None and int(data.get("epoch", 0)) != epoch):
                        await pipe.unwatch()
                        return False
                    pipe.multi()
                    pipe.hset(key, mapping={"state": new, "changed_at": self.clock(), "reason": reason})
                    pipe.hincrby(key, "epoch", 1)
                    pipe.delete(self._calls_key(provider))
                    await pipe.execute()
                    break
                except WatchError:
                    continue
        log.warning("circuit %s: %s -> %s (%s)", provider, current, new, reason)
        if self.metrics is not None:
            self.metrics.breaker_transitions.labels(provider, current, new).inc()
        return True

    async def allow(self, provider: str) -> Admission:
        if not self.config().enabled:
            return Admission(True, "disabled")
        try:
            s = await self.snapshot(provider)
            state, epoch = s["state"], s["epoch"]
            if state == "open" and self.clock() - s["changed_at"] >= self.config().open_seconds:
                if await self._transition(provider, ("open",), "half_open", "open period elapsed", epoch):
                    state, epoch = "half_open", epoch + 1
                else:  # another instance moved it first; act on what it is now
                    s = await self.snapshot(provider)
                    state, epoch = s["state"], s["epoch"]
            if state == "closed":
                return Admission(True, "closed", epoch=epoch)
            if state == "half_open":
                leased = await self.redis.set(f"{self._key(provider)}:probe", self.clock(), nx=True,
                                              ex=int(self.config().probe_lease_seconds))
                return Admission(bool(leased), "half_open", probe=bool(leased), epoch=epoch)
            return Admission(False, "open", epoch=epoch)
        except (RedisError, OSError) as exc:
            log.warning("breaker state unavailable for %s, letting the call through: %s", provider, exc)
            return Admission(True, "unknown")

    def watch(self, provider: str, admission: Admission) -> CallWatch | None:
        """v2: start the timer that lists the call as slow if it runs past slow_call_seconds. Returns
        None in v1, when the breaker is off, or when Redis was unreachable at admission."""
        config = self.config()
        if config.mode != "count_window" or admission.state in ("disabled", "unknown"):
            return None
        watch = CallWatch()

        async def flag():
            await asyncio.sleep(config.slow_call_seconds)
            watch.flagged = True
            await self._record(provider, SLOW, admission)

        watch.task = asyncio.create_task(flag())
        return watch

    async def _record(self, provider: str, entry: str, admission: Admission) -> None:
        """v2: list one outcome of a call admitted in the current epoch, then open, close or reopen the
        circuit if that outcome calls for it."""
        try:
            config = self.config()
            if admission.probe:
                if entry == SUCCESS:
                    await self._transition(provider, ("half_open",), "closed", "probe succeeded", admission.epoch)
                else:
                    what = "failed" if entry == FAILURE else f"ran past {config.slow_call_seconds:.0f} s"
                    await self._transition(provider, ("half_open",), "open", f"probe {what}", admission.epoch)
                return
            key, calls = self._key(provider), self._calls_key(provider)
            async with self.redis.pipeline(transaction=True) as pipe:
                while True:
                    try:
                        await pipe.watch(key)
                        data = {_text(k): _text(v) for k, v in (await pipe.hgetall(key)).items()}
                        if data.get("state", "closed") != "closed" or int(data.get("epoch", 0)) != admission.epoch:
                            await pipe.unwatch()
                            return  # admitted before the circuit last moved: a straggler, not evidence
                        pipe.multi()
                        pipe.lpush(calls, entry)
                        pipe.ltrim(calls, 0, config.window_calls - 1)
                        pipe.lrange(calls, 0, -1)
                        entries = [_text(e) for e in (await pipe.execute())[2]]
                        break
                    except WatchError:
                        continue
            reason = count_trip_reason(entries, config)
            if reason:
                await self._transition(provider, ("closed",), "open", reason, admission.epoch)
        except (RedisError, OSError) as exc:
            log.warning("breaker update failed for %s: %s", provider, exc)

    async def on_result(self, provider: str, outcome: str, admission: Admission,
                        duration: float | None = None, watch: CallWatch | None = None) -> None:
        """Called after the call's outcome was recorded in the health window. duration (seconds) and
        watch are used by v2 only."""
        if admission.state in ("disabled", "unknown"):
            return
        if self.config().mode == "count_window":
            if watch is not None and watch.task is not None:
                watch.task.cancel()
            if admission.probe:
                await self._release_probe(provider)
            if watch is not None and watch.flagged:
                return  # already listed as slow while it ran
            if outcome in FAILURE_KINDS:
                entry = FAILURE
            elif outcome == "ok":
                entry = SLOW if duration is not None and duration >= self.config().slow_call_seconds else SUCCESS
            else:
                return  # a bad request says nothing about the provider
            await self._record(provider, entry, admission)
            return
        try:
            if admission.probe:
                await self._release_probe(provider)
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

    async def _release_probe(self, provider: str) -> None:
        try:
            await self.redis.delete(f"{self._key(provider)}:probe")
        except (RedisError, OSError) as exc:
            log.warning("probe lease release failed for %s: %s", provider, exc)
