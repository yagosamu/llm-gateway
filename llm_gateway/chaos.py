"""Fault injection for chaos tests: a provider wrapper that adds latency and errors on demand.

The fault for each provider lives in Redis (llmgw:chaos:<provider>), so one call to the admin
endpoint degrades that provider for every gateway instance at once. A fault has an error rate, an
error kind from the gateway's taxonomy, an extra latency added to every call, and, for the kind
"timeout", how long a call hangs before it fails. The wrapper is only built by the chaos app
(harness/chaos/app.py); the served gateway in llm_gateway.main never wraps its providers."""
import asyncio
import random

from llm_gateway.providers.base import ERROR_KINDS, ProviderError, ProviderResult

KEY_PREFIX = "llmgw:chaos"
FIELDS = ("error_rate", "kind", "extra_latency_ms", "timeout_seconds")


class ChaosConfigError(ValueError):
    pass


def validate(fault: dict) -> dict:
    unknown = sorted(set(fault) - set(FIELDS))
    if unknown:
        raise ChaosConfigError(f"unknown fault fields {unknown}")
    rate = float(fault.get("error_rate", 0.0))
    kind = fault.get("kind", "server_error")
    latency = float(fault.get("extra_latency_ms", 0.0))
    timeout = float(fault.get("timeout_seconds", 10.0))
    if not 0 <= rate <= 1:
        raise ChaosConfigError("error_rate must be in [0, 1]")
    if kind not in ERROR_KINDS:
        raise ChaosConfigError(f"kind must be one of {ERROR_KINDS}")
    if latency < 0 or timeout <= 0:
        raise ChaosConfigError("extra_latency_ms must be >= 0 and timeout_seconds > 0")
    return {"error_rate": rate, "kind": kind, "extra_latency_ms": latency, "timeout_seconds": timeout}


class ChaosControl:
    """Reads and writes the faults in Redis; the admin endpoint and the wrappers share it."""

    def __init__(self, redis):
        self.redis = redis

    async def set(self, provider: str, fault: dict) -> dict:
        fault = validate(fault)
        await self.redis.hset(f"{KEY_PREFIX}:{provider}", mapping={k: str(v) for k, v in fault.items()})
        return fault

    async def get(self, provider: str) -> dict | None:
        data = await self.redis.hgetall(f"{KEY_PREFIX}:{provider}")
        if not data:
            return None
        data = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
                for k, v in data.items()}
        return validate(data)

    async def clear(self, provider: str | None = None) -> None:
        if provider:
            await self.redis.delete(f"{KEY_PREFIX}:{provider}")
            return
        keys = [k async for k in self.redis.scan_iter(f"{KEY_PREFIX}:*")]
        if keys:
            await self.redis.delete(*keys)


class ChaosProvider:
    def __init__(self, inner, provider: str, control: ChaosControl, rng: random.Random | None = None):
        self.inner, self.provider, self.control = inner, provider, control
        self.rng = rng or random.Random()

    async def complete(self, model, request) -> ProviderResult:
        fault = await self.control.get(self.provider)
        if fault is None:
            return await self.inner.complete(model, request)
        if fault["extra_latency_ms"]:
            await asyncio.sleep(fault["extra_latency_ms"] / 1000)
        if self.rng.random() < fault["error_rate"]:
            if fault["kind"] == "timeout":
                await asyncio.sleep(fault["timeout_seconds"])
            raise ProviderError(fault["kind"], self.provider, "injected by the chaos endpoint")
        return await self.inner.complete(model, request)
