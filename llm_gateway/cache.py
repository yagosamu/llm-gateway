"""Response cache, exact layer: an answer is reused only for the same request, from the same tenant,
under the same system prompt, model and parameters.

The namespace is everything about a request except the conversation itself: tenant, feature, the
model that will serve it (after routing), temperature, output limit, the system message and the
client's optional X-Prompt-Version. The exact key adds the messages, with whitespace runs collapsed.
Keying on the question alone would hand one customer an answer computed for another; keying on the
namespace means a changed system prompt, or a bumped prompt version, never meets yesterday's answers.

Keys: llmgw:cache/<tenant>/<feature>/<exact key>, a hash holding the answer, with a TTL. Tenant and
feature are identifiers that never contain "/", so invalidating one tenant or feature by pattern
cannot reach another whose name merely starts the same way.

Only answers that finished normally are stored (finish_reason "stop", non-empty), and only when the
model the request was meant for answered it: a fallback's answer is not filed under the primary.
Redis errors fail open: a lookup that cannot reach Redis is a miss, a store that cannot is skipped."""
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass

from redis.exceptions import RedisError

PREFIX = "llmgw:cache"
log = logging.getLogger(__name__)


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _sha(payload) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def namespace(tenant: str, feature: str, model_id: str, temperature: float | None, max_tokens: int | None,
              messages: list[dict], prompt_version: str | None) -> str:
    system = [collapse(m["content"]) for m in messages if m["role"] == "system"]
    return _sha({"tenant": tenant, "feature": feature, "model": model_id, "temperature": temperature,
                 "max_tokens": max_tokens, "system": system, "prompt_version": prompt_version})


def exact_key(space: str, messages: list[dict]) -> str:
    return _sha({"namespace": space, "messages": [[m["role"], collapse(m["content"])] for m in messages]})


@dataclass(frozen=True)
class CachedAnswer:
    text: str
    finish_reason: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float  # what the original call cost: what a hit saves
    created_at: float
    tenant: str


def storable(text: str, finish_reason: str) -> bool:
    return finish_reason == "stop" and bool(text.strip())


class ResponseCache:
    def __init__(self, redis, clock=time.time):
        self.redis, self.clock = redis, clock
        self.metrics = None

    @staticmethod
    def _key(tenant: str, feature: str, key: str) -> str:
        return f"{PREFIX}/{tenant}/{feature}/{key}"

    def _error(self, what: str, exc: Exception) -> None:
        log.warning("cache %s failed: %s", what, exc)
        if self.metrics is not None:
            self.metrics.cache_errors.labels(what).inc()

    async def get(self, tenant: str, feature: str, key: str) -> CachedAnswer | None:
        try:
            raw = await self.redis.hget(self._key(tenant, feature, key), "answer")
        except (RedisError, OSError) as exc:
            self._error("lookup", exc)
            return None
        if raw is None:
            return None
        answer = CachedAnswer(**json.loads(raw))
        return answer if answer.tenant == tenant else None  # the key already holds the tenant; belt and braces

    async def put(self, tenant: str, feature: str, key: str, space: str, answer: CachedAnswer, ttl_seconds: int) -> None:
        name = self._key(tenant, feature, key)
        try:
            pipe = self.redis.pipeline(transaction=True)
            pipe.hset(name, mapping={"answer": json.dumps(asdict(answer), ensure_ascii=False), "namespace": space})
            pipe.expire(name, ttl_seconds)
            await pipe.execute()
        except (RedisError, OSError) as exc:
            self._error("store", exc)

    async def invalidate(self, tenant: str | None = None, feature: str | None = None) -> int:
        """Delete every entry of a tenant, a feature, both, or all; returns how many."""
        pattern = f"{PREFIX}/{tenant or '*'}/{feature or '*'}/*"
        deleted, batch = 0, []
        async for name in self.redis.scan_iter(match=pattern, count=500):
            batch.append(name)
            if len(batch) == 500:
                deleted += await self.redis.delete(*batch)
                batch = []
        if batch:
            deleted += await self.redis.delete(*batch)
        return deleted
