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
Redis errors fail open: a lookup that cannot reach Redis is a miss, a store that cannot is skipped.

Semantic layer: a single-turn request that misses the exact layer is embedded (the user message,
context included) and compared with the stored answers of the same namespace only; the nearest one
is served if its cosine similarity reaches the configured floor. The search sits behind a small
index interface: RedisVectorIndex (Redis Query Engine, the vector stored in the answer's own hash, so
TTL and invalidation remove it too) serves the gateway; MemoryVectorIndex does the same exact search
in process for tests and the offline evaluation replay. A contract test holds them to the same answers."""
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass

import numpy as np
from redis.exceptions import RedisError, ResponseError

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


def single_turn_text(messages: list[dict]) -> str | None:
    """The user message of a single-turn request (system messages aside); None otherwise."""
    turns = [m for m in messages if m["role"] != "system"]
    return turns[0]["content"] if len(turns) == 1 and turns[0]["role"] == "user" else None


def _text(value):
    return value.decode() if isinstance(value, bytes) else value


class MemoryVectorIndex:
    """Exact cosine search in process. Vectors must be unit length (llm_gateway.embeddings returns them so)."""

    def __init__(self, clock=time.time):
        self.clock = clock
        self._spaces: dict[str, tuple[list[str], list[np.ndarray], list[float]]] = {}

    async def add(self, name: str, space: str, vector: np.ndarray, ttl_seconds: int) -> None:
        names, vectors, expiries = self._spaces.setdefault(space, ([], [], []))
        names.append(name)
        vectors.append(np.asarray(vector, dtype=np.float32))
        expiries.append(self.clock() + ttl_seconds)

    async def nearest(self, space: str, vector: np.ndarray) -> tuple[str, float] | None:
        if space not in self._spaces:
            return None
        names, vectors, expiries = self._spaces[space]
        now = self.clock()
        live = [i for i, e in enumerate(expiries) if e > now]
        if not live:
            return None
        sims = np.stack([vectors[i] for i in live]) @ np.asarray(vector, dtype=np.float32)
        best = int(np.argmax(sims))
        return names[live[best]], float(sims[best])


class RedisVectorIndex:
    """Redis Query Engine over the cache hashes: a TAG on the namespace and a FLAT (exact) cosine
    vector field. The vector lives in the answer's hash, so it expires and is invalidated with it."""
    INDEX = "llmgw:cache:idx"

    def __init__(self, redis, dimensions: int, index: str = INDEX, prefix: str = f"{PREFIX}/"):
        # The Query Engine indexes database 0 only; tests pass their own index name and key prefix.
        self.redis, self.dimensions, self.index, self.prefix = redis, dimensions, index, prefix
        self._ready = False

    async def ensure(self) -> None:
        if self._ready:
            return
        from redis.commands.search.field import TagField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType
        try:
            await self.redis.ft(self.index).create_index(
                [TagField("namespace"), VectorField("vec", "FLAT", {"TYPE": "FLOAT32", "DIM": self.dimensions,
                                                                    "DISTANCE_METRIC": "COSINE"})],
                definition=IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH))
        except ResponseError as exc:
            if "already exists" not in str(exc).lower():
                raise
        self._ready = True

    async def add(self, name: str, space: str, vector: np.ndarray, ttl_seconds: int) -> None:
        pass  # ResponseCache.put already wrote the vector into the hash, which the index follows

    async def nearest(self, space: str, vector: np.ndarray) -> tuple[str, float] | None:
        from redis.commands.search.query import Query
        await self.ensure()
        # Punctuation in a TAG value is query syntax unless escaped (a namespace is hex, but stay safe).
        tag = re.sub(r"([^A-Za-z0-9_])", r"\\\1", space)
        query = (Query(f"(@namespace:{{{tag}}})=>[KNN 1 @vec $v AS dist]").return_fields("dist")
                 .sort_by("dist").dialect(2))
        result = await self.redis.ft(self.index).search(
            query, query_params={"v": np.asarray(vector, dtype=np.float32).tobytes()})
        if not result.docs:
            return None
        doc = result.docs[0]
        return _text(doc.id), 1.0 - float(_text(doc.dist))  # cosine distance is 1 - cosine similarity


class ResponseCache:
    def __init__(self, redis, clock=time.time, index=None):
        self.redis, self.clock, self.index = redis, clock, index
        self.metrics = None

    @staticmethod
    def _key(tenant: str, feature: str, key: str) -> str:
        return f"{PREFIX}/{tenant}/{feature}/{key}"

    def _error(self, what: str, exc: Exception) -> None:
        log.warning("cache %s failed: %s", what, exc)
        if self.metrics is not None:
            self.metrics.cache_errors.labels(what).inc()

    async def _read(self, name: str, tenant: str) -> CachedAnswer | None:
        raw = await self.redis.hget(name, "answer")
        if raw is None:
            return None  # expired or invalidated since the index saw it
        answer = CachedAnswer(**json.loads(raw))
        return answer if answer.tenant == tenant else None  # the key already holds the tenant; belt and braces

    async def get(self, tenant: str, feature: str, key: str) -> CachedAnswer | None:
        try:
            return await self._read(self._key(tenant, feature, key), tenant)
        except (RedisError, OSError) as exc:
            self._error("lookup", exc)
            return None

    async def search(self, tenant: str, space: str, vector: np.ndarray) -> tuple[CachedAnswer | None, float | None]:
        """The nearest stored answer in this namespace and its cosine similarity; (None, None) when
        the namespace is empty or the index cannot be reached. The caller applies the floor."""
        try:
            found = await self.index.nearest(space, vector)
            if found is None:
                return None, None
            name, similarity = found
            return await self._read(name, tenant), similarity
        except (RedisError, OSError) as exc:
            self._error("semantic_lookup", exc)
            return None, None

    async def put(self, tenant: str, feature: str, key: str, space: str, answer: CachedAnswer, ttl_seconds: int,
                  vector: np.ndarray | None = None) -> None:
        name = self._key(tenant, feature, key)
        fields = {"answer": json.dumps(asdict(answer), ensure_ascii=False), "namespace": space}
        if vector is not None:
            fields["vec"] = np.asarray(vector, dtype=np.float32).tobytes()
        try:
            pipe = self.redis.pipeline(transaction=True)
            pipe.hset(name, mapping=fields)
            pipe.expire(name, ttl_seconds)
            await pipe.execute()
            if vector is not None and self.index is not None:
                await self.index.add(name, space, vector, ttl_seconds)
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
