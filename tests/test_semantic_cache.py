import asyncio
import os
import time
import uuid

import fakeredis
import numpy as np
import pytest
from fastapi.testclient import TestClient

from llm_gateway.api import create_app
from llm_gateway.cache import MemoryVectorIndex, RedisVectorIndex, ResponseCache, single_turn_text
from llm_gateway.embeddings import Embedding
from llm_gateway.providers.base import ProviderResult
from llm_gateway.registry import load_registry
from llm_gateway.request_log import RequestLog
from llm_gateway.routing import Router

REGISTRY = load_registry()
SYSTEM = {"role": "system", "content": "You are a helpful assistant."}


def unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


# Cosines to ORDER: REPHRASED 0.995, REFUND 0.8.
VECTORS = {"Where is my order?": unit(1, 0, 0), "Where's my order right now?": unit(1, 0.1, 0),
           "Where is my refund?": unit(0.8, 0.6, 0), "Tell me a joke": unit(0, 0, 1)}


class FakeEmbedder:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    async def embed(self, texts):
        self.calls.append(texts)
        if self.fail:
            raise RuntimeError("embedding service down")
        return Embedding(np.stack([VECTORS[t] for t in texts]), tokens=7, seconds=0.01)


class Provider:
    def __init__(self):
        self.calls = 0

    async def complete(self, model, request):
        self.calls += 1
        return ProviderResult(f"answer {self.calls}", 10, 5, "stop")


def gateway(tmp_path, threshold="0.95", embedder=None):
    path = tmp_path / "routing.yaml"
    semantic = f"  semantic_threshold: {threshold}\n" if threshold else ""
    path.write_text('version: "t"\npolicy: always_low\ntier_map:\n  low: gpt-6-luna\n  medium: claude-haiku-4-5\n'
                    f'  high: gpt-6.1-sol\ncache:\n  enabled: true\n{semantic}', encoding="utf-8")
    provider, log = Provider(), RequestLog(":memory:")
    embedder = embedder or FakeEmbedder()
    cache = ResponseCache(fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer()), index=MemoryVectorIndex())
    app = create_app(REGISTRY, {p: provider for p in ("openai", "anthropic", "groq")}, log,
                     router=Router(path, set(REGISTRY)), cache=cache, embedder=embedder)
    return TestClient(app), provider, embedder, log


def ask(client, text, tenant="tenant-a", system=SYSTEM, history=()):
    messages = ([system] if system else []) + list(history) + [{"role": "user", "content": text}]
    return client.post("/v1/chat/completions", json={"model": "auto", "messages": messages},
                       headers={"X-Tenant-Id": tenant, "X-Feature": "support", "X-Request-Id": "r"}).json()


def test_a_rephrased_question_above_the_floor_gets_the_stored_answer(tmp_path):
    client, provider, _, log = gateway(tmp_path)
    ask(client, "Where is my order?")
    hit = ask(client, "Where's my order right now?")
    assert provider.calls == 1 and hit["choices"][0]["message"]["content"] == "answer 1"
    assert hit["gateway"]["cache"]["layer"] == "semantic"
    assert hit["gateway"]["cache"]["similarity"] == pytest.approx(0.995, abs=1e-3)
    assert hit["gateway"]["cost_usd"] == pytest.approx(7 * 0.02 / 1e6)  # the embedding call only
    assert [r["cache"] for r in log.rows()] == ["miss", "semantic"]


def test_a_similar_but_different_question_below_the_floor_is_answered_fresh_and_stored(tmp_path):
    client, provider, embedder, log = gateway(tmp_path)
    ask(client, "Where is my order?")
    miss = ask(client, "Where is my refund?")
    assert provider.calls == 2 and miss["gateway"]["cache"] is None
    assert log.rows()[1]["cache_similarity"] == pytest.approx(0.8, abs=1e-3)
    assert ask(client, "Where is my refund?")["gateway"]["cache"]["layer"] == "exact"
    assert len(embedder.calls) == 2  # the exact hit needed no embedding


@pytest.mark.parametrize("change", [{"tenant": "tenant-b"}, {"system": {"role": "system", "content": "Be terse."}}])
def test_the_semantic_layer_never_crosses_a_tenant_or_a_system_prompt(tmp_path, change):
    client, provider, _, _ = gateway(tmp_path)
    ask(client, "Where is my order?")
    assert ask(client, "Where's my order right now?", **change)["gateway"]["cache"] is None
    assert provider.calls == 2


def test_multi_turn_requests_skip_the_semantic_layer(tmp_path):
    client, provider, embedder, _ = gateway(tmp_path)
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    ask(client, "Where is my order?", history=history)
    ask(client, "Where's my order right now?", history=history)
    assert embedder.calls == [] and provider.calls == 2


def test_without_a_floor_the_semantic_layer_does_not_run(tmp_path):
    client, provider, embedder, _ = gateway(tmp_path, threshold=None)
    ask(client, "Where is my order?")
    ask(client, "Where's my order right now?")
    assert embedder.calls == [] and provider.calls == 2


def test_an_embedding_failure_is_a_miss_not_a_failed_request(tmp_path):
    client, provider, _, _ = gateway(tmp_path, embedder=FakeEmbedder(fail=True))
    assert ask(client, "Where is my order?")["choices"][0]["message"]["content"] == "answer 1"
    assert 'llm_gateway_cache_errors_total{operation="embed"} 1.0' in client.get("/metrics").text


def test_single_turn_text_ignores_system_messages_only():
    assert single_turn_text([SYSTEM, {"role": "user", "content": "q"}]) == "q"
    assert single_turn_text([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]) is None
    assert single_turn_text([SYSTEM]) is None


def test_the_memory_index_forgets_expired_entries():
    now = [1000.0]
    index = MemoryVectorIndex(clock=lambda: now[0])
    asyncio.run(index.add("k", "ns", unit(1, 0), ttl_seconds=10))
    assert asyncio.run(index.nearest("ns", unit(1, 0)))[0] == "k"
    now[0] += 11
    assert asyncio.run(index.nearest("ns", unit(1, 0))) is None


# --- Contract: the Redis Query Engine index finds what the in-memory index finds. Needs a real Redis
# (docker compose up -d locally, a service container in CI); skipped when none answers.
REDIS_URL = os.environ.get("LLM_GATEWAY_TEST_REDIS_URL", "redis://127.0.0.1:6379/0")


def redis_or_skip():
    import redis
    try:
        redis.Redis.from_url(REDIS_URL, socket_connect_timeout=0.5).ping()
    except (redis.RedisError, OSError):
        pytest.skip(f"no Redis at {REDIS_URL}")


def test_the_redis_index_returns_the_same_nearest_neighbour_as_the_memory_index():
    redis_or_skip()

    async def run():
        import redis.asyncio as redis
        store = redis.Redis.from_url(REDIS_URL)
        run_id = uuid.uuid4().hex[:8]
        prefix, name = f"test:{run_id}/", f"test:{run_id}:idx"
        index, memory = RedisVectorIndex(store, 8, index=name, prefix=prefix), MemoryVectorIndex()
        await index.ensure()
        rng = np.random.default_rng(7)
        try:
            for i in range(60):
                space, vec = ("ns-a", "ns-b")[i % 2], rng.normal(size=8).astype(np.float32)
                vec /= np.linalg.norm(vec)
                key = f"{prefix}{i}"
                await store.hset(key, mapping={"namespace": space, "vec": vec.tobytes()})
                await store.expire(key, 60)
                await memory.add(key, space, vec, 60)
            await asyncio.sleep(0.5)  # indexing is asynchronous
            for _ in range(25):
                q = rng.normal(size=8).astype(np.float32)
                q /= np.linalg.norm(q)
                for space in ("ns-a", "ns-b", "ns-empty"):
                    got, expected = await index.nearest(space, q), await memory.nearest(space, q)
                    if expected is None:
                        assert got is None
                    else:
                        assert got[0] == expected[0] and got[1] == pytest.approx(expected[1], abs=1e-5)
        finally:
            await store.ft(name).dropindex(delete_documents=True)
            await store.aclose()

    asyncio.run(run())


def test_an_expired_entry_leaves_the_redis_index():
    redis_or_skip()

    async def run():
        import redis.asyncio as redis
        store = redis.Redis.from_url(REDIS_URL)
        run_id = uuid.uuid4().hex[:8]
        prefix, name = f"test:{run_id}/", f"test:{run_id}:idx"
        index = RedisVectorIndex(store, 2, index=name, prefix=prefix)
        await index.ensure()
        try:
            await store.hset(f"{prefix}x", mapping={"namespace": "ns", "vec": unit(1, 0).tobytes()})
            await store.expire(f"{prefix}x", 1)
            await asyncio.sleep(0.3)
            assert (await index.nearest("ns", unit(1, 0)))[0] == f"{prefix}x"
            time.sleep(1.5)
            assert await index.nearest("ns", unit(1, 0)) is None
        finally:
            await store.ft(name).dropindex(delete_documents=True)
            await store.aclose()

    asyncio.run(run())
