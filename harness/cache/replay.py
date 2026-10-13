"""Replay the slice 4 workload through the gateway, offline, once per similarity floor.

The real gateway code answers every query: create_app with the cache on (exact layer plus semantic
layer at the floor), recorded answers in place of the provider (harness.replay.ReplayProvider),
recorded embeddings in place of the embedding API, an in-memory vector index (the same contract as
the Redis one, tests/test_semantic_cache.py) and fakeredis for the cache entries. Nothing is called
and nothing is spent, so CI can run it.

All queries carry one X-Feature ("assistant"): harness/cache/prereg.py keys the cache on tenant,
system prompt version, model and parameters, not on a feature, and Dolly's category is not a product
surface. The category stays in each row for per-category reporting.

Each replayed query yields: the layer that answered (miss, exact, semantic), the stored text it
matched and the tenant that stored it, and the nearest similarity the semantic layer saw.

Usage: uv run python -m harness.cache.replay pairs   (writes data/cache/pairs.jsonl: the semantic-hit
pairs that occur at any floor, the labels the judges need)"""
import asyncio
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fakeredis
import httpx2
import numpy as np
import yaml

from harness import record as slice1
from harness.cache import prereg
from harness.cache.record import ANSWERS_PATH, EMBEDDINGS_INDEX, EMBEDDINGS_PATH, as_rows, workload_texts
from harness.cache.workload import WORKLOAD_PATH, read_jsonl, write_jsonl
from harness.replay import ReplayProvider, current_records
from llm_gateway.api import create_app
from llm_gateway.cache import MemoryVectorIndex, ResponseCache
from llm_gateway.embeddings import Embedding
from llm_gateway.registry import load_registry
from llm_gateway.request_log import RequestLog
from llm_gateway.routing import DEFAULT_CONFIG_PATH, Router

FEATURE = "assistant"
PAIRS_PATH = Path("data/cache/pairs.jsonl")


class RecordedEmbedder:
    """The embedding recorded for each text, with its recorded token count and latency."""

    def __init__(self, texts: list[dict]):
        index = {r["text_id"]: r for r in read_jsonl(EMBEDDINGS_INDEX)}
        vectors = np.load(EMBEDDINGS_PATH)
        self.by_text = {t["text"]: (vectors[index[t["text_id"]]["row"]], index[t["text_id"]]) for t in texts}

    async def embed(self, texts: list[str]) -> Embedding:
        rows = [self.by_text[t] for t in texts]
        return Embedding(np.stack([v for v, _ in rows]), sum(r["tokens"] for _, r in rows),
                         sum(r["latency_ms"] for _, r in rows) / 1000)


class TracingIndex(MemoryVectorIndex):
    """The in-memory index, remembering which text and tenant stored each entry and what the last
    search found, so a replayed hit can be traced to the request it reuses."""

    def __init__(self):
        super().__init__()
        self.current: tuple[str, str] | None = None
        self.owner: dict[str, tuple[str, str]] = {}
        self.last: tuple[str, float] | None = None

    async def add(self, name, space, vector, ttl_seconds):
        self.owner[name] = self.current
        await super().add(name, space, vector, ttl_seconds)

    async def nearest(self, space, vector):
        self.last = await super().nearest(space, vector)
        return self.last


class TracingCache(ResponseCache):
    """The cache, remembering which text and tenant stored each entry and which entry the last exact
    lookup found, so an exact hit can be traced too."""

    def __init__(self, redis, index: TracingIndex):
        super().__init__(redis, index=index)
        self.last_exact: str | None = None

    async def get(self, tenant, feature, key):
        answer = await super().get(tenant, feature, key)
        self.last_exact = self._key(tenant, feature, key) if answer else None
        return answer

    async def put(self, tenant, feature, key, space, answer, ttl_seconds, vector=None):
        self.index.owner[self._key(tenant, feature, key)] = self.index.current
        await super().put(tenant, feature, key, space, answer, ttl_seconds, vector)


def routing_file(threshold: float | None, directory: Path) -> Path:
    """The committed routing config with the cache on; threshold None keeps only the exact layer."""
    label = "exact-only" if threshold is None else f"{threshold:.2f}"
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["version"] = f"cache-replay-{label}"
    data["verifier"] = {"sample_rate": 0.0, "daily_budget_usd": 0.0}
    data["cache"] = {"enabled": True, "ttl_seconds": 86_400, "features": ["*"]}
    if threshold is not None:
        data["cache"]["semantic_threshold"] = threshold
    path = directory / f"routing-{label}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


class Replayer:
    def __init__(self, directory: Path):
        self.directory = directory
        self.registry = load_registry()
        self.texts = workload_texts()
        self.by_id = {t["text_id"]: t for t in self.texts}
        self.matrix = current_records(slice1.load_records(ANSWERS_PATH), self.registry)
        self.provider = ReplayProvider(self.matrix, as_rows(self.texts))
        self.embedder = RecordedEmbedder(self.texts)
        self.workload = read_jsonl(WORKLOAD_PATH)

    async def run(self, threshold: float | None, queries: list[dict] | None = None, app_state: dict | None = None,
                  headers: dict | None = None) -> list[dict]:
        """Replay `queries` (default: the workload) at one floor (None: exact layer only). Pass the same
        app_state dict to several calls to keep the cache between them (the system prompt change
        check). A query may carry "sent_as_tenant" to be sent under another tenant than its own."""
        state = app_state if app_state is not None else {}
        if "app" not in state:
            index = TracingIndex()
            cache = TracingCache(fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer()), index)
            state["index"], state["cache"] = index, cache
            state["app"] = create_app(self.registry, {p: self.provider for p in ("openai", "anthropic", "groq")},
                                      RequestLog(":memory:"), router=Router(routing_file(threshold, self.directory),
                                                                            set(self.registry)),
                                      cache=cache, embedder=self.embedder)
        index, cache, rows = state["index"], state["cache"], []
        transport = httpx2.ASGITransport(app=state["app"])
        async with httpx2.AsyncClient(transport=transport, base_url="http://gateway", timeout=None) as client:
            for q in queries if queries is not None else self.workload:
                index.current, index.last = (q["text_id"], q["tenant"]), None
                response = await client.post("/v1/chat/completions", headers={
                    "X-Tenant-Id": q.get("sent_as_tenant", q["tenant"]), "X-Feature": FEATURE,
                    "X-Request-Id": f"q{q['q']}", **(headers or {}),
                }, json={"model": "auto", "messages": q["messages"], "max_completion_tokens": slice1.OUTPUT_LIMIT})
                body = response.json()
                cache_info = body["gateway"]["cache"]
                layer = cache_info["layer"] if cache_info else "miss"
                found = {"semantic": index.last[0] if index.last else None, "exact": cache.last_exact}.get(layer)
                matched = index.owner.get(found) if found else None
                rows.append({"q": q["q"], "text_id": q["text_id"], "tenant": q["tenant"], "half": q["half"],
                             "kind": q["kind"], "category": q["category"], "layer": layer,
                             "matched_text_id": matched[0] if matched else None,
                             "matched_tenant": matched[1] if matched else None, "matched_entry": found,
                             "nearest_similarity": index.last[1] if index.last else None,
                             "lookup_ms": body["gateway"]["latency_ms"] if cache_info else None})
        return rows


def pair_id(query_text_id: str, stored_text_id: str) -> str:
    return f"{query_text_id}<-{stored_text_id}"


def build_pairs(replays: dict[float, list[dict]], replayer: Replayer) -> list[dict]:
    """Every distinct (query text, stored text) semantic hit at any floor, ready for the judges."""
    seen: dict[str, dict] = {}
    for threshold, rows in sorted(replays.items()):
        for r in rows:
            if r["layer"] != "semantic":
                continue
            pid = pair_id(r["text_id"], r["matched_text_id"])
            entry = seen.setdefault(pid, {"pair_id": pid, "query_text_id": r["text_id"], "stored_text_id": r["matched_text_id"],
                                          "similarity": round(r["nearest_similarity"], 6), "category": r["category"],
                                          "half": r["half"], "occurrences": Counter()})
            entry["occurrences"][f"{threshold:.2f}"] += 1
    pairs = []
    for entry in seen.values():
        query, stored = replayer.by_id[entry["query_text_id"]], replayer.by_id[entry["stored_text_id"]]
        fresh = replayer.matrix[(query["text_id"], "gpt-6-luna")]
        usable = fresh["finish_reason"] == "stop" and fresh["text"].strip()
        candidate = replayer.matrix[(stored["text_id"], "gpt-6-luna")]
        pairs.append(entry | {
            "occurrences": dict(entry["occurrences"]),
            "same_intent": bool(query.get("intent")) and query.get("intent") == stored.get("intent")
                           and query["kind"] == stored["kind"] == "intent",
            # Fields the slice 2 judge harness reads.
            "prompt_id": entry["pair_id"], "tier": "cache", "model": "gpt-6-luna",
            "reference_model": "gpt-6-luna" if usable else "dolly-human",
            "request": query["text"], "reference_text": fresh["text"] if usable else query["reference_response"],
            "candidate_text": candidate["text"], "auto_unacceptable": False})
    return sorted(pairs, key=lambda p: p["pair_id"])


def sample_for_labels(pairs: list[dict]) -> list[dict]:
    if len(pairs) <= prereg.MAX_LABELED_PAIRS:
        return pairs
    return sorted(random.Random(f"{prereg.SEED}:labels").sample(pairs, prereg.MAX_LABELED_PAIRS),
                  key=lambda p: p["pair_id"])


async def replay_all(directory: Path) -> tuple[Replayer, dict[float, list[dict]]]:
    replayer = Replayer(directory)
    replays = {}
    for t in prereg.THRESHOLDS:
        replays[t] = await replayer.run(t)
        layers = Counter(r["layer"] for r in replays[t])
        print(f"floor {t:.2f}: {dict(layers)}", flush=True)
    return replayer, replays


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:] != ["pairs"]:
        sys.exit(__doc__)
    directory = Path("var/cache/replay")
    directory.mkdir(parents=True, exist_ok=True)
    replayer, replays = asyncio.run(replay_all(directory))
    pairs = build_pairs(replays, replayer)
    labeled = sample_for_labels(pairs)
    write_jsonl(PAIRS_PATH, labeled)
    by_kind = Counter("same intent" if p["same_intent"] else "different requests" for p in pairs)
    print(f"{len(pairs)} distinct semantic-hit pairs at any floor ({dict(by_kind)}); {len(labeled)} written for "
          f"labelling to {PAIRS_PATH}")


if __name__ == "__main__":
    main()
