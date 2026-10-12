"""Record, once, what the slice 4 replay needs from paid APIs, for every distinct text of the workload.

  answers     gpt-6-luna's fresh answer to each text: the no-cache baseline (cost and latency), the
              answer a miss stores, and the judges' reference (harness/cache/prereg.py). Same path
              and output limit as slice 1 (harness.record), resumable, every call in the ledger.
  embeddings  each text embedded alone, one call at a time, so the recorded latency is what the
              semantic layer adds to a request; plus, for texts with a context block, the instruction
              alone, which the instruction-only counterfactual needs (batched; latency not used).

Writes data/cache/answers.jsonl, data/cache/embeddings.npy (rows in the order of
data/cache/embeddings.jsonl), data/cache/embeddings_instruction_only.npy (+ its .jsonl).

Usage: uv run --env-file .env python -m harness.cache.record <answers | embeddings> [--dry-run]"""
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from harness import budget
from harness import record as slice1
from harness.cache.workload import TEXTS_PATH, WORKLOAD_PATH, read_jsonl, write_jsonl
from llm_gateway import embeddings
from llm_gateway.registry import load_registry

MODEL = "gpt-6-luna"  # the low tier routing.yaml serves
DATA = Path("data/cache")
ANSWERS_PATH = DATA / "answers.jsonl"
EMBEDDINGS_PATH, EMBEDDINGS_INDEX = DATA / "embeddings.npy", DATA / "embeddings.jsonl"
INSTRUCTION_PATH, INSTRUCTION_INDEX = DATA / "embeddings_instruction_only.npy", DATA / "embeddings_instruction_only.jsonl"
CONTEXT_MARK = "\n\nContext:\n"  # how harness.traffic.sample.user_message joins instruction and context
SYSTEM_PROMPT_TENANT = "cache-eval"  # the answer does not depend on the tenant; one label for all


def workload_texts() -> list[dict]:
    """The distinct texts the workload asks, sorted by text id."""
    used = {q["text_id"] for q in read_jsonl(WORKLOAD_PATH)}
    return [t for t in read_jsonl(TEXTS_PATH) if t["text_id"] in used]


def as_rows(texts: list[dict]) -> list[dict]:
    """In the shape harness.record sends: id, tenant, category, messages."""
    from harness.traffic.sample import SYSTEM_PROMPT
    return [{"id": t["text_id"], "tenant": SYSTEM_PROMPT_TENANT, "category": t["category"],
             "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": t["text"]}]}
            for t in texts]


def instruction_only(text: str) -> str | None:
    """The instruction of a text with a context block; None when it has none."""
    return text.split(CONTEXT_MARK, 1)[0] if CONTEXT_MARK in text else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def record_embeddings(texts: list[dict]) -> None:
    embedder = embeddings.OpenAIEmbedder()
    vectors, index = [], []
    for i, t in enumerate(texts):
        result = await embedder.embed([t["text"]])
        cost = embeddings.cost_usd(result.tokens)
        budget.append(budget.LedgerEntry("slice4-embeddings", t["text_id"], embeddings.MODEL, result.tokens, 0, cost, _now()))
        vectors.append(result.vectors[0])
        index.append({"row": i, "text_id": t["text_id"], "tokens": result.tokens, "latency_ms": round(result.seconds * 1000, 3)})
        if (i + 1) % 100 == 0:
            print(f"{i + 1} of {len(texts)} embedded", flush=True)
    np.save(EMBEDDINGS_PATH, np.vstack(vectors).astype(np.float32))
    write_jsonl(EMBEDDINGS_INDEX, index)

    with_context = [(t["text_id"], instruction_only(t["text"])) for t in texts if instruction_only(t["text"])]
    result = await embedder.embed([s for _, s in with_context])
    budget.append(budget.LedgerEntry("slice4-embeddings", "instruction-only-batch", embeddings.MODEL, result.tokens, 0,
                                     embeddings.cost_usd(result.tokens), _now()))
    np.save(INSTRUCTION_PATH, result.vectors.astype(np.float32))
    write_jsonl(INSTRUCTION_INDEX, [{"row": i, "text_id": tid} for i, (tid, _) in enumerate(with_context)])
    print(f"{len(texts)} texts embedded one by one, {len(with_context)} instructions without their context")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    step, dry = (sys.argv[1] if len(sys.argv) > 1 else ""), "--dry-run" in sys.argv
    texts = workload_texts()
    if step == "answers":
        registry = load_registry()
        model = registry[MODEL]
        done = {r["key"] for r in slice1.load_records(ANSWERS_PATH) if r["status"] == "ok"}
        calls = [(row, model) for row in as_rows(texts)
                 if slice1.call_key(row["id"], model.id, slice1.params_for(model)) not in done]
        ceiling = slice1.plan_ceiling_usd(calls)
        left = budget.preflight("slice4-answers", ceiling)
        print(f"{len(calls)} answers to record on {MODEL}; worst case US$ {ceiling:.3f}; budget left after it "
              f"at least US$ {left:.2f}", flush=True)
        if dry or not calls:
            return
        from llm_gateway.api import create_app
        from llm_gateway.providers.factory import build_providers
        from llm_gateway.request_log import RequestLog
        app = create_app(registry, build_providers({model.provider}), RequestLog(Path("var/cache/record_requests.sqlite3")))
        records = asyncio.run(slice1.run(calls, app, "slice4-answers", recordings_path=ANSWERS_PATH))
        print(slice1.summarize(records, len(texts)))
    elif step == "embeddings":
        extra = [instruction_only(t["text"]) for t in texts if instruction_only(t["text"])]
        ceiling = embeddings.ceiling_usd([t["text"] for t in texts] + extra)
        left = budget.preflight("slice4-embeddings", ceiling)
        print(f"{len(texts)} texts and {len(extra)} instructions to embed; worst case US$ {ceiling:.4f}; "
              f"budget left after it at least US$ {left:.2f}", flush=True)
        if not dry:
            asyncio.run(record_embeddings(texts))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
