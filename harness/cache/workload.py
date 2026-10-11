"""Build the slice 4 workload described in harness/cache/prereg.py, in four steps.

  intents      choose the 160 intents and their tune or test half (no call)
  paraphrase   ask gpt-4.1-mini for 4 rewrites of each intent's instruction (paid, resumable)
  embed-dolly  embed every deduplicated Dolly user message, to find hard negatives (paid)
  build        mine hard negatives, draw the random tail, assemble and shuffle the 1,000 queries (no call)

Paid steps print their worst case and refuse to start past the project budget; --dry-run stops there.
Every paid call is written to the spend ledger.

Committed: data/cache/intents.jsonl, paraphrases.jsonl, texts.jsonl, workload.jsonl, workload_manifest.json.
Not committed: the Dolly embeddings (var/cache, about 90 MB), which only serve to choose the hard
negatives; the chosen ones, and their similarity to their intent, are in the workload.

Usage: uv run [--env-file .env] python -m harness.cache.workload <step> [--dry-run]"""
import asyncio
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from harness import budget
from harness.cache import prereg
from harness.routing.judge import GPT_41_MINI
from harness.traffic import dolly
from harness.traffic.sample import SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION, TENANTS, dedupe, user_message
from llm_gateway import embeddings
from llm_gateway.registry import call_ceiling_usd, cost_usd

RAW_PATH = Path("data/raw/databricks-dolly-15k.jsonl")
DATA = Path("data/cache")
INTENTS_PATH = DATA / "intents.jsonl"
PARAPHRASES_PATH = DATA / "paraphrases.jsonl"
TEXTS_PATH = DATA / "texts.jsonl"
WORKLOAD_PATH = DATA / "workload.jsonl"
MANIFEST_PATH = DATA / "workload_manifest.json"
DOLLY_VECTORS = Path("var/cache/dolly_embeddings.npy")
DOLLY_VECTOR_IDS = Path("var/cache/dolly_embeddings_index.json")
PARAPHRASE_MAX_TOKENS = 1024
EMBED_BATCH = 256
CATEGORIES = ("brainstorming", "classification", "closed_qa", "creative_writing", "general_qa",
              "information_extraction", "open_qa", "summarization")


def text_id(index: int, variant: int = 0) -> str:
    return f"dolly-{index:05d}" + (f"-p{variant}" if variant else "")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def unique_records() -> list[dict]:
    records, _ = dedupe(dolly.load_records(dolly.ensure_downloaded(RAW_PATH)))
    return records


# --- intents -------------------------------------------------------------------------------------

def select_intents(records: list[dict], per_category: int = prereg.INTENTS_PER_CATEGORY,
                   seed: int = prereg.SEED) -> list[dict]:
    """per_category records from each category, each its own seeded stream; the first half of each
    draw is tune, the second test."""
    chosen = []
    for category in CATEGORIES:
        pool = sorted((r for r in records if r["category"] == category), key=lambda r: r["index"])
        drawn = random.Random(f"{seed}:intents:{category}").sample(pool, per_category)
        chosen += [{"id": text_id(r["index"]), "index": r["index"], "category": category,
                    "half": "tune" if i < per_category // 2 else "test",
                    "instruction": r["instruction"].strip(), "context": r["context"].strip(),
                    "response": r["response"].strip()} for i, r in enumerate(drawn)]
    return chosen


# --- paraphrases ---------------------------------------------------------------------------------

def paraphrase_messages(instruction: str, n: int = prereg.N_PARAPHRASES) -> list[dict]:
    return [{"role": "user", "content": prereg.PARAPHRASE_INSTRUCTION.format(n=n) + "\n\nRequest:\n" + instruction}]


def parse_paraphrases(text: str) -> list[str]:
    """The JSON list in the model's reply, tolerating a code fence around it."""
    body = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fenced:
        body = fenced.group(1).strip()
    items = json.loads(body)
    if not isinstance(items, list) or not all(isinstance(s, str) for s in items):
        raise ValueError("not a JSON list of strings")
    return items


def clean_paraphrases(original: str, paraphrases: list[str]) -> list[str]:
    """Drop rewrites equal to the original or to an earlier rewrite, after whitespace normalization."""
    seen, kept = {normalize_ws(original)}, []
    for p in paraphrases:
        key = normalize_ws(p)
        if key and key not in seen:
            seen.add(key)
            kept.append(p.strip())
    return kept


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def paraphrase_all(intents: list[dict], done: set[str], concurrency: int = 8) -> None:
    import openai
    client = openai.AsyncOpenAI(timeout=60.0, max_retries=0)
    gate = asyncio.Semaphore(concurrency)
    PARAPHRASES_PATH.parent.mkdir(parents=True, exist_ok=True)

    async def one(intent: dict) -> None:
        async with gate:
            row = {"intent": intent["id"], "model": prereg.PARAPHRASE_MODEL}
            try:
                response = await client.chat.completions.create(
                    model=prereg.PARAPHRASE_MODEL, messages=paraphrase_messages(intent["instruction"]),
                    max_completion_tokens=PARAPHRASE_MAX_TOKENS)
            except openai.OpenAIError as exc:
                row |= {"error": f"{type(exc).__name__}: {exc}"[:300], "paraphrases": []}
            else:
                usage, choice = response.usage, response.choices[0]
                cost = cost_usd(GPT_41_MINI, usage.prompt_tokens, usage.completion_tokens)
                budget.append(budget.LedgerEntry("slice4-paraphrase", intent["id"], prereg.PARAPHRASE_MODEL,
                                                 usage.prompt_tokens, usage.completion_tokens, cost, _now()))
                row |= {"finish_reason": choice.finish_reason, "input_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens, "cost_usd": cost}
                try:
                    raw = parse_paraphrases(choice.message.content or "")
                    kept = clean_paraphrases(intent["instruction"], raw)
                    row |= {"returned": len(raw), "paraphrases": kept}
                except (ValueError, json.JSONDecodeError) as exc:
                    row |= {"error": f"unparseable: {exc}"[:300], "paraphrases": [], "raw": choice.message.content}
            with PARAPHRASES_PATH.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(i) for i in intents if i["id"] not in done))


# --- hard negatives, tail and assembly -----------------------------------------------------------

def nearest_neighbors(intent_rows: list[int], vectors: np.ndarray, k: int, excluded: set[int]) -> list[list[tuple[int, float]]]:
    """For each intent (a row of `vectors`), in the order given, the k most similar rows not excluded
    and not chosen for an earlier intent: [(row, cosine)]."""
    taken = set(excluded)
    out = []
    for row in intent_rows:
        sims = vectors @ vectors[row]
        picks = []
        for candidate in np.argsort(-sims):
            candidate = int(candidate)
            if candidate in taken:
                continue
            picks.append((candidate, float(sims[candidate])))
            taken.add(candidate)
            if len(picks) == k:
                break
        out.append(picks)
    return out


def zipf_head(intent_ids: list[str], n: int, exponent: float, seed: int) -> list[str]:
    """n draws of intents, intent at rank r (a seeded random order) weighted 1 / r ** exponent."""
    order = random.Random(f"{seed}:zipf-order").sample(intent_ids, len(intent_ids))
    weights = [1 / (rank + 1) ** exponent for rank in range(len(order))]
    return random.Random(f"{seed}:zipf-draws").choices(order, weights=weights, k=n)


def assemble(texts: dict[str, dict], intents: list[dict], hard: list[dict], tail: list[dict],
             seed: int = prereg.SEED) -> list[dict]:
    """The workload: head queries over intents' texts, then hard negatives and tail once each, each
    with a random tenant, shuffled. `texts` maps text id to {text, intent, ...}."""
    variants = {i["id"]: sorted(t for t, row in texts.items() if row.get("intent") == i["id"] and row["kind"] == "intent")
                for i in intents}
    pick = random.Random(f"{seed}:variants")
    head = [pick.choice(variants[i]) for i in zipf_head([i["id"] for i in intents], prereg.HEAD_QUERIES,
                                                        prereg.ZIPF_EXPONENT, seed)]
    ids = head + [h["id"] for h in hard] + [t["id"] for t in tail]
    tenants = random.Random(f"{seed}:tenants")
    queries = [{"text_id": t, "tenant": tenants.choice(TENANTS)} for t in ids]
    random.Random(f"{seed}:order").shuffle(queries)
    rows = []
    for q, query in enumerate(queries):
        text = texts[query["text_id"]]
        rows.append({"q": q, "text_id": query["text_id"], "kind": "head" if text["kind"] == "intent" else text["kind"],
                     "intent": text.get("intent"), "half": text["half"], "tenant": query["tenant"],
                     "category": text["category"], "source_index": text["source_index"],
                     "system_prompt_version": SYSTEM_PROMPT_VERSION,
                     "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text["text"]}],
                     "reference_response": text["reference_response"]})
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build() -> None:
    records = unique_records()
    by_index = {r["index"]: r for r in records}
    intents = read_jsonl(INTENTS_PATH)
    paraphrases = {p["intent"]: p for p in read_jsonl(PARAPHRASES_PATH)}
    missing = [i["id"] for i in intents if i["id"] not in paraphrases]
    if missing:
        raise SystemExit(f"{len(missing)} intents have no paraphrase row; run the paraphrase step first")
    order = json.loads(DOLLY_VECTOR_IDS.read_text(encoding="utf-8"))
    vectors = np.load(DOLLY_VECTORS)
    row_of = {index: row for row, index in enumerate(order)}

    texts: dict[str, dict] = {}
    for intent in intents:
        base = {"kind": "intent", "intent": intent["id"], "half": intent["half"], "category": intent["category"],
                "source_index": intent["index"], "reference_response": intent["response"]}
        texts[intent["id"]] = base | {"text": user_message(intent["instruction"], intent["context"]), "variant": 0}
        for v, p in enumerate(paraphrases[intent["id"]]["paraphrases"], start=1):
            texts[text_id(intent["index"], v)] = base | {"text": user_message(p, intent["context"]), "variant": v}

    intent_rows = [row_of[i["index"]] for i in intents]
    neighbors = nearest_neighbors(intent_rows, vectors, prereg.HARD_NEGATIVES_PER_INTENT, set(intent_rows))
    hard = []
    for intent, picks in zip(intents, neighbors):
        for row, sim in picks:
            r = by_index[order[row]]
            hard.append({"id": text_id(r["index"]), "intent": intent["id"], "half": intent["half"], "similarity": sim, "record": r})
    used = {i["index"] for i in intents} | {h["record"]["index"] for h in hard}
    pool = sorted((r for r in records if r["index"] not in used), key=lambda r: r["index"])
    drawn = random.Random(f"{prereg.SEED}:tail").sample(pool, prereg.RANDOM_TAIL)
    tail = [{"id": text_id(r["index"]), "half": "tune" if i < prereg.RANDOM_TAIL // 2 else "test", "record": r}
            for i, r in enumerate(drawn)]
    for kind, items in (("hard_negative", hard), ("tail", tail)):
        for item in items:
            r = item["record"]
            texts[item["id"]] = {"kind": kind, "intent": item.get("intent"), "half": item["half"], "category": r["category"],
                                 "source_index": r["index"], "reference_response": r["response"].strip(),
                                 "text": user_message(r["instruction"], r["context"]), "variant": 0,
                                 **({"similarity_to_intent": round(item["similarity"], 6)} if kind == "hard_negative" else {})}

    workload = assemble(texts, intents, hard, tail)
    write_jsonl(TEXTS_PATH, [{"text_id": t, **row} for t, row in sorted(texts.items())])
    write_jsonl(WORKLOAD_PATH, workload)
    sims = [h["similarity"] for h in hard]
    manifest = {
        "prereg": "harness/cache/prereg.py", "seed": prereg.SEED, "queries": len(workload),
        "by_kind": dict(Counter(q["kind"] for q in workload)), "by_half": dict(Counter(q["half"] for q in workload)),
        "by_tenant": dict(Counter(q["tenant"] for q in workload)),
        "distinct_texts": len(texts), "distinct_texts_in_workload": len({q["text_id"] for q in workload}),
        "intents": len(intents), "intents_in_workload": len({q["intent"] for q in workload if q["kind"] == "head"}),
        "paraphrases_kept": sum(len(p["paraphrases"]) for p in paraphrases.values()),
        "paraphrase_rows_with_error": sum(1 for p in paraphrases.values() if p.get("error")),
        "head_occurrences_of_top_intent": Counter(q["intent"] for q in workload if q["kind"] == "head").most_common(1)[0][1],
        "hard_negative_similarity": {"min": round(min(sims), 4), "median": round(float(np.median(sims)), 4),
                                     "max": round(max(sims), 4)},
        "embedding_model": embeddings.MODEL,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))


# --- CLI -----------------------------------------------------------------------------------------

async def embed_dolly(records: list[dict]) -> None:
    embedder = embeddings.OpenAIEmbedder()
    texts = [user_message(r["instruction"], r["context"]) for r in records]
    chunks, spent = [], 0.0
    for start in range(0, len(texts), EMBED_BATCH):
        result = await embedder.embed(texts[start:start + EMBED_BATCH])
        cost = embeddings.cost_usd(result.tokens)
        budget.append(budget.LedgerEntry("slice4-embed-dolly", f"batch-{start // EMBED_BATCH:03d}", embeddings.MODEL,
                                         result.tokens, 0, cost, _now()))
        chunks.append(result.vectors)
        spent += cost
        print(f"{min(start + EMBED_BATCH, len(texts))} of {len(texts)} embedded, US$ {spent:.4f}", flush=True)
    DOLLY_VECTORS.parent.mkdir(parents=True, exist_ok=True)
    np.save(DOLLY_VECTORS, np.vstack(chunks))
    DOLLY_VECTOR_IDS.write_text(json.dumps([r["index"] for r in records]), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    step, dry = (args[0] if args else ""), "--dry-run" in args
    if step == "intents":
        intents = select_intents(unique_records())
        write_jsonl(INTENTS_PATH, intents)
        print(f"{len(intents)} intents -> {INTENTS_PATH}: {dict(Counter(i['half'] for i in intents))}")
    elif step == "paraphrase":
        intents = read_jsonl(INTENTS_PATH)
        done = {p["intent"] for p in read_jsonl(PARAPHRASES_PATH)} if PARAPHRASES_PATH.exists() else set()
        todo = [i for i in intents if i["id"] not in done]
        ceiling = sum(call_ceiling_usd(GPT_41_MINI, paraphrase_messages(i["instruction"]), PARAPHRASE_MAX_TOKENS)
                      for i in todo)
        left = budget.preflight("slice4-paraphrase", ceiling)
        print(f"{len(todo)} intents to paraphrase; worst case US$ {ceiling:.3f}; budget left after it at least US$ {left:.2f}")
        if not dry:
            asyncio.run(paraphrase_all(todo, done))
    elif step == "embed-dolly":
        records = sorted(unique_records(), key=lambda r: r["index"])
        ceiling = embeddings.ceiling_usd([user_message(r["instruction"], r["context"]) for r in records])
        left = budget.preflight("slice4-embed-dolly", ceiling)
        print(f"{len(records)} Dolly texts to embed; worst case US$ {ceiling:.3f}; budget left after it at least US$ {left:.2f}")
        if not dry:
            asyncio.run(embed_dolly(records))
    elif step == "build":
        build()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
