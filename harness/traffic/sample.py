"""Pure functions that turn Dolly records into replay traffic rows. No file or network I/O here.

A record is one Dolly line plus its 0-based line index: {index, category, instruction, context,
response}. The traffic row is what the recorder sends through the gateway."""
import random

SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's request directly."
SYSTEM_PROMPT_VERSION = "v1"
TENANTS = ("tenant-a", "tenant-b", "tenant-c")
DATASET = "databricks/databricks-dolly-15k"


def _prompt_key(record: dict) -> tuple[str, str]:
    return record["instruction"].strip(), record["context"].strip()


def dedupe(records: list[dict]) -> tuple[list[dict], int]:
    """Keep the first record of each (instruction, context) pair, compared after stripping
    whitespace. Two identical prompts would be one cache entry and would double count one prompt
    in every per-model average. Returns (kept records in input order, number dropped)."""
    seen: set[tuple[str, str]] = set()
    kept = []
    for record in records:
        key = _prompt_key(record)
        if key not in seen:
            seen.add(key)
            kept.append(record)
    return kept, len(records) - len(kept)


def stratified_sample(records: list[dict], per_category: int, seed: int) -> list[dict]:
    """Draw `per_category` records from every category, without replacement, and give each a
    tenant. Each category has its own random stream seeded with "<seed>:<category>", so adding or
    removing a category never changes the draw of another. Tenants rotate over the draw order, so
    every tenant gets the same share of every category when per_category is a multiple of three.
    Returns the sample sorted by (category, index). Raises ValueError when a category is too small."""
    by_category: dict[str, list[dict]] = {}
    for record in records:
        by_category.setdefault(record["category"], []).append(record)
    sample = []
    for category in sorted(by_category):
        pool = sorted(by_category[category], key=lambda r: r["index"])
        if len(pool) < per_category:
            raise ValueError(f"category {category} has {len(pool)} records, fewer than {per_category}")
        drawn = random.Random(f"{seed}:{category}").sample(pool, per_category)
        sample += [{**r, "tenant": TENANTS[i % len(TENANTS)]} for i, r in enumerate(drawn)]
    return sorted(sample, key=lambda r: (r["category"], r["index"]))


def build_row(record: dict, revision: str) -> dict:
    """One traffic row: a stable id from the Dolly line index, provenance, the request class
    (Dolly's category), a fictional tenant, the chat messages and Dolly's human response kept as a
    reference. The context, when present, follows the instruction under a "Context:" label."""
    user = record["instruction"].strip()
    if record["context"].strip():
        user += "\n\nContext:\n" + record["context"].strip()
    return {
        "id": f"dolly-{record['index']:05d}",
        "source": {"dataset": DATASET, "revision": revision, "index": record["index"]},
        "category": record["category"],
        "tenant": record["tenant"],
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        "reference_response": record["response"],
    }
