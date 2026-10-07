"""Build data/traffic.jsonl and data/traffic_manifest.json from the pinned Dolly file.

The sample is stratified: the same number of prompts from each of Dolly's eight categories, not
Dolly's natural mix. Any cost figure computed over this traffic depends on that mix, and the
manifest records it next to Dolly's own category counts.

Usage: uv run python -m harness.traffic.build"""
import json
import sys
from collections import Counter
from pathlib import Path

from harness.traffic import dolly
from harness.traffic.sample import (
    DATASET,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERSION,
    TENANTS,
    build_row,
    dedupe,
    stratified_sample,
)

SEED = 20261007
PER_CATEGORY = 30
DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw" / "databricks-dolly-15k.jsonl"
TRAFFIC_PATH = DATA_DIR / "traffic.jsonl"
MANIFEST_PATH = DATA_DIR / "traffic_manifest.json"


def build(records: list[dict]) -> tuple[list[dict], dict]:
    unique, dropped = dedupe(records)
    rows = [build_row(r, dolly.REVISION) for r in stratified_sample(unique, PER_CATEGORY, SEED)]
    manifest = {
        "dataset": DATASET,
        "revision": dolly.REVISION,
        "sha256": dolly.SHA256,
        "license": dolly.LICENSE,
        "n_source": len(records),
        "n_duplicates_dropped": dropped,
        "source_category_counts": dict(sorted(Counter(r["category"] for r in unique).items())),
        "seed": SEED,
        "per_category": PER_CATEGORY,
        "n_rows": len(rows),
        "tenants": list(TENANTS),
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
    }
    return rows, manifest


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows, manifest = build(dolly.load_records(dolly.ensure_downloaded(RAW_PATH)))
    with TRAFFIC_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                             newline="\n")
    print(f"{len(rows)} rows -> {TRAFFIC_PATH}; {manifest['n_duplicates_dropped']} duplicates dropped")


if __name__ == "__main__":
    main()
