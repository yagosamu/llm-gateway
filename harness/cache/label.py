"""Label the slice 4 semantic-hit pairs (data/cache/pairs.jsonl, from harness.cache.replay).

  judge   both slice 2 judges grade each pair's cached answer against the fresh answer to the query,
          with slice 2's rubric and harness unchanged (paid, resumable; worst case checked first)
  human   the blind human sample: 15 pairs both judges accepted and 15 at least one rejected (fewer
          if fewer exist), drawn with the pre-registered seed; writes data/cache/human/ with the same
          browser page as slice 2

Usage: uv run --env-file .env python -m harness.cache.label <judge [--dry-run] | human>"""
import asyncio
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from harness.cache import prereg
from harness.cache.replay import PAIRS_PATH
from harness.cache.workload import read_jsonl
from harness.routing import judge
from harness.routing.human_page import build_page

VERDICTS_PATH = Path("data/cache/verdicts.jsonl")
HUMAN_DIR = Path("data/cache/human")


def verdicts_by_pair() -> dict[str, dict[str, bool]]:
    """pair id -> {judge: acceptable}, from the successful verdicts."""
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    for v in judge.load_verdicts(VERDICTS_PATH) if VERDICTS_PATH.exists() else []:
        if v["status"] == "ok":
            out[v["prompt_id"]][v["judge"]] = v["acceptable"]
    return out


def human_sample(pairs: list[dict], verdicts: dict[str, dict[str, bool]]) -> list[dict]:
    graded = [p for p in pairs if len(verdicts.get(p["pair_id"], {})) == len(prereg.JUDGES)]
    accepted = [p for p in graded if all(verdicts[p["pair_id"]].values())]
    rejected = [p for p in graded if not all(verdicts[p["pair_id"]].values())]
    rng = random.Random(f"{prereg.SEED}:human")
    half = prereg.N_HUMAN_ITEMS // 2
    chosen = rng.sample(accepted, min(half, len(accepted))) + rng.sample(rejected, min(half, len(rejected)))
    rng.shuffle(chosen)
    return chosen


def write_human(items: list[dict]) -> None:
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    with (HUMAN_DIR / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "request", "reference_answer", "candidate_answer", "acceptable", "note"])
        for i, p in enumerate(items, start=1):
            writer.writerow([i, p["request"], p["reference_text"], p["candidate_text"], "", ""])
    key = {i: {"pair_id": p["pair_id"], "similarity": p["similarity"]} for i, p in enumerate(items, start=1)}
    (HUMAN_DIR / "key.json").write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8", newline="\n")
    page = [{"item": str(i), "request": p["request"], "reference_answer": p["reference_text"],
             "candidate_answer": p["candidate_text"]} for i, p in enumerate(items, start=1)]
    (HUMAN_DIR / "grade.html").write_text(build_page(page), encoding="utf-8", newline="\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    step, dry = (sys.argv[1] if len(sys.argv) > 1 else ""), "--dry-run" in sys.argv
    pairs = read_jsonl(PAIRS_PATH)
    if step == "judge":
        done = {v["key"] for v in judge.load_verdicts(VERDICTS_PATH) if v["status"] == "ok"} if VERDICTS_PATH.exists() else set()
        calls = judge.plan(pairs, done)
        models = judge.judge_models()
        ceiling = judge.plan_ceiling_usd(calls, models)
        from harness import budget
        left = budget.preflight("slice4-judge", ceiling)
        print(f"{len(pairs)} pairs, {len(calls)} judge calls to make; worst case US$ {ceiling:.3f}; budget left "
              f"after it at least US$ {left:.2f}", flush=True)
        if dry or not calls:
            return
        records = asyncio.run(judge.run(calls, judge.Judges(), models, "slice4-judge", verdicts_path=VERDICTS_PATH))
        print(judge.summarize(records))
    elif step == "human":
        items = human_sample(pairs, verdicts_by_pair())
        write_human(items)
        print(f"{len(items)} blind items -> {HUMAN_DIR / 'grade.html'}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
