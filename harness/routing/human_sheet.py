"""Build the blind human-calibration sheet: 30 pairs, 15 per candidate tier, no model names.

The sheet goes to data/judgments/human/sheet.csv (UTF-8 with BOM, so Excel shows accents) for the
human to fill in the `acceptable` column with yes or no. Which model wrote each candidate answer is
kept apart in key.json, so the grader can stay blind to it. The sheet is built from the recording
alone, before any judge has run.

Usage: uv run python -m harness.routing.human_sheet"""
import csv
import json
import random
import sys
from pathlib import Path

from harness.record import RECORDINGS_PATH, load_records, load_rows
from harness.replay import current_records
from harness.routing import prereg
from harness.routing.pairs import build_pairs
from llm_gateway.registry import load_registry

HUMAN_DIR = Path("data/judgments/human")
COLUMNS = ("item", "request", "reference_answer", "candidate_answer", "acceptable", "note")


def sample_items(pairs: list[dict], n: int = prereg.N_HUMAN_ITEMS, seed: int = prereg.SEED) -> list[dict]:
    """n judged pairs, the same number from each candidate tier, in a shuffled order."""
    rng = random.Random(seed)
    per_tier = n // len(prereg.CANDIDATE_TIERS)
    chosen = []
    for tier in prereg.CANDIDATE_TIERS:
        judged = [p for p in pairs if p["tier"] == tier and not p["auto_unacceptable"]]
        chosen += rng.sample(judged, per_tier)
    rng.shuffle(chosen)
    return chosen


def write_sheet(items: list[dict], directory: Path = HUMAN_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for i, p in enumerate(items, start=1):
            writer.writerow([i, p["request"], p["reference_text"], p["candidate_text"], "", ""])
    key = {i: {"prompt_id": p["prompt_id"], "tier": p["tier"], "model": p["model"]}
           for i, p in enumerate(items, start=1)}
    (directory / "key.json").write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8", newline="\n")
    (directory / "INSTRUCTIONS.md").write_text(
        "# How to grade\n\n"
        "Open sheet.csv. For each row, read the request, the reference answer and the candidate answer, and "
        "write yes or no in the `acceptable` column. The `note` column is optional. Do not open key.json until "
        "you are done: it says which model wrote each candidate.\n\n"
        "The judges get exactly this rubric:\n\n" + prereg.RUBRIC + "\n",
        encoding="utf-8", newline="\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    matrix = current_records(load_records(RECORDINGS_PATH), load_registry())
    items = sample_items(build_pairs(load_rows(), matrix))
    write_sheet(items)
    print(f"{len(items)} items -> {HUMAN_DIR / 'sheet.csv'}")


if __name__ == "__main__":
    main()
