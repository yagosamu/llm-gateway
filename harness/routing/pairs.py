"""The (prompt, candidate) pairs the judges and the human grade, built from the recording."""
from harness.routing import prereg


def is_auto_unacceptable(record: dict) -> str | None:
    """The rule that rejects an answer without a judge, or None when it must be judged."""
    if record["finish_reason"] == "length":
        return "cut by the output limit"
    if not record["text"].strip():
        return "empty answer"
    return None


def reference_for(prompt_id: str, matrix: dict) -> dict | None:
    """The reference model's answer, the fallback's when it is unusable, or None when both are."""
    for model in (prereg.REFERENCE_MODEL, prereg.FALLBACK_REFERENCE_MODEL):
        record = matrix[(prompt_id, model)]
        if not is_auto_unacceptable(record):
            return record
    return None


def excluded_prompts(rows: list[dict], matrix: dict) -> list[str]:
    """Prompts with no usable reference answer; they are left out of the evaluation."""
    return [row["id"] for row in rows if reference_for(row["id"], matrix) is None]


def build_pairs(rows: list[dict], matrix: dict) -> list[dict]:
    """One pair per evaluated prompt and candidate tier, in traffic order then tier order."""
    pairs = []
    for row in rows:
        reference = reference_for(row["id"], matrix)
        if reference is None:
            continue
        for tier in prereg.CANDIDATE_TIERS:
            model = prereg.TIER_MAP[tier]
            candidate = matrix[(row["id"], model)]
            pairs.append({
                "prompt_id": row["id"], "category": row["category"], "tier": tier, "model": model,
                "request": row["messages"][1]["content"],
                "reference_model": reference["model"], "reference_text": reference["text"],
                "candidate_text": candidate["text"], "auto_unacceptable": is_auto_unacceptable(candidate),
            })
    return pairs
