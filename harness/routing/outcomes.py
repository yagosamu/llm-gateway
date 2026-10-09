"""Per-prompt outcomes: for each evaluated prompt, whether each tier's answer is acceptable and what
it cost. This is the table every policy is scored on.

Candidate tiers (low, medium): acceptable when the rules do not reject the answer and both judges
accept it. High tier: acceptable unless the rules reject gpt-6.1-sol's own answer. Cost is what the
tier's model cost on that prompt in the recording, which slice 1 showed equals the gateway's own
attribution."""
from harness.routing import prereg
from harness.routing.pairs import excluded_prompts, is_auto_unacceptable


class IncompleteVerdicts(ValueError):
    pass


def consensus(verdicts: list[dict]) -> dict[tuple[str, str], bool]:
    """(prompt_id, tier) -> True when every pre-registered judge accepted, from ok verdicts of the
    current rubric. Raises when a judged pair lacks a verdict from either judge."""
    by_pair: dict[tuple[str, str], dict[str, bool]] = {}
    for v in verdicts:
        if v["status"] == "ok" and v["rubric_version"] == prereg.RUBRIC_VERSION:
            by_pair.setdefault((v["prompt_id"], v["tier"]), {})[v["judge"]] = v["acceptable"]
    incomplete = sorted(k for k, judges in by_pair.items() if set(judges) != set(prereg.JUDGES))
    if incomplete:
        raise IncompleteVerdicts(f"{len(incomplete)} pairs lack a verdict from a judge, first {incomplete[:3]}")
    return {k: all(judges.values()) for k, judges in by_pair.items()}


def build_outcomes(rows: list[dict], matrix: dict, verdicts: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """prompt_id -> {category, messages, accept: {tier: bool}, cost: {tier: usd}}, plus the excluded ids."""
    excluded = excluded_prompts(rows, matrix)
    agreed = consensus(verdicts)
    outcomes = {}
    for row in rows:
        if row["id"] in excluded:
            continue
        accept, cost = {}, {}
        for tier, model in prereg.TIER_MAP.items():
            record = matrix[(row["id"], model)]
            cost[tier] = record["cost_usd"]
            if is_auto_unacceptable(record):
                accept[tier] = False
            elif tier == "high":
                accept[tier] = True
            else:
                if (row["id"], tier) not in agreed:
                    raise IncompleteVerdicts(f"{row['id']} {tier} was never judged")
                accept[tier] = agreed[(row["id"], tier)]
        outcomes[row["id"]] = {"category": row["category"], "messages": row["messages"], "accept": accept, "cost": cost}
    return outcomes, excluded
