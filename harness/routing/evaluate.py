"""Score the pre-registered routing policies out of fold and write results/slice2.json and slice2.md.

Every learned policy decides each prompt with a model fitted on the other folds, through the same
FeatureTable and LogisticClassifier classes the gateway serves with. Intervals are 95% percentile
bootstrap over prompts with the pre-registered seed and resample count, from one generator consumed
in a fixed order, so the JSON is deterministic and CI can regenerate it.

Usage: uv run python -m harness.routing.evaluate"""
import csv
import json
import math
import random
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from harness.routing import prereg
from harness.routing.judge import load_verdicts
from harness.routing.train import fit_classifier, fit_feature_table, load_outcomes
from llm_gateway.routing import FeatureTable, LogisticClassifier

RESULTS_DIR = Path("results")
JSON_PATH = RESULTS_DIR / "slice2.json"
MD_PATH = RESULTS_DIR / "slice2.md"
HUMAN_DIR = Path("data/judgments/human")
TIERS = ("low", "medium", "high")


def assign_folds(outcomes: dict[str, dict]) -> dict[str, int]:
    """Stratified by category: each category's prompts are shuffled with a seeded generator and dealt
    to the folds in turn."""
    folds = {}
    for category in sorted({o["category"] for o in outcomes.values()}):
        ids = sorted(pid for pid, o in outcomes.items() if o["category"] == category)
        random.Random(f"{prereg.SEED}:{category}").shuffle(ids)
        folds |= {pid: i % prereg.N_FOLDS for i, pid in enumerate(ids)}
    return folds


def route_all(outcomes: dict[str, dict], folds: dict[str, int]) -> dict[str, dict[str, str]]:
    """policy -> prompt_id -> tier."""
    ids = sorted(outcomes)
    routes = {p: {} for p in prereg.POLICIES}
    for pid in ids:
        for tier in TIERS:
            routes[f"always_{tier}"][pid] = tier
        o = outcomes[pid]
        acceptable = [t for t in TIERS if o["accept"][t]]
        routes["oracle"][pid] = min(acceptable, key=lambda t: o["cost"][t]) if acceptable else "high"
    for k in range(prereg.N_FOLDS):
        train = [pid for pid in ids if folds[pid] != k]
        table = FeatureTable(fit_feature_table(outcomes, train), prereg.DECISION_THRESHOLD)
        classifier = LogisticClassifier(fit_classifier(outcomes, train), prereg.DECISION_THRESHOLD)
        for pid in (p for p in ids if folds[p] == k):
            o = outcomes[pid]
            routes["feature_table"][pid] = table.decide(o["category"], o["messages"]).tier
            routes["classifier"][pid] = classifier.decide(o["category"], o["messages"]).tier
    return routes


def bootstrap_mean(values, rng) -> list[float]:
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(prereg.N_RESAMPLES, len(values)))
    low, high = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return [round(float(values.mean()), 6), round(float(low), 6), round(float(high), 6)]


def bootstrap_reduction(costs, baseline, rng) -> list[float]:
    """1 - sum(costs) / sum(baseline), resampling prompts with the pairing kept."""
    costs, baseline = np.asarray(costs, dtype=float), np.asarray(baseline, dtype=float)
    idx = rng.integers(0, len(costs), size=(prereg.N_RESAMPLES, len(costs)))
    resampled = 1 - costs[idx].sum(axis=1) / baseline[idx].sum(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return [round(float(1 - costs.sum() / baseline.sum()), 6), round(float(low), 6), round(float(high), 6)]


def mcnemar_exact(first: list[bool], second: list[bool]) -> dict:
    """Two-sided exact McNemar test. b: only first accepted; c: only second accepted."""
    b = sum(f and not s for f, s in zip(first, second))
    c = sum(s and not f for f, s in zip(first, second))
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return {"b": b, "c": c, "p": min(1.0, float(2 * Fraction(tail, 1 << n)))}


def agreement(first: list[bool], second: list[bool]) -> dict:
    """Raw agreement and Cohen's kappa. Kappa is None when chance agreement is 1 (both raters
    constant), and 0 when one rater is constant, however often they agree."""
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    p1, p2 = sum(first) / n, sum(second) / n
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    return {"n": n, "agreement": round(observed, 6), "kappa": None if kappa is None else round(kappa, 6),
            "first_yes": sum(first), "second_yes": sum(second)}


def judge_agreement(verdicts: list[dict]) -> dict:
    by_pair = {}
    for v in verdicts:
        if v["status"] == "ok" and v["rubric_version"] == prereg.RUBRIC_VERSION:
            by_pair.setdefault((v["prompt_id"], v["tier"]), {})[v["judge"]] = v["acceptable"]
    keys = sorted(by_pair)
    first, second = prereg.JUDGES
    return agreement([by_pair[k][first] for k in keys], [by_pair[k][second] for k in keys])


def human_agreement(outcomes: dict[str, dict], human_dir: Path = HUMAN_DIR) -> dict | None:
    answers_path = human_dir / "answers.csv"
    if not answers_path.exists():
        return None
    key = json.loads((human_dir / "key.json").read_text(encoding="utf-8"))
    with answers_path.open(encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        answers = {r["item"]: r for r in csv.DictReader(f, dialect=csv.Sniffer().sniff(sample, delimiters=",;"))}
    items = sorted(key, key=int)
    human = [answers[i]["acceptable"].strip().lower() == "yes" for i in items]
    judges = [outcomes[key[i]["prompt_id"]]["accept"][key[i]["tier"]] for i in items]
    result = agreement(human, judges)
    result["disagreements"] = [{"item": int(i), "prompt_id": key[i]["prompt_id"], "tier": key[i]["tier"],
                                "human": h, "judges": j} for i, h, j in zip(items, human, judges) if h != j]
    return result


def build_results(outcomes: dict[str, dict], excluded: list[str], verdicts: list[dict],
                  human_dir: Path = HUMAN_DIR) -> dict:
    ids = sorted(outcomes)
    routes = route_all(outcomes, assign_folds(outcomes))
    rng = np.random.default_rng(prereg.SEED)
    baseline = [outcomes[pid]["cost"]["high"] for pid in ids]
    policies, per_prompt = {}, {}
    for policy in prereg.POLICIES:
        tiers = [routes[policy][pid] for pid in ids]
        accepted = [outcomes[pid]["accept"][t] for pid, t in zip(ids, tiers)]
        costs = [outcomes[pid]["cost"][t] for pid, t in zip(ids, tiers)]
        per_prompt[policy] = (accepted, costs)
        acceptance = bootstrap_mean(accepted, rng)
        policies[policy] = {
            "acceptance": acceptance,
            "cost_per_1k_usd": bootstrap_mean([1000 * c for c in costs], rng),
            "cost_reduction_vs_always_high": bootstrap_reduction(costs, baseline, rng),
            "tier_share": {t: round(tiers.count(t) / len(tiers), 6) for t in TIERS},
            "meets_quality_bar": acceptance[1] >= prereg.QUALITY_BAR_LOWER_BOUND,
        }
    first, second = prereg.SECONDARY_COMPARISON
    (acc_a, cost_a), (acc_b, cost_b) = per_prompt[first], per_prompt[second]
    by_category = {}
    for category in sorted({o["category"] for o in outcomes.values()}):
        members = [pid for pid in ids if outcomes[pid]["category"] == category]
        by_category[category] = {t: round(sum(outcomes[p]["accept"][t] for p in members) / len(members), 6)
                                 for t in TIERS} | {"n": len(members)}
    return {
        "n_prompts": len(ids), "excluded": excluded, "seed": prereg.SEED, "n_resamples": prereg.N_RESAMPLES,
        "decision_threshold": prereg.DECISION_THRESHOLD, "quality_bar_lower_bound": prereg.QUALITY_BAR_LOWER_BOUND,
        "tier_map": prereg.TIER_MAP, "judges": list(prereg.JUDGES), "rubric_version": prereg.RUBRIC_VERSION,
        "policies": policies,
        "primary": {"policy": prereg.PRIMARY_POLICY,
                    "meets_quality_bar": policies[prereg.PRIMARY_POLICY]["meets_quality_bar"]},
        "secondary": {"pair": [first, second], "mcnemar": mcnemar_exact(acc_a, acc_b),
                      "cost_per_1k_difference_usd": bootstrap_mean(
                          [1000 * (a - b) for a, b in zip(cost_a, cost_b)], rng)},
        "acceptance_by_category": by_category,
        "judge_agreement": judge_agreement(verdicts),
        "human_agreement": human_agreement(outcomes, human_dir),
    }


def _ci(values, digits=3, scale=1.0) -> str:
    point, low, high = (v * scale for v in values)
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def render_markdown(r: dict) -> str:
    p = r["policies"]
    primary = p[r["primary"]["policy"]]
    verdict = "meets" if r["primary"]["meets_quality_bar"] else "does not meet"
    lines = [
        "# Slice 2 results", "",
        f"The pre-registered primary policy, `{r['primary']['policy']}`, {verdict} the quality bar: "
        f"{100 * primary['acceptance'][0]:.1f}% of answers acceptable (95% CI {100 * primary['acceptance'][1]:.1f} "
        f"to {100 * primary['acceptance'][2]:.1f}; the bar is a lower bound of {100 * r['quality_bar_lower_bound']:.0f}%), "
        f"at {100 * primary['cost_reduction_vs_always_high'][0]:.1f}% less cost than sending everything to "
        f"{r['tier_map']['high']}.", "",
        "## Policies", "",
        f"{r['n_prompts']} prompts, learned policies scored out of fold ({prereg.N_FOLDS} folds stratified by category). "
        f"95% bootstrap intervals, {r['n_resamples']:,} resamples, seed {r['seed']}. Tiers: "
        + ", ".join(f"{t} = {m}" for t, m in r["tier_map"].items()) + ".", "",
        "| policy | acceptable answers | cost per 1,000 requests, US$ | cost reduction vs always_high | low / medium / high | meets bar |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in p.items():
        share = " / ".join(f"{100 * m['tier_share'][t]:.0f}%" for t in TIERS)
        lines.append(f"| {name} | {_ci(m['acceptance'], 1, 100)}% | {_ci(m['cost_per_1k_usd'], 3)} | "
                     f"{_ci(m['cost_reduction_vs_always_high'], 1, 100)}% | {share} | {'yes' if m['meets_quality_bar'] else 'no'} |")
    s = r["secondary"]
    a, b = s["pair"]
    lines += ["", "## Secondary comparison", "",
              f"{a} against {b} on per-prompt acceptance, two-sided exact McNemar: only {a} acceptable on "
              f"{s['mcnemar']['b']} prompts, only {b} on {s['mcnemar']['c']}, p = {s['mcnemar']['p']:.4f}. "
              f"Cost difference per 1,000 requests ({a} minus {b}): US$ {_ci(s['cost_per_1k_difference_usd'], 3)}.",
              "", "## Acceptance by request class", "",
              "| class | n | low | medium | high |", "|---|---|---|---|---|"]
    for category, c in r["acceptance_by_category"].items():
        lines.append(f"| {category} | {c['n']} | " + " | ".join(f"{100 * c[t]:.0f}%" for t in TIERS) + " |")
    j = r["judge_agreement"]
    lines += ["", "## Judges and the human check", "",
              f"- The two judges agree on {100 * j['agreement']:.1f}% of the {j['n']} judged pairs (Cohen's kappa "
              f"{j['kappa']:.2f}); {r['judges'][0]} accepted {j['first_yes']}, {r['judges'][1]} accepted {j['second_yes']}."]
    h = r["human_agreement"]
    if h:
        kappa = "undefined" if h["kappa"] is None else f"{h['kappa']:.2f}"
        lines.append(f"- A blind human grader and the judge consensus agree on {h['agreement'] * h['n']:.0f} of {h['n']} "
                     f"sampled answers (kappa {kappa}). The human accepted {h['first_yes']} of {h['n']}; when one rater "
                     "never says no, kappa is 0 whatever the agreement, so the raw count is the informative number.")
        for d in h["disagreements"]:
            lines.append(f"  - item {d['item']} ({d['prompt_id']}, {d['tier']} tier): human "
                         f"{'yes' if d['human'] else 'no'}, judges {'yes' if d['judges'] else 'no'}")
    lines += ["", "## Method", "",
              f"- Excluded: {', '.join(r['excluded'])} (no usable answer from either reference model).",
              "- An answer cut by the output limit or empty is unacceptable by rule, for every tier including high.",
              f"- A candidate answer is acceptable only when both judges ({', '.join(r['judges'])}) accept it under "
              f"rubric {r['rubric_version']}.",
              f"- Learned policies route to the cheapest candidate tier whose estimated acceptance is at least "
              f"{r['decision_threshold']}, else high. The design is fixed in harness/routing/prereg.py.", ""]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    outcomes, excluded = load_outcomes()
    results = build_results(outcomes, excluded, load_verdicts())
    RESULTS_DIR.mkdir(exist_ok=True)
    JSON_PATH.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    text = render_markdown(results)
    MD_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
