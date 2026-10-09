"""Which model the low tier should fail over to, by answer quality and cost.

Descriptive, not pre-registered. Every model is scored the same way on the same 238 prompts: an
answer cut by the output limit or empty fails by rule, and otherwise gpt-4.1-mini's verdict decides.
That is the verifier's judge, more lenient than the slice 2 two-judge consensus, so the absolute
rates read higher than slice 2's; comparisons between models are like for like.

The candidates were claude-haiku-4-5 (its gpt-4.1-mini verdicts come from the slice 2 judging, where
it was the medium tier) and gpt-oss-20b (judged for this comparison). Haiku was chosen.

Usage: uv run python -m harness.routing.fallback_quality"""
import json
import sys
from pathlib import Path

import numpy as np

from harness.record import RECORDINGS_PATH, load_records, load_rows
from harness.replay import current_records
from harness.routing import prereg
from harness.routing.evaluate import bootstrap_mean, mcnemar_exact
from harness.routing.judge import FALLBACK_VERDICTS_PATH, VERDICTS_PATH, load_verdicts
from harness.routing.pairs import excluded_prompts, is_auto_unacceptable
from llm_gateway.registry import load_registry

JUDGE = "gpt-4.1-mini"
PRIMARY = prereg.TIER_MAP["low"]
CHOSEN = "claude-haiku-4-5"
# model -> (which verdict file, the label its verdicts carry)
CANDIDATES = {CHOSEN: ("slice2", "medium"), "gpt-oss-20b": ("fallback", "fallback-gpt-oss-20b")}
JSON_PATH = Path("results/slice3_fallback_quality.json")
MD_PATH = Path("results/slice3_fallback_quality.md")


def acceptance(ids, matrix, model, verdicts, tier_label) -> list[bool]:
    by_prompt = {v["prompt_id"]: v["acceptable"] for v in verdicts
                 if v["status"] == "ok" and v["judge"] == JUDGE and v["tier"] == tier_label}
    out = []
    for pid in ids:
        if is_auto_unacceptable(matrix[(pid, model)]):
            out.append(False)
        elif pid not in by_prompt:
            raise ValueError(f"{pid}: {model} was never judged by {JUDGE}")
        else:
            out.append(bool(by_prompt[pid]))
    return out


def build_results(rows, matrix, slice2_verdicts, fallback_verdicts) -> dict:
    excluded = set(excluded_prompts(rows, matrix))
    ids = [r["id"] for r in rows if r["id"] not in excluded]
    sources = {"slice2": slice2_verdicts, "fallback": fallback_verdicts}
    rng = np.random.default_rng(prereg.SEED)
    primary = acceptance(ids, matrix, PRIMARY, slice2_verdicts, "low")

    def describe(model: str, accepted: list[bool]) -> dict:
        return {"model": model, "acceptance": bootstrap_mean(accepted, rng),
                "cost_per_1k_usd": round(1000 * sum(matrix[(p, model)]["cost_usd"] for p in ids) / len(ids), 6),
                "cut_by_limit": sum(matrix[(p, model)]["finish_reason"] == "length" for p in ids)}

    result = {"n_prompts": len(ids), "judge": JUDGE, "chosen": CHOSEN, "primary": describe(PRIMARY, primary),
              "candidates": []}
    for model, (source, label) in CANDIDATES.items():
        accepted = acceptance(ids, matrix, model, sources[source], label)
        entry = describe(model, accepted)
        entry["paired_vs_primary"] = mcnemar_exact(primary, accepted)
        entry["difference_vs_primary"] = bootstrap_mean([float(a) - float(b) for a, b in zip(primary, accepted)], rng)
        result["candidates"].append(entry)
    return result


def render_markdown(r: dict) -> str:
    pct = lambda v: f"{100 * v[0]:.1f}% [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"
    p = r["primary"]
    lines = ["# Slice 3: which model the low tier fails over to", "",
             f"Scored by {r['judge']} alone, with cut or empty answers failing by rule, on the same {r['n_prompts']} "
             "prompts. Descriptive, not pre-registered.", "",
             "| model | role | acceptable (95% CI) | drop vs the low tier, points | cost per 1,000, US$ | cut by the limit |",
             "|---|---|---|---|---|---|",
             f"| {p['model']} | low tier | {pct(p['acceptance'])} | - | {p['cost_per_1k_usd']:.3f} | {p['cut_by_limit']} |"]
    for c in r["candidates"]:
        role = "failover (chosen)" if c["model"] == r["chosen"] else "failover (rejected)"
        d = c["difference_vs_primary"]
        lines.append(f"| {c['model']} | {role} | {pct(c['acceptance'])} | {100 * d[0]:.1f} [{100 * d[1]:.1f}, "
                     f"{100 * d[2]:.1f}] | {c['cost_per_1k_usd']:.3f} | {c['cut_by_limit']} |")
    lines += ["", "Paired exact McNemar against the low tier: " + "; ".join(
        f"{c['model']}: only the low tier acceptable on {c['paired_vs_primary']['b']} prompts, only {c['model']} on "
        f"{c['paired_vs_primary']['c']}, p = {c['paired_vs_primary']['p']:.4f}" for c in r["candidates"]) + ".", "",
        f"{r['chosen']} keeps the low tier's quality during an outage at about twelve times its cost; gpt-oss-20b "
        "would have kept the cost and lost quality. The cost only applies while the failover lasts.", "",
        f"{r['judge']} is more lenient than the slice 2 two-judge consensus, under which {r['chosen']} was accepted "
        "on 85.7% against gpt-6-luna's 95.8%; the equal rates here depend on the judge.", ""]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    matrix = current_records(load_records(RECORDINGS_PATH), load_registry())
    results = build_results(load_rows(), matrix, load_verdicts(VERDICTS_PATH), load_verdicts(FALLBACK_VERDICTS_PATH))
    JSON_PATH.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    text = render_markdown(results)
    MD_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
