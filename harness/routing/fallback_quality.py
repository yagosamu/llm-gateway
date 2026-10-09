"""How much answer quality the low tier gives up while it fails over from gpt-6-luna to gpt-oss-20b.

Descriptive, not pre-registered. Both models are scored the same way on the same 238 prompts: an
answer cut by the output limit or empty fails by rule, and otherwise gpt-4.1-mini's verdict decides.
That is the verifier's judge, more lenient than the slice 2 two-judge consensus, so the absolute
rates read higher than slice 2's; the comparison between the two models is like for like.

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
PRIMARY, FALLBACK = prereg.TIER_MAP["low"], "gpt-oss-20b"
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


def build_results(rows, matrix, primary_verdicts, fallback_verdicts) -> dict:
    excluded = set(excluded_prompts(rows, matrix))
    ids = [r["id"] for r in rows if r["id"] not in excluded]
    primary = acceptance(ids, matrix, PRIMARY, primary_verdicts, "low")
    fallback = acceptance(ids, matrix, FALLBACK, fallback_verdicts, f"fallback-{FALLBACK}")
    rng = np.random.default_rng(prereg.SEED)
    cut = lambda model: sum(matrix[(pid, model)]["finish_reason"] == "length" for pid in ids)
    return {"n_prompts": len(ids), "judge": JUDGE,
            "primary": {"model": PRIMARY, "acceptance": bootstrap_mean(primary, rng), "cut_by_limit": cut(PRIMARY)},
            "fallback": {"model": FALLBACK, "acceptance": bootstrap_mean(fallback, rng), "cut_by_limit": cut(FALLBACK)},
            "paired": mcnemar_exact(primary, fallback),
            "difference": bootstrap_mean([float(a) - float(b) for a, b in zip(primary, fallback)], rng)}


def render_markdown(r: dict) -> str:
    p, f, d = r["primary"], r["fallback"], r["difference"]
    pct = lambda v: f"{100 * v[0]:.1f}% [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"
    return "\n".join([
        "# Slice 3: answer quality during a low-tier failover", "",
        f"Scored by {r['judge']} alone, with cut or empty answers failing by rule, on the same {r['n_prompts']} prompts. "
        "Descriptive, not pre-registered.", "",
        "| model | role | acceptable (95% CI) | cut by the 1,024-token limit |", "|---|---|---|---|",
        f"| {p['model']} | low tier | {pct(p['acceptance'])} | {p['cut_by_limit']} |",
        f"| {f['model']} | its failover | {pct(f['acceptance'])} | {f['cut_by_limit']} |", "",
        f"While the low tier fails over, acceptance drops by {100 * d[0]:.1f} points (95% CI {100 * d[1]:.1f} to "
        f"{100 * d[2]:.1f}). Paired exact McNemar: only {p['model']} acceptable on {r['paired']['b']} prompts, only "
        f"{f['model']} on {r['paired']['c']}, p = {r['paired']['p']:.4f}.", "",
        f"{r['judge']} is the verifier's judge and more lenient than the slice 2 two-judge consensus, so both rates "
        "read higher than slice 2's; the comparison between the two models is like for like.", ""])


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
