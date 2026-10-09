"""What the sampled verifier costs and catches under always_low, from the recorded answers and verdicts.

Descriptive, not pre-registered: no hypothesis is tested. For each sample rate the numbers are exact
expectations over the 238 evaluated prompts (each prompt verified with probability equal to the
rate), not a simulation, so they carry no sampling noise of their own.

Per prompt: serving cost is gpt-6-luna's recorded cost. A verification costs nothing when the answer
is cut or empty (a failure by rule); otherwise it costs the reference model's recorded answer plus
gpt-4.1-mini's recorded verdict on that pair. The verifier flags a failure when the rule rejects the
answer or gpt-4.1-mini does; a true failure is one the evaluation's two-judge consensus rejected.

Usage: uv run python -m harness.routing.verifier_cost"""
import json
import math
import sys
from pathlib import Path

from harness.routing import prereg
from harness.routing.judge import load_verdicts
from harness.routing.pairs import is_auto_unacceptable
from harness.routing.train import load_outcomes
from harness.record import RECORDINGS_PATH, load_records
from harness.replay import current_records
from llm_gateway.registry import load_registry

SAMPLE_RATES = (0.01, 0.05, 0.10, 0.25)
DAILY_REQUESTS = 1_000  # used only to express how precisely a day's checks estimate the failure rate
JSON_PATH = Path("results/slice2_verifier.json")
MD_PATH = Path("results/slice2_verifier.md")
VERIFIER_JUDGE = "gpt-4.1-mini"


def per_prompt(outcomes: dict, matrix: dict, verdicts: list[dict]) -> list[dict]:
    judged = {(v["prompt_id"], v["tier"], v["judge"]): v for v in verdicts
              if v["status"] == "ok" and v["rubric_version"] == prereg.RUBRIC_VERSION}
    other_judge = next(j for j in prereg.JUDGES if j != VERIFIER_JUDGE)
    rows = []
    for pid in sorted(outcomes):
        o = outcomes[pid]
        candidate = matrix[(pid, prereg.TIER_MAP["low"])]
        reference = matrix[(pid, prereg.TIER_MAP["high"])]
        if is_auto_unacceptable(candidate):
            verify_cost, both_judges_cost, flagged = 0.0, 0.0, True
        elif is_auto_unacceptable(reference):
            verify_cost = both_judges_cost = reference["cost_usd"]  # no usable reference: no verdict
            flagged = False
        else:
            verdict = judged[(pid, "low", VERIFIER_JUDGE)]
            verify_cost, flagged = reference["cost_usd"] + verdict["cost_usd"], not verdict["acceptable"]
            both_judges_cost = verify_cost + judged[(pid, "low", other_judge)]["cost_usd"]
        rows.append({"serve": o["cost"]["low"], "high": o["cost"]["high"], "verify": verify_cost,
                     "verify_both_judges": both_judges_cost, "flagged": flagged, "failure": not o["accept"]["low"]})
    return rows


def build_results(rows: list[dict]) -> dict:
    n = len(rows)
    mean = lambda key: sum(r[key] for r in rows) / n
    failures = sum(r["failure"] for r in rows)
    caught = sum(r["failure"] and r["flagged"] for r in rows)
    failure_rate = failures / n
    serve_1k, high_1k, verify_each = 1000 * mean("serve"), 1000 * mean("high"), mean("verify")
    rates = []
    for r in SAMPLE_RATES:
        overhead_1k = 1000 * r * verify_each
        checks_per_day = r * DAILY_REQUESTS
        rates.append({
            "sample_rate": r,
            "verifications_per_1k": round(1000 * r, 6),
            "overhead_per_1k_usd": round(overhead_1k, 6),
            "overhead_vs_serving": round(overhead_1k / serve_1k, 6),
            "total_per_1k_usd": round(serve_1k + overhead_1k, 6),
            "reduction_vs_always_high": round(1 - (serve_1k + overhead_1k) / high_1k, 6),
            "failures_caught_per_1k": round(1000 * r * caught / n, 6),
            "daily_failure_rate_half_width": round(1.96 * math.sqrt(failure_rate * (1 - failure_rate) / checks_per_day), 6),
        })
    return {"n_prompts": n, "policy": "always_low", "verifier_judge": VERIFIER_JUDGE,
            "serving_per_1k_usd": round(serve_1k, 6), "always_high_per_1k_usd": round(high_1k, 6),
            "verification_cost_each_usd": round(verify_each, 8),
            "two_judge_verification_cost_ratio": round(mean("verify_both_judges") / verify_each, 4),
            "verifications_per_usd": round(1 / verify_each, 1),
            "failures": failures, "failure_rate": round(failure_rate, 6),
            "caught_by_verifier": caught,
            "verifier_sensitivity": round(caught / failures, 6) if failures else None,
            "daily_requests_assumed": DAILY_REQUESTS, "rates": rates}


def render_markdown(r: dict) -> str:
    lines = [
        "# Slice 2: what the verifier costs and catches", "",
        f"Under `{r['policy']}` gpt-6-luna serves at US$ {r['serving_per_1k_usd']:.3f} per 1,000 requests, and "
        f"{r['failures']} of {r['n_prompts']} answers ({100 * r['failure_rate']:.1f}%) are failures by the two-judge "
        f"consensus. One verification costs US$ {r['verification_cost_each_usd']:.5f} on average (the reference "
        f"answer plus a {r['verifier_judge']} verdict), about {r['verification_cost_each_usd'] * 1000 / r['serving_per_1k_usd']:.0f} "
        "times what serving the request cost.", "",
        f"The verifier's single judge flags {r['caught_by_verifier']} of the {r['failures']} failures "
        f"(sensitivity {100 * r['verifier_sensitivity']:.0f}%). It cannot raise a false alarm against the consensus by "
        "construction, since the consensus rejects whatever either judge rejects, so no false-alarm rate is reported. "
        f"Adding the evaluation's second judge to every check would cost {r['two_judge_verification_cost_ratio']:.1f} "
        "times as much per verification, from the recorded verdict costs.", "",
        "| sample rate | checks per 1,000 | overhead per 1,000, US$ | overhead vs serving | total per 1,000, US$ | "
        "reduction vs always_high | failures caught per 1,000 | daily failure-rate precision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for x in r["rates"]:
        lines.append(f"| {100 * x['sample_rate']:.0f}% | {x['verifications_per_1k']:.0f} | {x['overhead_per_1k_usd']:.3f} | "
                     f"{100 * x['overhead_vs_serving']:.0f}% | {x['total_per_1k_usd']:.3f} | "
                     f"{100 * x['reduction_vs_always_high']:.1f}% | {x['failures_caught_per_1k']:.1f} | "
                     f"± {100 * x['daily_failure_rate_half_width']:.1f} pp |")
    lines += ["", "## Reading it", "",
              "- Verification goes to the most expensive model, so even a small sample rate costs more than serving: "
              "the price gap that makes always_low cheap also makes checking it expensive.",
              "- At any rate it catches only that share of failures; the user has already received the answer. "
              "Sampled verification is a monitoring instrument: it measures the failure rate and collects routing "
              "failures as training examples, it does not fix individual answers.",
              f"- Daily failure-rate precision is the 95% half-width of the failure rate estimated from one day's checks "
              f"at {r['daily_requests_assumed']:,} requests a day, assuming the measured {100 * r['failure_rate']:.1f}% rate.",
              f"- The daily cap in routing.yaml bounds the overhead whatever the rate: US$ 1 buys about "
              f"{r['verifications_per_usd']:.0f} verifications.",
              "- Descriptive, not pre-registered. Expectations over the recorded prompts, not a simulation.", ""]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    outcomes, _ = load_outcomes()
    matrix = current_records(load_records(RECORDINGS_PATH), load_registry())
    results = build_results(per_prompt(outcomes, matrix, load_verdicts()))
    JSON_PATH.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    text = render_markdown(results)
    MD_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
