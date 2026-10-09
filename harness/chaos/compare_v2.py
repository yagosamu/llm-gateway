"""Judge breaker v2 against v1 by the criteria fixed in harness/chaos/prereg_v2.py, from the committed
runs of that study, and write results/slice3_breaker_v2.json and .md.

Deterministic, like the slice 3 analysis: the runs vary with the wall clock, this reading of them
does not, so a gate test regenerates the published verdict.

Usage: uv run python -m harness.chaos.compare_v2"""
import json
import statistics
import sys
from pathlib import Path

from harness.chaos import prereg_v2
from harness.chaos.analyze import build_results, load_runs, render_markdown

MANIFEST_PATH = Path("data/chaos/v2/manifest.json")
RUNS_DIR = Path("data/chaos/v2/runs")
JSON_PATH = Path("results/slice3_breaker_v2.json")
MD_PATH = Path("results/slice3_breaker_v2.md")
V1, V2 = prereg_v2.CONFIGS


def opens_total(timeline: list[dict]) -> int:
    """Times the circuit went into the open state at any point of the run."""
    states = [p["state"] for p in timeline]
    return sum(1 for prev, cur in zip(["closed"] + states, states) if cur == "open" and prev != "open")


def load_study(manifest_path: Path = MANIFEST_PATH, runs_dir: Path = RUNS_DIR) -> list[dict]:
    runs = load_runs(manifest_path, runs_dir)
    for run in runs:
        folder = runs_dir / run["name"]
        timeline = [json.loads(line) for line in (folder / "timeline.jsonl").read_text(encoding="utf-8").splitlines() if line]
        run["opens_total"] = opens_total(timeline)
    return runs


def _median(values):
    present = [v for v in values if v is not None]
    return round(float(statistics.median(present)), 3) if present else None


def criteria(runs: list[dict]) -> list[dict]:
    def of(scenario, config):
        return [r for r in runs if r["scenario"] == scenario and r["config"] == config]

    def med(scenario, config, metric):
        return _median([r["metrics"][metric] for r in of(scenario, config)])

    out = []
    for scenario in prereg_v2.DETECTION_SCENARIOS:
        v1, v2 = med(scenario, V1, "time_to_open_s"), med(scenario, V2, "time_to_open_s")
        out.append({"id": "C1", "scenario": scenario, "check": "median time to open lower than v1",
                    "v1": v1, "v2": v2, "pass": v2 is not None and (v1 is None or v2 < v1)})
    opened = sum(r["metrics"]["time_to_open_s"] is not None for r in of("partial", V2))
    out.append({"id": "C2", "scenario": "partial", "check": f"opens in at least {prereg_v2.MIN_PARTIAL_OPENS} of the repetitions",
                "v1": sum(r["metrics"]["time_to_open_s"] is not None for r in of("partial", V1)), "v2": opened,
                "pass": opened >= prereg_v2.MIN_PARTIAL_OPENS})
    for scenario in prereg_v2.FAULT_SCENARIOS:
        reps = [r for r in of(scenario, V2) if r["opens_total"] > 0]
        bad = [r["rep"] for r in reps if r["metrics"]["time_to_close_s"] is None or r["metrics"]["reopens_after_close"]]
        out.append({"id": "C3", "scenario": scenario, "check": "closes after the fault and stays closed, in every repetition where it opened",
                    "v1": [r["metrics"]["reopens_after_close"] for r in of(scenario, V1)],
                    "v2": [r["metrics"]["reopens_after_close"] for r in of(scenario, V2)],
                    "pass": not bad})
    for scenario in prereg_v2.CONTROL_SCENARIOS:
        opens = [r["opens_total"] for r in of(scenario, V2)]
        out.append({"id": "C4", "scenario": scenario, "check": "never opens", "v1": [r["opens_total"] for r in of(scenario, V1)],
                    "v2": opens, "pass": bool(opens) and not any(opens)})
    for scenario in prereg_v2.SCENARIOS:
        checks = [("error_rate", lambda a, b: b <= a + prereg_v2.ERROR_RATE_SLACK, "client errors at most v1 + 0.5 points"),
                  ("latency_p95_s", lambda a, b: b <= a * prereg_v2.P95_RATIO + prereg_v2.P95_SLACK_S, "p95 at most v1 x 1.05 + 0.1 s"),
                  ("wasted_calls", lambda a, b: b <= a, "wasted calls at most v1"),
                  ("error_rate_after", lambda a, b: b <= a + prereg_v2.ERROR_RATE_SLACK, "errors after recovery at most v1 + 0.5 points")]
        for metric, ok, text in checks:
            a, b = med(scenario, V1, metric), med(scenario, V2, metric)
            passed = (a is None and b is None) or (a is not None and b is not None and ok(a, b))
            out.append({"id": "C5", "scenario": scenario, "check": text, "v1": a, "v2": b, "pass": passed})
    return out


def build(runs: list[dict]) -> dict:
    checks = criteria(runs)
    table = build_results(runs, scenarios=prereg_v2.SCENARIOS, configs=prereg_v2.CONFIGS)
    table["opens_total"] = {s: {c: [r["opens_total"] for r in runs if r["scenario"] == s and r["config"] == c]
                                for c in prereg_v2.CONFIGS} for s in prereg_v2.SCENARIOS}
    return {"design": "harness/chaos/prereg_v2.py", "v2_better": all(c["pass"] for c in checks),
            "criteria": checks, "table": table}


def render(r: dict) -> str:
    verdict = "**v2 is better: every pre-registered criterion holds.**" if r["v2_better"] else \
        "**v2 is not better by the pre-registered criteria: at least one fails (marked below).**"
    lines = ["# Breaker v2 against v1", "",
             "v1 counts the shared 60 s health window; v2 counts the last 20 calls admitted since the circuit last "
             "moved, ignores calls admitted before that, and lists a call as slow while it runs. Same thresholds. "
             "v2 was designed after the slice 3 results; the criteria below were fixed in harness/chaos/prereg_v2.py "
             "before its runs, and both versions were run again, interleaved, in the same session.", "", verdict, "",
             "| criterion | scenario | check | v1 | v2 | holds |", "|---|---|---|---|---|---|"]
    for c in r["criteria"]:
        lines.append(f"| {c['id']} | {c['scenario']} | {c['check']} | {c['v1']} | {c['v2']} | {'yes' if c['pass'] else '**no**'} |")
    lines += ["", "Times in seconds; error rates as fractions; lists are one value per repetition.", "",
              "## Full measurements", ""]
    lines += render_markdown(r["table"]).splitlines()[4:]  # drop the slice 3 title and preamble
    lines += ["", "Openings over the whole run, per repetition: " + "; ".join(
        f"{s} v1 {o[V1]} v2 {o[V2]}" for s, o in r["table"]["opens_total"].items()) + ".", ""]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    result = build(load_study())
    JSON_PATH.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    text = render(result)
    MD_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
