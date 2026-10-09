"""Turn the committed chaos runs into results/slice3.json and slice3.md, offline and deterministically.

The runs themselves depend on the wall clock and vary a little; this analysis of what they recorded
does not, so CI regenerates the published numbers from data/chaos and compares them. Metric
definitions are in harness/chaos/prereg.py.

Usage: uv run python -m harness.chaos.analyze"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from harness.chaos import prereg

RUNS_DIR = Path("data/chaos/runs")
MANIFEST_PATH = Path("data/chaos/manifest.json")
JSON_PATH = Path("results/slice3.json")
MD_PATH = Path("results/slice3.md")
FAILED_ATTEMPTS = ("rate_limit", "timeout", "connection", "server_error", "auth", "bad_request")
METRICS = ("error_rate", "latency_p50_s", "latency_p95_s", "wasted_calls", "fallback_share",
           "time_to_open_s", "time_to_close_s", "error_rate_after", "reopens_after_close")


def _wasted(r: dict) -> bool:
    """The request made a call to the faulty provider that failed."""
    if r["status"] == 200:
        return r["attempts"][0] in FAILED_ATTEMPTS
    return r.get("error", "").startswith("upstream_") and r.get("error") != "upstream_circuit_open"


def run_metrics(requests: list[dict], timeline: list[dict]) -> dict:
    window = [r for r in requests if prereg.FAULT_START <= r["sent"] < prereg.FAULT_END]
    after = [r for r in requests if r["sent"] >= prereg.FAULT_END]
    answered = [r for r in window if r["status"] == 200]
    latencies = [r["done"] - r["sent"] for r in answered]
    opened = [p["t"] for p in timeline if p["t"] >= prereg.FAULT_START and p["state"] == "open"]
    closed = [p["t"] for p in timeline if opened and p["t"] >= max(prereg.FAULT_END, opened[0]) and p["state"] == "closed"]
    rnd = lambda x: None if x is None else round(float(x), 3)
    return {
        "requests_in_fault": len(window),
        "error_rate": rnd(1 - len(answered) / len(window)) if window else None,
        "latency_p50_s": rnd(np.percentile(latencies, 50)) if latencies else None,
        "latency_p95_s": rnd(np.percentile(latencies, 95)) if latencies else None,
        "wasted_calls": sum(_wasted(r) for r in window),
        "fallback_share": rnd(sum(r["model"] != prereg.PRIMARY_MODEL for r in answered) / len(answered)) if answered else None,
        "time_to_open_s": rnd(opened[0] - prereg.FAULT_START) if opened else None,
        "time_to_close_s": rnd(closed[0] - prereg.FAULT_END) if closed else None,
        "error_rate_after": rnd(1 - sum(r["status"] == 200 for r in after) / len(after)) if after else None,
        # Added after the runs, not pre-registered: time_to_close_s hides a circuit that closes and then
        # opens again, which the latency runs showed.
        "reopens_after_close": _reopens(timeline, closed[0]) if closed else None,
    }


def _reopens(timeline: list[dict], first_close: float) -> int:
    later = [p["state"] for p in timeline if p["t"] > first_close]
    return sum(1 for prev, cur in zip(["closed"] + later, later) if cur == "open" and prev != "open")


def summarize(values: list) -> dict:
    present = [v for v in values if v is not None]
    if not present:
        return {"median": None, "min": None, "max": None, "missing": len(values)}
    return {"median": round(float(statistics.median(present)), 3), "min": min(present), "max": max(present),
            "missing": len(values) - len(present)}


def build_results(runs: list[dict], notes: list[str] | None = None, scenarios: dict | None = None,
                  configs: tuple | None = None) -> dict:
    """runs: [{"scenario", "config", "rep", "metrics"}] -> per scenario and config, each metric's
    median, min and max over the repetitions. Scenarios and configurations default to the slice 3 study."""
    scenarios = prereg.SCENARIOS if scenarios is None else scenarios
    configs = prereg.CONFIGS if configs is None else configs
    table = {}
    for scenario in scenarios:
        for config in configs:
            reps = [r["metrics"] for r in runs if r["scenario"] == scenario and r["config"] == config]
            if not reps:
                continue
            table.setdefault(scenario, {})[config] = {"reps": len(reps)} | {m: summarize([x[m] for x in reps])
                                                                            for m in METRICS}
    return {"rate_rps": prereg.RATE, "fault_window_s": [prereg.FAULT_START, prereg.FAULT_END],
            "duration_s": prereg.DURATION, "scenarios": scenarios, "results": table,
            "notes": list(notes or [])}


def _cell(s: dict, scale=1.0, digits=1, unit="", absent="-") -> str:
    """absent: what to print when no repetition has a value ("never" for circuit times, "-" otherwise)."""
    if s["median"] is None:
        return absent
    text = f"{s['median'] * scale:.{digits}f}{unit}"
    if s["min"] != s["max"]:
        text += f" ({s['min'] * scale:.{digits}f} to {s['max'] * scale:.{digits}f})"
    if s["missing"]:
        text += f", never in {s['missing']}"
    return text


def render_markdown(r: dict) -> str:
    lines = ["# Slice 3 results: provider faults under load", "",
             f"Open-loop traffic at {r['rate_rps']:.0f} requests per second through two gateway instances sharing Redis, "
             f"over simulated providers at recorded latency. A fault hits openai (gpt-6-luna's provider) from "
             f"{r['fault_window_s'][0]:.0f} s to {r['fault_window_s'][1]:.0f} s of a {r['duration_s']:.0f} s run. "
             "Each cell is the median of the repetitions, with min to max in brackets when they differ. Design: "
             "harness/chaos/prereg.py.", ""]
    for scenario, by_config in r["results"].items():
        fault = r["scenarios"][scenario]
        lines += [f"## {scenario}: {json.dumps(fault) if fault else 'no fault injected'}", "",
                  "| configuration | client errors | p50 latency | p95 latency | wasted calls | served by fallback | "
                  "circuit opened after | closed after recovery | reopened after closing | errors after recovery |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for config, m in by_config.items():
            lines.append(f"| {config} | {_cell(m['error_rate'], 100, 1, '%')} | {_cell(m['latency_p50_s'], 1, 1, ' s')} | "
                         f"{_cell(m['latency_p95_s'], 1, 1, ' s')} | {_cell(m['wasted_calls'], 1, 0)} | "
                         f"{_cell(m['fallback_share'], 100, 0, '%')} | "
                         f"{_cell(m['time_to_open_s'], 1, 1, ' s', 'never')} | "
                         f"{_cell(m['time_to_close_s'], 1, 1, ' s', 'never')} | "
                         f"{_cell(m['reopens_after_close'], 1, 0)} | {_cell(m['error_rate_after'], 100, 1, '%')} |")
        lines.append("")
    lines += ["Wasted calls: requests in the fault window that made a failing call to openai. Circuit times come from "
              f"openai's state polled every {prereg.TIMELINE_INTERVAL} s, so they carry that resolution. Configurations "
              "without a breaker never open a circuit. \"Reopened after closing\" was added after the runs and is not "
              "part of the pre-registered metrics.", ""]
    if r.get("notes"):
        lines += ["## Measurement notes", ""] + [f"- {n}" for n in r["notes"]] + [""]
    return "\n".join(lines)


def load_notes(manifest_path: Path = MANIFEST_PATH) -> list[str]:
    return json.loads(manifest_path.read_text(encoding="utf-8")).get("notes", [])


def load_runs(manifest_path: Path = MANIFEST_PATH, runs_dir: Path = RUNS_DIR) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = []
    for entry in manifest["runs"]:
        folder = runs_dir / entry["name"]
        read = lambda name: [json.loads(line) for line in (folder / name).read_text(encoding="utf-8").splitlines() if line]
        runs.append(entry | {"metrics": run_metrics(read("requests.jsonl"), read("timeline.jsonl"))})
    return runs


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    results = build_results(load_runs(), load_notes())
    JSON_PATH.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    text = render_markdown(results)
    MD_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
