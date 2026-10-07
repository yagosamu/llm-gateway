"""Turn the recording and an offline replay into results/slice1.json and results/slice1.md.

Cost comes from the gateway's own request log during the replay, so the published cost is the
gateway's attribution, not a recomputation beside it; the report checks it equals the cost recorded
live. Latency and truncation come from the live recording, made once, one call at a time. Intervals
are 95% percentile bootstrap over prompts, 10,000 resamples from a fixed seed, so the JSON is
deterministic and the CI test can regenerate it and compare.

Usage: uv run python -m harness.report"""
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from harness import budget
from harness.record import OUTPUT_LIMIT
from harness.replay import run_replay
from llm_gateway.registry import ModelConfig, load_registry

SEED = 20261007
N_RESAMPLES = 10_000
RESULTS_DIR = Path("results")
JSON_PATH = RESULTS_DIR / "slice1.json"
MD_PATH = RESULTS_DIR / "slice1.md"


def interval(values, statistic, rng) -> list[float]:
    """[point estimate, 2.5th percentile, 97.5th percentile] of `statistic` over bootstrap resamples."""
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(N_RESAMPLES, len(values)))
    resampled = statistic(values[idx], axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return [round(float(statistic(values)), 6), round(float(low), 6), round(float(high), 6)]


def p50(values, axis=None):
    return np.percentile(values, 50, axis=axis)


def p95(values, axis=None):
    return np.percentile(values, 95, axis=axis)


def build_results(rows: list[dict], matrix: dict, log_rows: list[dict], registry: dict[str, ModelConfig]) -> dict:
    by_request = {r["request_id"]: r for r in log_rows}
    mismatched = [k for k, rec in matrix.items()
                  if abs(by_request[f"{k[0]}:{k[1]}"]["cost_usd"] - rec["cost_usd"]) > 1e-12]
    if mismatched:
        raise ValueError(f"replayed cost differs from recorded cost for {mismatched[:3]}")

    rng = np.random.default_rng(SEED)
    categories = sorted({row["category"] for row in rows})
    models, cost_by_category, truncated_by_category = {}, {}, {}
    for model in registry.values():
        recs = [matrix[(row["id"], model.id)] for row in rows]
        costs = [by_request[f"{row['id']}:{model.id}"]["cost_usd"] for row in rows]
        latencies = [r["latency_ms"] for r in recs]
        truncated = sum(r["finish_reason"] == "length" for r in recs)
        models[model.id] = {
            "provider": model.provider, "tier": model.tier, "n": len(recs),
            "cost_per_1k_usd": interval([1000 * c for c in costs], np.mean, rng),
            "mean_input_tokens": round(statistics.mean(r["input_tokens"] for r in recs), 1),
            "mean_output_tokens": round(statistics.mean(r["output_tokens"] for r in recs), 1),
            "truncated": truncated, "truncated_rate": round(truncated / len(recs), 4),
            "p50_latency_ms": interval(latencies, p50, rng),
            "p95_latency_ms": interval(latencies, p95, rng),
        }
        cost_by_category[model.id] = {
            c: round(1000 * statistics.mean(cost for row, cost in zip(rows, costs) if row["category"] == c), 6)
            for c in categories}
        truncated_by_category[model.id] = dict(sorted(Counter(
            row["category"] for row, r in zip(rows, recs) if r["finish_reason"] == "length").items()))

    with_metadata = sum(bool(r["tenant"] and r["feature"] and r["request_id"]) for r in log_rows)
    cost_by_tenant = defaultdict(float)
    for r in log_rows:
        cost_by_tenant[r["tenant"]] += r["cost_usd"]
    return {
        "n_prompts": len(rows), "n_models": len(registry), "output_limit": OUTPUT_LIMIT,
        "seed": SEED, "n_resamples": N_RESAMPLES,
        "models": models,
        "cost_per_1k_usd_by_category": cost_by_category,
        "truncated_by_category": truncated_by_category,
        "attribution": {"requests_logged": len(log_rows), "with_tenant_feature_and_request_id": with_metadata,
                        "cost_usd_by_tenant": {t: round(v, 6) for t, v in sorted(cost_by_tenant.items())},
                        "replayed_cost_equals_recorded": True},
    }


def _ci(values, scale=1.0, digits=2) -> str:
    point, low, high = (v * scale for v in values)
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def render_markdown(results: dict, overhead_ms: list[float], spent_usd: float) -> str:
    models = results["models"]
    cheapest = min(models, key=lambda m: models[m]["cost_per_1k_usd"][0])
    dearest = max(models, key=lambda m: models[m]["cost_per_1k_usd"][0])
    ratio = models[dearest]["cost_per_1k_usd"][0] / models[cheapest]["cost_per_1k_usd"][0]
    lines = [
        "# Slice 1 results",
        "",
        f"The same {results['n_prompts']} prompts cost US$ {models[cheapest]['cost_per_1k_usd'][0]:.2f} per 1,000 "
        f"requests on {cheapest} and US$ {models[dearest]['cost_per_1k_usd'][0]:.2f} on {dearest}, "
        f"{ratio:.0f} times more. Whether the cheaper answer is good enough is slice 2's question.",
        "",
        "## Per model",
        "",
        f"95% bootstrap intervals over prompts ({results['n_resamples']:,} resamples, seed {results['seed']}). "
        f"Cut: answers stopped by the {results['output_limit']}-token output limit.",
        "",
        "| model | tier | cost per 1,000 requests, US$ | mean output tokens | cut | p50 latency, s | p95 latency, s |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, m in models.items():
        lines.append(f"| {model} | {m['tier']} | {_ci(m['cost_per_1k_usd'])} | {m['mean_output_tokens']:.0f} | "
                     f"{m['truncated']} of {m['n']} | {_ci(m['p50_latency_ms'], 0.001, 1)} | "
                     f"{_ci(m['p95_latency_ms'], 0.001, 1)} |")
    categories = list(next(iter(results["cost_per_1k_usd_by_category"].values())))
    lines += ["", "## Cost per 1,000 requests by request class, US$", "",
              "| model | " + " | ".join(categories) + " |", "|" + "---|" * (len(categories) + 1)]
    for model, by_cat in results["cost_per_1k_usd_by_category"].items():
        lines.append(f"| {model} | " + " | ".join(f"{by_cat[c]:.2f}" for c in categories) + " |")
    lines += ["", "## Answers cut by the output limit, by request class", "",
              "| model | " + " | ".join(categories) + " |", "|" + "---|" * (len(categories) + 1)]
    for model, by_cat in results["truncated_by_category"].items():
        lines.append(f"| {model} | " + " | ".join(str(by_cat.get(c, 0)) for c in categories) + " |")
    a = results["attribution"]
    overhead = sorted(overhead_ms)
    lines += [
        "", "## Attribution and gateway overhead", "",
        f"- Requests replayed through the gateway: {a['requests_logged']:,}; logged with tenant, feature and "
        f"request id: {a['with_tenant_feature_and_request_id']:,}.",
        "- Cost attributed by tenant, US$: " + ", ".join(f"{t} {v:.4f}" for t, v in a["cost_usd_by_tenant"].items())
        + ". The replayed cost equals the cost recorded live for every call.",
        f"- Gateway overhead in process, provider time excluded: p50 {statistics.median(overhead):.1f} ms, "
        f"p95 {overhead[int(0.95 * (len(overhead) - 1))]:.1f} ms. Wall time on the machine that ran the report; "
        "informational, not gated.",
        "", "## Method", "",
        "- Traffic: 240 prompts from databricks-dolly-15k, 30 per category, see data/README.md.",
        "- Each prompt was sent once to each model, one call at a time, on 2026-10-07, from one machine. "
        "Latency is what that machine saw on that day and includes the network.",
        "- Prices are the registry prices checked on 2026-10-07 (llm_gateway/models.yaml).",
        f"- Total spent on live calls so far: US$ {spent_usd:.2f} of the US$ {budget.BUDGET_USD:.0f} project budget "
        "(data/spend_ledger.jsonl).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows, matrix, log_rows, overhead = run_replay()
    results = build_results(rows, matrix, log_rows, load_registry())
    RESULTS_DIR.mkdir(exist_ok=True)
    JSON_PATH.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    text = render_markdown(results, overhead, budget.spent_usd())
    MD_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
