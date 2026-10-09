"""Fit the two learned routing policies and write the estimates the gateway reads.

feature_table: the acceptance rate of each (X-Feature, candidate tier) in the training prompts.
classifier: one L2-regularized logistic regression per candidate tier, P(acceptable | prompt
features), fitted by Newton's method in numpy. The penalty is 1.0 on every weight except the
intercept; it was fixed before the first evaluation run and is not tuned. The features are the ones
llm_gateway.routing.prompt_features computes at serving time, so training and serving read the same
features from the same code.

`python -m harness.routing.train` fits both on every evaluated prompt and writes
llm_gateway/routing_estimates.json. The out-of-fold evaluation calls the same fit functions on each
training split.

Usage: uv run python -m harness.routing.train"""
import json
import sys
from pathlib import Path

import numpy as np

from harness.record import RECORDINGS_PATH, load_records, load_rows
from harness.replay import current_records
from harness.routing import prereg
from harness.routing.judge import load_verdicts
from harness.routing.outcomes import build_outcomes
from llm_gateway.registry import load_registry
from llm_gateway.routing import prompt_features

L2_PENALTY = 1.0
NEWTON_ITERATIONS = 50
ESTIMATES_PATH = Path("llm_gateway/routing_estimates.json")


def fit_feature_table(outcomes: dict[str, dict], ids: list[str]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, list[bool]]] = {}
    for pid in ids:
        o = outcomes[pid]
        for tier in prereg.CANDIDATE_TIERS:
            table.setdefault(o["category"], {}).setdefault(tier, []).append(o["accept"][tier])
    return {feature: {tier: round(sum(v) / len(v), 6) for tier, v in tiers.items()}
            for feature, tiers in sorted(table.items())}


def fit_logistic(x: np.ndarray, y: np.ndarray, penalty: float = L2_PENALTY) -> np.ndarray:
    """Weights for [intercept, features...] minimizing log loss plus penalty/2 * |w without intercept|^2.
    When every label is the same the unpenalized intercept has no finite optimum; the weights are then
    zero and the intercept is the log odds of the add-one smoothed rate, (sum + 0.5) / (n + 1)."""
    design = np.hstack([np.ones((len(x), 1)), x])
    w = np.zeros(design.shape[1])
    if y.min() == y.max():
        rate = (y.sum() + 0.5) / (len(y) + 1)
        w[0] = np.log(rate / (1 - rate))
        return w
    ridge = np.full(design.shape[1], penalty)
    ridge[0] = 0.0
    for _ in range(NEWTON_ITERATIONS):
        p = 1.0 / (1.0 + np.exp(-design @ w))
        gradient = design.T @ (p - y) + ridge * w
        hessian = design.T @ (design * (p * (1 - p))[:, None]) + np.diag(ridge)
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def fit_classifier(outcomes: dict[str, dict], ids: list[str]) -> dict[str, dict]:
    rows = [prompt_features(outcomes[pid]["messages"], outcomes[pid]["category"]) for pid in ids]
    names = sorted({name for r in rows for name in r})
    x = np.array([[r.get(name, 0.0) for name in names] for r in rows])
    weights = {}
    for tier in prereg.CANDIDATE_TIERS:
        y = np.array([float(outcomes[pid]["accept"][tier]) for pid in ids])
        w = fit_logistic(x, y)
        weights[tier] = {"intercept": round(float(w[0]), 10),
                         "coef": {name: round(float(c), 10) for name, c in zip(names, w[1:])}}
    return weights


def load_outcomes() -> tuple[dict[str, dict], list[str]]:
    matrix = current_records(load_records(RECORDINGS_PATH), load_registry())
    return build_outcomes(load_rows(), matrix, load_verdicts())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    outcomes, excluded = load_outcomes()
    ids = sorted(outcomes)
    estimates = {"trained_on": len(ids), "excluded": excluded,
                 "feature_table": fit_feature_table(outcomes, ids), "classifier": fit_classifier(outcomes, ids)}
    ESTIMATES_PATH.write_text(json.dumps(estimates, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"trained on {len(ids)} prompts -> {ESTIMATES_PATH}")


if __name__ == "__main__":
    main()
