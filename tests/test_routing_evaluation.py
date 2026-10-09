import json

import numpy as np
import pytest

from harness.routing import prereg
from harness.routing.evaluate import (
    JSON_PATH,
    agreement,
    assign_folds,
    bootstrap_reduction,
    build_results,
    mcnemar_exact,
    route_all,
)
from harness.routing.outcomes import IncompleteVerdicts, build_outcomes, consensus
from harness.routing.train import fit_classifier, fit_feature_table, fit_logistic
from llm_gateway.routing import LogisticClassifier


def verdict(pid, tier, judge, ok=True):
    return {"prompt_id": pid, "tier": tier, "judge": judge, "acceptable": ok, "status": "ok",
            "rubric_version": prereg.RUBRIC_VERSION}


def test_consensus_needs_both_judges_and_both_to_accept():
    vs = [verdict("p", "low", "gpt-4.1-mini"), verdict("p", "low", "claude-sonnet-5-5", ok=False),
          verdict("q", "low", "gpt-4.1-mini"), verdict("q", "low", "claude-sonnet-5-5")]
    assert consensus(vs) == {("p", "low"): False, ("q", "low"): True}
    with pytest.raises(IncompleteVerdicts):
        consensus(vs[:1])


def make_world(n_per_category=10, categories=("closed_qa", "open_qa")):
    """A small recording and verdict set: luna fails on every third closed_qa prompt, haiku never fails,
    and the high-tier model came back empty on one prompt."""
    rows, matrix, verdicts = [], {}, []
    for c in categories:
        for i in range(n_per_category):
            pid = f"{c}-{i}"
            text = f"Question {i}?" + ("\n\nContext:\nSome text." if c == "closed_qa" else "")
            rows.append({"id": pid, "category": c, "messages": [{"role": "user", "content": text}]})
            for model, cost in ((prereg.TIER_MAP["low"], 0.0001), (prereg.TIER_MAP["medium"], 0.001),
                                (prereg.TIER_MAP["high"], 0.002), (prereg.FALLBACK_REFERENCE_MODEL, 0.004)):
                empty = model == prereg.TIER_MAP["high"] and pid == "open_qa-0"
                matrix[(pid, model)] = {"model": model, "text": "" if empty else "answer", "finish_reason": "stop",
                                        "cost_usd": cost}
            for tier in prereg.CANDIDATE_TIERS:
                ok = not (tier == "low" and c == "closed_qa" and i % 3 == 0)
                verdicts += [verdict(pid, tier, j, ok) for j in prereg.JUDGES]
    return rows, matrix, verdicts


def test_outcomes_score_the_high_tier_by_rule_and_candidates_by_consensus():
    outcomes, excluded = build_outcomes(*make_world())
    assert excluded == []
    assert outcomes["open_qa-0"]["accept"]["high"] is False  # empty answer, even on the high tier
    assert outcomes["closed_qa-0"]["accept"]["low"] is False and outcomes["closed_qa-1"]["accept"]["low"] is True
    assert outcomes["closed_qa-1"]["cost"] == {"low": 0.0001, "medium": 0.001, "high": 0.002}


def test_folds_are_stratified_and_deterministic():
    outcomes, _ = build_outcomes(*make_world())
    folds = assign_folds(outcomes)
    assert folds == assign_folds(outcomes)
    for c in ("closed_qa", "open_qa"):
        counts = np.bincount([f for pid, f in folds.items() if pid.startswith(c)], minlength=prereg.N_FOLDS)
        assert set(counts) == {2}


def test_logistic_fit_recovers_a_separating_signal_and_stays_finite():
    x = np.array([[0.0], [0.0], [1.0], [1.0]] * 5)
    y = np.array([1.0, 1.0, 0.0, 0.0] * 5)
    w = fit_logistic(x, y)
    assert np.all(np.isfinite(w)) and w[1] < 0  # the penalty keeps a perfect separator finite


def test_trained_estimates_load_into_the_serving_classes():
    outcomes, _ = build_outcomes(*make_world())
    ids = sorted(outcomes)
    table = fit_feature_table(outcomes, ids)
    assert table["closed_qa"]["low"] == pytest.approx(6 / 10) and table["open_qa"]["low"] == 1.0
    weights = fit_classifier(outcomes, ids)
    classifier = LogisticClassifier(weights, prereg.DECISION_THRESHOLD)
    assert classifier.decide("open_qa", outcomes["open_qa-1"]["messages"]).tier in ("low", "medium", "high")
    assert set(weights) == set(prereg.CANDIDATE_TIERS) and "has_context" in weights["low"]["coef"]


def test_routes_cover_every_policy_and_the_oracle_picks_the_cheapest_acceptable_tier():
    outcomes, _ = build_outcomes(*make_world())
    routes = route_all(outcomes, assign_folds(outcomes))
    assert set(routes) == set(prereg.POLICIES)
    assert routes["oracle"]["closed_qa-0"] == "medium" and routes["oracle"]["open_qa-1"] == "low"
    assert routes["feature_table"]["closed_qa-1"] == "medium"  # luna's closed_qa rate is under 0.95


def test_mcnemar_and_agreement_edge_cases():
    assert mcnemar_exact([True, True], [True, True]) == {"b": 0, "c": 0, "p": 1.0}
    assert mcnemar_exact([True] * 6, [False] * 6)["p"] == pytest.approx(2 / 64)
    one_rater_constant = agreement([True] * 10, [True] * 9 + [False])
    assert one_rater_constant["agreement"] == 0.9 and one_rater_constant["kappa"] == 0
    assert agreement([True, True], [True, True])["kappa"] is None


def test_cost_reduction_is_one_minus_the_cost_ratio():
    rng = np.random.default_rng(0)
    point, low, high = bootstrap_reduction([1.0, 1.0], [4.0, 4.0], rng)
    assert point == pytest.approx(0.75) and low == pytest.approx(0.75) and high == pytest.approx(0.75)


def test_results_are_deterministic_on_a_small_world(tmp_path):
    rows, matrix, verdicts = make_world()
    outcomes, excluded = build_outcomes(rows, matrix, verdicts)
    first = build_results(outcomes, excluded, verdicts, human_dir=tmp_path)
    assert first == build_results(outcomes, excluded, verdicts, human_dir=tmp_path)
    assert first["human_agreement"] is None
    assert first["policies"]["always_high"]["cost_reduction_vs_always_high"][0] == 0.0
    assert first["policies"]["always_low"]["acceptance"][0] == pytest.approx(16 / 20)


@pytest.mark.gate
def test_the_committed_slice2_results_match_a_fresh_offline_evaluation():
    """Regenerates the published slice 2 numbers from the committed recording, verdicts and human
    labels. A change to the policies, the features, the labels or the statistics fails here until
    `python -m harness.routing.evaluate` is run and its output committed."""
    from harness.routing.judge import load_verdicts
    from harness.routing.train import load_outcomes
    outcomes, excluded = load_outcomes()
    fresh = json.loads(json.dumps(build_results(outcomes, excluded, load_verdicts())))
    assert fresh == json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert fresh["n_prompts"] == 238


@pytest.mark.gate
def test_the_committed_estimates_match_a_fresh_training_run():
    from harness.routing.train import ESTIMATES_PATH, load_outcomes
    outcomes, excluded = load_outcomes()
    ids = sorted(outcomes)
    committed = json.loads(ESTIMATES_PATH.read_text(encoding="utf-8"))
    assert committed["feature_table"] == fit_feature_table(outcomes, ids)
    assert committed["classifier"] == json.loads(json.dumps(fit_classifier(outcomes, ids)))
    assert committed["excluded"] == excluded
