import json
import random

import pytest
from fastapi.testclient import TestClient

from llm_gateway.api import create_app
from llm_gateway.providers.base import ProviderError, ProviderResult
from llm_gateway.registry import cost_usd, load_registry
from llm_gateway.request_log import RequestLog
from llm_gateway.routing import Router, load_config

REGISTRY = load_registry()
REFERENCE = REGISTRY["gpt-6.1-sol"]


def write_config(directory, policy="always_low", rate=1.0, budget=1.0):
    path = directory / "routing.yaml"
    path.write_text(f'version: "t"\npolicy: {policy}\ntier_map:\n  low: gpt-6-luna\n  medium: claude-haiku-4-5\n'
                    f"  high: gpt-6.1-sol\nverifier:\n  sample_rate: {rate}\n  daily_budget_usd: {budget}\n",
                    encoding="utf-8")
    return path


class Provider:
    """Answers per model: the routed model's answer is set per test, the reference always answers."""

    def __init__(self, routed=None, reference_error=None):
        self.routed = routed or ProviderResult("Paris.", 20, 10, "stop")
        self.reference_error = reference_error
        self.calls = []

    async def complete(self, model, request):
        self.calls.append(model.id)
        if model.id == REFERENCE.id:
            if self.reference_error:
                raise self.reference_error
            return ProviderResult("Paris is the capital.", 20, 30, "stop")
        return self.routed


class Judge:
    model = REGISTRY["gpt-6-luna"]  # any priced model; only its prices matter here
    rubric = "Grade it."
    max_output_tokens = 100

    def __init__(self, acceptable=False):
        self.acceptable, self.calls = acceptable, []

    async def grade(self, request, reference, candidate):
        self.calls.append((request, reference, candidate))
        return self.acceptable, "wrong city" if not self.acceptable else "fine", 200, 20


def gateway(tmp_path, provider=None, judge=None, **config):
    provider, judge = provider or Provider(), judge or Judge()
    log = RequestLog(tmp_path / "log.sqlite3")
    router = Router(write_config(tmp_path, **config), set(REGISTRY))
    app = create_app(REGISTRY, {p: provider for p in ("openai", "anthropic", "groq")}, log,
                     router=router, verifier_judge=judge, verifier_rng=random.Random(0))
    return TestClient(app), log, provider, judge


def ask(client, i=0, model="auto"):
    return client.post("/v1/chat/completions", json={"model": model, "messages": [
        {"role": "user", "content": "What is the capital of France?"}]},
        headers={"X-Tenant-Id": "tenant-a", "X-Feature": "open_qa", "X-Request-Id": f"r{i}"})


def test_a_sampled_routed_answer_is_verified_after_the_response(tmp_path):
    client, log, provider, judge = gateway(tmp_path)
    response = ask(client)
    assert response.status_code == 200 and response.json()["model"] == "gpt-6-luna"
    [row] = log.verifications()
    assert (row["outcome"], row["acceptable"], row["reason"]) == ("judged", 0, "wrong city")
    assert (row["routed_tier"], row["routed_model"], row["reference_model"]) == ("low", "gpt-6-luna", REFERENCE.id)
    assert row["cost_usd"] == pytest.approx(cost_usd(REFERENCE, 20, 30) + cost_usd(Judge.model, 200, 20))
    assert provider.calls == ["gpt-6-luna", REFERENCE.id]
    assert judge.calls == [("What is the capital of France?", "Paris is the capital.", "Paris.")]


def test_a_failure_keeps_routing_features_but_never_the_prompt_text(tmp_path):
    client, log, _, _ = gateway(tmp_path)
    ask(client)
    [row] = log.verifications()
    features = json.loads(row["prompt_features"])
    assert features["feature=open_qa"] == 1.0 and "has_context" in features
    assert "France" not in json.dumps(row)


def test_a_cut_answer_fails_by_rule_without_any_extra_call(tmp_path):
    client, log, provider, judge = gateway(tmp_path, provider=Provider(ProviderResult("Par", 20, 1024, "length")))
    ask(client)
    [row] = log.verifications()
    assert (row["outcome"], row["acceptable"], row["cost_usd"]) == ("rejected_by_rule", 0, 0.0)
    assert provider.calls == ["gpt-6-luna"] and judge.calls == []


def test_the_sample_rate_decides_how_many_requests_are_verified(tmp_path):
    client, log, _, _ = gateway(tmp_path, rate=0.25)
    for i in range(200):
        ask(client, i)
    assert 30 <= len(log.verifications()) <= 70  # 50 expected; random.Random(0) makes it repeatable


def test_requests_not_routed_below_high_are_never_verified(tmp_path):
    client, log, _, _ = gateway(tmp_path, policy="always_high")
    ask(client)
    ask(client, 1, model="gpt-6-luna")  # a named model bypasses routing
    assert log.verifications() == []


def test_the_daily_cap_admits_a_verification_only_if_its_worst_case_fits(tmp_path):
    """The cap compares what the day really spent plus this verification's worst case with the budget,
    so verifications stop once the next worst case would not fit, and the spend never passes it."""
    budget = 0.012
    client, log, _, _ = gateway(tmp_path, budget=budget)
    verifier = client.app.state.verifier
    worst = verifier.worst_case_usd(REFERENCE, _chat(), "Paris.")
    actual = cost_usd(REFERENCE, 20, 30) + cost_usd(Judge.model, 200, 20)
    expected = max(k for k in range(1, 100) if (k - 1) * actual + worst <= budget)
    for i in range(expected + 5):
        ask(client, i)
    rows = log.verifications()
    assert len(rows) == expected and verifier.skipped_for_budget == 5
    assert sum(r["cost_usd"] for r in rows) <= budget


def _chat():
    from llm_gateway.schemas import ChatCompletionRequest
    return ChatCompletionRequest.model_validate(
        {"model": "auto", "messages": [{"role": "user", "content": "What is the capital of France?"}]})


def test_a_reference_failure_is_recorded_as_an_error_not_as_a_verdict(tmp_path):
    error = ProviderError("timeout", "openai", "slow")
    client, log, _, judge = gateway(tmp_path, provider=Provider(reference_error=error))
    assert ask(client).status_code == 200  # the client never sees the verifier's failure
    [row] = log.verifications()
    assert (row["outcome"], row["acceptable"], row["reason"]) == ("error", None, "openai timeout")
    assert judge.calls == []


def test_verifications_are_counted_in_metrics(tmp_path):
    client, _, _, _ = gateway(tmp_path)
    ask(client)
    text = client.get("/metrics").text
    assert 'llm_gateway_verifications_total{acceptable="false",feature="open_qa",outcome="judged",tier="low"} 1.0' in text


@pytest.mark.parametrize("section, message", [("  sample_rate: 2\n", "sample_rate"),
                                              ("  daily_budget_usd: -1\n", "daily_budget_usd")])
def test_an_invalid_verifier_section_is_refused(tmp_path, section, message):
    path = tmp_path / "routing.yaml"
    path.write_text('version: "t"\npolicy: always_low\ntier_map:\n  low: gpt-6-luna\n  medium: claude-haiku-4-5\n'
                    "  high: gpt-6.1-sol\nverifier:\n" + section, encoding="utf-8")
    from llm_gateway.routing import RoutingConfigError
    with pytest.raises(RoutingConfigError, match=message):
        load_config(path, set(REGISTRY))


def test_verifier_cost_expectations_on_a_small_table():
    from harness.routing.verifier_cost import build_results
    rows = [{"serve": 0.0001, "high": 0.002, "verify": 0.0025, "verify_both_judges": 0.005, "flagged": f, "failure": x}
            for f, x in [(True, True), (False, True), (False, False), (False, False)]]
    r = build_results(rows)
    assert (r["failures"], r["caught_by_verifier"], r["verifier_sensitivity"]) == (2, 1, 0.5)
    assert r["two_judge_verification_cost_ratio"] == 2.0
    ten = next(x for x in r["rates"] if x["sample_rate"] == 0.10)
    assert ten["overhead_per_1k_usd"] == pytest.approx(0.25) and ten["overhead_vs_serving"] == pytest.approx(2.5)
    assert ten["failures_caught_per_1k"] == pytest.approx(1000 * 0.10 * 1 / 4)


@pytest.mark.gate
def test_the_committed_verifier_results_match_a_fresh_computation():
    from harness.record import RECORDINGS_PATH, load_records
    from harness.replay import current_records
    from harness.routing.judge import load_verdicts
    from harness.routing.train import load_outcomes
    from harness.routing.verifier_cost import JSON_PATH, build_results, per_prompt
    outcomes, _ = load_outcomes()
    matrix = current_records(load_records(RECORDINGS_PATH), REGISTRY)
    fresh = json.loads(json.dumps(build_results(per_prompt(outcomes, matrix, load_verdicts()))))
    assert fresh == json.loads(JSON_PATH.read_text(encoding="utf-8"))
