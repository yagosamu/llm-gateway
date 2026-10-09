import json
import math
import os

import pytest
from fastapi.testclient import TestClient

from llm_gateway.api import create_app
from llm_gateway.providers.base import ProviderResult
from llm_gateway.registry import load_registry
from llm_gateway.request_log import RequestLog
from llm_gateway.routing import (
    DEFAULT_CONFIG_PATH,
    FeatureTable,
    LogisticClassifier,
    Router,
    RoutingConfigError,
    load_config,
    prompt_features,
)

REGISTRY = load_registry()
IDS = set(REGISTRY)
TIER_MAP = {"low": "gpt-6-luna", "medium": "claude-haiku-4-5", "high": "gpt-6.1-sol"}
ESTIMATES = {
    "feature_table": {"open_qa": {"low": 0.97, "medium": 0.99}, "creative_writing": {"low": 0.80, "medium": 0.96},
                      "general_qa": {"low": 0.60, "medium": 0.70}},
    "classifier": {"low": {"intercept": 4.0, "coef": {"has_context": -6.0}},
                   "medium": {"intercept": 4.0, "coef": {}}},
}


def write_config(directory, policy="feature_table", tier_map=None, threshold=0.95, version="v1"):
    (directory / "estimates.json").write_text(json.dumps(ESTIMATES), encoding="utf-8")
    lines = [f'version: "{version}"', f"policy: {policy}", f"decision_threshold: {threshold}",
             "estimates: estimates.json", "tier_map:"]
    lines += [f"  {tier}: {model}" for tier, model in (tier_map or TIER_MAP).items()]
    path = directory / "routing.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def msgs(text):
    return [{"role": "system", "content": "Be brief."}, {"role": "user", "content": text}]


def test_the_committed_config_serves_the_measured_choice_with_the_verifier_on():
    config = load_config(DEFAULT_CONFIG_PATH, IDS)
    assert config.policy_name == "always_low" and config.tier_map["high"] == "gpt-6.1-sol"
    assert 0 < config.verifier.sample_rate <= 1 and config.verifier.daily_budget_usd > 0


def test_features_read_the_instruction_and_detect_a_context_block():
    with_context = prompt_features(msgs("When did it open?\n\nContext:\nWrite a story about the bridge."), "closed_qa")
    assert with_context["has_context"] == 1.0
    assert with_context["asks_creative"] == 0.0  # "Write ... story" is in the context, not the instruction
    assert with_context["feature=closed_qa"] == 1.0
    plain = prompt_features(msgs("Compare cats and dogs, then list three differences."), "general_qa")
    assert (plain["has_context"], plain["asks_compare"], plain["asks_list"]) == (0.0, 1.0, 1.0)
    assert plain["log_chars"] == pytest.approx(math.log1p(len("Compare cats and dogs, then list three differences.")))


def test_feature_table_routes_to_the_cheapest_tier_that_reaches_the_threshold():
    table = FeatureTable(ESTIMATES["feature_table"], 0.95)
    assert table.decide("open_qa", msgs("q")).tier == "low"
    assert table.decide("creative_writing", msgs("q")).tier == "medium"
    assert table.decide("general_qa", msgs("q")).tier == "high"
    assert table.decide("never_seen", msgs("q")).tier == "high"


def test_classifier_uses_the_logistic_probability_per_tier():
    classifier = LogisticClassifier(ESTIMATES["classifier"], 0.95)
    assert classifier.probability("low", {"has_context": 0.0}) == pytest.approx(1 / (1 + math.exp(-4)))
    assert classifier.decide("open_qa", msgs("Who wrote it?")).tier == "low"
    assert classifier.decide("closed_qa", msgs("Who?\n\nContext:\nSome text.")).tier == "medium"


@pytest.mark.parametrize("change, message", [
    ({"tier_map": {"low": "gpt-6-luna", "medium": "claude-haiku-4-5", "high": "gpt-9"}}, "outside the registry"),
    ({"tier_map": {"low": "gpt-6-luna", "high": "gpt-6.1-sol"}}, "exactly"),
    ({"policy": "cheapest"}, "policy must be"),
    ({"threshold": 1.5}, "decision_threshold"),
])
def test_an_invalid_config_is_refused_with_its_reason(tmp_path, change, message):
    path = write_config(tmp_path, **change)
    with pytest.raises(RoutingConfigError, match=message):
        load_config(path, IDS)


def touch_later(path):
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


def test_the_router_reloads_an_edited_config_without_a_restart(tmp_path):
    path = write_config(tmp_path, policy="always_low", version="v1")
    router = Router(path, IDS)
    assert router.route("open_qa", msgs("q"))[0] == "gpt-6-luna"
    write_config(tmp_path, policy="always_medium", version="v2")
    touch_later(path)
    model, decision, config = router.route("open_qa", msgs("q"))
    assert (model, decision.tier, config.version) == ("claude-haiku-4-5", "medium", "v2")


def test_a_bad_edit_is_refused_and_the_last_good_config_stays(tmp_path):
    path = write_config(tmp_path, policy="always_low", version="v1")
    router = Router(path, IDS)
    path.write_text("policy: [not, valid\n", encoding="utf-8")
    touch_later(path)
    model, _, config = router.route("open_qa", msgs("q"))
    assert (model, config.version, router.rejected_reloads) == ("gpt-6-luna", "v1", 1)


class Provider:
    async def complete(self, model, request):
        return ProviderResult(text=f"from {model.id}", input_tokens=10, output_tokens=5, finish_reason="stop")


@pytest.fixture
def client(tmp_path):
    router = Router(write_config(tmp_path), IDS)
    log = RequestLog(":memory:")
    app = create_app(REGISTRY, {p: Provider() for p in ("openai", "anthropic", "groq")}, log, router=router)
    return TestClient(app), log


def post(client, feature, text="Who wrote it?", model="auto"):
    return client.post("/v1/chat/completions", json={"model": model, "messages": msgs(text)},
                       headers={"X-Tenant-Id": "tenant-a", "X-Feature": feature, "X-Request-Id": f"r-{feature}"})


def test_auto_routes_by_policy_and_says_why(client):
    test_client, log = client
    data = post(test_client, "creative_writing").json()
    assert data["model"] == "claude-haiku-4-5"
    routing = data["gateway"]["routing"]
    assert (routing["tier"], routing["policy"], routing["config_version"]) == ("medium", "feature_table", "v1")
    assert "creative_writing" in routing["reason"]
    [row] = log.rows()
    assert (row["requested_model"], row["model"], row["routed_tier"], row["routing_policy"]) == (
        "auto", "claude-haiku-4-5", "medium", "feature_table")


def test_a_named_model_bypasses_the_router(client):
    test_client, log = client
    data = post(test_client, "creative_writing", model="gpt-oss-20b").json()
    assert data["model"] == "gpt-oss-20b" and data["gateway"]["routing"] is None
    assert log.rows()[0]["routed_tier"] is None


def test_routed_requests_are_counted_by_tier(client):
    test_client, _ = client
    post(test_client, "open_qa")
    post(test_client, "general_qa")
    text = test_client.get("/metrics").text
    assert 'llm_gateway_routed_total{feature="open_qa",policy="feature_table",tier="low"} 1.0' in text
    assert 'llm_gateway_routed_total{feature="general_qa",policy="feature_table",tier="high"} 1.0' in text


def test_auto_without_a_router_is_refused():
    app = create_app(REGISTRY, {p: Provider() for p in ("openai", "anthropic", "groq")})
    response = post(TestClient(app), "open_qa")
    assert response.status_code == 400 and response.json()["error"]["code"] == "routing_disabled"


def test_an_old_log_file_gains_the_routing_columns(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE requests (id INTEGER PRIMARY KEY AUTOINCREMENT, received_at TEXT NOT NULL, "
                 "outcome TEXT NOT NULL, http_status INTEGER NOT NULL)")
    conn.commit()
    conn.close()
    log = RequestLog(path)
    from llm_gateway.request_log import RequestRecord
    log.write(RequestRecord(received_at="t", outcome="ok", http_status=200, routed_tier="low"))
    assert log.rows()[0]["routed_tier"] == "low"
