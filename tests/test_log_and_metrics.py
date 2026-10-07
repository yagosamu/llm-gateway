import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from llm_gateway.api import create_app
from llm_gateway.metrics import GatewayMetrics
from llm_gateway.providers.base import ProviderError, ProviderResult
from llm_gateway.registry import cost_usd, load_registry
from llm_gateway.request_log import RequestLog, RequestRecord, prompt_sha256

REGISTRY = load_registry()
HEADERS = {"X-Tenant-Id": "tenant-a", "X-Feature": "open_qa", "X-Request-Id": "dolly-00001:gpt-6-luna"}
SECRET = "my account number is 12345"
BODY = {"model": "gpt-6-luna", "messages": [{"role": "user", "content": SECRET}]}


class Provider:
    def __init__(self, fail_kind=None):
        self.fail_kind = fail_kind

    async def complete(self, model, request):
        if self.fail_kind:
            raise ProviderError(self.fail_kind, model.provider, "boom")
        return ProviderResult(text="ok", input_tokens=100, output_tokens=40, finish_reason="stop")


@pytest.fixture
def gateway(tmp_path):
    def make(fail_kind=None):
        log, metrics = RequestLog(tmp_path / "requests.sqlite3"), GatewayMetrics()
        app = create_app(REGISTRY, {p: Provider(fail_kind) for p in ("openai", "anthropic", "groq")}, log, metrics)
        return TestClient(app), log
    return make


def samples(client) -> dict[tuple, float]:
    """(metric sample name, sorted label items) -> value, parsed from /metrics."""
    text = client.get("/metrics").text
    return {(s.name, tuple(sorted(s.labels.items()))): s.value
            for family in text_string_to_metric_families(text) for s in family.samples}


def test_an_answered_request_leaves_one_full_row_and_no_prompt_text(gateway, tmp_path):
    client, log = gateway()
    client.post("/v1/chat/completions", json=BODY, headers=HEADERS)
    [row] = log.rows()
    assert (row["tenant"], row["feature"], row["request_id"]) == ("tenant-a", "open_qa", "dolly-00001:gpt-6-luna")
    assert (row["model"], row["provider"], row["tier"], row["outcome"], row["http_status"]) == (
        "gpt-6-luna", "openai", "low", "ok", 200)
    assert (row["input_tokens"], row["output_tokens"], row["finish_reason"]) == (100, 40, "stop")
    assert row["cost_usd"] == pytest.approx(cost_usd(REGISTRY["gpt-6-luna"], 100, 40))
    assert row["latency_ms"] >= 0
    assert row["prompt_sha256"] == prompt_sha256(BODY["messages"])
    log.close()
    assert SECRET.encode() not in b"".join(p.read_bytes() for p in tmp_path.iterdir())


@pytest.mark.parametrize("headers, body, code, model", [
    ({k: v for k, v in HEADERS.items() if k != "X-Feature"}, BODY, "missing_metadata", None),
    (HEADERS, {**BODY, "model": "a-model-nobody-registered"}, "model_not_found", None),
    (HEADERS, {**BODY, "stream": True}, "stream_not_supported", "gpt-6-luna"),
])
def test_a_refused_request_is_logged_with_what_was_known(gateway, headers, body, code, model):
    client, log = gateway()
    client.post("/v1/chat/completions", json=body, headers=headers)
    [row] = log.rows()
    assert (row["outcome"], row["error_code"], row["model"]) == ("rejected", code, model)
    assert row["tenant"] == "tenant-a" and row["cost_usd"] is None


def test_a_provider_failure_is_logged_as_upstream_error(gateway):
    client, log = gateway(fail_kind="timeout")
    client.post("/v1/chat/completions", json=BODY, headers=HEADERS)
    [row] = log.rows()
    assert (row["outcome"], row["error_code"], row["http_status"]) == ("upstream_error", "upstream_timeout", 504)


def test_metrics_count_requests_tokens_and_cost_by_tenant_feature_and_model(gateway):
    client, _ = gateway()
    for _ in range(3):
        client.post("/v1/chat/completions", json=BODY, headers=HEADERS)
    client.post("/v1/chat/completions", json={**BODY, "model": "a-model-nobody-registered"}, headers=HEADERS)
    s = samples(client)
    base = {"tenant": "tenant-a", "feature": "open_qa"}
    key = lambda name, **labels: (name, tuple(sorted({**base, **labels}.items())))
    assert s[key("llm_gateway_requests_total", model="gpt-6-luna", outcome="ok", code="none")] == 3
    assert s[key("llm_gateway_requests_total", model="unknown", outcome="rejected", code="model_not_found")] == 1
    assert s[key("llm_gateway_tokens_total", model="gpt-6-luna", direction="output")] == 120
    assert s[key("llm_gateway_cost_usd_total", model="gpt-6-luna")] == pytest.approx(
        3 * cost_usd(REGISTRY["gpt-6-luna"], 100, 40))
    assert s[("llm_gateway_provider_latency_seconds_count", (("model", "gpt-6-luna"), ("provider", "openai")))] == 3


def test_an_unregistered_model_name_never_becomes_a_label(gateway):
    client, _ = gateway()
    client.post("/v1/chat/completions", json={**BODY, "model": "x" * 50}, headers=HEADERS)
    assert "x" * 50 not in client.get("/metrics").text


def test_the_log_refuses_an_unknown_outcome(tmp_path):
    with pytest.raises(ValueError):
        RequestLog(tmp_path / "r.sqlite3").write(RequestRecord(received_at="t", outcome="maybe"))
