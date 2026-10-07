import pytest
from fastapi.testclient import TestClient

from llm_gateway.api import create_app
from llm_gateway.providers.base import ProviderResult
from llm_gateway.registry import cost_usd, load_registry

HEADERS = {"X-Tenant-Id": "tenant-a", "X-Feature": "open_qa", "X-Request-Id": "dolly-00001"}
BODY = {"model": "gpt-6-luna", "messages": [{"role": "user", "content": "Name a river."}]}


class FakeProvider:
    def __init__(self):
        self.calls = []

    async def complete(self, model, request):
        self.calls.append((model.id, request))
        return ProviderResult(text="The Amazon.", input_tokens=120, output_tokens=30, finish_reason="stop")


@pytest.fixture
def fake():
    return FakeProvider()


@pytest.fixture
def client(fake):
    registry = load_registry()
    return TestClient(create_app(registry, {p: fake for p in ("anthropic", "openai", "groq")}))


def test_a_valid_request_returns_the_openai_shape_with_cost_attributed(client, fake):
    response = client.post("/v1/chat/completions", json=BODY, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion" and data["model"] == "gpt-6-luna"
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "The Amazon."}
    assert data["usage"] == {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    gateway = data["gateway"]
    assert (gateway["tenant"], gateway["feature"], gateway["request_id"]) == ("tenant-a", "open_qa", "dolly-00001")
    assert gateway["cost_usd"] == pytest.approx(cost_usd(load_registry()["gpt-6-luna"], 120, 30))
    assert response.headers["X-Request-Id"] == "dolly-00001"
    assert [model_id for model_id, _ in fake.calls] == ["gpt-6-luna"]


@pytest.mark.parametrize("header", ["X-Tenant-Id", "X-Feature", "X-Request-Id"])
def test_a_request_without_a_metadata_header_is_refused_before_any_call(client, fake, header):
    headers = {k: v for k, v in HEADERS.items() if k != header}
    response = client.post("/v1/chat/completions", json=BODY, headers=headers)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_metadata"
    assert response.json()["error"]["param"] == header
    assert fake.calls == []


def test_a_metadata_value_outside_the_allowed_characters_is_refused(client, fake):
    response = client.post("/v1/chat/completions", json=BODY, headers={**HEADERS, "X-Tenant-Id": "acme corp"})
    assert response.status_code == 400 and response.json()["error"]["code"] == "invalid_metadata"
    assert fake.calls == []


@pytest.mark.parametrize("extra, code, param", [
    ({"stream": True}, "stream_not_supported", "stream"),
    ({"tools": [{"type": "function"}]}, "tools_not_supported", "tools"),
    ({"top_p": 0.5}, "invalid_request", "top_p"),
    ({"temperature": 3}, "invalid_request", "temperature"),
])
def test_unsupported_features_are_refused_with_their_own_code(client, fake, extra, code, param):
    response = client.post("/v1/chat/completions", json={**BODY, **extra}, headers=HEADERS)
    assert response.status_code == 400
    assert (response.json()["error"]["code"], response.json()["error"]["param"]) == (code, param)
    assert fake.calls == []


def test_an_unknown_model_is_a_404_in_the_openai_envelope(client, fake):
    response = client.post("/v1/chat/completions", json={**BODY, "model": "gpt-9"}, headers=HEADERS)
    assert response.status_code == 404 and response.json()["error"]["code"] == "model_not_found"
    assert fake.calls == []


def test_a_body_that_is_not_json_is_refused(client):
    response = client.post("/v1/chat/completions", content=b"{not json", headers=HEADERS)
    assert response.status_code == 400 and response.json()["error"]["code"] == "invalid_json"


def test_max_tokens_and_max_completion_tokens_both_set_the_output_limit(client, fake):
    client.post("/v1/chat/completions", json={**BODY, "max_tokens": 64}, headers=HEADERS)
    client.post("/v1/chat/completions", json={**BODY, "max_completion_tokens": 32}, headers=HEADERS)
    assert [request.output_limit() for _, request in fake.calls] == [64, 32]


def test_models_lists_every_registry_entry_with_its_price(client):
    data = client.get("/v1/models").json()["data"]
    assert [m["id"] for m in data] == list(load_registry())
    luna = next(m for m in data if m["id"] == "gpt-6-luna")
    assert (luna["input_per_mtok"], luna["output_per_mtok"], luna["tier"]) == (0.10, 0.50, "low")
