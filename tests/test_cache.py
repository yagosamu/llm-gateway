import fakeredis
import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from llm_gateway.api import create_app
from llm_gateway.cache import ResponseCache
from llm_gateway.providers.base import ProviderError, ProviderResult
from llm_gateway.registry import load_registry
from llm_gateway.request_log import RequestLog
from llm_gateway.routing import Router, RoutingConfigError, load_config

REGISTRY = load_registry()
SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
ADMIN = {"X-Admin-Token": "secret"}


def write_config(directory, cache='  enabled: true\n  ttl_seconds: 600\n  features: ["*"]\n'):
    path = directory / "routing.yaml"
    path.write_text('version: "t"\npolicy: always_low\ntier_map:\n  low: gpt-6-luna\n  medium: claude-haiku-4-5\n'
                    "  high: gpt-6.1-sol\nfailover:\n  fallbacks:\n    low: [claude-haiku-4-5]\n"
                    f"cache:\n{cache}", encoding="utf-8")
    return path


class Fake:
    """Answers "answer N" (N counts calls) unless told to fail or to finish otherwise."""

    def __init__(self):
        self.calls, self.fail, self.finish, self.text = [], set(), "stop", None

    def provider(self, name):
        outer = self

        class One:
            async def complete(self, model, request):
                if name in outer.fail:
                    raise ProviderError("server_error", name, "down")
                outer.calls.append(model.id)
                text = outer.text if outer.text is not None else f"answer {len(outer.calls)}"
                return ProviderResult(text, 10, 5, outer.finish)
        return One()


def gateway(tmp_path, store=None, **config):
    fake, log = Fake(), RequestLog(":memory:")
    store = store if store is not None else fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())
    cache = ResponseCache(store)
    app = create_app(REGISTRY, {p: fake.provider(p) for p in ("openai", "anthropic", "groq")}, log,
                     router=Router(write_config(tmp_path, **config), set(REGISTRY)), cache=cache, admin_token="secret")
    return TestClient(app), fake, log, store


def ask(client, text="Where is my order?", tenant="tenant-a", feature="support", system=SYSTEM, headers=None, **body):
    messages = ([system] if system else []) + [{"role": "user", "content": text}]
    return client.post("/v1/chat/completions", json={"model": "auto", "messages": messages, **body},
                       headers={"X-Tenant-Id": tenant, "X-Feature": feature, "X-Request-Id": "r1", **(headers or {})})


def test_the_same_request_is_answered_once_then_from_the_cache_at_no_cost(tmp_path):
    client, fake, log, _ = gateway(tmp_path)
    first, second = ask(client).json(), ask(client).json()
    assert fake.calls == ["gpt-6-luna"]
    assert second["choices"][0]["message"]["content"] == first["choices"][0]["message"]["content"] == "answer 1"
    assert first["gateway"]["cache"] is None
    assert second["gateway"]["cache"]["layer"] == "exact" and second["gateway"]["cost_usd"] == 0.0
    assert second["gateway"]["cache"]["saved_usd"] == pytest.approx(first["gateway"]["cost_usd"])
    assert [r["cache"] for r in log.rows()] == ["miss", "exact"]


def test_whitespace_differences_still_hit(tmp_path):
    client, fake, _, _ = gateway(tmp_path)
    ask(client, "Where is my  order?")
    assert ask(client, "  Where is my\norder? ").json()["gateway"]["cache"]["layer"] == "exact"
    assert len(fake.calls) == 1


@pytest.mark.parametrize("change", [
    {"tenant": "tenant-b"},                                         # another customer
    {"feature": "checkout"},                                        # another product surface
    {"system": {"role": "system", "content": "You are terse."}},    # the system prompt was edited
    {"headers": {"X-Prompt-Version": "v2"}},                        # the prompt version was bumped
    {"temperature": 0.7},
    {"max_tokens": 50},
    {"text": "Where is my refund?"},
])
def test_anything_in_the_key_that_differs_misses(tmp_path, change):
    client, fake, _, _ = gateway(tmp_path)
    ask(client)
    response = ask(client, **change).json()
    assert response["gateway"]["cache"] is None and len(fake.calls) == 2


def test_the_namespace_alone_separates_system_prompts_and_versions():
    # The semantic layer searches within a namespace, so the namespace itself, not only the exact key,
    # has to change with the system prompt and its version.
    from llm_gateway.cache import namespace
    user = [{"role": "user", "content": "Where is my order?"}]
    base = namespace("t", "f", "gpt-6-luna", None, None, [SYSTEM] + user, None)
    assert namespace("t", "f", "gpt-6-luna", None, None, [{"role": "system", "content": "Be terse."}] + user, None) != base
    assert namespace("t", "f", "gpt-6-luna", None, None, [SYSTEM] + user, "v2") != base
    assert namespace("t", "f", "gpt-6-luna", None, None, [SYSTEM, {"role": "user", "content": "Other"}], None) == base


def test_a_named_model_and_auto_do_not_share_answers_across_models(tmp_path):
    client, fake, _, _ = gateway(tmp_path)
    ask(client)
    response = client.post("/v1/chat/completions", json={"model": "claude-haiku-4-5", "messages": [
        SYSTEM, {"role": "user", "content": "Where is my order?"}]},
        headers={"X-Tenant-Id": "tenant-a", "X-Feature": "support", "X-Request-Id": "r2"}).json()
    assert response["model"] == "claude-haiku-4-5" and response["gateway"]["cache"] is None


@pytest.mark.parametrize("finish, text", [("length", "cut off mid"), ("stop", "   ")])
def test_cut_or_empty_answers_are_not_stored(tmp_path, finish, text):
    client, fake, _, _ = gateway(tmp_path)
    fake.finish, fake.text = finish, text
    ask(client)
    ask(client)
    assert len(fake.calls) == 2


def test_a_fallback_answer_is_not_filed_under_the_primary(tmp_path):
    client, fake, _, _ = gateway(tmp_path)
    fake.fail.add("openai")
    assert ask(client).json()["model"] == "claude-haiku-4-5"
    fake.fail.clear()
    response = ask(client).json()
    assert response["model"] == "gpt-6-luna" and response["gateway"]["cache"] is None


def test_entries_carry_the_configured_ttl_and_a_key_scoped_by_tenant_and_feature(tmp_path):
    server = fakeredis.FakeServer()
    client, _, _, _ = gateway(tmp_path, store=fakeredis.FakeAsyncRedis(server=server))
    ask(client)
    sync = fakeredis.FakeRedis(server=server)
    [name] = sync.keys("llmgw:cache/*")
    assert name.decode().startswith("llmgw:cache/tenant-a/support/")
    assert 0 < sync.ttl(name) <= 600


def test_a_feature_outside_the_config_bypasses_the_cache(tmp_path):
    client, fake, log, _ = gateway(tmp_path, cache='  enabled: true\n  features: ["faq"]\n')
    ask(client)
    ask(client)
    assert len(fake.calls) == 2 and [r["cache"] for r in log.rows()] == [None, None]


def test_invalidation_by_tenant_reaches_only_that_tenant(tmp_path):
    client, fake, _, _ = gateway(tmp_path)
    ask(client, tenant="a")
    ask(client, tenant="a:b")
    deleted = client.delete("/admin/cache", params={"tenant": "a"}, headers=ADMIN).json()["deleted"]
    assert deleted == 1
    assert ask(client, tenant="a:b").json()["gateway"]["cache"]["layer"] == "exact"
    assert ask(client, tenant="a").json()["gateway"]["cache"] is None


def test_invalidation_by_feature_and_the_admin_guard(tmp_path):
    client, _, _, _ = gateway(tmp_path)
    ask(client, feature="support")
    ask(client, feature="faq")
    assert client.delete("/admin/cache", params={"feature": "faq"}).status_code == 403
    assert client.delete("/admin/cache", params={"feature": "faq"}, headers=ADMIN).json()["deleted"] == 1
    assert ask(client, feature="support").json()["gateway"]["cache"]["layer"] == "exact"


class DownRedis:
    async def hget(self, *args):
        raise RedisConnectionError("down")

    def pipeline(self, *args, **kwargs):
        raise RedisConnectionError("down")


def test_redis_down_means_a_miss_and_the_request_is_still_answered(tmp_path):
    client, fake, _, _ = gateway(tmp_path, store=DownRedis())
    assert ask(client).status_code == 200 and ask(client).status_code == 200
    assert len(fake.calls) == 2
    text = client.get("/metrics").text
    assert 'llm_gateway_cache_errors_total{operation="lookup"} 2.0' in text


def test_hits_are_counted_and_kept_out_of_provider_latency(tmp_path):
    client, _, _, _ = gateway(tmp_path)
    ask(client)
    ask(client)
    text = client.get("/metrics").text
    assert 'llm_gateway_cache_lookups_total{layer="exact",result="hit"} 1.0' in text
    assert 'llm_gateway_cache_lookups_total{layer="exact",result="miss"} 1.0' in text
    assert 'llm_gateway_provider_latency_seconds_count{model="gpt-6-luna",provider="openai"} 1.0' in text


@pytest.mark.parametrize("section, message", [
    ("  enabled: yes please\n", "cache.enabled"),
    ("  ttl_seconds: 0\n", "cache.ttl_seconds"),
    ("  features: []\n", "cache.features"),
    ("  threshold: 0.9\n", "unknown keys"),
])
def test_invalid_cache_sections_are_refused(tmp_path, section, message):
    with pytest.raises(RoutingConfigError, match=message):
        load_config(write_config(tmp_path, cache=section), set(REGISTRY))


def test_the_committed_config_serves_the_exact_layer_only():
    # The slice 4 decision (results/slice4.md): exact layer on, semantic layer off.
    from llm_gateway.routing import DEFAULT_CONFIG_PATH
    cache = load_config(DEFAULT_CONFIG_PATH, set(REGISTRY)).cache
    assert cache.enabled is True and cache.semantic_threshold is None and cache.features == ("*",)
