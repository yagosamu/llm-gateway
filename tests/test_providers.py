import asyncio
from types import SimpleNamespace

import anthropic
import httpx2
import openai
import pytest

from llm_gateway.providers.anthropic_messages import AnthropicProvider
from llm_gateway.providers.base import DEFAULT_MAX_OUTPUT_TOKENS, ProviderError, classify
from llm_gateway.providers.factory import MissingCredentials, build_providers
from llm_gateway.providers.openai_compat import OpenAICompatibleProvider
from llm_gateway.registry import load_registry
from llm_gateway.schemas import ChatCompletionRequest

REGISTRY = load_registry()
REQUEST = httpx2.Request("POST", "https://provider.test/v1")


def chat(**extra):
    return ChatCompletionRequest.model_validate({"model": "x", "messages": [
        {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Name a river."}], **extra})


class Recorder:
    """Stands in for an SDK resource: remembers the kwargs and returns or raises what it was given."""

    def __init__(self, result):
        self.result, self.kwargs = result, None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def openai_response(content="The Nile.", finish="stop", prompt=20, completion=40):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion))


def openai_provider(result):
    completions = Recorder(result)
    return OpenAICompatibleProvider("openai", SimpleNamespace(chat=SimpleNamespace(completions=completions))), completions


def anthropic_provider(result):
    messages = Recorder(result)
    return AnthropicProvider(SimpleNamespace(messages=messages)), messages


def anthropic_response(blocks, stop="end_turn", input_tokens=25, output_tokens=60):
    return SimpleNamespace(content=blocks, stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens))


def test_openai_adapter_sends_the_pinned_effort_and_the_default_output_limit():
    provider, completions = openai_provider(openai_response())
    result = asyncio.run(provider.complete(REGISTRY["gpt-6-luna"], chat()))
    assert completions.kwargs["model"] == "gpt-6-luna"
    assert completions.kwargs["reasoning_effort"] == "low"
    assert completions.kwargs["max_completion_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS
    assert "temperature" not in completions.kwargs
    assert completions.kwargs["messages"][0] == {"role": "system", "content": "Be brief."}
    assert (result.text, result.input_tokens, result.output_tokens, result.finish_reason) == ("The Nile.", 20, 40, "stop")


def test_groq_models_are_called_by_their_provider_model_id():
    provider, completions = openai_provider(openai_response())
    asyncio.run(provider.complete(REGISTRY["gpt-oss-20b"], chat(max_completion_tokens=64, temperature=0.2)))
    assert completions.kwargs["model"] == "openai/gpt-oss-20b"
    assert (completions.kwargs["max_completion_tokens"], completions.kwargs["temperature"]) == (64, 0.2)


def test_openai_adapter_turns_an_empty_content_into_an_empty_answer():
    provider, _ = openai_provider(openai_response(content=None, finish="length"))
    result = asyncio.run(provider.complete(REGISTRY["gpt-6-luna"], chat()))
    assert (result.text, result.finish_reason) == ("", "length")


def test_anthropic_adapter_moves_system_out_of_messages_and_keeps_only_text_blocks():
    blocks = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="The "),
              SimpleNamespace(type="text", text="Nile.")]
    provider, messages = anthropic_provider(anthropic_response(blocks))
    result = asyncio.run(provider.complete(REGISTRY["claude-sonnet-5-5"], chat()))
    assert messages.kwargs["system"] == "Be brief."
    assert messages.kwargs["messages"] == [{"role": "user", "content": "Name a river."}]
    assert messages.kwargs["output_config"] == {"effort": "medium"}
    assert messages.kwargs["max_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS
    assert (result.text, result.input_tokens, result.output_tokens) == ("The Nile.", 25, 60)


def test_anthropic_adapter_sends_no_effort_for_a_model_that_does_not_reason():
    provider, messages = anthropic_provider(anthropic_response([SimpleNamespace(type="text", text="Nile")]))
    asyncio.run(provider.complete(REGISTRY["claude-haiku-4-5"], chat()))
    assert "output_config" not in messages.kwargs


@pytest.mark.parametrize("stop, finish", [("end_turn", "stop"), ("max_tokens", "length"),
                                          ("refusal", "content_filter"), ("pause_turn", "pause_turn")])
def test_anthropic_stop_reasons_map_to_openai_finish_reasons(stop, finish):
    provider, _ = anthropic_provider(anthropic_response([], stop=stop))
    assert asyncio.run(provider.complete(REGISTRY["claude-haiku-4-5"], chat())).finish_reason == finish


def status_error(sdk, cls, status):
    return cls("boom", response=httpx2.Response(status, request=REQUEST), body=None)


@pytest.mark.parametrize("sdk", [openai, anthropic])
@pytest.mark.parametrize("make, kind, status", [
    (lambda sdk: status_error(sdk, sdk.RateLimitError, 429), "rate_limit", 429),
    (lambda sdk: status_error(sdk, sdk.AuthenticationError, 401), "auth", 401),
    (lambda sdk: status_error(sdk, sdk.PermissionDeniedError, 403), "auth", 403),
    (lambda sdk: status_error(sdk, sdk.InternalServerError, 500), "server_error", 500),
    (lambda sdk: status_error(sdk, sdk.APIStatusError, 529), "server_error", 529),
    (lambda sdk: status_error(sdk, sdk.BadRequestError, 400), "bad_request", 400),
    (lambda sdk: status_error(sdk, sdk.NotFoundError, 404), "bad_request", 404),
    (lambda sdk: sdk.APITimeoutError(request=REQUEST), "timeout", None),
    (lambda sdk: sdk.APIConnectionError(request=REQUEST), "connection", None),
])
def test_both_sdks_map_into_the_same_taxonomy(sdk, make, kind, status):
    error = classify(make(sdk), sdk, "p")
    assert (error.kind, error.status) == (kind, status)


def test_an_error_that_is_not_from_the_sdk_is_not_blamed_on_the_provider():
    with pytest.raises(KeyError):
        classify(KeyError("bug"), openai, "p")


def test_the_adapters_raise_the_gateway_error_not_the_sdk_error():
    provider, _ = openai_provider(status_error(openai, openai.RateLimitError, 429))
    with pytest.raises(ProviderError) as caught:
        asyncio.run(provider.complete(REGISTRY["gpt-6-luna"], chat()))
    assert (caught.value.kind, caught.value.provider) == ("rate_limit", "openai")
    provider, _ = anthropic_provider(anthropic.APITimeoutError(request=REQUEST))
    with pytest.raises(ProviderError) as caught:
        asyncio.run(provider.complete(REGISTRY["claude-haiku-4-5"], chat()))
    assert (caught.value.kind, caught.value.provider) == ("timeout", "anthropic")


def test_build_providers_names_every_missing_key_before_any_call():
    with pytest.raises(MissingCredentials, match="ANTHROPIC_API_KEY, GROQ_API_KEY"):
        build_providers({"openai", "anthropic", "groq"}, env={"OPENAI_API_KEY": "x"})


def test_build_providers_turns_off_sdk_retries():
    providers = build_providers({"openai", "anthropic", "groq"},
                                env={"OPENAI_API_KEY": "a", "ANTHROPIC_API_KEY": "b", "GROQ_API_KEY": "c"})
    assert all(p.client.max_retries == 0 for p in providers.values())
    assert str(providers["groq"].client.base_url).startswith("https://api.groq.com/openai/v1")
