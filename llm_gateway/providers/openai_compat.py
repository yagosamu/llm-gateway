"""Adapter for the OpenAI Chat Completions API. Groq exposes the same API, so one class serves both;
only the client's base URL and key differ."""
import openai

from llm_gateway.providers.base import ProviderResult, classify, output_limit
from llm_gateway.registry import ModelConfig
from llm_gateway.schemas import ChatCompletionRequest

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class OpenAICompatibleProvider:
    def __init__(self, name: str, client: openai.AsyncOpenAI):
        self.name, self.client = name, client

    async def complete(self, model: ModelConfig, request: ChatCompletionRequest) -> ProviderResult:
        kwargs = {
            "model": model.provider_model,
            "messages": [m.model_dump() for m in request.messages],
            "max_completion_tokens": output_limit(request),
            "reasoning_effort": model.reasoning_effort,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        try:
            response = await self.client.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            raise classify(exc, openai, self.name) from exc
        choice = response.choices[0]
        return ProviderResult(
            text=choice.message.content or "",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            finish_reason=choice.finish_reason,
        )
