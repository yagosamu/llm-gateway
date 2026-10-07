"""Adapter for the Anthropic Messages API.

Translations from the OpenAI shape: system messages move to the top-level `system` field;
reasoning effort goes in `output_config.effort` (omitted for a model that does not reason); only
text blocks form the answer, thinking blocks are dropped; stop reasons map to OpenAI's finish
reasons. Server-side refusal fallbacks are deliberately not enabled: they would let a different
model answer under this model's name and break per-model cost attribution."""
import anthropic

from llm_gateway.providers.base import ProviderResult, classify, output_limit
from llm_gateway.registry import ModelConfig
from llm_gateway.schemas import ChatCompletionRequest

FINISH_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "refusal": "content_filter"}


class AnthropicProvider:
    def __init__(self, client: anthropic.AsyncAnthropic, name: str = "anthropic"):
        self.name, self.client = name, client

    async def complete(self, model: ModelConfig, request: ChatCompletionRequest) -> ProviderResult:
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        kwargs = {
            "model": model.provider_model,
            "max_tokens": output_limit(request),
            "messages": [{"role": m.role, "content": m.content} for m in request.messages if m.role != "system"],
        }
        if system:
            kwargs["system"] = system
        if model.reasoning_effort != "none":
            kwargs["output_config"] = {"effort": model.reasoning_effort}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        try:
            response = await self.client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise classify(exc, anthropic, self.name) from exc
        return ProviderResult(
            text="".join(block.text for block in response.content if block.type == "text"),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=FINISH_REASONS.get(response.stop_reason, response.stop_reason),
        )
