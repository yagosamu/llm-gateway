"""What the gateway needs from a provider adapter, and nothing more."""
from dataclasses import dataclass
from typing import Protocol

from llm_gateway.registry import ModelConfig
from llm_gateway.schemas import ChatCompletionRequest


@dataclass(frozen=True)
class ProviderResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str


class Provider(Protocol):
    async def complete(self, model: ModelConfig, request: ChatCompletionRequest) -> ProviderResult: ...
