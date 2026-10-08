"""The subset of the OpenAI chat completions shape the gateway accepts and returns.

Unknown request fields are rejected rather than dropped: a client that sends top_p or n and gets an
answer computed without them would be misled. Streaming and tools are rejected in the API layer
with their own error codes, so a client learns exactly which feature is missing."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False
    tools: list[Any] | None = None

    def output_limit(self) -> int | None:
        """max_completion_tokens is the current OpenAI name; max_tokens is the legacy one."""
        return self.max_completion_tokens or self.max_tokens


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: str


class RoutingInfo(BaseModel):
    """Present when the request asked for model "auto": which tier was chosen, by what, and why."""
    tier: str
    policy: str
    reason: str
    config_version: str


class GatewayInfo(BaseModel):
    """Extra top-level field; OpenAI clients ignore keys they do not know."""
    request_id: str
    tenant: str
    feature: str
    provider: str
    cost_usd: float
    latency_ms: float
    routing: RoutingInfo | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    gateway: GatewayInfo
