"""What the gateway needs from a provider adapter, and the error taxonomy every adapter maps into.

The taxonomy is the gateway's, not the SDKs': health tracking and failover (slice 3) count failures
by kind, and a kind has to mean the same thing whichever provider produced it."""
from dataclasses import dataclass
from typing import Protocol

from llm_gateway.registry import ModelConfig
from llm_gateway.schemas import ChatCompletionRequest

# Used when the client sets no output limit. Reasoning tokens count against it, which is what makes
# the cost of a call bounded before it is made.
DEFAULT_MAX_OUTPUT_TOKENS = 1024

ERROR_KINDS = ("rate_limit", "timeout", "connection", "server_error", "auth", "bad_request")


@dataclass(frozen=True)
class ProviderResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str  # "stop", "length" or "content_filter"; anything else is passed through


class ProviderError(Exception):
    def __init__(self, kind: str, provider: str, message: str, status: int | None = None):
        if kind not in ERROR_KINDS:
            raise ValueError(f"unknown error kind {kind!r}")
        super().__init__(f"{provider} {kind}: {message}")
        self.kind, self.provider, self.message, self.status = kind, provider, message, status


class Provider(Protocol):
    async def complete(self, model: ModelConfig, request: ChatCompletionRequest) -> ProviderResult: ...


def output_limit(request: ChatCompletionRequest) -> int:
    return request.output_limit() or DEFAULT_MAX_OUTPUT_TOKENS


def classify(exc: Exception, sdk, provider: str) -> ProviderError:
    """Map an exception from the openai or anthropic SDK (both expose the same class names) to a
    ProviderError. Timeout is checked before connection because the SDKs subclass one from the
    other. Anything that is not an SDK API error is re-raised untouched: a bug in the adapter must
    not be counted as the provider failing."""
    if isinstance(exc, sdk.APITimeoutError):
        return ProviderError("timeout", provider, str(exc))
    if isinstance(exc, sdk.APIConnectionError):
        return ProviderError("connection", provider, str(exc))
    if isinstance(exc, sdk.APIStatusError):
        status = exc.status_code
        if status == 429:
            kind = "rate_limit"
        elif status in (401, 403):
            kind = "auth"
        elif status >= 500:
            kind = "server_error"
        else:
            kind = "bad_request"
        return ProviderError(kind, provider, str(exc), status)
    raise exc
