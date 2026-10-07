"""The gateway's HTTP surface: an OpenAI-shaped /v1/chat/completions plus /v1/models.

Every request must say who it is for (X-Tenant-Id), which product feature sent it (X-Feature) and
carry its own id (X-Request-Id). Without the first two, cost cannot be attributed; without the
third, a retry cannot be told apart from a new request. A request missing any of them is refused
before its body is read."""
import json
import re
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from llm_gateway.providers.base import Provider, ProviderError
from llm_gateway.registry import ModelConfig, cost_usd
from llm_gateway.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    GatewayInfo,
    Usage,
)

REQUIRED_HEADERS = {"X-Tenant-Id": "tenant", "X-Feature": "feature", "X-Request-Id": "request_id"}
# Tenant and feature become metric labels later, so they are kept short and free of separators.
IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
# How a provider failure reaches the client. The provider's own message is not forwarded: it can
# name the gateway's account or key, which the client has no business seeing.
UPSTREAM_STATUS = {
    "rate_limit": (429, "the provider is rate limiting this gateway; retry later."),
    "timeout": (504, "the provider did not answer in time."),
    "connection": (502, "the provider could not be reached."),
    "server_error": (502, "the provider returned a server error."),
    "auth": (502, "the gateway's credentials for this provider were refused."),
    "bad_request": (400, "the provider rejected the request as sent; check the parameters."),
}


def error(status: int, code: str, message: str, param: str | None = None) -> JSONResponse:
    """The OpenAI error envelope, so existing clients surface the message as they would OpenAI's."""
    return JSONResponse(status_code=status, content={
        "error": {"message": message, "type": "invalid_request_error", "param": param, "code": code}})


def read_metadata(request: Request) -> dict[str, str] | JSONResponse:
    metadata = {}
    for header, key in REQUIRED_HEADERS.items():
        value = request.headers.get(header)
        if not value:
            return error(400, "missing_metadata", f"Header {header} is required on every request.", header)
        if not IDENTIFIER.match(value):
            return error(400, "invalid_metadata",
                         f"Header {header} must be 1 to 64 characters from A-Z, a-z, 0-9, '.', '_', ':' or '-'.",
                         header)
        metadata[key] = value
    return metadata


def _validation_message(exc: ValidationError) -> tuple[str, str | None]:
    first = exc.errors()[0]
    param = ".".join(str(part) for part in first["loc"]) or None
    return f"{param}: {first['msg']}" if param else first["msg"], param


def create_app(registry: dict[str, ModelConfig], providers: dict[str, Provider]) -> FastAPI:
    app = FastAPI(title="llm-gateway")

    @app.get("/v1/models")
    async def list_models() -> dict:
        return {"object": "list", "data": [
            {"id": m.id, "object": "model", "owned_by": m.provider, "tier": m.tier,
             "input_per_mtok": m.input_per_mtok, "output_per_mtok": m.output_per_mtok,
             "price_checked": m.price_checked.isoformat()}
            for m in registry.values()]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        metadata = read_metadata(request)
        if isinstance(metadata, JSONResponse):
            return metadata
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return error(400, "invalid_json", "The request body is not valid JSON.")
        try:
            chat = ChatCompletionRequest.model_validate(body)
        except ValidationError as exc:
            message, param = _validation_message(exc)
            return error(400, "invalid_request", message, param)
        if chat.stream:
            return error(400, "stream_not_supported", "This gateway does not stream; send stream=false.", "stream")
        if chat.tools:
            return error(400, "tools_not_supported", "This gateway does not forward tools.", "tools")
        model = registry.get(chat.model)
        if model is None:
            return error(404, "model_not_found",
                         f"Model {chat.model!r} is not in the registry. GET /v1/models lists the options.", "model")

        started = time.perf_counter()
        try:
            result = await providers[model.provider].complete(model, chat)
        except ProviderError as exc:
            status, hint = UPSTREAM_STATUS[exc.kind]
            return error(status, f"upstream_{exc.kind}", f"{model.provider} failed ({exc.kind}): {hint}", "model")
        latency_ms = (time.perf_counter() - started) * 1000

        response = ChatCompletionResponse(
            id=f"chatcmpl-{metadata['request_id']}",
            created=int(time.time()),
            model=model.id,
            choices=[Choice(message=AssistantMessage(content=result.text), finish_reason=result.finish_reason)],
            usage=Usage(prompt_tokens=result.input_tokens, completion_tokens=result.output_tokens,
                        total_tokens=result.input_tokens + result.output_tokens),
            gateway=GatewayInfo(**metadata, provider=model.provider,
                                cost_usd=cost_usd(model, result.input_tokens, result.output_tokens),
                                latency_ms=latency_ms),
        )
        return JSONResponse(response.model_dump(), headers={"X-Request-Id": metadata["request_id"]})

    return app
