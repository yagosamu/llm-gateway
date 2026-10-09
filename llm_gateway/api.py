"""The gateway's HTTP surface: an OpenAI-shaped /v1/chat/completions, /v1/models, /v1/health and
/metrics.

Every request must say who it is for (X-Tenant-Id), which product feature sent it (X-Feature) and
carry its own id (X-Request-Id). Without the first two, cost cannot be attributed; without the
third, a retry cannot be told apart from a new request. A request missing any of them is refused
before its body is read.

Every request, answered or refused, leaves exactly one row in the request log and one increment in
the metrics, written in one place after the response is decided."""
import asyncio
import json
import math
import random
import re
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from redis.exceptions import RedisError

from llm_gateway.breaker import STATES as BREAKER_STATES
from llm_gateway.breaker import CircuitBreaker
from llm_gateway.health import HealthTracker
from llm_gateway.metrics import GatewayMetrics
from llm_gateway.providers.base import Provider, ProviderError
from llm_gateway.registry import ModelConfig, cost_usd
from llm_gateway.request_log import RequestLog, RequestRecord, now_utc, prompt_sha256
from llm_gateway.routing import Router
from llm_gateway.verifier import Judge, Verifier
from llm_gateway.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    FailoverAttempt,
    GatewayInfo,
    RoutingInfo,
    Usage,
)

AUTO_MODEL = "auto"
# Errors that send an "auto" request on to the next model. A bad request would fail the same way
# anywhere, so it is returned at once.
FAILOVER_KINDS = ("rate_limit", "timeout", "connection", "server_error", "auth")

REQUIRED_HEADERS = {"X-Tenant-Id": "tenant", "X-Feature": "feature", "X-Request-Id": "request_id"}
# Tenant and feature are metric labels, so they are kept short and free of separators.
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


def _validation_message(exc: ValidationError) -> tuple[str, str | None]:
    first = exc.errors()[0]
    param = ".".join(str(part) for part in first["loc"]) or None
    return f"{param}: {first['msg']}" if param else first["msg"], param


def create_app(registry: dict[str, ModelConfig], providers: dict[str, Provider],
               request_log: RequestLog | None = None, metrics: GatewayMetrics | None = None,
               router: Router | None = None, verifier_judge: Judge | None = None,
               verifier_rng: random.Random | None = None, health: HealthTracker | None = None,
               breaker: CircuitBreaker | None = None) -> FastAPI:
    request_log = request_log or RequestLog(":memory:")
    metrics = metrics or GatewayMetrics()
    verifier = (Verifier(router, registry, providers, verifier_judge, request_log, metrics, verifier_rng)
                if router is not None and verifier_judge is not None else None)
    if health is not None and health.metrics is None:
        health.metrics = metrics
    if breaker is not None and breaker.metrics is None:
        breaker.metrics = metrics
    provider_names = sorted({m.provider for m in registry.values()})
    app = FastAPI(title="llm-gateway")
    app.state.request_log, app.state.metrics, app.state.verifier = request_log, metrics, verifier
    app.state.health = health

    async def refresh_health_gauges() -> None:
        if health is None:
            return
        try:
            for provider in provider_names:
                metrics.observe_health(await health.snapshot(provider))
                if breaker is not None:
                    state, _ = await breaker.state(provider)
                    metrics.breaker_state.labels(provider).set(BREAKER_STATES.index(state))
        except (RedisError, OSError):
            pass  # the scrape still returns the local metrics; the shared view is just stale

    @app.get("/v1/health")
    async def provider_health():
        if health is None:
            return error(503, "health_disabled", "This gateway has no health store configured.")
        try:
            snapshots = [await health.snapshot(p) for p in provider_names]
            states = {p: (await breaker.state(p))[0] for p in provider_names} if breaker else {}
        except (RedisError, OSError):
            return error(503, "health_store_unavailable", "The shared health store cannot be reached.")
        return {"window_seconds": snapshots[0].window_seconds if snapshots else None,
                "providers": {s.provider: s.as_dict() | ({"circuit": states[s.provider]} if states else {})
                              for s in snapshots}}

    @app.get("/v1/models")
    async def list_models() -> dict:
        return {"object": "list", "data": [
            {"id": m.id, "object": "model", "owned_by": m.provider, "tier": m.tier,
             "input_per_mtok": m.input_per_mtok, "output_per_mtok": m.output_per_mtok,
             "price_checked": m.price_checked.isoformat()}
            for m in registry.values()]}

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        await refresh_health_gauges()
        return Response(metrics.render(), media_type=metrics.content_type)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        record = RequestRecord(received_at=now_utc())
        response = await handle(request, record)
        record.http_status = response.status_code
        request_log.write(record)
        metrics.observe(record)
        return response

    async def handle(request: Request, record: RequestRecord) -> JSONResponse:
        def reject(status: int, code: str, message: str, param: str | None = None) -> JSONResponse:
            record.outcome, record.error_code = "rejected", code
            return error(status, code, message, param)

        problem = None
        for header, key in REQUIRED_HEADERS.items():
            value = request.headers.get(header)
            if value and IDENTIFIER.match(value):
                setattr(record, key, value)
            elif problem is None and not value:
                problem = ("missing_metadata", f"Header {header} is required on every request.", header)
            elif problem is None:
                problem = ("invalid_metadata", f"Header {header} must be 1 to 64 characters from A-Z, a-z, "
                                               "0-9, '.', '_', ':' or '-'.", header)
        if problem:
            return reject(400, *problem)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return reject(400, "invalid_json", "The request body is not valid JSON.")
        try:
            chat = ChatCompletionRequest.model_validate(body)
        except ValidationError as exc:
            message, param = _validation_message(exc)
            return reject(400, "invalid_request", message, param)
        messages = [m.model_dump() for m in chat.messages]
        record.prompt_sha256 = prompt_sha256(messages)
        record.requested_model = chat.model
        routing = None
        if chat.model == AUTO_MODEL:
            if router is None:
                return reject(400, "routing_disabled", "This gateway has no routing config; name a model.", "model")
            model_id, decision, config = router.route(record.feature, messages)
            record.routed_tier, record.routing_policy, record.routing_version = (
                decision.tier, config.policy_name, config.version)
            routing = RoutingInfo(tier=decision.tier, policy=config.policy_name, reason=decision.reason,
                                  config_version=config.version)
            model = registry[model_id]
        else:
            model = registry.get(chat.model)
        if model is None:
            return reject(404, "model_not_found",
                          f"Model {chat.model!r} is not in the registry. GET /v1/models lists the options, "
                          f"or send {AUTO_MODEL!r} to let the gateway choose.", "model")
        record.model, record.provider, record.tier = model.id, model.provider, model.tier
        if chat.stream:
            return reject(400, "stream_not_supported", "This gateway does not stream; send stream=false.", "stream")
        if chat.tools:
            return reject(400, "tools_not_supported", "This gateway does not forward tools.", "tools")

        # The tier's model first, then its fallbacks: only for "auto" requests, where the gateway chose
        # the model in the first place.
        candidates = [model]
        failover = router.config.failover if routing is not None else None
        if failover is not None:
            candidates += [registry[m] for m in failover.fallbacks.get(routing.tier, ())]
        deadline = time.perf_counter() + (failover.deadline_seconds if failover else math.inf)
        attempts, result = [], None
        for candidate in candidates:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                attempts.append(FailoverAttempt(model=candidate.id, outcome="deadline"))
                break
            admission = await breaker.allow(candidate.provider) if breaker is not None else None
            if admission is not None and not admission.allowed:
                attempts.append(FailoverAttempt(model=candidate.id, outcome="circuit_open"))
                continue
            started = time.perf_counter()
            try:
                call = providers[candidate.provider].complete(candidate, chat)
                result = await (asyncio.wait_for(call, remaining) if math.isfinite(remaining) else call)
            except (ProviderError, asyncio.TimeoutError) as exc:
                kind = exc.kind if isinstance(exc, ProviderError) else "timeout"
                if health is not None:
                    await health.record(candidate.provider, kind)
                if admission is not None:
                    await breaker.on_result(candidate.provider, kind, admission)
                attempts.append(FailoverAttempt(model=candidate.id, outcome=kind))
                if kind not in FAILOVER_KINDS:
                    break
                continue
            record.latency_ms = (time.perf_counter() - started) * 1000
            if health is not None:
                await health.record(candidate.provider, "ok", record.latency_ms / 1000)
            if admission is not None:
                await breaker.on_result(candidate.provider, "ok", admission)
            attempts.append(FailoverAttempt(model=candidate.id, outcome="ok"))
            model = candidate
            break
        record.attempts = len(attempts)
        record.model, record.provider = model.id, model.provider
        if result is None:
            failed = [a for a in attempts if a.outcome not in ("circuit_open", "deadline")]
            if not failed and attempts[-1].outcome == "circuit_open":
                record.outcome, record.error_code = "upstream_error", "upstream_circuit_open"
                return error(503, record.error_code, "Every provider for this request has an open circuit; "
                                                     "the gateway is not sending them traffic.", "model")
            kind = "timeout" if attempts[-1].outcome == "deadline" else failed[-1].outcome
            status, hint = UPSTREAM_STATUS[kind]
            tried = ", ".join(f"{a.model}: {a.outcome}" for a in attempts)
            record.outcome, record.error_code = "upstream_error", f"upstream_{kind}"
            return error(status, record.error_code, f"No model could answer ({tried}): {hint}", "model")
        if model.id != candidates[0].id:
            record.failover_from = candidates[0].id
            metrics.failovers.labels(candidates[0].id, model.id, attempts[0].outcome).inc()
        record.outcome, record.finish_reason = "ok", result.finish_reason
        record.input_tokens, record.output_tokens = result.input_tokens, result.output_tokens
        record.cost_usd = cost_usd(model, result.input_tokens, result.output_tokens)

        response = ChatCompletionResponse(
            id=f"chatcmpl-{record.request_id}",
            created=int(time.time()),
            model=model.id,
            choices=[Choice(message=AssistantMessage(content=result.text), finish_reason=result.finish_reason)],
            usage=Usage(prompt_tokens=result.input_tokens, completion_tokens=result.output_tokens,
                        total_tokens=result.input_tokens + result.output_tokens),
            gateway=GatewayInfo(request_id=record.request_id, tenant=record.tenant, feature=record.feature,
                                provider=model.provider, cost_usd=record.cost_usd, latency_ms=record.latency_ms,
                                routing=routing, attempts=attempts if len(attempts) > 1 else None),
        )
        # The verification, when sampled, runs after the response is sent; the client never waits for it.
        task = verifier.maybe_schedule(record, chat, result) if verifier else None
        return JSONResponse(response.model_dump(), headers={"X-Request-Id": record.request_id}, background=task)

    return app
