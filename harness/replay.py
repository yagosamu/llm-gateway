"""Replay the traffic through the gateway with recorded answers in place of the providers.

Offline and free: the ReplayProvider looks each call up in the recording by (prompt hash, model,
output limit, reasoning effort) and returns what the provider answered when it was recorded. A call
the recording does not hold raises ReplayMiss instead of reaching a provider, the same way the
Groundtruth embedding cache refused a miss. Because the provider step takes no time, the wall time
of each request is the gateway's own overhead, measured in process without an HTTP server.

Usage: uv run python -m harness.replay"""
import asyncio
import time

import httpx2

from harness.record import OUTPUT_LIMIT, RECORDINGS_PATH, load_records, load_rows
from llm_gateway.api import create_app
from llm_gateway.providers.base import ProviderResult, output_limit
from llm_gateway.registry import ModelConfig, load_registry
from llm_gateway.request_log import RequestLog, prompt_sha256
from llm_gateway.schemas import ChatCompletionRequest


class ReplayMiss(RuntimeError):
    pass


def current_records(records: list[dict], registry: dict[str, ModelConfig]) -> dict[tuple[str, str], dict]:
    """(prompt_id, model) -> the last successful record made with the model's current parameters.
    Records made with other parameters (an older output limit or effort) are ignored."""
    latest = {}
    for r in records:
        model = registry.get(r["model"])
        if r["status"] != "ok" or model is None:
            continue
        if r["params"] == {"reasoning_effort": model.reasoning_effort, "max_completion_tokens": OUTPUT_LIMIT}:
            latest[(r["prompt_id"], r["model"])] = r
    return latest


def require_complete(matrix: dict, rows: list[dict], registry: dict[str, ModelConfig]) -> None:
    missing = [(row["id"], m) for row in rows for m in registry if (row["id"], m) not in matrix]
    if missing:
        raise ReplayMiss(f"the recording lacks {len(missing)} prompt-model pairs, first {missing[:3]}; "
                         "run python -m harness.record to fill them")


class ReplayProvider:
    def __init__(self, matrix: dict[tuple[str, str], dict], rows: list[dict]):
        self.prompt_ids = {prompt_sha256(row["messages"]): row["id"] for row in rows}
        self.matrix = matrix

    async def complete(self, model: ModelConfig, request: ChatCompletionRequest) -> ProviderResult:
        prompt_id = self.prompt_ids.get(prompt_sha256([m.model_dump() for m in request.messages]))
        record = self.matrix.get((prompt_id, model.id))
        if record is None or output_limit(request) != record["params"]["max_completion_tokens"]:
            raise ReplayMiss(f"no recorded answer for prompt {prompt_id} on {model.id} "
                             f"with output limit {output_limit(request)}")
        return ProviderResult(text=record["text"], input_tokens=record["input_tokens"],
                              output_tokens=record["output_tokens"], finish_reason=record["finish_reason"])


async def replay(rows: list[dict], matrix: dict, registry: dict[str, ModelConfig]) -> tuple[list[dict], list[float]]:
    """Send every prompt to every model through the gateway. Returns the gateway's request log rows
    and the in-process wall time of each request in milliseconds."""
    log = RequestLog(":memory:")
    app = create_app(registry, {p: ReplayProvider(matrix, rows) for p in ("openai", "anthropic", "groq")}, log)
    overhead_ms = []
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://gateway") as client:
        for row in rows:
            for model in registry.values():
                started = time.perf_counter()
                response = await client.post("/v1/chat/completions", headers={
                    "X-Tenant-Id": row["tenant"], "X-Feature": row["category"],
                    "X-Request-Id": f"{row['id']}:{model.id}",
                }, json={"model": model.id, "messages": row["messages"], "max_completion_tokens": OUTPUT_LIMIT})
                overhead_ms.append((time.perf_counter() - started) * 1000)
                response.raise_for_status()
    return log.rows(), overhead_ms


def run_replay() -> tuple[list[dict], dict, list[dict], list[float]]:
    """Load the committed traffic and recording, check the recording is complete, replay it.
    Returns (rows, matrix, gateway log rows, overhead in ms)."""
    registry, rows = load_registry(), load_rows()
    matrix = current_records(load_records(RECORDINGS_PATH), registry)
    require_complete(matrix, rows, registry)
    log_rows, overhead = asyncio.run(replay(rows, matrix, registry))
    return rows, matrix, log_rows, overhead
