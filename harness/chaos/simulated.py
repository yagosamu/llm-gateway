"""A provider that answers from the slice 1 recording, taking as long as the real call took.

Chaos runs need many calls and must cost nothing, so they run against this instead of the real
providers: for each request it finds the recorded answer by prompt hash and model and sleeps the
latency recorded for that call before returning it. Latency can be scaled (0 for unit tests).
Unlike harness.replay.ReplayProvider, which measures the gateway's own overhead, this one keeps the
providers' real latency distribution, which is what the breaker's p95 trigger and the failover
timings depend on."""
import asyncio

from harness.record import RECORDINGS_PATH, load_records, load_rows
from harness.replay import ReplayMiss, current_records
from llm_gateway.providers.base import ProviderResult
from llm_gateway.registry import load_registry
from llm_gateway.request_log import prompt_sha256


class SimulatedProvider:
    def __init__(self, matrix: dict, rows: list[dict], latency_scale: float = 1.0):
        self.matrix, self.latency_scale = matrix, latency_scale
        self.prompt_ids = {prompt_sha256(row["messages"]): row["id"] for row in rows}

    async def complete(self, model, request) -> ProviderResult:
        prompt_id = self.prompt_ids.get(prompt_sha256([m.model_dump() for m in request.messages]))
        record = self.matrix.get((prompt_id, model.id))
        if record is None:
            raise ReplayMiss(f"no recorded answer for prompt {prompt_id} on {model.id}")
        if self.latency_scale:
            await asyncio.sleep(record["latency_ms"] / 1000 * self.latency_scale)
        return ProviderResult(text=record["text"], input_tokens=record["input_tokens"],
                              output_tokens=record["output_tokens"], finish_reason=record["finish_reason"])


def from_recording(latency_scale: float = 1.0) -> SimulatedProvider:
    return SimulatedProvider(current_records(load_records(RECORDINGS_PATH), load_registry()), load_rows(), latency_scale)
