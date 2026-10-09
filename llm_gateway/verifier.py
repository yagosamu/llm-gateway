"""Sampled, budgeted verification of routed answers, after the response has been sent.

For a sampled request that the router sent below the high tier, the verifier asks the high tier's
model the same request, has a judge decide whether the routed answer was an acceptable substitute,
and writes one row to the verifications table. An answer cut by the output limit or empty is a
failure by rule, with no extra call. Rows with acceptable = 0 are routing failures; they carry the
prompt's routing features, never its text, so they can be added to the classifier's training data.

Verification sends traffic to the most expensive model, so it is bounded twice: by the sample rate
and by a daily cap. Before a verification starts, the day's spend so far (read from the table, so it
survives a restart) plus what in-flight verifications have reserved plus this one's worst case must
stay under the cap; otherwise the request is not verified. Both numbers live in routing.yaml.

The worst case bounds the reference call by its output limit and its input bytes, as the gateway
does everywhere. The judge's input includes the reference answer, which does not exist yet; it is
bounded by two judge tokens per reference output token, a margin rather than a proof."""
import json
import random
import threading
from typing import Protocol

from starlette.background import BackgroundTask

from llm_gateway.providers.base import Provider, ProviderError, ProviderResult, output_limit
from llm_gateway.registry import ModelConfig, call_ceiling_usd, cost_usd, input_token_bound
from llm_gateway.request_log import RequestLog, RequestRecord, VerificationRecord, now_utc
from llm_gateway.routing import Router, prompt_features, user_text
from llm_gateway.schemas import ChatCompletionRequest

JUDGE_TOKENS_PER_REFERENCE_TOKEN = 2


class Judge(Protocol):
    model: ModelConfig
    rubric: str
    max_output_tokens: int

    async def grade(self, request: str, reference: str, candidate: str) -> tuple[bool, str, int, int]:
        """(acceptable, reason, input tokens, output tokens)."""
        ...


def _unusable(result: ProviderResult) -> str | None:
    if result.finish_reason == "length":
        return "cut by the output limit"
    if not result.text.strip():
        return "empty answer"
    return None


class Verifier:
    def __init__(self, router: Router, registry: dict[str, ModelConfig], providers: dict[str, Provider],
                 judge: Judge, request_log: RequestLog, metrics=None, rng: random.Random | None = None):
        self.router, self.registry, self.providers, self.judge = router, registry, providers, judge
        self.request_log, self.metrics = request_log, metrics
        self.rng = rng or random.Random()
        self._lock = threading.Lock()
        self._reserved = 0.0
        self.skipped_for_budget = 0

    def worst_case_usd(self, reference: ModelConfig, chat: ChatCompletionRequest, candidate_text: str) -> float:
        messages = [m.model_dump() for m in chat.messages]
        limit = output_limit(chat)
        judge_input = (input_token_bound([{"content": self.judge.rubric}, {"content": user_text(messages)},
                                          {"content": candidate_text}], self.judge.model.provider)
                       + JUDGE_TOKENS_PER_REFERENCE_TOKEN * limit)
        return (call_ceiling_usd(reference, messages, limit)
                + cost_usd(self.judge.model, judge_input, self.judge.max_output_tokens))

    def maybe_schedule(self, record: RequestRecord, chat: ChatCompletionRequest,
                       result: ProviderResult) -> BackgroundTask | None:
        """A background task that verifies this request, or None when it is not routed below high, is
        not drawn by the sample, or would push the day past the cap."""
        if record.routed_tier in (None, "high") or record.outcome != "ok":
            return None
        config = self.router.config
        if config.verifier.sample_rate <= 0 or self.rng.random() >= config.verifier.sample_rate:
            return None
        reference = self.registry[config.tier_map["high"]]
        ceiling = self.worst_case_usd(reference, chat, result.text)
        with self._lock:
            spent = self.request_log.verification_spend(now_utc()[:10])
            if spent + self._reserved + ceiling > config.verifier.daily_budget_usd:
                self.skipped_for_budget += 1
                return None
            self._reserved += ceiling
        return BackgroundTask(self.verify, record, chat, result, reference, ceiling)

    async def verify(self, record: RequestRecord, chat: ChatCompletionRequest, result: ProviderResult,
                     reference: ModelConfig, reserved: float) -> None:
        messages = [m.model_dump() for m in chat.messages]
        row = VerificationRecord(
            verified_at=now_utc(), request_id=record.request_id, feature=record.feature,
            routed_tier=record.routed_tier, routed_model=record.model, reference_model=reference.id,
            outcome="error", acceptable=None, reason="", cost_usd=0.0, prompt_sha256=record.prompt_sha256,
            prompt_features=json.dumps(prompt_features(messages, record.feature or "")))
        try:
            rule = _unusable(result)
            if rule:
                row.outcome, row.acceptable, row.reason = "rejected_by_rule", False, rule
                return
            answer = await self.providers[reference.provider].complete(reference, chat)
            row.cost_usd += cost_usd(reference, answer.input_tokens, answer.output_tokens)
            if _unusable(answer):
                row.outcome, row.reason = "no_reference", f"reference {_unusable(answer)}"
                return
            acceptable, reason, judge_in, judge_out = await self.judge.grade(user_text(messages), answer.text, result.text)
            row.cost_usd += cost_usd(self.judge.model, judge_in, judge_out)
            row.outcome, row.acceptable, row.reason = "judged", acceptable, reason
        except ProviderError as exc:
            row.reason = f"{exc.provider} {exc.kind}"
        except (ValueError, KeyError) as exc:  # a malformed verdict from the judge
            row.reason = f"invalid verdict: {type(exc).__name__}"
        finally:
            row.verified_at = now_utc()
            self.request_log.write_verification(row)
            with self._lock:
                self._reserved -= reserved
            if self.metrics is not None:
                self.metrics.observe_verification(row)
