"""Prometheus metrics for /metrics.

Each app gets its own CollectorRegistry, so two apps in one process (tests, the replay harness)
never share counters. Label values are bounded on purpose: tenant and feature are validated short
identifiers, model is a registry id or "unknown", and the request id is never a label, because one
label value per request would grow the series without limit."""
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

from llm_gateway.request_log import RequestRecord

UNKNOWN = "unknown"
# LLM calls take from a fraction of a second to a minute; the default buckets stop at 10 s.
LATENCY_BUCKETS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


class GatewayMetrics:
    content_type = CONTENT_TYPE_LATEST

    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter("llm_gateway_requests", "Requests by outcome.",
                                ["tenant", "feature", "model", "outcome", "code"], registry=self.registry)
        self.latency = Histogram("llm_gateway_provider_latency_seconds", "Provider call latency of answered requests.",
                                 ["provider", "model"], buckets=LATENCY_BUCKETS, registry=self.registry)
        self.tokens = Counter("llm_gateway_tokens", "Tokens billed, by direction.",
                              ["tenant", "feature", "model", "direction"], registry=self.registry)
        self.cost = Counter("llm_gateway_cost_usd", "Cost at registry prices.",
                            ["tenant", "feature", "model"], registry=self.registry)

    def observe(self, record: RequestRecord) -> None:
        tenant, feature, model = record.tenant or UNKNOWN, record.feature or UNKNOWN, record.model or UNKNOWN
        self.requests.labels(tenant, feature, model, record.outcome, record.error_code or "none").inc()
        if record.outcome != "ok":
            return
        self.latency.labels(record.provider, model).observe(record.latency_ms / 1000)
        self.tokens.labels(tenant, feature, model, "input").inc(record.input_tokens)
        self.tokens.labels(tenant, feature, model, "output").inc(record.output_tokens)
        self.cost.labels(tenant, feature, model).inc(record.cost_usd)

    def render(self) -> bytes:
        return generate_latest(self.registry)
