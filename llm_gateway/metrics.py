"""Prometheus metrics for /metrics.

Each app gets its own CollectorRegistry, so two apps in one process (tests, the replay harness)
never share counters. Label values are bounded on purpose: tenant and feature are validated short
identifiers, model is a registry id or "unknown", and the request id is never a label, because one
label value per request would grow the series without limit."""
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

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
        self.routed = Counter("llm_gateway_routed", "Requests sent with model auto, by chosen tier.",
                              ["feature", "tier", "policy"], registry=self.registry)
        self.verifications = Counter("llm_gateway_verifications", "Sampled verifications of routed answers.",
                                     ["feature", "tier", "outcome", "acceptable"], registry=self.registry)
        self.verification_cost = Counter("llm_gateway_verification_cost_usd", "Spent on verification.",
                                         registry=self.registry)
        # The shared sliding-window view kept in Redis (llm_gateway.health), refreshed at each scrape.
        # Every instance reports the same values; dashboards should take one, not sum them.
        self.provider_success_rate = Gauge("llm_gateway_provider_success_rate",
                                           "Share of provider calls that succeeded in the shared window.",
                                           ["provider"], registry=self.registry)
        self.provider_calls_window = Gauge("llm_gateway_provider_window_calls",
                                           "Provider calls in the shared window, by outcome.",
                                           ["provider", "outcome"], registry=self.registry)
        self.provider_latency_quantile = Gauge("llm_gateway_provider_window_latency_ms",
                                               "Latency quantiles of successful calls in the shared window.",
                                               ["provider", "quantile"], registry=self.registry)
        self.health_write_errors = Counter("llm_gateway_health_write_errors",
                                           "Health writes lost because Redis was unreachable.",
                                           registry=self.registry)

    def observe_health(self, snapshot) -> None:
        if snapshot.success_rate is not None:
            self.provider_success_rate.labels(snapshot.provider).set(snapshot.success_rate)
        for outcome, count in snapshot.outcomes.items():
            self.provider_calls_window.labels(snapshot.provider, outcome).set(count)
        for name, value in (("0.5", snapshot.p50_ms), ("0.95", snapshot.p95_ms), ("0.99", snapshot.p99_ms)):
            if value is not None:
                self.provider_latency_quantile.labels(snapshot.provider, name).set(value)

    def observe_verification(self, row) -> None:
        acceptable = "unknown" if row.acceptable is None else str(bool(row.acceptable)).lower()
        self.verifications.labels(row.feature or UNKNOWN, row.routed_tier or UNKNOWN, row.outcome, acceptable).inc()
        self.verification_cost.inc(row.cost_usd)

    def observe(self, record: RequestRecord) -> None:
        tenant, feature, model = record.tenant or UNKNOWN, record.feature or UNKNOWN, record.model or UNKNOWN
        self.requests.labels(tenant, feature, model, record.outcome, record.error_code or "none").inc()
        if record.routed_tier:
            self.routed.labels(feature, record.routed_tier, record.routing_policy).inc()
        if record.outcome != "ok":
            return
        self.latency.labels(record.provider, model).observe(record.latency_ms / 1000)
        self.tokens.labels(tenant, feature, model, "input").inc(record.input_tokens)
        self.tokens.labels(tenant, feature, model, "output").inc(record.output_tokens)
        self.cost.labels(tenant, feature, model).inc(record.cost_usd)

    def render(self) -> bytes:
        return generate_latest(self.registry)
