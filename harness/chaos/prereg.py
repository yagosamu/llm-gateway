"""Pre-registered design of the slice 3 chaos measurement, fixed before any run.

The question: when a provider fails, what do clients see, and what do failover and the circuit
breaker each change? Every run sends the same open-loop traffic (RATE requests per second, model
"auto", always_low, so the low tier's gpt-6-luna on openai serves everything until it fails) through
two gateway instances sharing one Redis, over simulated providers that answer from the recording at
the recorded latency. A fault is injected into openai at FAULT_START and removed at FAULT_END.

Configurations differ only in the routing file the instances read:
- none: no fallbacks, breaker disabled; the client sees whatever openai does.
- failover: the committed fallbacks (low -> claude-haiku-4-5), breaker disabled; every request still
  tries openai first and fails over after its error.
- failover_breaker: fallbacks and the committed breaker; once the circuit opens, requests skip openai.

Metrics, per run, over requests SENT in the fault window [FAULT_START, FAULT_END):
- client error rate: share of requests answered with a non-200 status;
- client latency p50 and p95 of answered requests (finish minus send);
- wasted calls: requests whose first attempt went to openai and failed;
- served by fallback: share of answered requests served by another model.
And from the circuit timeline (openai's state polled every TIMELINE_INTERVAL seconds):
- time to open: from FAULT_START to the first open state;
- time to close: from FAULT_END to the first closed state after it had opened.
Over requests sent after FAULT_END: the client error rate.
Each scenario-configuration pair runs REPS times; results are reported as median, min and max."""

RATE = 4.0
WARMUP_SECONDS = 20.0
FAULT_SECONDS = 40.0
RECOVERY_SECONDS = 40.0
FAULT_START = WARMUP_SECONDS
FAULT_END = WARMUP_SECONDS + FAULT_SECONDS
DURATION = WARMUP_SECONDS + FAULT_SECONDS + RECOVERY_SECONDS
REPS = 3
TIMELINE_INTERVAL = 0.5
FAULTY_PROVIDER = "openai"
PRIMARY_MODEL = "gpt-6-luna"
PORTS = (8000, 8001)

# Faults injected into openai during [FAULT_START, FAULT_END). The latency spike sits above the
# breaker's 30 s p95 budget; the timeout hangs a call for 10 s before it fails.
SCENARIOS = {
    "outage": {"error_rate": 1.0, "kind": "server_error"},
    "partial": {"error_rate": 0.5, "kind": "server_error"},
    "timeouts": {"error_rate": 1.0, "kind": "timeout", "timeout_seconds": 10.0},
    "latency": {"error_rate": 0.0, "extra_latency_ms": 35_000.0},
}
CONFIGS = ("none", "failover", "failover_breaker")
