"""Pre-registered comparison of breaker v2 (count_window) against v1 (time_window), fixed before any v2
run. v2 was designed after seeing the v1 results, so this study is post hoc with respect to the slice 3
measurement; what is fixed here, before its own runs, is how v2 will be judged.

Same traffic, instances, simulated providers, timing and metrics as harness/chaos/prereg.py. Both
configurations fail over (low -> claude-haiku-4-5) and differ only in breaker.mode; every threshold
is the same (error_rate_threshold 0.5, min_calls 10, 30 s for slowness, open_seconds 15), so any
difference comes from the mechanism, not from a more sensitive setting.

Scenarios: the four slice 3 faults, plus two controls the v1 study did not have, because a breaker
that detects faster but also opens on noise would be worse, not better:
- healthy: no fault at all;
- mild: 10% of openai calls fail with a server error during the fault window, a degradation the
  0.5 threshold is meant to tolerate.

Runs are interleaved (repetition, then scenario, then configuration), so both versions see the same
machine conditions, and v1 is run again here rather than taken from the slice 3 study.

v2 counts as better only if every criterion holds:
C1 faster detection: v2's median time to open is lower than v1's in outage, timeouts and latency.
C2 partial outage: v2 opens in at least 2 of the 3 repetitions of partial.
C3 clean recovery: in every repetition of every fault scenario where v2 opened, it closes after the
   fault ends and does not open again afterwards.
C4 no false trips: v2 never opens, at any time, in any repetition of healthy or mild.
C5 no regression for clients, in every scenario, comparing medians: client error rate in the fault
   window at most v1's plus 0.5 points; p95 latency at most v1's times 1.05 plus 0.1 s; wasted calls
   at most v1's; client error rate after recovery at most v1's plus 0.5 points."""

from harness.chaos import prereg

SCENARIOS = dict(prereg.SCENARIOS) | {"healthy": None, "mild": {"error_rate": 0.1, "kind": "server_error"}}
CONFIGS = ("breaker_v1", "breaker_v2")
MODES = {"breaker_v1": "time_window", "breaker_v2": "count_window"}
REPS = 3
DETECTION_SCENARIOS = ("outage", "timeouts", "latency")
CONTROL_SCENARIOS = ("healthy", "mild")
FAULT_SCENARIOS = tuple(prereg.SCENARIOS)
MIN_PARTIAL_OPENS = 2
ERROR_RATE_SLACK = 0.005
P95_RATIO, P95_SLACK_S = 1.05, 0.1
