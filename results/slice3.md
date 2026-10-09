# Slice 3 results: provider faults under load

Open-loop traffic at 4 requests per second through two gateway instances sharing Redis, over simulated providers at recorded latency. A fault hits openai (gpt-6-luna's provider) from 20 s to 60 s of a 100 s run. Each cell is the median of the repetitions, with min to max in brackets when they differ. Design: harness/chaos/prereg.py.

## outage: {"error_rate": 1.0, "kind": "server_error"}

| configuration | client errors | p50 latency | p95 latency | wasted calls | served by fallback | circuit opened after | closed after recovery | reopened after closing | errors after recovery |
|---|---|---|---|---|---|---|---|---|---|
| none | 100.0% | - | - | 160 (159 to 161) | - | never | never | - | 0.6% (0.0 to 0.6) |
| failover | 0.0% | 2.9 s (2.9 to 2.9) | 4.9 s (4.9 to 4.9) | 161 (160 to 161) | 100% | never | never | - | 0.0% |
| failover_breaker | 0.0% | 2.9 s (2.9 to 2.9) | 4.9 s (4.9 to 4.9) | 80 (80 to 81) | 100% | 20.2 s (19.9 to 20.3) | 13.4 s (11.7 to 13.7) | 0 | 0.0% |

## partial: {"error_rate": 0.5, "kind": "server_error"}

| configuration | client errors | p50 latency | p95 latency | wasted calls | served by fallback | circuit opened after | closed after recovery | reopened after closing | errors after recovery |
|---|---|---|---|---|---|---|---|---|---|
| none | 48.8% (48.4 to 50.6) | 2.4 s (2.2 to 2.5) | 6.2 s (5.5 to 7.6) | 78 (77 to 81) | 0% | never | never | - | 0.0% |
| failover | 0.0% (0.0 to 0.6) | 2.5 s (2.4 to 2.5) | 5.2 s (5.1 to 5.3) | 80 (76 to 85) | 50% (48 to 54) | never | never | - | 0.0% |
| failover_breaker | 0.0% | 2.4 s (2.4 to 2.5) | 5.1 s (5.1 to 5.4) | 82 (80 to 85) | 51% (50 to 53) | never | never | - | 0.0% |

## timeouts: {"error_rate": 1.0, "kind": "timeout", "timeout_seconds": 10.0}

| configuration | client errors | p50 latency | p95 latency | wasted calls | served by fallback | circuit opened after | closed after recovery | reopened after closing | errors after recovery |
|---|---|---|---|---|---|---|---|---|---|
| none | 100.0% (99.4 to 100.0) | 3.7 s, never in 2 | 3.7 s, never in 2 | 160 (160 to 161) | 0%, never in 2 | never | never | - | 0.0% |
| failover | 0.0% (0.0 to 0.6) | 12.9 s (12.9 to 12.9) | 14.9 s (14.9 to 14.9) | 160 (159 to 161) | 100% (99 to 100) | never | never | - | 0.0% |
| failover_breaker | 0.0% | 11.7 s (11.7 to 11.8) | 14.8 s (14.8 to 14.8) | 120 (119 to 120) | 100% | 30.2 s (29.9 to 30.2) | 8.8 s (8.5 to 8.9) | 0 | 0.0% |

## latency: {"error_rate": 0.0, "extra_latency_ms": 35000.0}

| configuration | client errors | p50 latency | p95 latency | wasted calls | served by fallback | circuit opened after | closed after recovery | reopened after closing | errors after recovery |
|---|---|---|---|---|---|---|---|---|---|
| none | 0.0% | 37.4 s (37.3 to 37.4) | 41.0 s (41.0 to 41.0) | 0 | 0% | never | never | - | 0.0% |
| failover | 0.0% | 37.3 s (37.3 to 37.3) | 41.0 s (41.0 to 41.0) | 0 | 0% | never | never | - | 0.0% |
| failover_breaker | 0.0% | 37.3 s (37.3 to 37.3) | 41.0 s (41.0 to 41.1) | 0 | 7% (7 to 8) | 37.5 s (37.5 to 37.6) | 14.5 s (14.5 to 14.5) | 1 | 0.0% |

Wasted calls: requests in the fault window that made a failing call to openai. Circuit times come from openai's state polled every 0.5 s, so they carry that resolution. Configurations without a breaker never open a circuit. "Reopened after closing" was added after the runs and is not part of the pre-registered metrics.

## Measurement notes

- The latency scenario was run a second time. In the first run the load generator's HTTP client kept its default pool of 100 connections, and with calls taking about 37 s up to 155 requests were in flight, so some waited in the client before reaching the gateway and the loop was no longer open. The pool limit was removed (harness/chaos/load.py) and only that scenario was repeated, with the same design. The other scenarios peaked at 61 requests in flight and were not affected.
