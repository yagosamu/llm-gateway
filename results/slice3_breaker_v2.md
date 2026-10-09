# Breaker v2 against v1

v1 counts the shared 60 s health window; v2 counts the last 20 calls admitted since the circuit last moved, ignores calls admitted before that, and lists a call as slow while it runs. Same thresholds. v2 was designed after the slice 3 results; the criteria below were fixed in harness/chaos/prereg_v2.py before its runs, and both versions were run again, interleaved, in the same session.

**v2 is better: every pre-registered criterion holds.**

| criterion | scenario | check | v1 | v2 | holds |
|---|---|---|---|---|---|
| C1 | outage | median time to open lower than v1 | 20.218 | 2.709 | yes |
| C1 | timeouts | median time to open lower than v1 | 30.246 | 12.526 | yes |
| C1 | latency | median time to open lower than v1 | 37.522 | 32.51 | yes |
| C2 | partial | opens in at least 2 of the repetitions | 0 | 3 | yes |
| C3 | outage | closes after the fault and stays closed, in every repetition where it opened | [0, 0, 0] | [0, 0, 0] | yes |
| C3 | partial | closes after the fault and stays closed, in every repetition where it opened | [None, None, None] | [0, 0, 0] | yes |
| C3 | timeouts | closes after the fault and stays closed, in every repetition where it opened | [0, 0, 0] | [0, 0, 0] | yes |
| C3 | latency | closes after the fault and stays closed, in every repetition where it opened | [1, 1, 1] | [0, 0, 0] | yes |
| C4 | healthy | never opens | [0, 0, 0] | [0, 0, 0] | yes |
| C4 | mild | never opens | [0, 0, 0] | [0, 0, 0] | yes |
| C5 | outage | client errors at most v1 + 0.5 points | 0.0 | 0.0 | yes |
| C5 | outage | p95 at most v1 x 1.05 + 0.1 s | 4.89 | 4.903 | yes |
| C5 | outage | wasted calls at most v1 | 81.0 | 11.0 | yes |
| C5 | outage | errors after recovery at most v1 + 0.5 points | 0.0 | 0.0 | yes |
| C5 | partial | client errors at most v1 + 0.5 points | 0.0 | 0.0 | yes |
| C5 | partial | p95 at most v1 x 1.05 + 0.1 s | 5.508 | 4.917 | yes |
| C5 | partial | wasted calls at most v1 | 84.0 | 15.0 | yes |
| C5 | partial | errors after recovery at most v1 + 0.5 points | 0.0 | 0.0 | yes |
| C5 | timeouts | client errors at most v1 + 0.5 points | 0.0 | 0.0 | yes |
| C5 | timeouts | p95 at most v1 x 1.05 + 0.1 s | 14.751 | 14.552 | yes |
| C5 | timeouts | wasted calls at most v1 | 120.0 | 51.0 | yes |
| C5 | timeouts | errors after recovery at most v1 + 0.5 points | 0.0 | 0.0 | yes |
| C5 | latency | client errors at most v1 + 0.5 points | 0.0 | 0.0 | yes |
| C5 | latency | p95 at most v1 x 1.05 + 0.1 s | 41.005 | 41.024 | yes |
| C5 | latency | wasted calls at most v1 | 0.0 | 0.0 | yes |
| C5 | latency | errors after recovery at most v1 + 0.5 points | 0.0 | 0.0 | yes |
| C5 | healthy | client errors at most v1 + 0.5 points | 0.0 | 0.0 | yes |
| C5 | healthy | p95 at most v1 x 1.05 + 0.1 s | 6.022 | 6.032 | yes |
| C5 | healthy | wasted calls at most v1 | 0.0 | 0.0 | yes |
| C5 | healthy | errors after recovery at most v1 + 0.5 points | 0.0 | 0.0 | yes |
| C5 | mild | client errors at most v1 + 0.5 points | 0.0 | 0.0 | yes |
| C5 | mild | p95 at most v1 x 1.05 + 0.1 s | 5.54 | 5.565 | yes |
| C5 | mild | wasted calls at most v1 | 19.0 | 15.0 | yes |
| C5 | mild | errors after recovery at most v1 + 0.5 points | 0.0 | 0.0 | yes |

Times in seconds; error rates as fractions; lists are one value per repetition.

## Full measurements

## outage: {"error_rate": 1.0, "kind": "server_error"}

| configuration | client errors | p50 latency | p95 latency | wasted calls | served by fallback | circuit opened after | closed after recovery | reopened after closing | errors after recovery |
|---|---|---|---|---|---|---|---|---|---|
| breaker_v1 | 0.0% | 2.9 s (2.9 to 2.9) | 4.9 s (4.9 to 4.9) | 81 (80 to 81) | 100% | 20.2 s (20.1 to 20.2) | 13.5 s (11.6 to 13.5) | 0 | 0.0% |
| breaker_v2 | 0.0% | 2.9 s (2.9 to 2.9) | 4.9 s (4.9 to 4.9) | 11 (11 to 12) | 100% | 2.7 s (2.6 to 2.7) | 12.6 s (9.7 to 12.8) | 0 | 0.0% |

## partial: {"error_rate": 0.5, "kind": "server_error"}

| configuration | client errors | p50 latency | p95 latency | wasted calls | served by fallback | circuit opened after | closed after recovery | reopened after closing | errors after recovery |
|---|---|---|---|---|---|---|---|---|---|
| breaker_v1 | 0.0% | 2.5 s (2.5 to 2.7) | 5.5 s (5.3 to 6.0) | 84 (75 to 85) | 52% (47 to 53) | never | never | - | 0.0% |
| breaker_v2 | 0.0% | 2.9 s (2.8 to 2.9) | 4.9 s (4.8 to 5.0) | 15 (12 to 19) | 91% (90 to 94) | 5.2 s (3.8 to 5.5) | 15.5 s (12.2 to 16.3) | 0 | 0.0% |

## timeouts: {"error_rate": 1.0, "kind": "timeout", "timeout_seconds": 10.0}

| configuration | client errors | p50 latency | p95 latency | wasted calls | served by fallback | circuit opened after | closed after recovery | reopened after closing | errors after recovery |
|---|---|---|---|---|---|---|---|---|---|
| breaker_v1 | 0.0% (0.0 to 1.2) | 11.7 s (11.7 to 11.8) | 14.8 s (14.7 to 14.8) | 120 (119 to 120) | 100% | 30.2 s (30.0 to 30.3) | 8.7 s (8.6 to 9.3) | 0 | 0.0% |
| breaker_v2 | 0.0% | 3.5 s (3.4 to 3.5) | 14.6 s (14.5 to 14.6) | 51 (50 to 52) | 100% | 12.5 s (12.5 to 12.8) | 15.0 s (13.8 to 15.2) | 0 | 0.0% |

## latency: {"error_rate": 0.0, "extra_latency_ms": 35000.0}

| configuration | client errors | p50 latency | p95 latency | wasted calls | served by fallback | circuit opened after | closed after recovery | reopened after closing | errors after recovery |
|---|---|---|---|---|---|---|---|---|---|
| breaker_v1 | 0.0% | 37.3 s (37.3 to 37.3) | 41.0 s (41.0 to 41.0) | 0 | 7% (6 to 7) | 37.5 s (37.3 to 37.7) | 14.5 s (13.7 to 14.6) | 1 | 0.0% |
| breaker_v2 | 0.0% | 37.1 s (37.0 to 37.1) | 41.0 s (41.0 to 41.0) | 0 | 19% (19 to 20) | 32.5 s (32.3 to 32.6) | 9.8 s (9.6 to 10.1) | 0 | 0.0% |

## healthy: no fault injected

| configuration | client errors | p50 latency | p95 latency | wasted calls | served by fallback | circuit opened after | closed after recovery | reopened after closing | errors after recovery |
|---|---|---|---|---|---|---|---|---|---|
| breaker_v1 | 0.0% | 2.3 s (2.3 to 2.4) | 6.0 s (6.0 to 6.0) | 0 | 0% | never | never | - | 0.0% |
| breaker_v2 | 0.0% | 2.4 s (2.3 to 2.4) | 6.0 s (6.0 to 6.1) | 0 | 0% | never | never | - | 0.0% |

## mild: {"error_rate": 0.1, "kind": "server_error"}

| configuration | client errors | p50 latency | p95 latency | wasted calls | served by fallback | circuit opened after | closed after recovery | reopened after closing | errors after recovery |
|---|---|---|---|---|---|---|---|---|---|
| breaker_v1 | 0.0% | 2.4 s (2.4 to 2.4) | 5.5 s (5.5 to 5.6) | 19 (16 to 24) | 12% (10 to 15) | never | never | - | 0.0% |
| breaker_v2 | 0.0% | 2.4 s (2.3 to 2.4) | 5.6 s (5.5 to 5.6) | 15 (15 to 18) | 9% (9 to 11) | never | never | - | 0.0% |

Wasted calls: requests in the fault window that made a failing call to openai. Circuit times come from openai's state polled every 0.5 s, so they carry that resolution. Configurations without a breaker never open a circuit. "Reopened after closing" was added after the runs and is not part of the pre-registered metrics.

Openings over the whole run, per repetition: outage v1 [1, 1, 1] v2 [1, 2, 1]; partial v1 [0, 0, 0] v2 [3, 2, 1]; timeouts v1 [1, 1, 1] v2 [2, 2, 2]; latency v1 [2, 2, 2] v2 [1, 1, 1]; healthy v1 [0, 0, 0] v2 [0, 0, 0]; mild v1 [0, 0, 0] v2 [0, 0, 0].

## What the criteria did not cover: noise at moderate error rates

Added after the runs, not pre-registered. The mild control (10% errors) never tripped v2, but a 20-call window is a small sample: under a steady error rate well below the 50% threshold it can still see half its calls fail. Exact binomial probability that one evaluation trips, and the outage detection time at 4 requests per second:

| window, calls | outage detection | trip probability at 10% errors | trip probability at 20% errors | trip probability at 30% errors |
|---|---|---|---|---|
| 20 | 2.5 s | 7.2e-06 | 2.6e-03 | 4.8e-02 |
| 40 | 5.0 s | 1.9e-10 | 2.2e-05 | 6.3e-03 |
| 60 | 7.5 s | 5.6e-15 | 2.1e-07 | 9.1e-04 |

Every evaluation follows a call, so at 4 requests per second there are 14,400 an hour; overlapping windows are correlated, so trips per hour are lower than that product. A trip sends traffic to the fallback for the open period: clients see no error, but the fallback costs more.
