# Slice 3: deferrable work through a double outage

openai and anthropic both fail every call from 20 s to 60 s, so the low tier and its fallback are down together. Deferrable jobs arrive at 2 per second for 80 s, each submitted twice with the same Idempotency-Key; one of the two workers is killed at 65 s. Design: harness/chaos/deferrable.py.

## run1_sequential_worker

Pre-registered run. Workers processed one job at a time, about 0.4 jobs per second each against 2 arriving per second, so the stream backed up and jobs were still unread when the run stopped at 300 s; the queue lost nothing but had no throughput. Jobs also died after 8 attempts inside the outage, because an open circuit fails an attempt at once.

| rep | submissions | deduplicated | jobs | done | failed | unfinished | jobs answered more than once | attempts median | attempts max | drained after fault s | interactive error rate in fault |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rep1 | 320 | 160 | 160 | 79 | 23 | 58 | 0 | 6.0 | 8 | - | 1.0 |
| rep2 | 320 | 160 | 160 | 80 | 26 | 54 | 0 | 6.0 | 8 | - | 1.0 |
| rep3 | 320 | 160 | 160 | 74 | 16 | 70 | 0 | 6.0 | 8 | - | 1.0 |

Jobs answered more than once count jobs whose request id shows two or more successful model calls in the gateway's request log, the at-least-once duplicate window. Drained after fault: seconds from the fault ending to the last job finishing, reported only when no job was left unfinished.
