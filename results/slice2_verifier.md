# Slice 2: what the verifier costs and catches

Under `always_low` gpt-6-luna serves at US$ 0.089 per 1,000 requests, and 10 of 238 answers (4.2%) are failures by the two-judge consensus. One verification costs US$ 0.00233 on average (the reference answer plus a gpt-4.1-mini verdict), about 26 times what serving the request cost.

The verifier's single judge flags 7 of the 10 failures (sensitivity 70%). It cannot raise a false alarm against the consensus by construction, since the consensus rejects whatever either judge rejects, so no false-alarm rate is reported. Adding the evaluation's second judge to every check would cost 2.4 times as much per verification, from the recorded verdict costs.

| sample rate | checks per 1,000 | overhead per 1,000, US$ | overhead vs serving | total per 1,000, US$ | reduction vs always_high | failures caught per 1,000 | daily failure-rate precision |
|---|---|---|---|---|---|---|---|
| 1% | 10 | 0.023 | 26% | 0.112 | 94.5% | 0.3 | ± 12.4 pp |
| 5% | 50 | 0.116 | 131% | 0.206 | 90.0% | 1.5 | ± 5.6 pp |
| 10% | 100 | 0.233 | 261% | 0.322 | 84.3% | 2.9 | ± 3.9 pp |
| 25% | 250 | 0.582 | 653% | 0.671 | 67.2% | 7.4 | ± 2.5 pp |

## Reading it

- Verification goes to the most expensive model, so even a small sample rate costs more than serving: the price gap that makes always_low cheap also makes checking it expensive.
- At any rate it catches only that share of failures; the user has already received the answer. Sampled verification is a monitoring instrument: it measures the failure rate and collects routing failures as training examples, it does not fix individual answers.
- Daily failure-rate precision is the 95% half-width of the failure rate estimated from one day's checks at 1,000 requests a day, assuming the measured 4.2% rate.
- The daily cap in routing.yaml bounds the overhead whatever the rate: US$ 1 buys about 430 verifications.
- Descriptive, not pre-registered. Expectations over the recorded prompts, not a simulation.
