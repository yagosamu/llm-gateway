# Slice 1 results

The same 240 prompts cost US$ 0.09 per 1,000 requests on gpt-6-luna and US$ 4.21 on claude-sonnet-5-5, 46 times more. Whether the cheaper answer is good enough is slice 2's question.

## Per model

95% bootstrap intervals over prompts (10,000 resamples, seed 20261007). Cut: answers stopped by the 1024-token output limit.

| model | tier | cost per 1,000 requests, US$ | mean output tokens | cut | p50 latency, s | p95 latency, s |
|---|---|---|---|---|---|---|
| gpt-6.1-sol | high | 2.12 [1.88, 2.36] | 185 | 3 of 240 | 4.7 [4.1, 5.4] | 16.5 [13.8, 20.1] |
| claude-sonnet-5-5 | high | 4.21 [3.85, 4.57] | 381 | 7 of 240 | 3.4 [3.0, 3.8] | 10.1 [8.5, 11.3] |
| claude-haiku-4-5 | medium | 1.13 [1.05, 1.21] | 197 | 0 of 240 | 2.3 [1.8, 2.9] | 4.7 [4.5, 5.2] |
| gpt-oss-120b | medium | 0.32 [0.29, 0.35] | 483 | 62 of 240 | 1.3 [1.1, 1.6] | 2.9 [2.7, 3.2] |
| gpt-6-luna | low | 0.09 [0.08, 0.10] | 158 | 2 of 240 | 2.3 [2.1, 2.5] | 6.1 [5.2, 8.7] |
| gpt-oss-20b | low | 0.12 [0.11, 0.13] | 357 | 18 of 240 | 0.6 [0.6, 0.7] | 1.7 [1.5, 1.9] |

## Cost per 1,000 requests by request class, US$

| model | brainstorming | classification | closed_qa | creative_writing | general_qa | information_extraction | open_qa | summarization |
|---|---|---|---|---|---|---|---|---|
| gpt-6.1-sol | 3.33 | 0.87 | 1.12 | 3.71 | 3.17 | 1.38 | 1.47 | 1.89 |
| claude-sonnet-5-5 | 5.67 | 2.26 | 2.30 | 7.17 | 6.56 | 2.52 | 3.67 | 3.52 |
| claude-haiku-4-5 | 1.35 | 0.51 | 0.92 | 1.73 | 1.46 | 0.90 | 0.92 | 1.22 |
| gpt-oss-120b | 0.42 | 0.10 | 0.23 | 0.47 | 0.55 | 0.19 | 0.27 | 0.32 |
| gpt-6-luna | 0.14 | 0.05 | 0.05 | 0.14 | 0.12 | 0.07 | 0.07 | 0.09 |
| gpt-oss-20b | 0.18 | 0.04 | 0.07 | 0.19 | 0.21 | 0.07 | 0.10 | 0.11 |

## Answers cut by the output limit, by request class

| model | brainstorming | classification | closed_qa | creative_writing | general_qa | information_extraction | open_qa | summarization |
|---|---|---|---|---|---|---|---|---|
| gpt-6.1-sol | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| claude-sonnet-5-5 | 1 | 0 | 0 | 5 | 1 | 0 | 0 | 0 |
| claude-haiku-4-5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gpt-oss-120b | 13 | 0 | 2 | 11 | 22 | 2 | 6 | 6 |
| gpt-6-luna | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| gpt-oss-20b | 8 | 0 | 0 | 5 | 4 | 0 | 1 | 0 |

## Attribution and gateway overhead

- Requests replayed through the gateway: 1,440; logged with tenant, feature and request id: 1,440.
- Cost attributed by tenant, US$: tenant-a 0.5902, tenant-b 0.6660, tenant-c 0.6603. The replayed cost equals the cost recorded live for every call.
- Gateway overhead in process, provider time excluded: p50 0.4 ms, p95 0.6 ms. Wall time on the machine that ran the report; informational, not gated.

## Method

- Traffic: 240 prompts from databricks-dolly-15k, 30 per category, see data/README.md.
- Each prompt was sent once to each model, one call at a time, on 2026-10-07, from one machine. Latency is what that machine saw on that day and includes the network.
- Prices are the registry prices checked on 2026-10-07 (llm_gateway/models.yaml).
- Total spent on live calls so far: US$ 1.92 of the US$ 10 project budget (data/spend_ledger.jsonl).
