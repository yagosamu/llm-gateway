# Slice 3: which model the low tier fails over to

Scored by gpt-4.1-mini alone, with cut or empty answers failing by rule, on the same 238 prompts. Descriptive, not pre-registered.

| model | role | acceptable (95% CI) | drop vs the low tier, points | cost per 1,000, US$ | cut by the limit |
|---|---|---|---|---|---|
| gpt-6-luna | low tier | 97.1% [94.5, 99.2] | - | 0.089 | 2 |
| claude-haiku-4-5 | failover (chosen) | 97.1% [94.5, 99.2] | 0.0 [-2.5, 2.5] | 1.104 | 0 |
| gpt-oss-20b | failover (rejected) | 85.3% [80.7, 89.5] | 11.8 [7.6, 16.4] | 0.120 | 16 |

Paired exact McNemar against the low tier: claude-haiku-4-5: only the low tier acceptable on 5 prompts, only claude-haiku-4-5 on 5, p = 1.0000; gpt-oss-20b: only the low tier acceptable on 31 prompts, only gpt-oss-20b on 3, p = 0.0000.

claude-haiku-4-5 keeps the low tier's quality during an outage at about twelve times its cost; gpt-oss-20b would have kept the cost and lost quality. The cost only applies while the failover lasts.

gpt-4.1-mini is more lenient than the slice 2 two-judge consensus, under which claude-haiku-4-5 was accepted on 85.7% against gpt-6-luna's 95.8%; the equal rates here depend on the judge.
