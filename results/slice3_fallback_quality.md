# Slice 3: answer quality during a low-tier failover

Scored by gpt-4.1-mini alone, with cut or empty answers failing by rule, on the same 238 prompts. Descriptive, not pre-registered.

| model | role | acceptable (95% CI) | cut by the 1,024-token limit |
|---|---|---|---|
| gpt-6-luna | low tier | 97.1% [94.5, 99.2] | 2 |
| gpt-oss-20b | its failover | 85.3% [80.7, 89.5] | 16 |

While the low tier fails over, acceptance drops by 11.8 points (95% CI 7.6 to 16.4). Paired exact McNemar: only gpt-6-luna acceptable on 31 prompts, only gpt-oss-20b on 3, p = 0.0000.

gpt-4.1-mini is the verifier's judge and more lenient than the slice 2 two-judge consensus, so both rates read higher than slice 2's; the comparison between the two models is like for like.
