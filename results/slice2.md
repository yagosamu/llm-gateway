# Slice 2 results

The pre-registered primary policy, `classifier`, meets the quality bar: 95.8% of answers acceptable (95% CI 93.3 to 98.3; the bar is a lower bound of 90%), at 61.6% less cost than sending everything to gpt-6.1-sol.

## Policies

238 prompts, learned policies scored out of fold (5 folds stratified by category). 95% bootstrap intervals, 10,000 resamples, seed 20261008. Tiers: low = gpt-6-luna, medium = claude-haiku-4-5, high = gpt-6.1-sol.

| policy | acceptable answers | cost per 1,000 requests, US$ | cost reduction vs always_high | low / medium / high | meets bar |
|---|---|---|---|---|---|
| always_high | 99.6 [98.7, 100.0]% | 2.046 [1.833, 2.273] | 0.0 [0.0, 0.0]% | 0% / 0% / 100% | yes |
| always_low | 95.8 [93.3, 98.3]% | 0.089 [0.080, 0.099] | 95.6 [95.3, 95.9]% | 100% / 0% / 0% | yes |
| always_medium | 85.7 [81.1, 89.9]% | 1.104 [1.030, 1.178] | 46.1 [42.0, 49.7]% | 0% / 100% / 0% | no |
| feature_table | 95.4 [92.4, 97.9]% | 0.566 [0.424, 0.723] | 72.3 [65.6, 78.8]% | 75% / 13% / 12% | yes |
| classifier | 95.8 [93.3, 98.3]% | 0.786 [0.612, 0.974] | 61.6 [53.6, 69.3]% | 67% / 0% / 33% | yes |
| oracle | 99.6 [98.7, 100.0]% | 0.190 [0.112, 0.305] | 90.7 [85.8, 94.3]% | 96% / 3% / 2% | yes |

## Secondary comparison

classifier against feature_table on per-prompt acceptance, two-sided exact McNemar: only classifier acceptable on 1 prompts, only feature_table on 0, p = 1.0000. Cost difference per 1,000 requests (classifier minus feature_table): US$ 0.220 [0.098, 0.361].

## Acceptance by request class

| class | n | low | medium | high |
|---|---|---|---|---|
| brainstorming | 30 | 93% | 77% | 97% |
| classification | 30 | 97% | 90% | 100% |
| closed_qa | 30 | 93% | 97% | 100% |
| creative_writing | 28 | 93% | 93% | 100% |
| general_qa | 30 | 100% | 80% | 100% |
| information_extraction | 30 | 97% | 83% | 100% |
| open_qa | 30 | 97% | 80% | 100% |
| summarization | 30 | 97% | 87% | 100% |

## Judges and the human check

- The two judges agree on 93.2% of the 474 judged pairs (Cohen's kappa 0.36); gpt-4.1-mini accepted 462, claude-sonnet-5-5 accepted 434.
- A blind human grader and the judge consensus agree on 27 of 30 sampled answers (kappa 0.00). The human accepted 30 of 30; when one rater never says no, kappa is 0 whatever the agreement, so the raw count is the informative number.
  - item 14 (dolly-12245, medium tier): human yes, judges no
  - item 17 (dolly-00477, medium tier): human yes, judges no
  - item 29 (dolly-09763, medium tier): human yes, judges no

## Method

- Excluded: dolly-02945, dolly-06056 (no usable answer from either reference model).
- An answer cut by the output limit or empty is unacceptable by rule, for every tier including high.
- A candidate answer is acceptable only when both judges (gpt-4.1-mini, claude-sonnet-5-5) accept it under rubric v1.
- Learned policies route to the cheapest candidate tier whose estimated acceptance is at least 0.95, else high. The design is fixed in harness/routing/prereg.py.
