# Slice 4 results: response cache

1000 queries replayed through the gateway; design in harness/cache/prereg.py. Semantic hits are judged against the served model's fresh answer by two judges; a hit is false if either rejects it.

**Chosen floor 0.96** (lowest with at most 0.5% false hits over at least 50 labeled hits on the tune half).

Test half at that floor: 0 false of 53 semantic hits, 0.0%, exact 95% interval [0.0%, 6.7%]. Below 1%: yes; upper bound below 1%: no. Over distinct pairs: 0 of 11, [0.0%, 28.5%].

| | no cache | exact layer only | exact + semantic at the chosen floor |
|---|---|---|---|
| hit rate | - | 23.3% | 27.7% (167 exact, 110 semantic) |
| cost per 1,000 queries, US$ | 0.0985 | 0.0790 (-19.8%) | 0.0770 (-21.9%) |
| p50 latency, ms | 2112 | 1728 | 2035 |
| p95 latency, ms | 6200 | 5990 | 6314 |

## Precision by floor

| floor | semantic hits | false | false-hit rate | tune: hits / false | test: hits / false |
|---|---|---|---|---|---|
| 0.80 | 269 | 8 | 3.0% | 127 / 5 | 142 / 3 |
| 0.81 | 268 | 7 | 2.6% | 126 / 4 | 142 / 3 |
| 0.82 | 265 | 7 | 2.6% | 125 / 4 | 140 / 3 |
| 0.83 | 260 | 6 | 2.3% | 122 / 4 | 138 / 2 |
| 0.84 | 252 | 6 | 2.4% | 114 / 4 | 138 / 2 |
| 0.85 | 233 | 6 | 2.6% | 109 / 4 | 124 / 2 |
| 0.86 | 225 | 5 | 2.2% | 108 / 4 | 117 / 1 |
| 0.87 | 217 | 2 | 0.9% | 100 / 1 | 117 / 1 |
| 0.88 | 188 | 2 | 1.1% | 81 / 1 | 107 / 1 |
| 0.89 | 178 | 2 | 1.1% | 78 / 1 | 100 / 1 |
| 0.90 | 169 | 2 | 1.2% | 73 / 1 | 96 / 1 |
| 0.91 | 154 | 1 | 0.6% | 72 / 1 | 82 / 0 |
| 0.92 | 130 | 1 | 0.8% | 68 / 1 | 62 / 0 |
| 0.93 | 128 | 1 | 0.8% | 67 / 1 | 61 / 0 |
| 0.94 | 130 | 1 | 0.8% | 67 / 1 | 63 / 0 |
| 0.95 | 128 | 1 | 0.8% | 66 / 1 | 62 / 0 |
| 0.96 | 110 | 0 | 0.0% | 57 / 0 | 53 / 0 |
| 0.97 | 61 | 0 | 0.0% | 43 / 0 | 18 / 0 |
| 0.98 | 40 | 0 | 0.0% | 26 / 0 | 14 / 0 |
| 0.99 | 7 | 0 | 0.0% | 7 / 0 | 0 / 0 |

## False hits

| pair (query <- stored) | similarity | same intent | category | half |
|---|---|---|---|---|
| dolly-00286<-dolly-06472 | 0.959 | no | closed_qa | tune |
| dolly-00905<-dolly-10012 | 0.905 | no | open_qa | test |
| dolly-10012<-dolly-00905 | 0.905 | no | open_qa | test |
| dolly-10012-p3<-dolly-00905 | 0.876 | no | open_qa | test |
| dolly-14058-p4<-dolly-14058-p2 | 0.864 | yes | brainstorming | tune |
| dolly-04426<-dolly-00445 | 0.855 | no | brainstorming | test |
| dolly-14646<-dolly-01077 | 0.801 | no | classification | tune |

## Counterfactuals at the chosen floor

- Key on the question alone (no tenant): 228 of 371 hits would have served an answer stored for another tenant (147 exact, 81 semantic).
- System prompt version changed for the first 200 queries: 0 hits on answers stored under the old version (28 hits on answers those same queries stored under the new version, when one repeated another); the same queries again under the old version: 198 hits. The pre-registration said every one must miss, which overlooked repeats within the 200; the check that matters is the first number.
- Context left out of the embedding: among 25320 pairs of texts with different context blocks, 1 would match on the instruction alone, 0 match with the context embedded (not judged).

## Judges and the human check

The two judges agree on 150 of 150 pairs. Blind human labels agree with the judges on 19 of 22 sampled pairs: {'human_no_judges_no': 6, 'human_no_judges_yes': 2, 'human_yes_judges_no': 1, 'human_yes_judges_yes': 13}.

False pairs by category, over all labeled pairs: brainstorming 2/13, classification 1/5, closed_qa 1/37, creative_writing 0/9, general_qa 0/8, information_extraction 0/10, open_qa 3/45, summarization 0/23.

Latency is built from recorded latencies (the model call and the embedding call); the cache lookup is not included. Exact hits skip the embedding call; misses pay it.
