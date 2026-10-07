# Replay traffic

`traffic.jsonl` holds 240 prompts sampled from
[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), revision
`bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, written by Databricks employees and released under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). `traffic.jsonl` is a derivative of
that dataset and is distributed under the same license.

What changed from the source: exact duplicate prompts (same instruction and context after
trimming whitespace) were dropped, 30 prompts were drawn per category with a fixed seed, the
context was appended to the instruction under a `Context:` label, a fixed system prompt was added,
and each row was assigned one of three fictional tenants. `traffic_manifest.json` records the
source hash, the seed and the counts.

The sample has the same number of prompts per category, which is not Dolly's natural mix. Any cost
figure computed over it depends on that mix.

Rebuild with `uv run python -m harness.traffic.build`. The raw file is downloaded to `data/raw/`,
which is not committed.
