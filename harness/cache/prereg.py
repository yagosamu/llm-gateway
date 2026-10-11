"""Pre-registered design of the slice 4 cache evaluation.

Everything below was fixed before any paid call of the slice: the workload, the embedding model, the
cache key, what a false hit is and who judges it, the threshold grid, the tune and test split, the
rule that picks the threshold and the claim that is tested. The commit history of this file is the
public record of that. A value changed after the first paid call makes the analysis post hoc, and the
report has to say so.

The question: with an exact layer and a semantic layer in front of the served model, how much of a
production-shaped workload is answered from the cache, what does that save in cost and latency, and
how often is a semantic hit an answer the user should not have received?

Two things this design does not measure, on purpose:
- The hit rate is a property of the workload built here (60% of queries repeat a small set of
  questions, the share in the use case this slice follows); Dolly has no natural repetition. It is
  reported, never presented as a finding about real traffic.
- Provider-side prefix caching is out of scope: the system block is about 15 tokens, far below the
  minimum prefix length the providers cache (cited from their documentation in the README).
"""
from harness.routing import prereg as routing_prereg

# --- The workload: 1,000 queries, built from the pinned Dolly file (data/raw, revision in
# data/traffic_manifest.json), deduplicated as in slice 1. ---------------------------------------
SEED = 20261011
N_QUERIES = 1000

# Intents: Dolly records whose question is asked repeatedly, in different words.
INTENTS_PER_CATEGORY = 20  # 160 intents across Dolly's eight categories
N_PARAPHRASES = 4          # so each intent has 5 texts: the original and 4 paraphrases
HEAD_QUERIES = 600         # 60% of the workload repeats an intent
ZIPF_EXPONENT = 1.0        # intent i (rank in a seeded random order) is drawn with weight 1 / i
# Each head query picks one of its intent's 5 texts uniformly at random, so the same text can come
# back (an exact hit) or a different wording of it (a semantic hit, or a miss).

# Paraphrases are written once by PARAPHRASE_MODEL, one call per intent returning N_PARAPHRASES
# rewrites of the instruction. A context block, when present, is kept verbatim after the rewritten
# instruction, as in slice 1 ("Context:" label). Paraphrases identical to the original or to each
# other after whitespace normalization are dropped and not replaced. Paraphrases are not checked for
# meaning: a rewrite that changed the question is a query like any other, and if the cache serves it
# the wrong answer the judges count a false hit, as a real user would experience it.
PARAPHRASE_MODEL = "gpt-4.1-mini"
PARAPHRASE_INSTRUCTION = (
    "Rewrite the request below in {n} different ways, as {n} different people might ask for the same "
    "thing. Keep its meaning, every named entity, number and constraint. Vary the wording, the "
    "sentence structure and the length. Do not answer the request. Return only a JSON list of {n} strings.")

# Hard negatives: for each intent, the HARD_NEGATIVES_PER_INTENT Dolly records nearest to it by
# embedding (cosine over the full user message, context included), excluding every intent and any
# record already chosen. Similar wording, usually a different question: the false-hit risk.
HARD_NEGATIVES_PER_INTENT = 2  # 320 queries, each asked once
# Random tail: Dolly records drawn uniformly from the rest, each asked once.
RANDOM_TAIL = 80
assert HEAD_QUERIES + INTENTS_PER_CATEGORY * 8 * HARD_NEGATIVES_PER_INTENT + RANDOM_TAIL == N_QUERIES

# Every query gets a tenant uniformly at random from slice 1's three, and the 1,000 are shuffled.
# All use slice 1's system prompt, version v1.

# --- Tune and test halves -----------------------------------------------------------------------
# Intents are split in half, stratified by category (10 and 10 per category); an intent's hard
# negatives go with it, and the random tail is split 40 and 40. A hit belongs to the half of the
# query that received it. The threshold is chosen on the tune half and measured on the test half.

# --- The cache -----------------------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"  # price checked on OpenAI's page before the first paid run
EMBEDDING_INPUT = "the user message, instruction and context together"  # never the instruction alone
# Exact layer key: sha256 over tenant, system prompt version, the model that serves the request (after
# routing), temperature, max_tokens and the messages with whitespace runs collapsed. The semantic
# layer searches only among entries with the same key minus the messages, so a hit never crosses a
# tenant, a system prompt version, a model or a parameter. Only single-turn requests use the semantic
# layer. Only answers that finished normally (finish_reason "stop", non-empty) are stored.

# --- What a false hit is, and who decides ---------------------------------------------------------
# A semantic hit serves query Q the stored answer A, written for an earlier text Q'. Reference: the
# served model's own fresh answer to Q (gpt-6-luna, the low tier routing.yaml serves, recorded once
# per distinct text with the slice 1 output limit). When that fresh answer was cut or came back empty,
# the reference is Dolly's human response for Q's source record. The judges grade A as a candidate
# for request Q against that reference, with slice 2's rubric and judges unchanged. The hit is
# correct only if both accept; a false hit if either rejects.
JUDGES = routing_prereg.JUDGES  # ("gpt-4.1-mini", "claude-sonnet-5-5")
RUBRIC_VERSION = routing_prereg.RUBRIC_VERSION
RUBRIC = routing_prereg.RUBRIC
FALSE_HIT_IF = "either judge rejects"
# Exact hits are not judged: the stored answer is the served model's answer to the same text.

# --- Thresholds and labels ------------------------------------------------------------------------
# Cosine similarity floors from 0.80 to 0.99 in steps of 0.01. The workload is replayed offline once
# per floor (the cache fills as it goes, so which entry a query meets depends on the floor). Every
# distinct (Q, Q') semantic-hit pair that occurs at any floor is a label; if there are more than
# MAX_LABELED_PAIRS, a seeded uniform sample of that many is judged and rates are computed over it.
THRESHOLDS = tuple(round(0.80 + 0.01 * i, 2) for i in range(20))
MAX_LABELED_PAIRS = 800

# --- The decision and the claim -------------------------------------------------------------------
# Threshold: the lowest floor at which the tune half's false-hit rate is at most TUNE_MAX_FALSE_HIT,
# over at least TUNE_MIN_LABELED labeled semantic hits. If no floor qualifies, the semantic layer is
# reported as not shippable and the served config keeps only the exact layer.
TUNE_MAX_FALSE_HIT = 0.005
TUNE_MIN_LABELED = 50
# Primary claim: at the chosen floor, the test half's false-hit rate is below 1%, reported with its
# exact (Clopper-Pearson) 95% interval. The upper end is published whether or not it is below 1%.
CLAIM_MAX_FALSE_HIT = 0.01
ALPHA = 0.05

# Descriptive, not tested, on the full 1,000 at the chosen floor: hit rate (exact, semantic, total);
# cost against no cache (fresh answers avoided, minus the embedding calls, at registry prices); p50 and
# p95 latency against no cache (a miss costs the embedding call, the lookup and the model call, a hit
# the embedding call and the lookup; embedding latency recorded once per text, lookups timed in the
# replay); false-hit rate per Dolly category on all labeled pairs.

# Counterfactuals, descriptive:
# - Question-only key: hits the same replay would serve across tenants if the key held only the
#   message. Every one is an answer computed for another customer.
# - Instruction-only embedding: semantic matches above the chosen floor between texts whose
#   instructions match but whose contexts differ, had the context been left out of the embedding.
#   Counted, not judged.
# - System prompt change: after the replay, the first 200 queries are sent again with system prompt
#   version v2; every one must miss. Then again with v1, as a control that they hit.

# Human calibration: 30 labeled pairs, 15 that both judges accepted and 15 that at least one rejected
# (fewer if fewer exist), drawn with SEED, shown blind. Reported as raw agreement with the judges.
N_HUMAN_ITEMS = 30
