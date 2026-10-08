"""Pre-registered design of the slice 2 routing evaluation.

Everything below was fixed before any judge verdict existed: the reference model, the tier map, the
judges and how they combine, the rubric, the policies, the decision threshold, the folds, the quality
bar and the comparisons. The commit history of this file is the public record of that. Changing a
value after verdicts exist turns the analysis into a post hoc one, and the report would have to say so.

The question: routing each request to the cheapest model that serves it, how much cost is saved
against sending everything to the reference model, and what share of answers stays acceptable?"""

# The answer every candidate is compared with. When the reference answer was cut by the output limit
# or came back empty (3 of 240 prompts in the recording), the fallback reference is used instead.
# A prompt for which neither gave a usable answer has no reference and is excluded from the evaluation
# (2 of 240 in the recording, both creative_writing: dolly-02945 and dolly-06056); the report lists them.
REFERENCE_MODEL = "gpt-6.1-sol"
FALLBACK_REFERENCE_MODEL = "claude-sonnet-5-5"

# Tier map evaluated in this slice. The router sends a request to one of these three models.
TIER_MAP = {"low": "gpt-6-luna", "medium": "claude-haiku-4-5", "high": "gpt-6.1-sol"}
CANDIDATE_TIERS = ("low", "medium")
# The high tier's own answer is not acceptable by definition: it is acceptable unless the
# AUTO_UNACCEPTABLE rules reject it. So always_high fails where gpt-6.1-sol came back empty.

# Two judges from two providers, neither of which is a candidate, so no judge grades its own answer.
# A candidate answer is acceptable only when both judges say so.
JUDGES = ("gpt-4.1-mini", "claude-sonnet-5-5")
CONSENSUS = "both"

# Answers rejected by rule, without a judge call: cut by the output limit, or empty. This follows the
# slice 1 decision that a cut answer is a failure of that model under the 1,024-token policy.
AUTO_UNACCEPTABLE = ("finish_reason == 'length'", "empty text")

RUBRIC_VERSION = "v1"
RUBRIC = """You grade whether a CANDIDATE answer is an acceptable substitute for a REFERENCE answer to the same request.

Acceptable means that the person who sent the request would be about as well served by the candidate as by the reference:
- It does what the request asks, including any format, length or style constraint stated in the request.
- It makes no factual claim that contradicts the reference or the context given in the request, and invents no fact that matters to the request.
- It is complete. An answer that stops in the middle is not acceptable.

Do not reject for differences in wording, style, order or length when the request sets no constraint on them.
For open-ended requests (brainstorming, creative writing, opinions) there is no single right answer: judge whether the candidate is an on-task answer of similar usefulness, not whether it resembles the reference.
The reference can be wrong. If the candidate is right where the reference is wrong, the candidate is acceptable.

Give your reason first, in one or two sentences, then the verdict."""

# Policies, all evaluated out of fold. A per-tier decision routes to the cheapest candidate tier whose
# estimated chance of an acceptable answer is at least DECISION_THRESHOLD, and to high otherwise.
POLICIES = ("always_high", "always_low", "always_medium", "feature_table", "classifier", "oracle")
DECISION_THRESHOLD = 0.95
# feature_table: the estimate is the acceptance rate of (X-Feature, tier) in the training folds.
# classifier: one logistic regression per candidate tier, P(acceptable | prompt features), trained on
# the training folds. Features: log of prompt characters, whether a context block is present, whether
# the instruction asks for a list, a comparison, a creative piece or a summary, and the X-Feature value.
CLASSIFIER_FEATURES = ("log_chars", "has_context", "asks_list", "asks_compare", "asks_creative",
                       "asks_summary", "feature_one_hot")
N_FOLDS = 5  # stratified by category: 6 prompts per category per fold, 5 in some folds of a category that lost prompts
SEED = 20261008

# Quality bar, set by the project owner: a policy meets it when the lower end of the 95% bootstrap
# interval of its acceptance rate over the evaluated prompts is at least 0.90.
QUALITY_BAR_LOWER_BOUND = 0.90
N_RESAMPLES = 10_000
ALPHA = 0.05

# Primary claim: the classifier policy meets the quality bar; reported with its cost reduction against
# always_high, both with 95% bootstrap intervals. Secondary: classifier against feature_table on
# per-prompt acceptance, two-sided exact McNemar test, and the paired bootstrap interval of the cost
# difference. These are the only two hypothesis tests, so no multiplicity correction is applied.
PRIMARY_POLICY = "classifier"
SECONDARY_COMPARISON = ("classifier", "feature_table")

# Human calibration: 30 candidate answers, 15 per candidate tier, drawn with SEED from the answers that
# reach the judges (cut and empty answers are excluded), shown without model names. Reported as raw
# agreement and Cohen's kappa between the human labels and the judge consensus.
N_HUMAN_ITEMS = 30
