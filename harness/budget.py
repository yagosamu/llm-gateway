"""The project's own spending cap: US$ 10 across every paid run, recorded in a committed ledger.

Before a run starts, its worst case is added to what the ledger already holds; if the sum passes the
cap, the run does not start. Output tokens are bounded by the max_completion_tokens the run sends.
Input tokens are bounded by the UTF-8 byte count of the messages plus a per-message allowance: a
byte-level BPE tokenizer, as OpenAI's and Groq's models use, never emits more tokens than bytes.
Anthropic does not document its tokenizer, so its byte count is doubled as a margin, not a proof."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_gateway.registry import ModelConfig, cost_usd

BUDGET_USD = 10.0
LEDGER_PATH = Path("data/spend_ledger.jsonl")
# Role markers and separators the providers add around each message, generously rounded up.
TOKENS_PER_MESSAGE = 16
INPUT_SAFETY = {"anthropic": 2.0}


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class LedgerEntry:
    run: str
    request_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    recorded_at: str


def input_token_bound(messages: list[dict], provider: str) -> int:
    raw = sum(len(m["content"].encode("utf-8")) + TOKENS_PER_MESSAGE for m in messages)
    return int(raw * INPUT_SAFETY.get(provider, 1.0))


def call_ceiling_usd(model: ModelConfig, messages: list[dict], max_output_tokens: int) -> float:
    return cost_usd(model, input_token_bound(messages, model.provider), max_output_tokens)


def spent_usd(path: Path = LEDGER_PATH) -> float:
    if not path.exists():
        return 0.0
    with path.open(encoding="utf-8") as f:
        return sum(json.loads(line)["cost_usd"] for line in f if line.strip())


def preflight(run: str, ceiling_usd: float, path: Path = LEDGER_PATH, budget_usd: float = BUDGET_USD) -> float:
    """Raise BudgetExceeded when what was already spent plus this run's worst case passes the
    budget. Returns what is left of the budget after the worst case."""
    spent = spent_usd(path)
    if spent + ceiling_usd > budget_usd:
        raise BudgetExceeded(
            f"run {run!r} could cost up to US$ {ceiling_usd:.2f}; with US$ {spent:.2f} already spent "
            f"that passes the US$ {budget_usd:.2f} budget. Run fewer calls or lower the output limit.")
    return budget_usd - spent - ceiling_usd


def append(entry: LedgerEntry, path: Path = LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(asdict(entry)) + "\n")
