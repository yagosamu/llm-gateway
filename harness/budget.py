"""The project's own spending cap: US$ 10 across every paid run, recorded in a committed ledger.

Before a run starts, its worst case is added to what the ledger already holds; if the sum passes the
cap, the run does not start. The worst case of a call is llm_gateway.registry.call_ceiling_usd, the
same bound the gateway's verifier uses for its daily cap."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_gateway.registry import TOKENS_PER_MESSAGE, call_ceiling_usd, input_token_bound  # noqa: F401

BUDGET_USD = 10.0
LEDGER_PATH = Path("data/spend_ledger.jsonl")


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
