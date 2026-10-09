"""One SQLite row per request the gateway receives, including the ones it refuses.

The prompt is stored only as a sha256 of its messages: the log answers who spent what on which
model, and can match two identical prompts, without keeping anyone's text. SQLite runs in WAL mode
so a reader (a report, a dashboard) never blocks the writer. Writes are synchronous and serialized
by a lock; at this gateway's volume a write takes well under a millisecond, and a second process
writing to the same file is the point at which this should move to Postgres."""
import datetime
import hashlib
import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at TEXT NOT NULL,
    request_id TEXT,
    tenant TEXT,
    feature TEXT,
    model TEXT,
    provider TEXT,
    tier TEXT,
    outcome TEXT NOT NULL,
    http_status INTEGER NOT NULL,
    error_code TEXT,
    finish_reason TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    latency_ms REAL,
    prompt_sha256 TEXT,
    requested_model TEXT,
    routed_tier TEXT,
    routing_policy TEXT,
    routing_version TEXT,
    failover_from TEXT,
    attempts INTEGER
)"""
OUTCOMES = ("ok", "rejected", "upstream_error")
# One row per sampled verification. Rows with acceptable = 0 are the routing failures; they keep the
# prompt's routing features (never its text), which is what retraining the classifier needs.
VERIFICATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    verified_at TEXT NOT NULL,
    request_id TEXT,
    feature TEXT,
    routed_tier TEXT,
    routed_model TEXT,
    reference_model TEXT,
    outcome TEXT NOT NULL,
    acceptable INTEGER,
    reason TEXT,
    cost_usd REAL NOT NULL,
    prompt_sha256 TEXT,
    prompt_features TEXT
)"""
VERIFICATION_OUTCOMES = ("judged", "rejected_by_rule", "no_reference", "error")


@dataclass
class VerificationRecord:
    verified_at: str
    request_id: str | None
    feature: str | None
    routed_tier: str | None
    routed_model: str | None
    reference_model: str
    outcome: str
    acceptable: bool | None
    reason: str
    cost_usd: float
    prompt_sha256: str | None
    prompt_features: str  # JSON of llm_gateway.routing.prompt_features


@dataclass
class RequestRecord:
    received_at: str
    request_id: str | None = None
    tenant: str | None = None
    feature: str | None = None
    model: str | None = None  # only a registry id; a requested model the registry lacks stays None
    provider: str | None = None
    tier: str | None = None
    outcome: str = "rejected"
    http_status: int = 0
    error_code: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    prompt_sha256: str | None = None
    requested_model: str | None = None  # "auto" or a registry id; free text from the client otherwise
    routed_tier: str | None = None
    routing_policy: str | None = None
    routing_version: str | None = None
    failover_from: str | None = None  # the tier's model when another one served the request
    attempts: int | None = None  # models tried, including turned-away open circuits


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def prompt_sha256(messages: list[dict]) -> str:
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RequestLog:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(SCHEMA)
        self._conn.execute(VERIFICATION_SCHEMA)
        self._lock = threading.Lock()
        self._columns = [f.name for f in fields(RequestRecord)]
        # A log file created by an older version lacks the newer columns; add them in place.
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(requests)")}
        for column in self._columns:
            if column not in existing:
                self._conn.execute(f"ALTER TABLE requests ADD COLUMN {column}")

    def write(self, record: RequestRecord) -> None:
        if record.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {record.outcome!r}")
        values = asdict(record)
        placeholders = ", ".join("?" for _ in self._columns)
        with self._lock:
            self._conn.execute(f"INSERT INTO requests ({', '.join(self._columns)}) VALUES ({placeholders})",
                               [values[c] for c in self._columns])

    def rows(self) -> list[dict]:
        with self._lock:
            cursor = self._conn.execute(f"SELECT {', '.join(self._columns)} FROM requests ORDER BY id")
            return [dict(zip(self._columns, row)) for row in cursor.fetchall()]

    def write_verification(self, record: VerificationRecord) -> None:
        if record.outcome not in VERIFICATION_OUTCOMES:
            raise ValueError(f"unknown verification outcome {record.outcome!r}")
        values = asdict(record)
        columns = [f.name for f in fields(VerificationRecord)]
        with self._lock:
            self._conn.execute(f"INSERT INTO verifications ({', '.join(columns)}) VALUES "
                               f"({', '.join('?' for _ in columns)})", [values[c] for c in columns])

    def verifications(self) -> list[dict]:
        columns = [f.name for f in fields(VerificationRecord)]
        with self._lock:
            cursor = self._conn.execute(f"SELECT {', '.join(columns)} FROM verifications ORDER BY id")
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def verification_spend(self, day: str) -> float:
        """What verification cost on a UTC day (YYYY-MM-DD). Read from the table, so the daily cap
        survives a restart."""
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM verifications "
                                     "WHERE substr(verified_at, 1, 10) = ?", (day,)).fetchone()
        return float(row[0])

    def close(self) -> None:
        self._conn.close()
