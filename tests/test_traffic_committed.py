"""Checks on the committed data/traffic.jsonl against its manifest. Offline: no download needed."""
import json
from collections import Counter
from pathlib import Path

import pytest

from harness.traffic import dolly
from harness.traffic.build import MANIFEST_PATH, TRAFFIC_PATH
from harness.traffic.sample import SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    with Path(TRAFFIC_PATH).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_manifest_matches_the_pinned_source(manifest):
    assert (manifest["revision"], manifest["sha256"]) == (dolly.REVISION, dolly.SHA256)
    assert manifest["system_prompt_version"] == SYSTEM_PROMPT_VERSION


def test_every_category_has_per_category_rows_and_ids_are_unique(rows, manifest):
    assert len(rows) == manifest["n_rows"] == len({r["id"] for r in rows})
    counts = Counter(r["category"] for r in rows)
    assert set(counts) == set(manifest["source_category_counts"])
    assert set(counts.values()) == {manifest["per_category"]}


def test_every_tenant_gets_an_equal_share_of_every_category(rows, manifest):
    cells = Counter((r["category"], r["tenant"]) for r in rows)
    assert set(cells.values()) == {manifest["per_category"] // len(manifest["tenants"])}


def test_every_row_is_a_system_plus_user_chat_from_the_pinned_revision(rows):
    for r in rows:
        assert r["source"]["revision"] == dolly.REVISION
        assert r["id"] == f"dolly-{r['source']['index']:05d}"
        assert [m["role"] for m in r["messages"]] == ["system", "user"]
        assert r["messages"][0]["content"] == SYSTEM_PROMPT
        assert r["messages"][1]["content"].strip()
