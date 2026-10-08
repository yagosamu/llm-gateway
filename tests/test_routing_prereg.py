import csv
import json

import pytest

from harness.record import RECORDINGS_PATH, load_records, load_rows
from harness.replay import current_records
from harness.routing import prereg
from harness.routing.human_sheet import sample_items, write_sheet
from harness.routing.pairs import build_pairs, excluded_prompts, is_auto_unacceptable, reference_for
from llm_gateway.registry import load_registry

REGISTRY = load_registry()


@pytest.fixture(scope="module")
def pairs():
    return build_pairs(load_rows(), current_records(load_records(RECORDINGS_PATH), REGISTRY))


def test_the_tier_map_and_references_are_registry_models():
    assert set(prereg.TIER_MAP.values()) <= set(REGISTRY)
    assert prereg.TIER_MAP["high"] == prereg.REFERENCE_MODEL
    assert prereg.FALLBACK_REFERENCE_MODEL in REGISTRY


def test_no_judge_grades_its_own_answers():
    candidates = {prereg.TIER_MAP[t] for t in prereg.CANDIDATE_TIERS}
    assert not candidates & set(prereg.JUDGES)


def test_folds_divide_every_category_evenly():
    assert 30 % prereg.N_FOLDS == 0 and prereg.N_HUMAN_ITEMS % len(prereg.CANDIDATE_TIERS) == 0


def test_auto_rules_reject_cut_and_empty_answers_only():
    assert is_auto_unacceptable({"finish_reason": "length", "text": "long..."}) == "cut by the output limit"
    assert is_auto_unacceptable({"finish_reason": "stop", "text": "  "}) == "empty answer"
    assert is_auto_unacceptable({"finish_reason": "stop", "text": "Paris."}) is None


def test_an_unusable_reference_falls_back_to_the_second_model():
    matrix = {("p", prereg.REFERENCE_MODEL): {"model": prereg.REFERENCE_MODEL, "finish_reason": "length", "text": ""},
              ("p", prereg.FALLBACK_REFERENCE_MODEL): {"model": prereg.FALLBACK_REFERENCE_MODEL,
                                                       "finish_reason": "stop", "text": "ok"}}
    assert reference_for("p", matrix)["model"] == prereg.FALLBACK_REFERENCE_MODEL


def test_a_prompt_with_no_usable_reference_is_excluded():
    matrix = {("p", m): {"model": m, "finish_reason": "length", "text": "cut"}
              for m in (prereg.REFERENCE_MODEL, prereg.FALLBACK_REFERENCE_MODEL)}
    assert reference_for("p", matrix) is None


def test_the_recording_gives_one_pair_per_evaluated_prompt_and_candidate_tier(pairs):
    matrix = current_records(load_records(RECORDINGS_PATH), REGISTRY)
    assert excluded_prompts(load_rows(), matrix) == ["dolly-02945", "dolly-06056"]
    assert len(pairs) == 238 * len(prereg.CANDIDATE_TIERS)
    assert sum(p["reference_model"] == prereg.FALLBACK_REFERENCE_MODEL for p in pairs) == 1 * 2
    assert all(p["reference_text"].strip() for p in pairs)


def test_the_human_sample_is_balanced_deterministic_and_excludes_auto_rejections(pairs):
    items = sample_items(pairs)
    assert items == sample_items(pairs)
    assert [sum(i["tier"] == t for i in items) for t in prereg.CANDIDATE_TIERS] == [15, 15]
    assert not any(i["auto_unacceptable"] for i in items)
    assert len({(i["prompt_id"], i["tier"]) for i in items}) == 30


def test_the_sheet_names_no_model(pairs, tmp_path):
    items = sample_items(pairs)
    write_sheet(items, tmp_path)
    text = (tmp_path / "sheet.csv").read_text(encoding="utf-8-sig")
    assert not any(model in text for model in REGISTRY)
    with (tmp_path / "sheet.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30 and all(r["acceptable"] == "" for r in rows)
    key = json.loads((tmp_path / "key.json").read_text(encoding="utf-8"))
    assert {k["model"] for k in key.values()} == {prereg.TIER_MAP[t] for t in prereg.CANDIDATE_TIERS}
