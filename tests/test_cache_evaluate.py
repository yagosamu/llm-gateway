import asyncio
import json

import pytest

from harness.cache import evaluate


def test_clopper_pearson_matches_the_closed_forms_at_the_edges():
    lo, hi = evaluate.clopper_pearson(0, 53)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 53), abs=1e-9)
    lo, hi = evaluate.clopper_pearson(53, 53)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 53), abs=1e-9)
    lo, hi = evaluate.clopper_pearson(5, 100)
    assert 0.016 < lo < 0.017 and 0.112 < hi < 0.113  # the textbook 5/100 interval


def table(labeled, false):
    return {"labeled": labeled, "rate": false / labeled if labeled else None}


def test_the_floor_is_the_lowest_meeting_both_tune_rules():
    per_floor = {"0.80": {"tune": table(120, 3)}, "0.85": {"tune": table(100, 0)},
                 "0.90": {"tune": table(60, 0)}, "0.95": {"tune": table(40, 0)}}
    assert evaluate.choose_threshold(per_floor) == "0.85"
    per_floor["0.85"]["tune"] = table(49, 0)  # too few labeled hits to count
    assert evaluate.choose_threshold(per_floor) == "0.90"
    assert evaluate.choose_threshold({"0.80": {"tune": table(200, 2)}}) is None


def test_false_hits_count_occurrences_and_distinct_pairs():
    rows = [{"layer": "semantic", "half": "test", "text_id": "a", "matched_text_id": "b"},
            {"layer": "semantic", "half": "test", "text_id": "a", "matched_text_id": "b"},
            {"layer": "semantic", "half": "tune", "text_id": "c", "matched_text_id": "d"},
            {"layer": "exact", "half": "test", "text_id": "a", "matched_text_id": "a"}]
    t = evaluate.false_hit_table(rows, {"a<-b": False, "c<-d": True}, "test")
    assert (t["hits"], t["false"], t["rate"], t["distinct_pairs"], t["distinct_false"]) == (2, 2, 1.0, 1, 1)


@pytest.mark.gate
def test_the_committed_slice4_results_match_a_fresh_replay(tmp_path):
    if not evaluate.JSON_PATH.exists():
        pytest.skip("no slice 4 results committed yet")
    fresh = asyncio.run(evaluate.build(tmp_path))
    assert fresh == json.loads(evaluate.JSON_PATH.read_text(encoding="utf-8"))
