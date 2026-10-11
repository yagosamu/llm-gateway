from collections import Counter

import numpy as np
import pytest

from harness.cache import prereg, workload


def records(per_category=25):
    return [{"index": c * 100 + i, "category": cat, "instruction": f"q {cat} {i}", "context": "", "response": "r"}
            for c, cat in enumerate(workload.CATEGORIES) for i in range(per_category)]


def test_intents_are_twenty_per_category_split_in_halves_and_reproducible():
    intents = workload.select_intents(records())
    assert len(intents) == 160
    assert Counter(i["category"] for i in intents) == {c: 20 for c in workload.CATEGORIES}
    assert Counter((i["category"], i["half"]) for i in intents)[("closed_qa", "tune")] == 10
    assert intents == workload.select_intents(records())


def test_paraphrases_parse_through_a_code_fence_and_drop_repeats():
    raw = '```json\n["Where is my order?", "Where  is my  order?", "Track my parcel", "Where is my order"]\n```'
    parsed = workload.parse_paraphrases(raw)
    assert workload.clean_paraphrases("Where is my order", parsed) == ["Where is my order?", "Track my parcel"]
    with pytest.raises(ValueError):
        workload.parse_paraphrases('{"a": 1}')


def test_neighbors_skip_excluded_rows_and_rows_taken_by_an_earlier_intent():
    vectors = np.array([[1, 0], [0.99, 0.14], [0.98, 0.2], [0, 1], [0.1, 0.99]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    picks = workload.nearest_neighbors([0, 3], vectors, 1, excluded={0, 3})
    assert [p[0][0] for p in picks] == [1, 4]
    picks = workload.nearest_neighbors([0, 1], vectors, 1, excluded={0, 1})
    assert [p[0][0] for p in picks] == [2, 4]  # row 2 went to intent 0, so intent 1 gets the next one


def test_zipf_head_favours_the_first_ranks():
    draws = workload.zipf_head([f"i{n}" for n in range(160)], 600, 1.0, 1)
    counts = Counter(draws).most_common()
    assert len(draws) == 600 and counts[0][1] > 60 and len(counts) < 160


def test_the_workload_has_the_preregistered_shape():
    intents = workload.select_intents(records())
    texts = {}
    for i in intents:
        for v in range(5):
            texts[workload.text_id(i["index"], v)] = {"kind": "intent", "intent": i["id"], "half": i["half"],
                                                      "category": i["category"], "source_index": i["index"],
                                                      "reference_response": "r", "text": f"{i['id']} v{v}"}
    extra = [{"id": workload.text_id(10_000 + n), "half": "tune"} for n in range(400)]
    hard, tail = extra[:320], extra[320:]
    for kind, items in (("hard_negative", hard), ("tail", tail)):
        for item in items:
            texts[item["id"]] = {"kind": kind, "half": "tune", "category": "open_qa", "source_index": 0,
                                 "reference_response": "r", "text": item["id"]}
    rows = workload.assemble(texts, intents, hard, tail)
    assert len(rows) == prereg.N_QUERIES
    assert Counter(r["kind"] for r in rows) == {"head": 600, "hard_negative": 320, "tail": 80}
    once = Counter(r["text_id"] for r in rows if r["kind"] != "head")
    assert set(once.values()) == {1}
    assert {r["tenant"] for r in rows} == {"tenant-a", "tenant-b", "tenant-c"}
    assert [r["q"] for r in rows] == list(range(1000))
    assert rows == workload.assemble(texts, intents, hard, tail)
