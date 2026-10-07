import pytest

from harness.traffic.sample import (
    SYSTEM_PROMPT,
    TENANTS,
    build_row,
    dedupe,
    stratified_sample,
)


def record(index, category="open_qa", instruction="q", context=""):
    return {"index": index, "category": category, "instruction": instruction, "context": context,
            "response": f"r{index}"}


def test_dedupe_keeps_the_first_occurrence_of_an_instruction_and_context_pair():
    rows = [record(0, instruction="a"), record(1, instruction="b"), record(2, instruction=" a ")]
    kept, dropped = dedupe(rows)
    assert [r["index"] for r in kept] == [0, 1]
    assert dropped == 1


def test_dedupe_treats_a_different_context_as_a_different_prompt():
    rows = [record(0, instruction="a", context="x"), record(1, instruction="a", context="y")]
    kept, dropped = dedupe(rows)
    assert len(kept) == 2 and dropped == 0


def make_pool():
    return [record(i, category=c) for c in ("open_qa", "summarization") for i in range(100 if c == "open_qa" else 50)]


def test_stratified_sample_takes_n_per_category_without_repeats():
    sample = stratified_sample(make_pool(), per_category=10, seed=7)
    by_category = {}
    for r in sample:
        by_category.setdefault(r["category"], []).append(r["index"])
    assert {c: len(ids) for c, ids in by_category.items()} == {"open_qa": 10, "summarization": 10}
    assert all(len(set(ids)) == len(ids) for ids in by_category.values())


def test_stratified_sample_is_deterministic_for_a_seed():
    first = stratified_sample(make_pool(), per_category=10, seed=7)
    second = stratified_sample(make_pool(), per_category=10, seed=7)
    assert first == second
    assert first != stratified_sample(make_pool(), per_category=10, seed=8)


def test_stratified_sample_of_one_category_does_not_depend_on_the_others():
    pool = make_pool()
    alone = stratified_sample([r for r in pool if r["category"] == "summarization"], per_category=10, seed=7)
    together = stratified_sample(pool, per_category=10, seed=7)
    assert [r for r in together if r["category"] == "summarization"] == alone


def test_stratified_sample_refuses_a_category_smaller_than_n():
    with pytest.raises(ValueError, match="summarization"):
        stratified_sample(make_pool(), per_category=60, seed=7)


def test_stratified_sample_spreads_tenants_evenly_inside_each_category():
    sample = stratified_sample(make_pool(), per_category=9, seed=7)
    for category in ("open_qa", "summarization"):
        tenants = [r["tenant"] for r in sample if r["category"] == category]
        assert sorted(tenants) == sorted(TENANTS * 3)


def test_build_row_without_context_sends_the_instruction_alone():
    row = build_row({**record(5, instruction="Name three rivers."), "tenant": "tenant-a"}, revision="abc")
    assert row["id"] == "dolly-00005"
    assert row["messages"] == [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": "Name three rivers."}]
    assert row["source"] == {"dataset": "databricks/databricks-dolly-15k", "revision": "abc", "index": 5}
    assert row["reference_response"] == "r5"


def test_build_row_with_context_appends_it_under_a_label():
    rec = {**record(12, category="closed_qa", instruction="When?", context="In 2000."), "tenant": "tenant-b"}
    row = build_row(rec, revision="abc")
    assert row["messages"][1]["content"] == "When?\n\nContext:\nIn 2000."
    assert row["category"] == "closed_qa" and row["tenant"] == "tenant-b"
