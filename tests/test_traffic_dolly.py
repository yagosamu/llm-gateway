import json

import pytest

from harness.traffic.dolly import DollyChecksumMismatch, file_sha256, load_records, verify


def test_verify_refuses_a_file_whose_hash_differs(tmp_path):
    path = tmp_path / "dolly.jsonl"
    path.write_bytes(b"not dolly\n")
    with pytest.raises(DollyChecksumMismatch, match="expected abc"):
        verify(path, expected="abc")


def test_verify_accepts_the_expected_hash(tmp_path):
    path = tmp_path / "dolly.jsonl"
    path.write_bytes(b"x\n")
    verify(path, expected=file_sha256(path))


def test_load_records_adds_the_zero_based_line_index(tmp_path):
    path = tmp_path / "dolly.jsonl"
    lines = [{"instruction": "a", "context": "", "response": "r", "category": "open_qa"},
             {"instruction": "b", "context": "c", "response": "s", "category": "closed_qa"}]
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    assert [r["index"] for r in load_records(path)] == [0, 1]
    assert load_records(path)[1]["context"] == "c"
