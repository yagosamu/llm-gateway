import json

import pytest
import yaml

from harness.chaos import deferrable
from harness.chaos.scenarios import routing_for
from llm_gateway.routing import DEFAULT_CONFIG_PATH


def write(folder, name, rows):
    (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_rep_metrics_count_jobs_duplicates_and_drain_time(tmp_path):
    wall = 1_000.0
    write(tmp_path, "events", [{"t": 0.0, "event": "start", "wall": wall}, {"t": 20.0, "event": "fault on"},
                               {"t": 60.0, "event": "fault off"}, {"t": 65.0, "event": "worker-1 killed"}])
    write(tmp_path, "submissions", [
        {"i": 0, "copy": 0, "sent": 1.0, "status": 202, "job": "a", "deduplicated": False},
        {"i": 0, "copy": 1, "sent": 1.5, "status": 202, "job": "a", "deduplicated": True},
        {"i": 1, "copy": 0, "sent": 30.0, "status": 202, "job": "b", "deduplicated": False},
        {"i": 1, "copy": 1, "sent": 30.5, "status": 202, "job": "b", "deduplicated": True}])
    write(tmp_path, "interactive", [{"i": 0, "sent": 25.0, "status": 502}, {"i": 1, "sent": 30.0, "status": 503},
                                    {"i": 2, "sent": 70.0, "status": 200}])
    write(tmp_path, "jobs", [{"id": "a", "status": "done", "attempts": 1, "created_at": wall + 1, "updated_at": wall + 3},
                             {"id": "b", "status": "done", "attempts": 5, "created_at": wall + 30, "updated_at": wall + 72.5}])
    write(tmp_path, "executions", [{"job": "a", "outcome": "ok"}, {"job": "b", "outcome": "upstream_error"},
                                   {"job": "b", "outcome": "ok"}, {"job": "b", "outcome": "ok"}])
    m = deferrable.rep_metrics(tmp_path)
    assert (m["submissions"], m["deduplicated"], m["jobs"], m["done"], m["failed"], m["unfinished"]) == (4, 2, 2, 2, 0, 0)
    assert m["jobs_answered_more_than_once"] == 1  # b answered twice: the at-least-once window
    assert (m["attempts_median"], m["attempts_max"]) == (3.0, 5)
    assert m["drained_after_fault_s"] == 12.5
    assert m["interactive_error_rate_in_fault"] == 1.0


def test_the_served_configuration_is_the_committed_one():
    committed = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    served = routing_for("served")
    assert {k: v for k, v in served.items() if k != "version"} == {k: v for k, v in committed.items() if k != "version"}


@pytest.mark.gate
def test_the_committed_deferrable_results_match_the_committed_runs():
    if not deferrable.JSON_PATH.exists():
        pytest.skip("no deferrable study committed yet")
    assert json.loads(json.dumps(deferrable.analyze())) == json.loads(deferrable.JSON_PATH.read_text(encoding="utf-8"))
