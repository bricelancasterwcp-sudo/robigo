# tests/test_record.py
from __future__ import annotations

import json
from pathlib import Path

from robigo.loop import RunResult
from robigo.record import RunRecorder, next_run_id


def test_run_ids_increment_without_collision(tmp_path: Path):
    assert next_run_id(tmp_path, "fog") == "fog-1"
    (tmp_path / ".robigo" / "runs" / "fog-1").mkdir(parents=True)
    assert next_run_id(tmp_path, "fog") == "fog-2"


def test_a_turn_stores_the_reply_byte_for_byte(tmp_path: Path):
    recorder = RunRecorder(tmp_path, "fog-1")
    # Trailing whitespace and CRLF are exactly what breaks SEARCH blocks,
    # so the record must not normalise anything.
    raw = "patch a.py\r\n```\nx = 1   \n```\n"
    recorder.turn("the prompt", raw, "pytest said no")
    stored = (tmp_path / ".robigo" / "runs" / "fog-1" / "turn-01-reply.txt")
    assert stored.read_bytes() == raw.encode("utf-8")


def test_finish_writes_machine_readable_meta(tmp_path: Path):
    recorder = RunRecorder(tmp_path, "fog-1")
    recorder.turn("p", "r", "a")
    recorder.finish(RunResult("pass", 1, 0, "robigo/fog-1", "tests pass"),
                    model="m", window=8192, codec="search_replace")
    meta = json.loads((tmp_path / ".robigo" / "runs" / "fog-1" / "meta.json").read_text())
    assert meta["outcome"] == "pass"
    assert meta["turns"] == 1
    assert meta["model"] == "m"
    assert meta["window"] == 8192
    assert meta["branch"] == "robigo/fog-1"


def test_turns_are_numbered_in_order(tmp_path: Path):
    recorder = RunRecorder(tmp_path, "fog-1")
    for i in range(3):
        recorder.turn(f"p{i}", f"r{i}", "a")
    names = sorted(p.name for p in (tmp_path / ".robigo" / "runs" / "fog-1").glob("turn-*-reply.txt"))
    assert names == ["turn-01-reply.txt", "turn-02-reply.txt", "turn-03-reply.txt"]


def test_records_never_enter_the_users_history(tmp_path: Path):
    # The Critical: snapshot does `git add -A`, so a record written inside
    # the repo would be committed into the user's own history from run 2 on.
    recorder = RunRecorder(tmp_path, "fog-1")
    recorder.turn("p", "r", "a")
    assert (tmp_path / ".robigo" / ".gitignore").read_text() == "*\n"


def test_a_surrogate_in_a_reply_does_not_crash(tmp_path: Path):
    recorder = RunRecorder(tmp_path, "fog-1")
    recorder.turn("p", "bad \ud800 reply", "a")   # must not raise
    assert recorder.error is not None
    assert "cannot write" in recorder.error


def test_nothing_is_created_until_something_is_written(tmp_path: Path):
    RunRecorder(tmp_path, "fog-1")
    assert not (tmp_path / ".robigo").exists()


def test_an_unwritable_root_is_remembered_not_raised(tmp_path: Path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        recorder = RunRecorder(blocked, "fog-1")
        recorder.turn("p", "r", "a")              # must not raise
        assert recorder.error is not None
        assert "cannot create" in recorder.error
    finally:
        blocked.chmod(0o700)
