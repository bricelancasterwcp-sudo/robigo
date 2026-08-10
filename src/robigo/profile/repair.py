# src/robigo/profile/repair.py
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from robigo.adapters.python_ import PythonAdapter
from robigo.loop import RunResult, run
from robigo.profile.corpus_io import CorpusRecord
from robigo.profile.verify import (
    Baseline, Runner, SuiteState, _package_name, _resolve_in_clone,
    pytest_runner, suite_state,
)

_INFRA_OUTCOMES = frozenset({"infrastructure"})
"""Only `infrastructure` is excluded. `stalled`, `refused` and
`budget_exhausted` are REAL model failures -- a model that cannot get a
patch past the safety layer, or that burns its turn cap, failed to repair,
and the gate's number must say so. Excluding those would be scoring the
tool on the subset of tasks it already handles."""


@dataclass(frozen=True)
class Attempt:
    """One (record, seed) repair attempt. `excluded` non-None means this
    attempt never gave the model a fair chance (spec 4.3.4) and belongs in
    NEITHER the numerator nor the denominator of any rate."""

    record: str
    seed: int
    passed: bool
    outcome: str
    turns: int
    repeats: int
    excluded: str | None


def task_for(record: CorpusRecord) -> str:
    """The task the model is given. Names ONLY the failing test (spec 4.2).
    The record also carries `path`, `line` and `fixed`; putting any of them
    here would measure a tool nobody has."""
    return f"the test {record.test_id} fails; make it pass"


def _anchor_path(record: CorpusRecord, repo: Path) -> Path:
    """The test file the anchor hash guards -- the file part of the pytest
    node id, resolved inside the clone."""
    return _resolve_in_clone(repo, Path(record.test_id.split("::")[0]))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reset_clone(repo: Path) -> None:
    """Discard everything the previous attempt did. Runs before EVERY
    attempt, not once per record (spec 4.3.3)."""
    subprocess.run(["git", "checkout", "--", "."], cwd=repo, check=True,
                   capture_output=True)
    subprocess.run(["git", "clean", "-fdq"], cwd=repo, check=True,
                   capture_output=True)


def break_it(record: CorpusRecord, repo: Path) -> None:
    """Write `record.broken` at `record.line`, reproducing the corpus's
    defective tree. `line` is 1-based and `broken` carries its own line
    ending, so `splitlines(keepends=True)` is the only correct split."""
    target = _resolve_in_clone(repo, record.path)
    lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[record.line - 1] = record.broken
    target.write_text("".join(lines), encoding="utf-8")


def attempt_repair(
    record: CorpusRecord,
    repo: Path,
    client,
    *,
    seed: int,
    codec: str,
    base: Baseline,
    turn_cap: int = 8,
    runner: Runner = pytest_runner,
) -> Attempt:
    def excluded(why: str, outcome: str = "", turns: int = 0) -> Attempt:
        return Attempt(record.name, seed, False, outcome, turns, 0, why)

    try:
        reset_clone(repo)
        break_it(record, repo)
    except (OSError, subprocess.CalledProcessError, IndexError) as exc:
        return excluded(f"could not stage the defect: {exc}")

    anchor = _anchor_path(record, repo)
    before = _sha(anchor)

    # The SHIPPED tool: real turn cap, real codec, git on, test edits off
    # (spec 4.3.1). A defect on any of those paths counts against the tool,
    # because a user meets it.
    result: RunResult = run(
        task_for(record), repo, client, PythonAdapter(),
        codec=codec, turn_cap=turn_cap, allow_test_edits=False, use_git=True,
    )
    if result.outcome in _INFRA_OUTCOMES:
        return excluded(f"loop infrastructure: {result.detail}",
                        result.outcome, result.turns)

    try:
        state: SuiteState = suite_state(repo, runner, _package_name(record.path))
    except Exception as exc:
        return excluded(f"suite did not run: {exc}", result.outcome, result.turns)

    if state.incomplete is not None:
        return excluded(f"suite run incomplete: {state.incomplete}",
                        result.outcome, result.turns)
    if state.executed != base.executed:
        return excluded(
            f"executed total {state.executed} != baseline {base.executed}",
            result.outcome, result.turns)

    anchor_intact = _sha(anchor) == before
    passed = (
        result.outcome == "pass"
        and state.broken == 0
        and record.test_id not in state.broken_ids
        and anchor_intact
    )
    return Attempt(record.name, seed, passed, result.outcome, result.turns,
                   0, None)
