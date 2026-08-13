# tests/test_suite_state.py
"""Tests for `suite_state()`, task 3's public accessor over verify.py's
existing private parsers -- the primitive stage 4 needs to ask "is this
suite green, and did the run that says so actually finish" without a
second module reimplementing the same regexes (plan 01's process lesson 2,
`docs/CARRIED-DEBT.md`). Offline throughout: every runner here is a canned
callable exactly like the ones in `test_corpus_verify.py`, so nothing in
this file spawns a real pytest subprocess or touches the network."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from robigo.profile.verify import SuiteState, WrongTreeError, suite_state

CLEAN = "MODULE_UNDER_TEST=/clone/src/pkg/__init__.py\n120 passed\nEXIT_CODE=0\n"
ONE_BAD = (
    "MODULE_UNDER_TEST=/clone/src/pkg/__init__.py\n"
    "FAILED tests/test_mod.py::test_x\n"
    "119 passed, 1 failed\nEXIT_CODE=1\n"
)
INTERRUPTED = (
    "MODULE_UNDER_TEST=/clone/src/pkg/__init__.py\n"
    "Interrupted: 1 error during collection\n"
    "1 error\nEXIT_CODE=2\n"
)


# ---------------------------------------------------------------------------
# SuiteState -- frozen, and the field carries what it claims to
# ---------------------------------------------------------------------------


def test_suite_state_is_frozen():
    # Fails if `frozen=True` is dropped from the dataclass decorator -- the
    # same convention `test_corpus_verify.py` already pins for `Baseline`
    # and `Verdict`.
    s = SuiteState(broken=0, executed=1, broken_ids=(), incomplete=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.broken = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# suite_state -- composes exactly the parsers `verify.py` already has
# ---------------------------------------------------------------------------


def test_clean_suite(tmp_path: Path):
    s = suite_state(tmp_path, lambda repo, pkg: CLEAN.replace("/clone", str(tmp_path)), "pkg")
    assert s.broken == 0 and s.executed == 120
    assert s.broken_ids == () and s.incomplete is None


def test_one_failure_is_identified_by_id(tmp_path: Path):
    s = suite_state(
        tmp_path, lambda repo, pkg: ONE_BAD.replace("/clone", str(tmp_path)), "pkg"
    )
    assert s.broken == 1
    assert s.broken_ids == ("tests/test_mod.py::test_x",)
    assert s.incomplete is None


def test_an_interrupted_run_is_reported_not_counted(tmp_path: Path):
    # This task's brief (task-3-brief.md) specified `assert "Interrupted"
    # in s.incomplete` here. That does not hold against the real
    # `_run_did_not_complete`: it checks the EXIT_CODE marker FIRST and
    # returns as soon as it sees a code outside {0, 1}, before it ever
    # inspects `_INTERRUPTED_MARKERS` ("Interrupted:"/"INTERNALERROR") --
    # so for this exact input (`EXIT_CODE=2`) the message is "pytest did
    # not complete normally (exit code 2)", which contains no substring
    # "Interrupted" at all. Verified directly against the real function
    # (`.venv/bin/python -c ...`) before writing this assertion; see
    # task-3-report.md. What this task actually needs -- `incomplete` is
    # non-None, and names why, so no caller compares `broken`/`executed`
    # against a baseline as though the run finished -- is what this test
    # pins instead.
    s = suite_state(
        tmp_path, lambda repo, pkg: INTERRUPTED.replace("/clone", str(tmp_path)), "pkg"
    )
    assert s.incomplete is not None
    assert "exit code 2" in s.incomplete


def test_a_run_outside_the_clone_raises(tmp_path: Path):
    # Fails for an implementation that skips (or misorders) the
    # `_assert_in_clone` call -- invariant 7 must reject before any of the
    # four fields is trusted, exactly as `baseline`/`verify` already
    # require of themselves.
    outside = CLEAN.replace("/clone", "/somewhere/else")
    with pytest.raises(WrongTreeError):
        suite_state(tmp_path, lambda repo, pkg: outside, "pkg")
