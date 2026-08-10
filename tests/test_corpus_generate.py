# tests/test_corpus_generate.py
from __future__ import annotations

import dataclasses
import socket
import subprocess
import time
from pathlib import Path

import pytest

from robigo.profile.corpus import candidates
from robigo.profile.generate import (
    GenerationResult,
    TargetOutcome,
    _TARGET_ABANDON_AFTER,
    generate_corpus,
    render_report,
)
from robigo.profile.verify import Baseline

# ---------------------------------------------------------------------------
# Offline guarantee: this whole module never shells out (`generate_corpus`
# only ever calls the injected `runner`) -- matching the convention already
# established by test_corpus_verify.py and test_corpus_io.py for this plan.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("socket.socket.connect must never be called in this test suite")

    monkeypatch.setattr(socket.socket, "connect", _blocked)


# executed=2: every runner in this module reports against a 2-test pool
# (calc.py + other.py, one test each) whenever a candidate is actually
# KEPT (`_kept_one_survives_other_runner`, `eager_runner`) -- the scenarios
# that report a different total (1, from a single-file blind runner) never
# expect a keep at all, so they are unaffected by this baseline not
# matching their own total exactly (whole-branch review C2's
# executed-total check only ever downgrades an already-non-kept verdict's
# REASON text there, never its kept/not-kept outcome -- see progress notes
# for the per-scenario accounting).
_BASE = Baseline(broken=0, executed=2, seconds=1.0)

# Two files, each with exactly one real repair candidate (an off_by_one on
# a single int literal) -- deliberately simple so a test's own runner logic
# never has to disambiguate between several candidates touching the same
# file at once.
_CALC_SOURCE = "def bump(n):\n    return n + 1\n"
_CALC_PATH = Path("src/mylib/calc.py")
_OTHER_SOURCE = "def double(n):\n    return n * 2\n"
_OTHER_PATH = Path("src/mylib/other.py")


def _marker(repo: Path) -> str:
    # `_assert_in_clone` only ever resolves and range-checks this path --
    # it need not exist on disk, so a plain path under `repo` is enough.
    # `EXIT_CODE=0` -- whole-branch review C1/C2: `verify()` now refuses a
    # keep unless the runner's report carries a normal (0 or 1) exit code;
    # every canned runner in this module represents an ordinary completed
    # pytest run, so this is the correct default for all of them.
    return f"MODULE_UNDER_TEST={repo / 'src' / 'mylib' / '__init__.py'}\nEXIT_CODE=0\n"


def _mutated_file(source: str, mutant) -> str:
    """The full file text after applying one `Mutant` -- the same
    substitution `robigo.profile.corpus._apply` performs, rederived here
    (not imported) so a test runner can recognise exactly which candidate
    a clone's on-disk content corresponds to."""
    lines = source.splitlines(keepends=True)
    lines[mutant.line - 1] = mutant.mutated
    return "".join(lines)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "mylib").mkdir(parents=True)
    (tmp_path / _CALC_PATH).write_text(_CALC_SOURCE)
    (tmp_path / _OTHER_PATH).write_text(_OTHER_SOURCE)
    return tmp_path


# ---------------------------------------------------------------------------
# The happy path: one kept, one survived, across two targets
# ---------------------------------------------------------------------------


_CALC_OFF_BY_ONE = next(
    m for m in candidates(_CALC_SOURCE, _CALC_PATH) if m.operator == "off_by_one"
)
_CALC_OFF_BY_ONE_BROKEN = _mutated_file(_CALC_SOURCE, _CALC_OFF_BY_ONE)
"""calc.py's `candidates()` offers TWO real candidates here (`dropped_return`
in addition to `off_by_one`, since `return n + 1` is a bare `return <expr>`)
-- this fixture's runner only reports breakage for THIS SPECIFIC one
(matched by exact content, not "differs from the original"), so the tests
using it get a deterministic 1-of-2-tried result rather than accidentally
keeping both candidates a file happens to offer."""


def _kept_one_survives_other_runner(repo: Path):
    # calc.py's off_by_one candidate breaks exactly one test (kept); every
    # OTHER candidate anywhere (calc.py's own dropped_return, and
    # everything other.py offers) survives.
    def runner(r: Path, package: str) -> str:
        calc = (r / _CALC_PATH).read_text()
        if calc == _CALC_OFF_BY_ONE_BROKEN:
            return _marker(r) + (
                "FAILED tests/test_calc.py::test_bump - AssertionError\n"
                "1 failed, 1 passed in 0.01s\n"
            )
        return _marker(r) + "0 failed, 2 passed in 0.01s\n"

    return runner


def test_a_kept_candidate_becomes_a_correctly_mapped_corpus_record(repo: Path):
    # Fails if broken/fixed are ever swapped (broken must be the MUTATED
    # line, fixed the ORIGINAL -- corpus_io.py's own invariant 8), if
    # test_id/diagnostic are not threaded from the Verdict, or if
    # source_repo/source_sha are not passed through verbatim.
    mutant = next(m for m in candidates(_CALC_SOURCE, _CALC_PATH) if m.operator == "off_by_one")
    result = generate_corpus(
        repo, [_CALC_PATH], _BASE, _kept_one_survives_other_runner(repo),
        max_records=50, time_budget=60.0,
        source_repo="git@example.com:someone/mylib.git", source_sha="deadbeef" * 5,
    )
    assert len(result.records) == 1
    record = result.records[0]
    assert record.path == mutant.path
    assert record.line == mutant.line
    assert record.broken == mutant.mutated
    assert record.fixed == mutant.original
    assert record.broken != record.fixed  # guards against a vacuous equal-strings pass
    assert record.operator == "off_by_one"
    assert record.test_id == "tests/test_calc.py::test_bump"
    assert record.diagnostic == "exactly one net new failure"
    assert record.source_repo == "git@example.com:someone/mylib.git"
    assert record.source_sha == "deadbeef" * 5
    assert record.name == "calc-off_by_one-2"


def test_a_survived_candidate_is_named_in_dropped_not_silently_absent(repo: Path):
    result = generate_corpus(
        repo, [_OTHER_PATH], _BASE, _kept_one_survives_other_runner(repo),
        max_records=50, time_budget=60.0, source_repo="r", source_sha="s",
    )
    assert result.records == ()
    assert len(result.dropped) == 2  # other.py offers 2 real candidates, both survive
    assert all("src/mylib/other.py:2" in d and "survived" in d for d in result.dropped)
    joined = " ".join(result.dropped)
    assert "dropped_return" in joined
    assert "off_by_one" in joined


def test_per_target_outcomes_are_kept_separate_not_pooled(repo: Path):
    # Invariant 13: a target that kept something and a fully barren target
    # must stay visibly different, not averaged into one pooled figure.
    result = generate_corpus(
        repo, [_CALC_PATH, _OTHER_PATH], _BASE, _kept_one_survives_other_runner(repo),
        max_records=50, time_budget=60.0, source_repo="r", source_sha="s",
    )
    by_path = {t.path: t for t in result.targets}
    assert by_path[_CALC_PATH] == TargetOutcome(_CALC_PATH, 2, 2, 1, False)
    assert by_path[_OTHER_PATH] == TargetOutcome(_OTHER_PATH, 2, 2, 0, False)
    assert sum(t.kept for t in result.targets) == len(result.records)


def test_the_mutated_file_is_restored_after_generation(repo: Path):
    # generate_corpus must leave the clone byte-identical to how it found
    # it -- verify()'s own restore is exercised here through the real
    # wiring, not assumed to still hold.
    generate_corpus(
        repo, [_CALC_PATH, _OTHER_PATH], _BASE, _kept_one_survives_other_runner(repo),
        max_records=50, time_budget=60.0, source_repo="r", source_sha="s",
    )
    assert (repo / _CALC_PATH).read_text() == _CALC_SOURCE
    assert (repo / _OTHER_PATH).read_text() == _OTHER_SOURCE


# ---------------------------------------------------------------------------
# A target offering no real candidate
# ---------------------------------------------------------------------------


def test_a_target_with_no_candidates_contributes_nothing_to_dropped(repo: Path):
    blank = Path("src/mylib/blank.py")
    (repo / blank).write_text("x = 'hello'\n")

    def runner(r: Path, package: str) -> str:
        return _marker(r) + "0 failed, 1 passed\n"

    result = generate_corpus(
        repo, [blank], _BASE, runner,
        max_records=50, time_budget=60.0, source_repo="r", source_sha="s",
    )
    assert result.records == ()
    assert result.dropped == ()
    assert result.targets == (TargetOutcome(blank, 0, 0, 0, False),)


def test_an_unreadable_target_is_named_in_dropped_and_the_run_continues(repo: Path):
    missing = Path("src/mylib/does_not_exist.py")

    def runner(r: Path, package: str) -> str:
        return _marker(r) + "0 failed, 2 passed\n"

    result = generate_corpus(
        repo, [missing, _CALC_PATH], _BASE, runner,
        max_records=50, time_budget=60.0, source_repo="r", source_sha="s",
    )
    # The unreadable target contributes a dropped note and a zeroed
    # outcome, but does NOT stop the next target from being processed.
    assert any("does_not_exist.py" in d and "could not read" in d for d in result.dropped)
    assert result.targets[0] == TargetOutcome(missing, 0, 0, 0, False)
    assert result.targets[1].path == _CALC_PATH


# ---------------------------------------------------------------------------
# Abandonment: a barren target with more candidates than the abandon bound
# ---------------------------------------------------------------------------


_MANY_SOURCE = (
    "def many(a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7, i=8, j=9, k=10, l=11):\n"
    "    return a\n"
)
_MANY_PATH = Path("src/mylib/many.py")


@pytest.fixture()
def barren_repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "mylib").mkdir(parents=True)
    (tmp_path / _MANY_PATH).write_text(_MANY_SOURCE)
    return tmp_path


def test_a_fully_barren_target_is_abandoned_after_the_bound_not_ground_through(
    barren_repo: Path,
):
    # _MANY_SOURCE offers 13 real candidates (proven separately below) --
    # more than _TARGET_ABANDON_AFTER -- and this runner never reports
    # breakage for any of them. Fails for an implementation that tries all
    # 13 anyway (defeats invariant 13's whole purpose) or that abandons too
    # early (before _TARGET_ABANDON_AFTER unproductive tries).
    calls: list[int] = []

    def blind_runner(r: Path, package: str) -> str:
        calls.append(1)
        return _marker(r) + "0 failed, 1 passed\n"

    total_candidates = len(candidates(_MANY_SOURCE, _MANY_PATH))
    assert total_candidates > _TARGET_ABANDON_AFTER  # the premise this test needs

    result = generate_corpus(
        barren_repo, [_MANY_PATH], _BASE, blind_runner,
        max_records=50, time_budget=60.0, source_repo="r", source_sha="s",
    )
    assert len(calls) == _TARGET_ABANDON_AFTER
    outcome = result.targets[0]
    assert outcome.tried == _TARGET_ABANDON_AFTER
    assert outcome.kept == 0
    assert outcome.abandoned is True
    assert outcome.proposed == total_candidates
    assert any("abandoned as barren" in d for d in result.dropped)


def test_a_target_tried_to_completion_below_the_abandon_bound_is_not_abandoned(repo: Path):
    # THE plan's own measured fact, in spirit: mutating context/scope.py
    # kept 0 of 7 -- all seven genuinely tried, none cut short. A source
    # offering fewer candidates than the abandon bound must run to
    # completion and report barren, not "abandoned". `x = a` (no `return`)
    # keeps this to exactly the 7 off_by_one candidates, with no extra
    # dropped_return candidate to make the count depend on that operator
    # too.
    seven_source = (
        "def f(a=0, b=1, c=2, d=3, e=4, f=5, g=6):\n"
        "    x = a\n"
    )
    seven_path = Path("src/mylib/seven.py")
    (repo / seven_path).write_text(seven_source)
    total = len(candidates(seven_source, seven_path))
    assert total == 7
    assert total < _TARGET_ABANDON_AFTER

    def blind_runner(r: Path, package: str) -> str:
        return _marker(r) + "0 failed, 1 passed\n"

    result = generate_corpus(
        repo, [seven_path], _BASE, blind_runner,
        max_records=50, time_budget=60.0, source_repo="r", source_sha="s",
    )
    outcome = result.targets[0]
    assert (outcome.tried, outcome.kept, outcome.abandoned) == (7, 0, False)
    assert not any("abandoned" in d for d in result.dropped)


# ---------------------------------------------------------------------------
# Global stopping rules: max_records and time_budget
# ---------------------------------------------------------------------------


def test_max_records_stops_generation_the_moment_it_is_reached(repo: Path):
    # calc.py's candidate is kept; other.py's own candidate would be too
    # (both driven by the SAME "differs from original -> 1 failed" rule
    # here, unlike the earlier fixture) -- with max_records=1, only the
    # FIRST kept record may exist, and the run must stop calling the
    # runner immediately afterward rather than continuing to evaluate
    # other.py's candidate too.
    calls: list[Path] = []

    def eager_runner(r: Path, package: str) -> str:
        calc = (r / _CALC_PATH).read_text()
        other = (r / _OTHER_PATH).read_text()
        calls.append(1)
        if calc != _CALC_SOURCE or other != _OTHER_SOURCE:
            return _marker(r) + (
                "FAILED tests/test_x.py::test_y - AssertionError\n"
                "1 failed, 1 passed\n"
            )
        return _marker(r) + "0 failed, 2 passed\n"

    result = generate_corpus(
        repo, [_CALC_PATH, _OTHER_PATH], _BASE, eager_runner,
        max_records=1, time_budget=60.0, source_repo="r", source_sha="s",
    )
    assert len(result.records) == 1
    assert len(calls) == 1  # stopped before a second verify() call
    assert any("max_records" in d for d in result.dropped)


def test_time_budget_stops_generation_the_moment_it_is_exceeded(repo: Path):
    def slow_survivor(r: Path, package: str) -> str:
        time.sleep(0.05)
        return _marker(r) + "0 failed, 2 passed\n"

    result = generate_corpus(
        repo, [_CALC_PATH, _OTHER_PATH], _BASE, slow_survivor,
        max_records=50, time_budget=0.01, source_repo="r", source_sha="s",
    )
    assert any("time budget" in d for d in result.dropped)
    # At least one target never got a chance to run at all.
    total_tried = sum(t.tried for t in result.targets)
    assert total_tried < 2  # candidates() sees 1 real candidate per target


def test_seconds_reflects_real_wall_clock_not_a_placeholder(repo: Path):
    def slow_runner(r: Path, package: str) -> str:
        time.sleep(0.05)
        return _marker(r) + "0 failed, 2 passed\n"

    result = generate_corpus(
        repo, [_CALC_PATH], _BASE, slow_runner,
        max_records=50, time_budget=60.0, source_repo="r", source_sha="s",
    )
    assert result.seconds >= 0.05


# ---------------------------------------------------------------------------
# I6 (whole-branch review 2026-08-10) -- one hanging candidate must not
# discard every record already verified
# ---------------------------------------------------------------------------


def test_a_timing_out_candidate_is_dropped_and_generation_continues_to_the_next_one(
    repo: Path,
):
    # Reachable in robigo's own source (`inverted_condition` on a `while
    # ... and not ...:` loop); stood in for here with calc.py's own two
    # real candidates -- one raises `TimeoutExpired`, the other lands a
    # real keep. Before I6, the exception would propagate straight out of
    # `generate_corpus` (and past `cli.corpus_main`'s own handler, which
    # returns before `write_corpus` is ever called), losing this run
    # entirely rather than just the one candidate.
    off_by_one = next(
        m for m in candidates(_CALC_SOURCE, _CALC_PATH) if m.operator == "off_by_one"
    )
    dropped_return = next(
        m for m in candidates(_CALC_SOURCE, _CALC_PATH) if m.operator == "dropped_return"
    )
    off_by_one_broken = _mutated_file(_CALC_SOURCE, off_by_one)
    dropped_return_broken = _mutated_file(_CALC_SOURCE, dropped_return)

    def runner(r: Path, package: str) -> str:
        calc = (r / _CALC_PATH).read_text()
        if calc == off_by_one_broken:
            raise subprocess.TimeoutExpired(cmd=["pytest", "-q"], timeout=300)
        if calc == dropped_return_broken:
            return _marker(r) + (
                "FAILED tests/test_calc.py::test_bump - AssertionError\n"
                "1 failed, 1 passed in 0.01s\n"
            )
        return _marker(r) + "0 failed, 2 passed in 0.01s\n"

    result = generate_corpus(
        repo, [_CALC_PATH], _BASE, runner,
        max_records=50, time_budget=60.0, source_repo="r", source_sha="s",
    )
    # The timed-out candidate produces no record, but is not silently
    # absent either, and the OTHER real candidate on the same file is
    # still tried and kept -- generation did not abort.
    assert len(result.records) == 1
    assert result.records[0].operator == "dropped_return"
    assert any("timed out" in d and "I6" in d for d in result.dropped)
    assert any("off_by_one" in d for d in result.dropped)
    outcome = result.targets[0]
    assert outcome.tried == 2
    assert outcome.kept == 1


def test_records_from_an_earlier_target_survive_a_later_targets_timeout(repo: Path):
    # The literal claim: "one hanging mutant discards every record
    # produced so far" -- other.py's candidate is kept FIRST (targets are
    # walked in order), then EVERY one of calc.py's own candidates times
    # out. The already-verified other.py record must still be in the
    # result -- exactly what propagating the exception straight out of
    # `generate_corpus` would have lost.
    other_off_by_one = next(
        m for m in candidates(_OTHER_SOURCE, _OTHER_PATH) if m.operator == "off_by_one"
    )
    other_off_by_one_broken = _mutated_file(_OTHER_SOURCE, other_off_by_one)

    def runner(r: Path, package: str) -> str:
        calc = (r / _CALC_PATH).read_text()
        if calc != _CALC_SOURCE:
            raise subprocess.TimeoutExpired(cmd=["pytest", "-q"], timeout=300)
        other = (r / _OTHER_PATH).read_text()
        if other == other_off_by_one_broken:
            return _marker(r) + (
                "FAILED tests/test_other.py::test_double - AssertionError\n"
                "1 failed, 1 passed in 0.01s\n"
            )
        return _marker(r) + "0 failed, 2 passed in 0.01s\n"

    result = generate_corpus(
        repo, [_OTHER_PATH, _CALC_PATH], _BASE, runner,
        max_records=50, time_budget=60.0, source_repo="r", source_sha="s",
    )
    assert len(result.records) == 1
    assert result.records[0].path == _OTHER_PATH
    assert result.records[0].operator == "off_by_one"
    assert any("timed out" in d and "I6" in d for d in result.dropped)


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


def test_target_outcome_is_frozen():
    outcome = TargetOutcome(Path("a.py"), 1, 1, 1, False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.kept = 0  # type: ignore[misc]


def test_generation_result_is_frozen():
    result = GenerationResult((), (), (), 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.seconds = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# render_report — content, not merely "something was printed"
# (verification standard item 5)
# ---------------------------------------------------------------------------


def _result() -> GenerationResult:
    return GenerationResult(
        records=(),
        dropped=("context/scope.py: abandoned as barren after 10 unproductive candidates",),
        targets=(
            TargetOutcome(Path("context/scope.py"), 7, 7, 0, False),
            TargetOutcome(Path("src/mylib/calc.py"), 5, 5, 4, False),
        ),
        seconds=12.5,
    )


def test_render_report_states_the_keep_rate_per_target():
    text = render_report(_result(), name="mylib-v1")
    assert "target context/scope.py  0/7 kept" in text
    assert "target src/mylib/calc.py  4/5 kept" in text


def test_render_report_states_totals_and_wall_clock():
    text = render_report(_result(), name="mylib-v1")
    assert "proposed 12" in text  # 7 + 5
    assert "tried 12" in text
    assert "kept 4" in text
    assert "rejected 8" in text
    assert "12.5s" in text


def test_render_report_states_which_phases_the_wall_clock_excludes():
    # I3 (whole-branch review 2026-08-10): the printed wall-clock must
    # state its own scope -- `result.seconds` covers only this module's
    # candidate loop, not the clone/sentinel/baseline phases that run
    # earlier in `cli.corpus_main` and are never threaded into
    # `GenerationResult` at all.
    text = render_report(_result(), name="mylib-v1")
    assert "excludes" in text
    assert "clone" in text
    assert "sentinel" in text
    assert "baseline" in text


def test_render_report_names_the_corpus_and_carries_dropped_reasons_verbatim():
    text = render_report(_result(), name="mylib-v1")
    assert "corpus mylib-v1" in text
    assert "context/scope.py: abandoned as barren after 10 unproductive candidates" in text


def test_render_report_flags_an_abandoned_target_distinctly():
    result = GenerationResult(
        records=(), dropped=(),
        targets=(TargetOutcome(Path("hot.py"), 50, 10, 0, True),),
        seconds=1.0,
    )
    text = render_report(result, name="v1")
    assert "abandoned as barren" in text


def test_render_report_would_fail_if_the_rate_were_hardcoded():
    # Guards against a vacuous "prints SOME percentage" implementation:
    # two DIFFERENT result objects must render DIFFERENT keep-rate text.
    barren = GenerationResult(
        records=(), dropped=(), targets=(TargetOutcome(Path("a.py"), 3, 3, 0, False),),
        seconds=1.0,
    )
    productive = GenerationResult(
        records=(), dropped=(), targets=(TargetOutcome(Path("a.py"), 3, 3, 3, False),),
        seconds=1.0,
    )
    assert render_report(barren, name="v1") != render_report(productive, name="v1")
    assert "0/3 kept" in render_report(barren, name="v1")
    assert "3/3 kept" in render_report(productive, name="v1")
