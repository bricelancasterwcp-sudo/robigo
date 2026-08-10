# tests/test_corpus_verify.py
from __future__ import annotations

import dataclasses
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from robigo.profile.corpus import Mutant
from robigo.profile.verify import (
    Baseline,
    Verdict,
    WrongTreeError,
    _assert_in_clone,
    _broken_count,
    _broken_ids,
    _find_line,
    _module_path,
    _resolve_in_clone,
    baseline,
    pytest_runner,
    sentinel_ok,
    verify,
)

# ---------------------------------------------------------------------------
# Offline guarantee: every test in this module runs with real sockets
# blocked. `pytest_runner` is the only production code here that touches
# subprocess, and its own tests fake `subprocess.run` entirely -- nothing
# in this file should ever need a real socket, so any attempt is a bug.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("socket.socket.connect must never be called in this test suite")

    monkeypatch.setattr(socket.socket, "connect", _blocked)


# ---------------------------------------------------------------------------
# Fixture repo: a minimal on-disk tree shaped enough like a clone of robigo
# for `sentinel_ok`/`baseline`/`verify` to act on for real -- real file
# reads/writes inside tmp_path, never the working tree, and never anything
# that runs actual pytest.
# ---------------------------------------------------------------------------

BUDGET_SOURCE = (
    "CHARS_PER_TOKEN = 3.3\n"
    "\n"
    "\n"
    "def estimate_tokens(text: str) -> int:\n"
    "    return int(len(text) / CHARS_PER_TOKEN) + 1\n"
)
TARGET_RELATIVE = Path("src/robigo/other/thing.py")
TARGET_SOURCE = "def f():\n    return 1\n"


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "robigo" / "context").mkdir(parents=True)
    (tmp_path / "src" / "robigo" / "context" / "budget.py").write_text(BUDGET_SOURCE)
    (tmp_path / "src" / "robigo" / "other").mkdir(parents=True, exist_ok=True)
    (tmp_path / TARGET_RELATIVE).write_text(TARGET_SOURCE)
    return tmp_path


def _output(text: str, *, module_path: object) -> str:
    """Builds runner output the way a well-formed runner does: a
    `ROBIGO_MODULE=` marker line followed by pytest-shaped text.
    `module_path=None` omits the marker entirely -- a runner that never
    says where it ran, exercised by the dedicated "no marker" tests."""
    if module_path is None:
        return text
    return f"ROBIGO_MODULE={module_path}\n{text}"


def _inside(repo: Path) -> Path:
    return repo / "src" / "robigo" / "__init__.py"


def _outside(repo: Path) -> Path:
    return repo.parent / "definitely-not-the-clone" / "src" / "robigo" / "__init__.py"


# ---------------------------------------------------------------------------
# Baseline / Verdict — frozen, and the field carries what it claims to
# ---------------------------------------------------------------------------


def test_baseline_is_frozen():
    # Fails if `frozen=True` is dropped from the dataclass decorator.
    b = Baseline(broken=0, seconds=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.broken = 1  # type: ignore[misc]


def test_verdict_is_frozen():
    # Fails if `frozen=True` is dropped from the dataclass decorator.
    v = Verdict(kept=True, failures=1, test_id="x", reason="y")
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.kept = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _broken_count — invariant 5's arithmetic, pinned directly
# ---------------------------------------------------------------------------


def test_broken_count_sums_failed_and_error_independently():
    # Fails for an implementation using one combined regex requiring both
    # clauses present, or one that only reads the first number it finds.
    assert _broken_count("2 failed, 1 error, 400 passed in 3.21s") == 3


def test_broken_count_handles_an_error_only_report_with_no_failed_substring_at_all():
    # THE brief's required acceptance test for invariant 5: this exact
    # input scored as clean (0) in the implementer's own first
    # measurement. Fails for an implementation that requires "failed" to
    # be present before it will look for "error" at all.
    assert _broken_count("1 error in 0.01s") == 1
    assert "failed" not in "1 error in 0.01s"


def test_broken_count_is_zero_on_a_clean_report():
    # Fails for an implementation that misreads a passing summary as
    # nonzero (e.g. matching digits from "430 passed").
    assert _broken_count("430 passed in 12.34s") == 0


# ---------------------------------------------------------------------------
# _broken_ids — invariant 6's identity extraction, pinned directly
# ---------------------------------------------------------------------------


def test_broken_ids_extracts_failed_and_error_node_ids_in_order():
    # Fails for an implementation that only recognises FAILED or only
    # ERROR, or that captures the trailing " - reason" text as part of
    # the id.
    text = (
        "FAILED tests/test_a.py::test_one - AssertionError: boom\n"
        "ERROR tests/test_b.py::test_two\n"
        "2 failed, 1 error, 400 passed in 3.21s\n"
    )
    assert _broken_ids(text) == ("tests/test_a.py::test_one", "tests/test_b.py::test_two")


def test_broken_ids_is_empty_on_a_clean_report():
    # Fails for an implementation that returns a spurious id from summary
    # text that isn't actually a FAILED/ERROR line.
    assert _broken_ids("430 passed in 12.34s") == ()


def test_broken_ids_requires_the_marker_at_the_start_of_the_line():
    # Fails for an implementation using a bare substring search instead of
    # a line-anchored one -- "the FAILED build" here is prose, not a
    # pytest summary line, and must not be read as one.
    text = "note: the FAILED build from yesterday is unrelated\n430 passed in 1.00s\n"
    assert _broken_ids(text) == ()


# ---------------------------------------------------------------------------
# _module_path / _assert_in_clone — invariant 7's parsing and check
# ---------------------------------------------------------------------------


def test_module_path_returns_none_when_no_marker_is_present():
    # Fails for an implementation that raises or returns a bogus default
    # instead of a clean "not reported at all" signal.
    assert _module_path("430 passed in 1.00s\n") is None


def test_module_path_resolves_the_marked_path(repo: Path):
    resolved = _module_path(f"ROBIGO_MODULE={_inside(repo)}\n430 passed\n")
    assert resolved == _inside(repo).resolve()


def test_assert_in_clone_passes_for_a_path_inside_the_repo(repo: Path):
    _assert_in_clone(_output("0 failed\n", module_path=_inside(repo)), repo)  # must not raise


def test_assert_in_clone_raises_for_a_path_outside_the_repo(repo: Path):
    # Fails for an implementation that trusts any marker present without
    # checking it actually resolves inside `repo`.
    with pytest.raises(WrongTreeError):
        _assert_in_clone(_output("0 failed\n", module_path=_outside(repo)), repo)


def test_assert_in_clone_raises_when_no_marker_is_present(repo: Path):
    # Fails for an implementation that defaults to trusting an unmarked
    # runner instead of treating "didn't say" the same as "said wrong".
    with pytest.raises(WrongTreeError):
        _assert_in_clone("0 failed, 430 passed in 1.00s\n", repo)


# ---------------------------------------------------------------------------
# _resolve_in_clone — the pathlib-join trap, and traversal
# ---------------------------------------------------------------------------


def test_resolve_in_clone_places_a_relative_path_under_the_repo(repo: Path):
    assert _resolve_in_clone(repo, TARGET_RELATIVE) == (repo / TARGET_RELATIVE).resolve()


def test_resolve_in_clone_rejects_an_absolute_path(repo: Path):
    # First, prove the trap is real: pathlib's `/` operator discards the
    # left operand when the right one is absolute, so a naive
    # `repo / mutant.path` would silently point outside the clone.
    assert repo / Path("/etc/passwd") == Path("/etc/passwd")
    with pytest.raises(ValueError):
        _resolve_in_clone(repo, Path("/etc/passwd"))


def test_resolve_in_clone_rejects_a_relative_path_that_escapes_via_traversal(repo: Path):
    # Fails for an implementation relying solely on `is_absolute()`,
    # which this input does not trip -- only the post-join
    # `is_relative_to` check catches it.
    with pytest.raises(ValueError):
        _resolve_in_clone(repo, Path("../outside.py"))


# ---------------------------------------------------------------------------
# sentinel_ok — invariant 4
# ---------------------------------------------------------------------------


def test_sentinel_ok_returns_true_when_the_runner_reports_real_breakage(repo: Path):
    def runner(r: Path) -> str:
        return _output("18 failed, 412 passed in 4.00s\n", module_path=_inside(r))

    assert sentinel_ok(repo, runner) is True
    # And the file must be restored, byte-identical, afterward.
    assert (repo / "src" / "robigo" / "context" / "budget.py").read_text() == BUDGET_SOURCE


def test_sentinel_ok_returns_false_for_a_blind_harness_that_always_reports_zero(repo: Path):
    # THE brief's required acceptance test for invariant 4: a harness that
    # never reports breakage, no matter what was applied, must not be
    # trusted -- measured 2026-08-10, a harness with this exact defect
    # scored 8 of 8 real mutants as survivors.
    def blind_runner(r: Path) -> str:
        return _output("0 failed, 430 passed in 4.00s\n", module_path=_inside(r))

    assert sentinel_ok(repo, blind_runner) is False
    assert (repo / "src" / "robigo" / "context" / "budget.py").read_text() == BUDGET_SOURCE


def test_sentinel_ok_returns_false_when_the_breakage_is_reported_on_the_wrong_tree(repo: Path):
    # Fails for an implementation that only checks `_broken_count > 0`
    # and never checks whose tree the breakage was reported against.
    def wrong_tree_runner(r: Path) -> str:
        return _output("18 failed, 412 passed in 4.00s\n", module_path=_outside(r))

    assert sentinel_ok(repo, wrong_tree_runner) is False
    assert (repo / "src" / "robigo" / "context" / "budget.py").read_text() == BUDGET_SOURCE


def test_sentinel_ok_raises_if_its_target_line_has_moved_or_changed(repo: Path):
    # Fails for an implementation that silently no-ops (and therefore
    # tautologically "detects no breakage") instead of surfacing that its
    # own hardcoded sentinel no longer matches the file it targets.
    (repo / "src" / "robigo" / "context" / "budget.py").write_text(
        "def estimate_tokens(text: str) -> int:\n    return 999\n"
    )
    with pytest.raises(RuntimeError):
        sentinel_ok(repo, lambda r: _output("0 failed\n", module_path=_inside(r)))


def test_sentinel_ok_restores_the_file_even_when_the_runner_raises(repo: Path):
    # Failure injection on the restore path (CARRIED-DEBT's own named
    # pattern: "write_atomic's cleanup path under failure injection").
    def exploding_runner(r: Path) -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        sentinel_ok(repo, exploding_runner)
    assert (repo / "src" / "robigo" / "context" / "budget.py").read_text() == BUDGET_SOURCE


def test_a_disciplined_caller_would_have_aborted_on_the_measured_blind_harness(repo: Path):
    """Verification standard item 3, told as the story it actually is.
    A blind runner (always reports zero breakage) is run through BOTH
    `sentinel_ok` and, bypassing the check a real caller would never
    bypass, `verify` on a real mutant -- proving `verify` alone, with no
    sentinel gate in front of it, reports exactly the measured false
    negative ("survived" for a mutant that never really ran under a
    working harness), and that `sentinel_ok` on that SAME repo/runner
    correctly returns False, which is what a disciplined caller checks
    BEFORE trusting any `verify` result at all."""

    def blind_runner(r: Path) -> str:
        return _output("0 failed, 430 passed in 4.00s\n", module_path=_inside(r))

    mutant = Mutant(TARGET_RELATIVE, 2, "    return 1\n", "    return 2\n", "off_by_one")
    base = Baseline(broken=0, seconds=1.0)
    verdict = verify(mutant, repo, base, blind_runner)
    assert verdict.kept is False
    assert "survived" in verdict.reason

    assert sentinel_ok(repo, blind_runner) is False


# ---------------------------------------------------------------------------
# baseline — invariant 5
# ---------------------------------------------------------------------------


def test_baseline_counts_failures_and_errors_combined(repo: Path):
    def runner(r: Path) -> str:
        return _output("3 failed, 2 error, 425 passed in 5.00s\n", module_path=_inside(r))

    assert baseline(repo, runner).broken == 5


def test_baseline_is_not_assumed_zero(repo: Path):
    # Measured 2026-08-10: a git-archive copy of this project's own repo
    # baselined at 6. Fails for an implementation that hardcodes 0.
    def runner(r: Path) -> str:
        return _output("6 failed, 400 passed in 5.00s\n", module_path=_inside(r))

    assert baseline(repo, runner).broken == 6


def test_baseline_counts_an_error_only_report_with_no_failed_substring_at_all(repo: Path):
    # THE brief's required acceptance test, at the public `baseline()`
    # level rather than just the private `_broken_count` unit.
    def runner(r: Path) -> str:
        return _output("1 error in 0.01s\n", module_path=_inside(r))

    result = baseline(repo, runner)
    assert result.broken == 1


def test_baseline_measures_real_wall_clock_seconds(repo: Path):
    # Fails for an implementation that reports 0.0 unconditionally, or
    # that tries to parse a "seconds" figure out of the runner's own text
    # instead of timing the call itself.
    def slow_runner(r: Path) -> str:
        time.sleep(0.05)
        return _output("0 failed, 430 passed in 0.01s\n", module_path=_inside(r))

    result = baseline(repo, slow_runner)
    assert result.seconds >= 0.05


def test_baseline_raises_wrong_tree_error_for_a_path_outside_the_repo(repo: Path):
    def runner(r: Path) -> str:
        return _output("0 failed, 430 passed\n", module_path=_outside(r))

    with pytest.raises(WrongTreeError):
        baseline(repo, runner)


def test_baseline_raises_wrong_tree_error_when_no_marker_is_present(repo: Path):
    with pytest.raises(WrongTreeError):
        baseline(repo, lambda r: "0 failed, 430 passed in 1.00s\n")


# ---------------------------------------------------------------------------
# verify — invariants 5, 6, 7
# ---------------------------------------------------------------------------


def _mutant() -> Mutant:
    return Mutant(TARGET_RELATIVE, 2, "    return 1\n", "    return 9\n", "off_by_one")


def test_verify_keeps_a_mutant_with_exactly_one_net_new_failure_and_records_its_id(repo: Path):
    # The "1" case, invariant 6's core: the id must be captured, not just
    # the count.
    def runner(r: Path) -> str:
        return _output(
            "FAILED tests/test_x.py::test_y - AssertionError\n1 failed, 429 passed in 2.00s\n",
            module_path=_inside(r),
        )

    verdict = verify(_mutant(), repo, Baseline(broken=0, seconds=1.0), runner)
    assert verdict.kept is True
    assert verdict.failures == 1
    assert verdict.test_id == "tests/test_x.py::test_y"


def test_verify_rejects_a_mutant_with_zero_net_new_failures(repo: Path):
    # The "0" case.
    def runner(r: Path) -> str:
        return _output("0 failed, 430 passed in 2.00s\n", module_path=_inside(r))

    verdict = verify(_mutant(), repo, Baseline(broken=0, seconds=1.0), runner)
    assert verdict.kept is False
    assert verdict.test_id is None
    assert verdict.failures == 0
    assert "survived" in verdict.reason


def test_verify_rejects_a_mutant_that_breaks_many_tests(repo: Path):
    # The "many" case, using the exact figure ("18 failed") measured
    # 2026-08-10 for a deliberate central-code sentinel.
    def runner(r: Path) -> str:
        lines = "\n".join(f"FAILED tests/test_x.py::test_{i}" for i in range(18))
        return _output(f"{lines}\n18 failed, 412 passed in 4.00s\n", module_path=_inside(r))

    verdict = verify(_mutant(), repo, Baseline(broken=0, seconds=1.0), runner)
    assert verdict.kept is False
    assert verdict.test_id is None
    assert verdict.failures == 18
    assert "too many" in verdict.reason


def test_verify_does_not_keep_when_a_nonzero_baseline_makes_the_new_failure_ambiguous(repo: Path):
    # Measured 2026-08-10: a baseline of 6 pre-existing broken tests. Net
    # is 1 (7 - 6), but a real runner's report names all 7 currently-
    # broken tests, not a delta -- so the new one cannot be isolated by id
    # without guessing, and invariant 6 requires the id, not just the
    # count, so this must NOT be kept.
    def runner(r: Path) -> str:
        lines = "\n".join(f"FAILED tests/test_x.py::test_{i}" for i in range(7))
        return _output(f"{lines}\n7 failed, 423 passed in 4.00s\n", module_path=_inside(r))

    verdict = verify(_mutant(), repo, Baseline(broken=6, seconds=1.0), runner)
    assert verdict.kept is False
    assert verdict.test_id is None
    assert verdict.failures == 7
    assert "cannot isolate" in verdict.reason


def test_verify_rejects_a_survivor_reported_on_the_wrong_tree_rather_than_calling_it_survived(
    repo: Path,
):
    # A "0 failed" report that would otherwise read as a clean survivor,
    # but the module path is outside the clone.
    def runner(r: Path) -> str:
        return _output("0 failed, 430 passed in 2.00s\n", module_path=_outside(r))

    verdict = verify(_mutant(), repo, Baseline(broken=0, seconds=1.0), runner)
    assert verdict.kept is False
    assert "survived" not in verdict.reason
    assert "clone" in verdict.reason or "tree" in verdict.reason.lower()


def test_verify_rejects_a_looks_kept_result_reported_on_the_wrong_tree(repo: Path):
    # The more dangerous case: a report that would otherwise satisfy
    # invariant 6 outright (exactly one id, exactly one net new failure),
    # but on the wrong tree -- must not be reported as kept.
    def runner(r: Path) -> str:
        return _output(
            "FAILED tests/test_x.py::test_y - AssertionError\n1 failed, 429 passed in 2.00s\n",
            module_path=_outside(r),
        )

    verdict = verify(_mutant(), repo, Baseline(broken=0, seconds=1.0), runner)
    assert verdict.kept is False
    assert verdict.test_id is None


def test_verify_raises_valueerror_for_an_absolute_mutant_path(repo: Path):
    bad = Mutant(Path("/etc/passwd"), 1, "a\n", "b\n", "off_by_one")
    with pytest.raises(ValueError):
        verify(bad, repo, Baseline(broken=0, seconds=1.0), lambda r: "430 passed\n")


def test_verify_raises_valueerror_for_a_mutant_path_that_escapes_via_traversal(repo: Path):
    bad = Mutant(Path("../outside.py"), 1, "a\n", "b\n", "off_by_one")
    with pytest.raises(ValueError):
        verify(bad, repo, Baseline(broken=0, seconds=1.0), lambda r: "430 passed\n")


def test_verify_actually_writes_the_mutation_before_the_runner_is_called(repo: Path):
    # Not just that the Verdict's shape looks right afterward -- that the
    # file on disk, AT THE MOMENT the runner ran, held `mutant.mutated`.
    seen: dict[str, str] = {}

    def spy_runner(r: Path) -> str:
        seen["content"] = (r / TARGET_RELATIVE).read_text()
        return _output("0 failed, 430 passed in 1.00s\n", module_path=_inside(r))

    m = _mutant()
    verify(m, repo, Baseline(broken=0, seconds=1.0), spy_runner)
    assert seen["content"] == TARGET_SOURCE.replace(m.original, m.mutated)


def test_verify_restores_the_file_after_a_kept_result(repo: Path):
    def runner(r: Path) -> str:
        return _output(
            "FAILED tests/test_x.py::test_y - AssertionError\n1 failed, 429 passed in 2.00s\n",
            module_path=_inside(r),
        )

    verify(_mutant(), repo, Baseline(broken=0, seconds=1.0), runner)
    assert (repo / TARGET_RELATIVE).read_text() == TARGET_SOURCE


def test_verify_restores_the_file_after_a_rejected_result(repo: Path):
    def runner(r: Path) -> str:
        return _output("0 failed, 430 passed in 2.00s\n", module_path=_inside(r))

    verify(_mutant(), repo, Baseline(broken=0, seconds=1.0), runner)
    assert (repo / TARGET_RELATIVE).read_text() == TARGET_SOURCE


def test_verify_restores_the_file_even_when_the_runner_raises(repo: Path):
    def exploding_runner(r: Path) -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        verify(_mutant(), repo, Baseline(broken=0, seconds=1.0), exploding_runner)
    assert (repo / TARGET_RELATIVE).read_text() == TARGET_SOURCE


def test_verify_uses_apply_and_raises_if_the_mutants_original_does_not_match_the_clone(repo: Path):
    # Reuses corpus.py's own `_apply` guard (task 1) rather than a second
    # copy -- proven here by feeding a mutant whose `original` does not
    # match what's actually on disk in the clone.
    stale = Mutant(TARGET_RELATIVE, 2, "    return 999\n", "    return 2\n", "off_by_one")
    with pytest.raises(ValueError):
        verify(stale, repo, Baseline(broken=0, seconds=1.0), lambda r: "430 passed\n")


# ---------------------------------------------------------------------------
# _find_line
# ---------------------------------------------------------------------------


def test_find_line_locates_the_single_matching_line():
    assert _find_line("a\nb\nc\n", "b\n") == 2


def test_find_line_raises_when_the_text_is_absent():
    with pytest.raises(RuntimeError):
        _find_line("a\nb\nc\n", "z\n")


def test_find_line_raises_when_the_text_appears_more_than_once():
    with pytest.raises(RuntimeError):
        _find_line("a\nb\na\n", "a\n")


# ---------------------------------------------------------------------------
# pytest_runner — the real runner, exercised with `subprocess.run` faked so
# no real pytest ever executes inside this test suite.
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_pytest_runner_never_invokes_a_real_subprocess(
    monkeypatch: pytest.MonkeyPatch, repo: Path
):
    # Establishes the harness this whole section relies on: `subprocess
    # .run` is replaced before `pytest_runner` is called at all.
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> _FakeCompleted:
        calls.append((cmd, kwargs))
        if "-c" in cmd:
            return _FakeCompleted(stdout=str(repo / "src" / "robigo" / "__init__.py"))
        return _FakeCompleted(stdout="0 failed, 430 passed in 4.00s\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pytest_runner(repo)
    assert len(calls) == 2


def test_pytest_runner_sets_pythondontwritebytecode(monkeypatch: pytest.MonkeyPatch, repo: Path):
    # Stale bytecode has confused two implementers on this project
    # (constraints section) -- pin this env var directly. The ambient
    # environment this test suite itself runs under may already carry
    # this variable (this module's own verification runs did), which
    # would let a `pytest_runner` that forgot to set it pass by accident
    # via `os.environ.copy()` -- cleared explicitly so the assertion can
    # only pass if `pytest_runner` sets it itself.
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> _FakeCompleted:
        captured.setdefault("env", kwargs.get("env"))
        if "-c" in cmd:
            return _FakeCompleted(stdout=str(repo / "src" / "robigo" / "__init__.py"))
        return _FakeCompleted(stdout="0 failed, 430 passed\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pytest_runner(repo)
    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"  # type: ignore[index]


def test_pytest_runner_forces_pythonpath_to_the_clones_src(
    monkeypatch: pytest.MonkeyPatch, repo: Path
):
    # The editable-install trap, measured 2026-08-10: without this, a
    # subprocess started inside a copied tree still imports the real
    # repo's source and every mutant appears to survive. Cleared for the
    # same ambient-leakage reason as the PYTHONDONTWRITEBYTECODE test
    # above -- this process may already have a PYTHONPATH set.
    monkeypatch.delenv("PYTHONPATH", raising=False)
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> _FakeCompleted:
        captured.setdefault("env", kwargs.get("env"))
        if "-c" in cmd:
            return _FakeCompleted(stdout=str(repo / "src" / "robigo" / "__init__.py"))
        return _FakeCompleted(stdout="0 failed, 430 passed\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pytest_runner(repo)
    assert captured["env"]["PYTHONPATH"] == str(repo / "src")  # type: ignore[index]


def test_pytest_runner_returns_the_module_marker_followed_by_pytest_output(
    monkeypatch: pytest.MonkeyPatch, repo: Path
):
    module_file = repo / "src" / "robigo" / "__init__.py"

    def fake_run(cmd: list[str], **kwargs: object) -> _FakeCompleted:
        if "-c" in cmd:
            return _FakeCompleted(stdout=str(module_file) + "\n")
        return _FakeCompleted(stdout="1 failed, 429 passed in 2.00s\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    text = pytest_runner(repo)
    assert text.startswith(f"ROBIGO_MODULE={module_file}\n")
    assert "1 failed, 429 passed in 2.00s" in text
    _assert_in_clone(text, repo)  # must not raise -- proves the marker round-trips


def test_pytest_runner_omits_the_marker_when_the_import_check_fails(
    monkeypatch: pytest.MonkeyPatch, repo: Path
):
    # A broken clone (uninstalled package) must not fabricate a marker --
    # downstream `_assert_in_clone` then correctly rejects the run for
    # having no marker at all, rather than trusting a guessed path.
    def fake_run(cmd: list[str], **kwargs: object) -> _FakeCompleted:
        if "-c" in cmd:
            return _FakeCompleted(
                stderr="ModuleNotFoundError: no module named robigo", returncode=1
            )
        return _FakeCompleted(stdout="0 failed, 430 passed\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    text = pytest_runner(repo)
    assert "ROBIGO_MODULE=" not in text
    with pytest.raises(WrongTreeError):
        _assert_in_clone(text, repo)


def test_pytest_runner_combines_stdout_and_stderr_of_the_pytest_invocation(
    monkeypatch: pytest.MonkeyPatch, repo: Path
):
    def fake_run(cmd: list[str], **kwargs: object) -> _FakeCompleted:
        if "-c" in cmd:
            return _FakeCompleted(stdout=str(repo / "src" / "robigo" / "__init__.py"))
        return _FakeCompleted(stdout="1 failed\n", stderr="internal warning: something\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    text = pytest_runner(repo)
    assert "1 failed" in text
    assert "internal warning: something" in text


def test_pytest_runner_passes_short_summary_flags_pytest_would_need_for_ids(
    monkeypatch: pytest.MonkeyPatch, repo: Path
):
    # `_broken_ids` depends on the short test summary info section, which
    # requires `-rfE` (or equivalent) actually being passed to pytest.
    seen_pytest_cmd: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> _FakeCompleted:
        if "-c" in cmd:
            return _FakeCompleted(stdout=str(repo / "src" / "robigo" / "__init__.py"))
        seen_pytest_cmd.extend(cmd)
        return _FakeCompleted(stdout="0 failed, 430 passed\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pytest_runner(repo)
    assert "-rfE" in seen_pytest_cmd
    assert any("pytest" in part for part in seen_pytest_cmd) or "-m" in seen_pytest_cmd


def test_pytest_runner_uses_a_bounded_timeout_for_both_subprocess_calls(
    monkeypatch: pytest.MonkeyPatch, repo: Path
):
    # A hung subprocess must not hang this module forever.
    timeouts: list[object] = []

    def fake_run(cmd: list[str], **kwargs: object) -> _FakeCompleted:
        timeouts.append(kwargs.get("timeout"))
        if "-c" in cmd:
            return _FakeCompleted(stdout=str(repo / "src" / "robigo" / "__init__.py"))
        return _FakeCompleted(stdout="0 failed, 430 passed\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pytest_runner(repo)
    assert len(timeouts) == 2
    assert all(isinstance(t, (int, float)) and t > 0 for t in timeouts)


def test_pytest_runner_runs_python_as_the_interpreter_that_launched_this_process(
    monkeypatch: pytest.MonkeyPatch, repo: Path
):
    # Uses `sys.executable`, not a bare "python"/"python3" that could
    # resolve to a different interpreter (and a different, possibly
    # non-editable-install, environment) than this one.
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> _FakeCompleted:
        seen.append(list(cmd))
        if "-c" in cmd:
            return _FakeCompleted(stdout=str(repo / "src" / "robigo" / "__init__.py"))
        return _FakeCompleted(stdout="0 failed, 430 passed\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pytest_runner(repo)
    assert all(cmd[0] == sys.executable for cmd in seen)
