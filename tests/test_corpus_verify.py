# tests/test_corpus_verify.py
from __future__ import annotations

import dataclasses
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

import robigo.profile.verify as verify_module
from robigo.profile.corpus import Mutant
from robigo.profile.verify import (
    Baseline,
    Verdict,
    WrongTreeError,
    _apply_and_run,
    _assert_in_clone,
    _broken_count,
    _broken_ids,
    _excluded_dir,
    _find_line,
    _is_test_path,
    _module_path,
    _package_name,
    _primary_package,
    _resolve_in_clone,
    _sentinel_fast_path,
    _sentinel_via_search,
    _source_files,
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
    `MODULE_UNDER_TEST=` marker line followed by pytest-shaped text.
    `module_path=None` omits the marker entirely -- a runner that never
    says where it ran, exercised by the dedicated "no marker" tests."""
    if module_path is None:
        return text
    return f"MODULE_UNDER_TEST={module_path}\n{text}"


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
    resolved = _module_path(f"MODULE_UNDER_TEST={_inside(repo)}\n430 passed\n")
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
# _package_name / _primary_package — invariant 7's marker asks about the
# package being mutated, never a fixed name (coordinator review, 2026-08-10:
# the marker was still hardcoded to "robigo" even after the sentinel's
# TARGET was generalised, so every foreign repo was rejected for asking the
# wrong question).
# ---------------------------------------------------------------------------


def test_package_name_derives_from_a_src_layout_package_directory():
    assert _package_name(Path("src/mylib/calc.py")) == "mylib"


def test_package_name_derives_from_a_flat_layout_package_directory():
    # No `src/` prefix at all -- still finds the top-level directory.
    assert _package_name(Path("mylib/calc.py")) == "mylib"


def test_package_name_derives_from_a_bare_top_level_module_under_src():
    # No package DIRECTORY to name -- the module's own name, `.py` stripped.
    assert _package_name(Path("src/single.py")) == "single"


def test_package_name_derives_from_a_bare_top_level_module_without_src():
    assert _package_name(Path("single.py")) == "single"


def test_package_name_returns_robigo_for_robigos_own_sentinel_path_unchanged():
    # The exact input the marker checked before this generalisation --
    # pins that robigo-on-robigo is a SPECIAL CASE of this rule, not a
    # second, separately-maintained code path.
    assert _package_name(Path("src/robigo/context/budget.py")) == "robigo"


def test_package_name_raises_for_a_path_with_no_components():
    with pytest.raises(ValueError):
        _package_name(Path("."))


def test_primary_package_derives_from_the_first_real_source_file(repo: Path):
    assert _primary_package(repo) == "robigo"


def test_primary_package_raises_when_the_repo_offers_no_real_source(tmp_path: Path):
    # Fails for an implementation that guesses (e.g. defaulting to the
    # repo's own directory name) instead of refusing outright -- there is
    # nothing here to derive a package from.
    with pytest.raises(ValueError):
        _primary_package(tmp_path)


def test_primary_package_picks_the_alphabetically_first_package_in_a_multi_package_repo(
    tmp_path: Path,
):
    (tmp_path / "src" / "aaa_pkg").mkdir(parents=True)
    (tmp_path / "src" / "aaa_pkg" / "mod.py").write_text("x = 1\n")
    (tmp_path / "src" / "zzz_pkg").mkdir(parents=True)
    (tmp_path / "src" / "zzz_pkg" / "mod.py").write_text("x = 1\n")
    assert _primary_package(tmp_path) == "aaa_pkg"


# ---------------------------------------------------------------------------
# _apply_and_run derives and passes the package for a mutant's own path --
# the mechanism `verify`, `_sentinel_fast_path`, and `_sentinel_via_search`
# all share, so pinning it here covers all three at once.
# ---------------------------------------------------------------------------


def test_apply_and_run_passes_the_mutants_own_derived_package_to_the_runner(repo: Path):
    seen: dict[str, str] = {}

    def spy_runner(r: Path, package: str) -> str:
        seen["package"] = package
        return _output("0 failed, 430 passed\n", module_path=_inside(r))

    mutant = Mutant(TARGET_RELATIVE, 2, "    return 1\n", "    return 2\n", "off_by_one")
    _apply_and_run(repo, mutant, spy_runner)
    assert seen["package"] == "robigo"


def test_apply_and_run_derives_a_foreign_package_name_not_robigo(tmp_path: Path):
    # THE fix, proven directly: a mutant from a repo that isn't robigo at
    # all gets ITS OWN package name, not a hardcoded "robigo".
    (tmp_path / "src" / "mylib").mkdir(parents=True)
    (tmp_path / "src" / "mylib" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    seen: dict[str, str] = {}

    def spy_runner(r: Path, package: str) -> str:
        seen["package"] = package
        return _output("0 failed, 1 passed\n", module_path=r / "src" / "mylib" / "__init__.py")

    mutant = Mutant(
        Path("src/mylib/calc.py"), 2, "    return a + b\n", "    return a - b\n", "off_by_one"
    )
    _apply_and_run(tmp_path, mutant, spy_runner)
    assert seen["package"] == "mylib"


# ---------------------------------------------------------------------------
# sentinel_ok — invariant 4
# ---------------------------------------------------------------------------


def test_sentinel_ok_returns_true_when_the_runner_reports_real_breakage(repo: Path):
    def runner(r: Path, package: str) -> str:
        return _output("18 failed, 412 passed in 4.00s\n", module_path=_inside(r))

    assert sentinel_ok(repo, runner) is True
    # And the file must be restored, byte-identical, afterward.
    assert (repo / "src" / "robigo" / "context" / "budget.py").read_text() == BUDGET_SOURCE


def test_sentinel_ok_returns_false_for_a_blind_harness_that_always_reports_zero(repo: Path):
    # THE brief's required acceptance test for invariant 4: a harness that
    # never reports breakage, no matter what was applied, must not be
    # trusted -- measured 2026-08-10, a harness with this exact defect
    # scored 8 of 8 real mutants as survivors.
    def blind_runner(r: Path, package: str) -> str:
        return _output("0 failed, 430 passed in 4.00s\n", module_path=_inside(r))

    assert sentinel_ok(repo, blind_runner) is False
    assert (repo / "src" / "robigo" / "context" / "budget.py").read_text() == BUDGET_SOURCE


def test_sentinel_ok_returns_false_when_the_breakage_is_reported_on_the_wrong_tree(repo: Path):
    # Fails for an implementation that only checks `_broken_count > 0`
    # and never checks whose tree the breakage was reported against.
    def wrong_tree_runner(r: Path, package: str) -> str:
        return _output("18 failed, 412 passed in 4.00s\n", module_path=_outside(r))

    assert sentinel_ok(repo, wrong_tree_runner) is False
    assert (repo / "src" / "robigo" / "context" / "budget.py").read_text() == BUDGET_SOURCE


def test_sentinel_ok_falls_back_to_the_search_when_the_known_target_has_moved(repo: Path):
    # The fast path must NOT raise when it doesn't apply -- "doesn't
    # apply" is the ordinary case for any repo that isn't robigo itself
    # (Task 4's `--repo` points this at arbitrary repos), not a bug. It
    # must fall through to the general search instead, which for THIS
    # fixture (a blind runner, module content unchanged) finds nothing
    # to break and returns `False` -- not raise, not a false `True`.
    (repo / "src" / "robigo" / "context" / "budget.py").write_text(
        "def estimate_tokens(text: str) -> int:\n    return 999\n"
    )

    def blind_runner(r: Path, package: str) -> str:
        return _output("0 failed, 430 passed\n", module_path=_inside(r))

    assert sentinel_ok(repo, blind_runner) is False


def test_sentinel_ok_restores_the_file_even_when_the_runner_raises(repo: Path):
    # Failure injection on the restore path (CARRIED-DEBT's own named
    # pattern: "write_atomic's cleanup path under failure injection").
    def exploding_runner(r: Path, package: str) -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        sentinel_ok(repo, exploding_runner)
    assert (repo / "src" / "robigo" / "context" / "budget.py").read_text() == BUDGET_SOURCE


# ---------------------------------------------------------------------------
# sentinel_ok generalised to any repo, not just robigo's own -- Task 4's
# `--repo` (spec 5.1: black-oxide's 1327-test suite named as a corpus mine)
# ---------------------------------------------------------------------------


@pytest.fixture()
def scratch_repo(tmp_path: Path) -> Path:
    """A repo shaped nothing like robigo -- no `context/budget.py`, no
    `estimate_tokens` anywhere -- so `_sentinel_fast_path` can never apply
    and every test using this fixture genuinely exercises the general
    search, not the known-good shortcut."""
    (tmp_path / "src" / "widget").mkdir(parents=True)
    (tmp_path / "src" / "widget" / "__init__.py").write_text("")
    (tmp_path / "src" / "widget" / "core.py").write_text(
        "def double(n):\n    return n * 2\n"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_core.py").write_text(
        "def test_double():\n    assert True\n"
    )
    return tmp_path


def test_sentinel_ok_prefers_the_fast_path_and_skips_the_search_when_it_finds_breakage(
    monkeypatch: pytest.MonkeyPatch, repo: Path
):
    # Pins dispatch order directly: fails for an implementation that
    # always falls through to the (much more expensive) general search
    # regardless of whether the free fast path already answered.
    def exploding_search(r: Path, runner: object) -> bool:
        raise AssertionError("the general search must not run when the fast path applies")

    monkeypatch.setattr(verify_module, "_sentinel_via_search", exploding_search)

    def runner(r: Path, package: str) -> str:
        return _output("18 failed, 412 passed in 4.00s\n", module_path=_inside(r))

    assert sentinel_ok(repo, runner) is True


def test_sentinel_ok_trusts_the_fast_paths_own_false_without_falling_through_to_search(
    monkeypatch: pytest.MonkeyPatch, repo: Path
):
    # The other half of the same pin: `False` from the fast path (a
    # genuinely blind harness) must not be second-guessed by also
    # running the general search -- fails for an implementation like
    # `if fast is not None and fast: return fast`, which would silently
    # let a passing general-search result overrule a correctly-detected
    # blind harness.
    def exploding_search(r: Path, runner: object) -> bool:
        raise AssertionError("the general search must not run once the fast path answered False")

    monkeypatch.setattr(verify_module, "_sentinel_via_search", exploding_search)

    def blind_runner(r: Path, package: str) -> str:
        return _output("0 failed, 430 passed in 4.00s\n", module_path=_inside(r))

    assert sentinel_ok(repo, blind_runner) is False


def test_sentinel_ok_works_on_a_repo_shaped_nothing_like_robigo(scratch_repo: Path):
    # THE coordinator's first reported failure, reproduced and fixed:
    # pointing sentinel_ok at a scratch repo used to raise
    # FileNotFoundError unconditionally (the fast path's hardcoded
    # target). A runner that reports breakage for ANY mutation to
    # widget/core.py (checking the file's actual on-disk content, so this
    # proves the search really did apply a real candidate, not just that
    # something was called) must make this return True.
    def runner(r: Path, package: str) -> str:
        content = (r / "src" / "widget" / "core.py").read_text()
        broken = content != "def double(n):\n    return n * 2\n"
        report = "1 failed, 0 passed\n" if broken else "0 failed, 1 passed\n"
        return _output(report, module_path=r / "src" / "widget" / "__init__.py")

    assert sentinel_ok(scratch_repo, runner) is True
    # Restored afterward, whichever candidate happened to hit.
    assert (scratch_repo / "src" / "widget" / "core.py").read_text() == (
        "def double(n):\n    return n * 2\n"
    )


def test_sentinel_ok_asks_the_runner_about_widget_not_robigo_on_a_foreign_repo(
    scratch_repo: Path,
):
    # THE coordinator's SECOND reported failure, reproduced and fixed: the
    # marker check used to ask "does robigo resolve inside the clone" even
    # here, which fails by construction for every repo that isn't robigo
    # (robigo is never the code under test on a foreign repo, so of
    # course it resolves elsewhere). Recording every `package` the runner
    # was called with proves the real derived name ("widget") reaches the
    # runner throughout the whole search, never the fixed "robigo".
    seen_packages: list[str] = []

    def runner(r: Path, package: str) -> str:
        seen_packages.append(package)
        content = (r / "src" / "widget" / "core.py").read_text()
        broken = content != "def double(n):\n    return n * 2\n"
        report = "1 failed, 0 passed\n" if broken else "0 failed, 1 passed\n"
        return _output(report, module_path=r / "src" / "widget" / "__init__.py")

    assert sentinel_ok(scratch_repo, runner) is True
    assert seen_packages == ["widget"]
    assert "robigo" not in seen_packages


def test_sentinel_ok_never_tries_a_mutation_to_a_test_file(scratch_repo: Path):
    # Fails for an implementation that draws candidates from the WHOLE
    # tree including test-shaped files -- a runner that reports breakage
    # for ANY change to a test file (which candidates() can mutate just
    # as validly as source: `assert True` -> `assert False` parses fine)
    # would make this wrongly return True, because a test-file mutation
    # is collected by pytest regardless of whether the harness resolves
    # `import widget` into this clone at all -- it proves nothing about
    # invariant 7. A test-shaped file placed INSIDE `src/widget/` itself
    # (not just the top-level `tests/` directory `scratch_repo` also has)
    # is what actually exercises `_is_test_path`'s filename check here,
    # since `src/`-preference alone would already keep `tests/` out. Its
    # content (`assert 1 == 1`) is deliberately chosen to offer a REAL
    # candidate (`flipped_comparison`: `==` -> `!=`) -- if the exclusion
    # were missing, that candidate would actually be tried and this
    # runner would detect it, flipping the assertion below.
    (scratch_repo / "src" / "widget" / "test_extra.py").write_text(
        "def test_extra():\n    assert 1 == 1\n"
    )

    def runner(r: Path, package: str) -> str:
        outer = (r / "tests" / "test_core.py").read_text()
        inner = (r / "src" / "widget" / "test_extra.py").read_text()
        broken = (
            outer != "def test_double():\n    assert True\n"
            or inner != "def test_extra():\n    assert 1 == 1\n"
        )
        report = "1 failed, 0 passed\n" if broken else "0 failed, 1 passed\n"
        return _output(report, module_path=r / "src" / "widget" / "__init__.py")

    assert sentinel_ok(scratch_repo, runner) is False


def test_sentinel_via_search_tries_multiple_real_candidates_in_file_order(scratch_repo: Path):
    # `src/widget/core.py` alone offers a candidate the runner below never
    # reacts to; a second, alphabetically-LATER source file with the only
    # candidate the runner detects proves the search doesn't stop after
    # the first file's candidates find nothing -- it keeps going.
    later_file = scratch_repo / "src" / "widget" / "zzz_later.py"
    later_file.write_text("def triple(n):\n    return n * 3\n")

    def runner(r: Path, package: str) -> str:
        content = later_file.read_text()
        broken = content != "def triple(n):\n    return n * 3\n"
        report = "1 failed, 0 passed\n" if broken else "0 failed, 1 passed\n"
        return _output(report, module_path=r / "src" / "widget" / "__init__.py")

    assert _sentinel_via_search(scratch_repo, runner) is True
    assert later_file.read_text() == "def triple(n):\n    return n * 3\n"


def test_sentinel_via_search_is_bounded_and_does_not_exhaust_every_candidate(
    monkeypatch: pytest.MonkeyPatch, scratch_repo: Path
):
    # A hot file could offer far more candidates than are worth trying
    # (task 1's report: loop.py alone yields 125). Bounding the attempt
    # count is what keeps a blind harness's search a matter of minutes,
    # not an unbounded grind through the whole repo.
    monkeypatch.setattr(verify_module, "_SENTINEL_SEARCH_LIMIT", 1)
    calls: list[int] = []

    def blind_runner(r: Path, package: str) -> str:
        calls.append(1)
        return _output("0 failed, 1 passed\n", module_path=r / "src" / "widget" / "__init__.py")

    assert _sentinel_via_search(scratch_repo, blind_runner) is False
    assert len(calls) == 1


def test_sentinel_fast_path_returns_none_when_the_target_file_is_missing(scratch_repo: Path):
    # Fails for an implementation that raises (or crashes) instead of
    # signalling "doesn't apply here" -- `scratch_repo` has no
    # `context/budget.py` at all.
    assert _sentinel_fast_path(scratch_repo, lambda r, package: "430 passed\n") is None


def test_sentinel_fast_path_returns_none_when_the_line_has_changed(repo: Path):
    # A `budget.py` that exists but no longer reads the exact sentinel
    # line -- still "doesn't apply", not an error.
    (repo / "src" / "robigo" / "context" / "budget.py").write_text(
        "def estimate_tokens(text: str) -> int:\n    return 999\n"
    )
    assert _sentinel_fast_path(repo, lambda r, package: "430 passed\n") is None


def test_sentinel_fast_path_returns_true_for_the_known_good_sentinel(repo: Path):
    def runner(r: Path, package: str) -> str:
        return _output("18 failed, 412 passed\n", module_path=_inside(r))

    assert _sentinel_fast_path(repo, runner) is True


# ---------------------------------------------------------------------------
# _source_files / _is_test_path / _excluded_dir
# ---------------------------------------------------------------------------


def test_source_files_prefers_src_when_present(tmp_path: Path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "a.py").write_text("x = 1\n")
    (tmp_path / "notsrc.py").write_text("y = 2\n")  # outside src, must be ignored
    assert _source_files(tmp_path) == (Path("src/pkg/a.py"),)


def test_source_files_falls_back_to_the_whole_tree_without_a_src_directory(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    assert _source_files(tmp_path) == (Path("pkg/a.py"),)


def test_source_files_excludes_test_shaped_paths(tmp_path: Path):
    # Fails for an implementation that lets a mutation to a test file
    # serve as a sentinel candidate -- see `test_sentinel_ok_never_tries_
    # a_mutation_to_a_test_file` for why that would be a real bug, not a
    # cosmetic one.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    (tmp_path / "pkg" / "test_a.py").write_text("x = 1\n")
    (tmp_path / "pkg" / "b_test.py").write_text("x = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_c.py").write_text("x = 1\n")
    assert _source_files(tmp_path) == (Path("pkg/a.py"),)


def test_source_files_excludes_hidden_and_build_directories(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hooks.py").write_text("x = 1\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "generated.py").write_text("x = 1\n")
    assert _source_files(tmp_path) == (Path("pkg/a.py"),)


def test_source_files_is_sorted_for_determinism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # `Path.rglob` is monkeypatched to hand back results in a DELIBERATELY
    # unsorted order, rather than relying on this filesystem's own
    # directory-entry order happening to already look sorted (it does, on
    # this box, for a two-entry directory -- a first version of this test
    # relying on that passed even with `_source_files`'s own `sorted()`
    # call removed, which is exactly the false-negative this rewrite
    # closes).
    (tmp_path / "pkg").mkdir()
    z = tmp_path / "pkg" / "z.py"
    a = tmp_path / "pkg" / "a.py"
    z.write_text("x = 1\n")
    a.write_text("x = 1\n")

    def fake_rglob(self: Path, pattern: str) -> list[Path]:
        return [z, a]

    monkeypatch.setattr(Path, "rglob", fake_rglob)
    assert _source_files(tmp_path) == (Path("pkg/a.py"), Path("pkg/z.py"))


def test_is_test_path_recognises_the_conventional_shapes():
    assert _is_test_path(Path("pkg/test_a.py"))
    assert _is_test_path(Path("pkg/a_test.py"))
    assert _is_test_path(Path("tests/a.py"))
    assert _is_test_path(Path("test/a.py"))
    assert not _is_test_path(Path("pkg/a.py"))
    assert not _is_test_path(Path("pkg/latest.py"))  # contains "test" but isn't test-shaped


def test_excluded_dir_recognises_hidden_and_build_names():
    assert _excluded_dir(".git")
    assert _excluded_dir("__pycache__")
    assert _excluded_dir("build")
    assert _excluded_dir("widget.egg-info")
    assert not _excluded_dir("widget")


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

    def blind_runner(r: Path, package: str) -> str:
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
    def runner(r: Path, package: str) -> str:
        return _output("3 failed, 2 error, 425 passed in 5.00s\n", module_path=_inside(r))

    assert baseline(repo, runner).broken == 5


def test_baseline_is_not_assumed_zero(repo: Path):
    # Measured 2026-08-10: a git-archive copy of this project's own repo
    # baselined at 6. Fails for an implementation that hardcodes 0.
    def runner(r: Path, package: str) -> str:
        return _output("6 failed, 400 passed in 5.00s\n", module_path=_inside(r))

    assert baseline(repo, runner).broken == 6


def test_baseline_counts_an_error_only_report_with_no_failed_substring_at_all(repo: Path):
    # THE brief's required acceptance test, at the public `baseline()`
    # level rather than just the private `_broken_count` unit.
    def runner(r: Path, package: str) -> str:
        return _output("1 error in 0.01s\n", module_path=_inside(r))

    result = baseline(repo, runner)
    assert result.broken == 1


def test_baseline_measures_real_wall_clock_seconds(repo: Path):
    # Fails for an implementation that reports 0.0 unconditionally, or
    # that tries to parse a "seconds" figure out of the runner's own text
    # instead of timing the call itself.
    def slow_runner(r: Path, package: str) -> str:
        time.sleep(0.05)
        return _output("0 failed, 430 passed in 0.01s\n", module_path=_inside(r))

    result = baseline(repo, slow_runner)
    assert result.seconds >= 0.05


def test_baseline_raises_wrong_tree_error_for_a_path_outside_the_repo(repo: Path):
    def runner(r: Path, package: str) -> str:
        return _output("0 failed, 430 passed\n", module_path=_outside(r))

    with pytest.raises(WrongTreeError):
        baseline(repo, runner)


def test_baseline_raises_wrong_tree_error_when_no_marker_is_present(repo: Path):
    with pytest.raises(WrongTreeError):
        baseline(repo, lambda r, package: "0 failed, 430 passed in 1.00s\n")


def test_baseline_derives_and_passes_the_repos_own_package_not_a_fixed_name(repo: Path):
    seen: dict[str, str] = {}

    def spy_runner(r: Path, package: str) -> str:
        seen["package"] = package
        return _output("0 failed, 430 passed\n", module_path=_inside(r))

    baseline(repo, spy_runner)
    assert seen["package"] == "robigo"


def test_baseline_propagates_primary_packages_valueerror_on_an_empty_repo(tmp_path: Path):
    with pytest.raises(ValueError):
        baseline(tmp_path, lambda r, package: "430 passed\n")


# ---------------------------------------------------------------------------
# verify — invariants 5, 6, 7
# ---------------------------------------------------------------------------


def _mutant() -> Mutant:
    return Mutant(TARGET_RELATIVE, 2, "    return 1\n", "    return 9\n", "off_by_one")


def test_verify_keeps_a_mutant_with_exactly_one_net_new_failure_and_records_its_id(repo: Path):
    # The "1" case, invariant 6's core: the id must be captured, not just
    # the count.
    def runner(r: Path, package: str) -> str:
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
    def runner(r: Path, package: str) -> str:
        return _output("0 failed, 430 passed in 2.00s\n", module_path=_inside(r))

    verdict = verify(_mutant(), repo, Baseline(broken=0, seconds=1.0), runner)
    assert verdict.kept is False
    assert verdict.test_id is None
    assert verdict.failures == 0
    assert "survived" in verdict.reason


def test_verify_rejects_a_mutant_that_breaks_many_tests(repo: Path):
    # The "many" case, using the exact figure ("18 failed") measured
    # 2026-08-10 for a deliberate central-code sentinel.
    def runner(r: Path, package: str) -> str:
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
    def runner(r: Path, package: str) -> str:
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
    def runner(r: Path, package: str) -> str:
        return _output("0 failed, 430 passed in 2.00s\n", module_path=_outside(r))

    verdict = verify(_mutant(), repo, Baseline(broken=0, seconds=1.0), runner)
    assert verdict.kept is False
    assert "survived" not in verdict.reason
    assert "clone" in verdict.reason or "tree" in verdict.reason.lower()


def test_verify_rejects_a_looks_kept_result_reported_on_the_wrong_tree(repo: Path):
    # The more dangerous case: a report that would otherwise satisfy
    # invariant 6 outright (exactly one id, exactly one net new failure),
    # but on the wrong tree -- must not be reported as kept.
    def runner(r: Path, package: str) -> str:
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
        verify(bad, repo, Baseline(broken=0, seconds=1.0), lambda r, package: "430 passed\n")


def test_verify_raises_valueerror_for_a_mutant_path_that_escapes_via_traversal(repo: Path):
    bad = Mutant(Path("../outside.py"), 1, "a\n", "b\n", "off_by_one")
    with pytest.raises(ValueError):
        verify(bad, repo, Baseline(broken=0, seconds=1.0), lambda r, package: "430 passed\n")


def test_verify_actually_writes_the_mutation_before_the_runner_is_called(repo: Path):
    # Not just that the Verdict's shape looks right afterward -- that the
    # file on disk, AT THE MOMENT the runner ran, held `mutant.mutated`.
    seen: dict[str, str] = {}

    def spy_runner(r: Path, package: str) -> str:
        seen["content"] = (r / TARGET_RELATIVE).read_text()
        return _output("0 failed, 430 passed in 1.00s\n", module_path=_inside(r))

    m = _mutant()
    verify(m, repo, Baseline(broken=0, seconds=1.0), spy_runner)
    assert seen["content"] == TARGET_SOURCE.replace(m.original, m.mutated)


def test_verify_derives_and_passes_the_mutants_own_package_not_a_fixed_name(repo: Path):
    seen: dict[str, str] = {}

    def spy_runner(r: Path, package: str) -> str:
        seen["package"] = package
        return _output("0 failed, 430 passed in 1.00s\n", module_path=_inside(r))

    verify(_mutant(), repo, Baseline(broken=0, seconds=1.0), spy_runner)
    assert seen["package"] == "robigo"


def test_verify_restores_the_file_after_a_kept_result(repo: Path):
    def runner(r: Path, package: str) -> str:
        return _output(
            "FAILED tests/test_x.py::test_y - AssertionError\n1 failed, 429 passed in 2.00s\n",
            module_path=_inside(r),
        )

    verify(_mutant(), repo, Baseline(broken=0, seconds=1.0), runner)
    assert (repo / TARGET_RELATIVE).read_text() == TARGET_SOURCE


def test_verify_restores_the_file_after_a_rejected_result(repo: Path):
    def runner(r: Path, package: str) -> str:
        return _output("0 failed, 430 passed in 2.00s\n", module_path=_inside(r))

    verify(_mutant(), repo, Baseline(broken=0, seconds=1.0), runner)
    assert (repo / TARGET_RELATIVE).read_text() == TARGET_SOURCE


def test_verify_restores_the_file_even_when_the_runner_raises(repo: Path):
    def exploding_runner(r: Path, package: str) -> str:
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
        verify(stale, repo, Baseline(broken=0, seconds=1.0), lambda r, package: "430 passed\n")


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
# no real pytest ever executes inside this test suite. A REAL, un-faked
# proof that this works against an actual foreign repo (not just against
# canned `subprocess.run` output) is not possible here by design -- see the
# task report for a standalone, manually-run acceptance script instead.
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
    pytest_runner(repo, "robigo")
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
    pytest_runner(repo, "robigo")
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
    pytest_runner(repo, "robigo")
    assert captured["env"]["PYTHONPATH"] == str(repo / "src")  # type: ignore[index]


def test_pytest_runner_imports_the_given_package_not_a_fixed_name(
    monkeypatch: pytest.MonkeyPatch, repo: Path
):
    # THE coordinator's second fix, pinned directly on the real runner:
    # fails for an implementation that still hardcodes "robigo" in the
    # import-check command regardless of what `package` it was called
    # with.
    seen_import_cmd: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> _FakeCompleted:
        if "-c" in cmd:
            seen_import_cmd.extend(cmd)
            return _FakeCompleted(stdout=str(repo / "src" / "widget" / "__init__.py"))
        return _FakeCompleted(stdout="0 failed, 430 passed\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pytest_runner(repo, "widget")
    code = seen_import_cmd[seen_import_cmd.index("-c") + 1]
    assert "import widget" in code
    assert "widget.__file__" in code
    assert "robigo" not in code


def test_pytest_runner_returns_the_module_marker_followed_by_pytest_output(
    monkeypatch: pytest.MonkeyPatch, repo: Path
):
    module_file = repo / "src" / "robigo" / "__init__.py"

    def fake_run(cmd: list[str], **kwargs: object) -> _FakeCompleted:
        if "-c" in cmd:
            return _FakeCompleted(stdout=str(module_file) + "\n")
        return _FakeCompleted(stdout="1 failed, 429 passed in 2.00s\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    text = pytest_runner(repo, "robigo")
    assert text.startswith(f"MODULE_UNDER_TEST={module_file}\n")
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
    text = pytest_runner(repo, "robigo")
    assert "MODULE_UNDER_TEST=" not in text
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
    text = pytest_runner(repo, "robigo")
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
    pytest_runner(repo, "robigo")
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
    pytest_runner(repo, "robigo")
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
    pytest_runner(repo, "robigo")
    assert all(cmd[0] == sys.executable for cmd in seen)
