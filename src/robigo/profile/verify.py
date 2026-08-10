# src/robigo/profile/verify.py
"""The verifier: decides whether a `Mutant` (task 1) becomes a corpus
record. Everything downstream -- stage 4's number, and the project's 40%
kill criterion -- inherits its trustworthiness from this module, so every
function here is built to REFUSE to certify a result it cannot back up,
rather than to report the friendliest-looking answer.

`runner: Callable[[Path, str], str]` is injected everywhere real work
happens (`sentinel_ok`, `baseline`, `verify`), so this module's own tests
are offline and instant -- no daemon, no GPU, no network, no real pytest
subprocess. `pytest_runner`, below, is the one real implementation: it
shells out to pytest inside `repo` and is what a production caller (task
4's CLI) passes in place of a test's canned callable. The second
argument is the top-level package the caller wants invariant 7 checked
against -- `_apply_and_run`/`baseline` derive it (from a mutant's path,
or from the clone's own source when no specific mutant is in scope) and
pass it down; a runner never has to guess.

Four invariants, each one closing a false result this project actually
produced while measuring plan 04 (2026-08-10):

  4. `sentinel_ok` must prove the harness can SEE a break before any
     survival is believed -- a blind harness scored 8 of 8 mutants as
     survivors, a perfect false negative. This has to work against ANY
     repo `--repo` points it at (spec 5.1 names black-oxide's suite as a
     corpus mine by name), not just robigo's own -- so beyond a fast
     path known to work here, the sentinel draws a real candidate from
     the target repo's OWN source (task 1's `candidates()`) and proves
     the harness sees THAT break, rather than assuming a hardcoded
     robigo-specific target exists.
  5. Breakage is `failures + errors`, measured against a baseline that is
     NOT assumed to be zero -- a `git archive` copy (no `.git`) baselined
     at 6, and counting only `"N failed"` reads a syntax-breaking mutant
     (reported as an "error") as a clean run.
  6. "Exactly one" means exactly one, and the failing test's identity is
     recorded -- without it there is no diagnostic to check a repair
     against later, and the corpus's own verification property becomes
     uncheckable.
  7. The code under test must be the mutated clone, not the editable
     install's real source. Without `PYTHONPATH` forced, a subprocess
     started inside a copied tree still imports the real repo, and every
     mutant appears to survive. This check must ask about the PACKAGE
     BEING MUTATED, not a fixed name -- a marker that always asks "where
     does robigo resolve from" answers a question that has nothing to do
     with a foreign repo's own code, and rejects every foreign repo for
     the wrong reason (robigo resolving to the real install, not the
     clone, because robigo was never the code under test there).

Every one of the four is checked BEFORE a caller is allowed to trust the
number it protects: `sentinel_ok`/`baseline`/`verify` never assume, always
extract the runner's own report of what it did and reject a result whose
own report doesn't add up.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from robigo.profile.corpus import Mutant, _apply, candidates

Runner = Callable[[Path, str], str]
"""A runner takes the repo root it should test and the top-level package
name invariant 7 should be checked against, and returns the raw text of
what it did there. The real runner (`pytest_runner`) returns the pytest
short-test-summary output, prefixed with a `MODULE_UNDER_TEST=<path>`
marker line naming what `import <package>` resolved to in that same
subprocess environment -- the fact invariant 7 checks before anything
else is believed. `<package>` is never hardcoded: `_apply_and_run` derives
it from whichever `Mutant.path` is in scope, and `baseline` derives it
from the clone's own source when no specific mutant is in scope, so the
same runner works identically against robigo's own repo (package
`robigo`) or any foreign repo `--repo` points at (package `mylib`, or
whatever its own top-level package is named). A test's canned runner
constructs that same text by hand, which is what makes every test in this
module able to run with no daemon, no GPU, no network, and no real pytest
subprocess."""


class WrongTreeError(RuntimeError):
    """Raised when a runner's own report says the code it tested is not
    inside the clone it was asked to test. Never caught silently outside
    `sentinel_ok` (which turns it into `False`, its one legitimate "not
    proven" signal) and `verify` (which turns it into a rejected
    `Verdict`) -- `baseline` lets it propagate, because a `Baseline` value
    that might describe the wrong tree has no safe fallback to return."""


# ---------------------------------------------------------------------------
# Public data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Baseline:
    """The repo's own broken-test count and wall-clock cost, measured once
    with no mutation applied, before any mutant is judged. `broken` is
    `failures + errors` (invariant 5) and is NOT assumed to be zero -- a
    `git archive` copy of this project's own repo baselined at 6, because
    a copy with no `.git` fails every git-dependent test. `seconds` is the
    real wall-clock time `runner` took, measured by this module, not
    parsed out of the runner's own text -- so it is honest even against a
    runner whose text says nothing about timing at all.

    `executed` is `passed + broken` -- every test this baseline run actually
    RAN, as opposed to one a collection error or an early exit (`-x`) kept
    from ever starting (whole-branch review C2, ruled 2026-08-10). `verify`
    compares a mutant run's own `executed` total against this figure before
    trusting any failure count: a real single-test regression changes the
    passed/broken split without changing the total (`600` stays `600`
    whether it reads `600 passed` or `599 passed, 1 failed`), while a
    collection error or a `PYTEST_ADDOPTS=-x` early exit changes the total
    itself, and no comparison of `broken` alone against this baseline could
    ever catch that."""

    broken: int
    executed: int
    seconds: float


@dataclass(frozen=True)
class Verdict:
    """The decision for one `Mutant`. `kept` is `True` only when the
    mutant broke EXACTLY one test net of the baseline AND that test's
    identity was captured -- either half missing means `kept=False`
    (invariant 6: a kept mutant with no diagnostic test id is not
    verifiable later, so this module never produces one).

    `failures` is the mutant run's own total broken count (`failures +
    errors`, invariant 5's `broken` -- the field is named `failures`
    because that is the name the brief's interface gives it, but it holds
    the COMBINED count, not the pytest `"N failed"` component alone;
    task 3's `CorpusRecord.broken` is this value, verbatim, for a kept
    mutant). It is NOT adjusted for the baseline -- a caller that wants
    the net figure computes `failures - baseline.broken` itself, and
    still has the raw total besides.

    `test_id` is the pytest node id of the one new failure for a kept
    mutant, `None` otherwise -- including when the raw counts said
    "exactly one net new failure" but the runner's own output didn't let
    that one failure be isolated by id (a baseline with pre-existing
    breakage can produce this; see `verify`'s docstring).

    `reason` is always populated, whether kept or not -- a human-readable
    account of which branch produced this `Verdict`, so a caller
    inspecting a rejected mutant never has to re-derive why."""

    kept: bool
    failures: int
    test_id: str | None
    reason: str


# ---------------------------------------------------------------------------
# Parsing the runner's report
# ---------------------------------------------------------------------------

_FAILED_COUNT = re.compile(r"(\d+) failed")
_ERROR_COUNT = re.compile(r"(\d+) error")
_PASSED_COUNT = re.compile(r"(\d+) passed")
"""Independent, unanchored searches -- deliberately NOT one combined
pattern like `r"(\\d+) failed, (\\d+) error"`, because pytest's own summary
line omits either clause entirely when its count is zero (`"1 error in
0.01s"` has no `"failed"` substring anywhere). A combined pattern would
simply fail to match that text and silently score it as zero broken --
exactly the false-clean-run failure mode invariant 5 exists to prevent, and
the one this module's tests pin directly (`test_baseline_counts_an_
error_only_report_with_no_failed_substring_at_all`)."""

_BROKEN_ID = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.MULTILINE)
"""Pytest's short test summary info (`-rfE`) prints one line per broken
test, `"FAILED <nodeid> - <reason>"` or `"ERROR <nodeid>"` (setup/
collection errors often carry no ` - reason` suffix), each anchored at the
start of its own line. `\\S+` stops at the first whitespace, which is the
node id's own boundary for every id this project's suite produces."""

_EXIT_CODE = re.compile(r"^EXIT_CODE=(-?\d+)$", re.MULTILINE)
"""The exit-code marker `pytest_runner` prepends (alongside `MODULE_UNDER_
TEST=`), naming pytest's own real return code for that run -- whole-branch
review C1/C2, ruled 2026-08-10: the old `pytest_runner` shelled out to
pytest and then threw the return code away entirely, so a run pytest itself
marked INTERRUPTED (a collection error aborts the whole session, exit code
2, not 0 or 1) was scored purely on its failure/error counts, same as an
ordinary completed run. A canned test runner that doesn't care about this
check still includes a well-formed `EXIT_CODE=0` (via this test file's
`_output` helper) unless it is specifically exercising the abnormal-exit
path."""

_INTERRUPTED_MARKERS = ("Interrupted:", "INTERNALERROR")
"""Substrings pytest's own text prints when a run did not complete
normally: `"!!!!!!!!!!!!!!!!!!! Interrupted: N errors during collection
!!!!!!!!!!!!!!!!!!!"` for a collection error (measured 2026-08-10 against a
real `swapped_args` mutation to `action/codec.py`'s module-level
`re.compile(pattern, flags)` -- exit code 2, this exact text), and
`"INTERNALERROR>"` for a crash inside pytest itself. Checked as plain
substrings, not anchored patterns -- pytest does not print either at a
fixed column, and a false positive here only ever REJECTS a result that an
exit-code check below would very likely have rejected anyway."""

_MODULE_MARKER = re.compile(r"^MODULE_UNDER_TEST=(.+)$", re.MULTILINE)
"""The marker `pytest_runner` prepends, naming what `import <package>`
resolved to in the SAME subprocess environment pytest itself ran in --
`<package>` is whatever `pytest_runner` was asked to check, never a fixed
name. A canned test runner that doesn't care about invariant 7 still
includes a well-formed marker (via this test file's `_output` helper)
pointing inside the fixture repo -- only the tests that specifically
exercise invariant 7 give it something else."""


def _broken_count(text: str) -> int:
    """`failures + errors` (invariant 5), each counted independently so a
    report naming only one of the two is not misread as reporting zero of
    the other."""
    failed = int(m.group(1)) if (m := _FAILED_COUNT.search(text)) else 0
    errors = int(m.group(1)) if (m := _ERROR_COUNT.search(text)) else 0
    return failed + errors


def _broken_ids(text: str) -> tuple[str, ...]:
    """Every broken test's node id, in the order the runner reported
    them. Used only to isolate the single new failure a kept mutant must
    name (invariant 6); see `verify`."""
    return tuple(_BROKEN_ID.findall(text))


def _passed_count(text: str) -> int:
    """The `"N passed"` figure alone, 0 if pytest's summary never printed
    one at all (a fully-interrupted collection error reports none)."""
    match = _PASSED_COUNT.search(text)
    return int(match.group(1)) if match else 0


def _executed_total(text: str) -> int:
    """`passed + broken` -- every test this run actually EXECUTED, matched
    against `Baseline.executed` by `verify` (whole-branch review C2). A
    real single-test regression moves one test from the passed side to the
    broken side without changing this total; a collection error or a
    `PYTEST_ADDOPTS=-x` early exit shrinks it, which is exactly the
    signal invariant 6's "exactly one" arithmetic alone cannot see."""
    return _passed_count(text) + _broken_count(text)


def _exit_code(text: str) -> int | None:
    """The `EXIT_CODE=` marker's value, or `None` if the runner's report
    carries no such marker at all -- treated the same as "did not
    complete normally" by `verify`, exactly the convention `_module_path`
    already uses for a missing `MODULE_UNDER_TEST=` marker."""
    match = _EXIT_CODE.search(text)
    return int(match.group(1)) if match else None


def _run_did_not_complete(text: str) -> str | None:
    """`None` if `text` reports a normal, comparable pytest run (exit code
    0 or 1, no interruption/crash text); otherwise the reason it does not
    -- whole-branch review C1/C2's shared predicate, checked before
    `verify` ever asks "exactly one". A collection error exits 2
    ("Interrupted: N errors during collection"), never 0 or 1, and prints
    `"Interrupted:"` -- measured directly against a real `swapped_args`
    mutation to `action/codec.py`'s module-level `re.compile(pattern,
    flags)` (whole-branch review C1). `PYTEST_ADDOPTS=-x` does NOT trip
    this check on its own (an early exit after the first failure still
    exits 1, with no interruption text) -- that case is caught downstream
    by `verify`'s own executed-total comparison against the baseline, not
    here."""
    exit_code = _exit_code(text)
    if exit_code not in (0, 1):
        return f"pytest did not complete normally (exit code {exit_code!r})"
    if any(marker in text for marker in _INTERRUPTED_MARKERS):
        return "pytest reported an interrupted run or an internal error"
    return None


def _module_path(text: str) -> Path | None:
    """The resolved path the `MODULE_UNDER_TEST=` marker names, or `None`
    if the runner's report carries no such marker at all -- treated
    identically to an out-of-tree path by `_assert_in_clone`: a runner
    that never says where it ran has proven nothing about invariant 7,
    the same as one that says the wrong place."""
    match = _MODULE_MARKER.search(text)
    if match is None:
        return None
    return Path(match.group(1).strip()).resolve()


def _assert_in_clone(text: str, repo: Path) -> None:
    """Invariant 7's check: raises `WrongTreeError` unless the runner's
    own report names a module path resolving inside `repo`. Every caller
    of `runner` in this module runs its result through this before
    trusting any broken-test count -- an editable install (robigo's own,
    measured 2026-08-10: 8 of 8 mutants appeared to survive without
    `PYTHONPATH` forced) means a subprocess started in a copied tree can
    still import a REAL install's source instead of the clone's, for
    whichever package was checked. This does not care which package the
    marker names, only whether the path it resolved to is inside `repo`
    -- `pytest_runner` is what decides which package to ask about."""
    resolved_repo = repo.resolve()
    module_path = _module_path(text)
    if module_path is None or not module_path.is_relative_to(resolved_repo):
        raise WrongTreeError(
            f"runner reported the code under test as {module_path!r}, which "
            f"is not inside the clone {resolved_repo} -- refusing to trust "
            f"this result rather than silently scoring the real repo's "
            f"source as though it were the mutated copy"
        )


# ---------------------------------------------------------------------------
# Applying a mutation to a file inside the clone, safely
# ---------------------------------------------------------------------------


def _resolve_in_clone(repo: Path, relative: Path) -> Path:
    """Where `relative` (normally `Mutant.path`) lands inside `repo`,
    checked two ways before a single byte is written:

    An absolute `relative` is rejected outright and with a specific
    message, because pathlib's `/` operator does not join an absolute
    right-hand path onto a left-hand one -- it DISCARDS the left side and
    returns the absolute path unchanged (`Path("/tmp/clone") /
    Path("/etc/passwd") == Path("/etc/passwd")`). Silently doing that here
    would write into whatever `mutant.path` names verbatim, which is
    precisely the "operates on the wrong tree" failure mode invariant 7
    exists to catch, self-inflicted rather than measured.

    The join is then resolved and re-checked against `repo` with
    `is_relative_to` -- which alone is what actually stops both an
    absolute path (pathlib's rule above means an unresolved-away absolute
    input reaches this check as a path outside `repo`) and a relative path
    that escapes via `..`. The `is_absolute()` guard above is not the only
    thing standing between an absolute `mutant.path` and disaster; it is
    the one that fails fast with a message naming the actual mechanism,
    which `is_relative_to` alone would not say."""
    if relative.is_absolute():
        raise ValueError(
            f"mutant.path {relative} is absolute -- verify() requires a "
            f"path relative to the clone; `repo / mutant.path` would "
            f"silently discard `repo` and point at the absolute path "
            f"instead (pathlib's `/` operator joins that way), which is "
            f"the wrong-tree failure mode invariant 7 exists to prevent"
        )
    target = (repo / relative).resolve()
    resolved_repo = repo.resolve()
    if not target.is_relative_to(resolved_repo):
        raise ValueError(f"mutant.path {relative} escapes the clone {resolved_repo}")
    return target


def _package_name(relative: Path) -> str:
    """The top-level importable name for a repo-relative path like
    `Mutant.path`: `src/mylib/calc.py` -> `mylib`; `mylib/calc.py` (no
    `src` layout) -> `mylib`; a bare top-level module file such as
    `src/single.py` or `single.py` (no package directory to name instead)
    -> `single`, its own name with `.py` stripped.

    This is invariant 7's answer to "which package does this mutation
    belong to" -- for robigo's own sentinel (`src/robigo/context/
    budget.py`) this returns `robigo`, the exact name the marker checked
    before this was generalised to any repo (coordinator review,
    2026-08-10: the marker was still hardcoded to ask about robigo even
    after the sentinel target itself was generalised, so every foreign
    repo was rejected for asking the wrong question rather than for a
    real problem). Robigo-on-robigo is a SPECIAL CASE of this rule, not a
    second code path.

    Raises `ValueError` if `relative` has no path components at all (an
    empty `Path`) -- there is nothing to derive a name from, and guessing
    would risk asking the wrong question silently."""
    parts = relative.parts
    if not parts:
        raise ValueError(f"{relative!r} has no path components to derive a package from")
    if parts[0] == "src" and len(parts) > 1:
        parts = parts[1:]
    if len(parts) == 1:
        return Path(parts[0]).stem
    return parts[0]


def _apply_and_run(repo: Path, mutant: Mutant, runner: Runner) -> str:
    """Applies `mutant` to its file inside `repo` (via `_resolve_in_clone`,
    so an absolute or escaping `mutant.path` is refused before anything is
    written), calls `runner` with the package `_package_name(mutant.path)`
    derives, and restores the file to its exact original content --
    always, even if `runner` raises. The one shared "apply / run /
    restore" implementation `verify` and both of `sentinel_ok`'s
    strategies use, rather than three copies of the same three steps free
    to drift apart (this project's own `CARRIED-DEBT.md` names that
    pattern as a recurring defect source) -- and the one place that
    derives invariant 7's package name from a mutant, so `verify` and
    both sentinel strategies get the fix for free rather than needing
    their own copy of this logic."""
    target = _resolve_in_clone(repo, mutant.path)
    source = target.read_text()
    target.write_text(_apply(source, mutant))
    package = _package_name(mutant.path)
    try:
        return runner(repo, package)
    finally:
        target.write_text(source)


def _detects_breakage(repo: Path, text: str) -> bool:
    """Whether `text` -- a runner's report after some mutation was
    applied -- shows real, trustworthy breakage: resolves inside `repo`
    (invariant 7; a wrong-tree report proves nothing, however many
    failures it claims) AND `_broken_count` is nonzero. Shared by both of
    `sentinel_ok`'s strategies, which differ only in WHICH mutation they
    try, never in how a result is judged."""
    try:
        _assert_in_clone(text, repo)
    except WrongTreeError:
        return False
    return _broken_count(text) > 0


# ---------------------------------------------------------------------------
# The sentinel
# ---------------------------------------------------------------------------

_SENTINEL_PATH = Path("src/robigo/context/budget.py")
_SENTINEL_ORIGINAL = "    return int(len(text) / CHARS_PER_TOKEN) + 1\n"
_SENTINEL_MUTATED = "    return 0\n"
"""`estimate_tokens` forced to always return 0 -- the exact sentinel
measured 2026-08-10 (`docs/superpowers/plans/2026-08-10-robigo-04-mutation-
corpus.md`): a VALID, parseable change with a real semantic effect (18
failures, once the editable-install trap was worked around), deliberately
not a syntax break. A syntax break is reported by pytest as a collection
*error*, not a *failure*, and a first attempt at this sentinel that
happened to be malformed reported 0 breakage and would have certified a
blind harness as working -- the sentinel must actually run and actually
matter, not merely fail to parse.

This is `sentinel_ok`'s FAST PATH ONLY -- a free, known-good shortcut when
`repo` happens to be a clone of robigo itself. It is an optimisation, not
a requirement: Task 4's `--repo` points this module at arbitrary repos
(the spec names black-oxide's 1327-test suite as a corpus mine by name),
and this exact function/line is not expected to exist in any of them. See
`_sentinel_via_search` for the general case."""


def sentinel_ok(repo: Path, runner: Runner) -> bool:
    """Invariant 4: proves the harness can see a real source mutation in
    `repo` before any survival result is believed. Tries the free fast
    path above first; if it doesn't apply here (`_SENTINEL_PATH` missing,
    or present but not reading `_SENTINEL_ORIGINAL`), falls through to
    `_sentinel_via_search` -- never raises for "doesn't apply", because
    not applying is the ordinary case for any repo that isn't robigo
    itself, not a bug.

    Returns `False` for every way a result can fail to be trusted: the
    harness reported zero breakage for every mutation tried (blind --
    measured 2026-08-10, a blind harness reported 8 of 8 mutants
    surviving), or it reported breakage but on the wrong tree (invariant
    7 -- a "broken" result on the real repo's source proves nothing about
    this clone). Both are legitimate "not proven" outcomes a caller
    should abort on the same way; neither is distinguished in the return
    value, because both blind and wrong-tree candidates the search tries
    return `False` from `_detects_breakage` identically.

    Propagates whatever `runner` itself raises -- a runner blowing up is
    a real error, not a "not proven" result, and is not swallowed here or
    in either strategy below."""
    fast = _sentinel_fast_path(repo, runner)
    if fast is not None:
        return fast
    return _sentinel_via_search(repo, runner)


def _sentinel_fast_path(repo: Path, runner: Runner) -> bool | None:
    """Robigo's own known-fatal `estimate_tokens` change, tried only if
    `repo`'s copy of `_SENTINEL_PATH` exists and still reads
    `_SENTINEL_ORIGINAL` at exactly one line. Returns `None` -- NOT
    `False` -- when it doesn't apply, so `sentinel_ok` can tell "this
    fast path is unavailable here" apart from "this fast path ran and
    proved nothing", and fall through to the general search instead of
    reporting a hardcoded-robigo-only optimisation's absence as though it
    were evidence about THIS repo's harness."""
    target = repo / _SENTINEL_PATH
    if not target.is_file():
        return None
    source = target.read_text()
    try:
        line = _find_line(source, _SENTINEL_ORIGINAL)
    except RuntimeError:
        return None
    mutant = Mutant(_SENTINEL_PATH, line, _SENTINEL_ORIGINAL, _SENTINEL_MUTATED, "sentinel")
    text = _apply_and_run(repo, mutant, runner)
    return _detects_breakage(repo, text)


_SENTINEL_SEARCH_LIMIT = 8
"""Bounded attempts for the general sentinel search -- each attempt is one
full `runner` call, measured at ~15s against robigo's own suite (this
plan's "Measured before planning" section), so this bounds a worst-case
search to roughly two minutes.

Does NOT double as a data point on the target's own keep rate (I3,
whole-branch review 2026-08-10, correcting an earlier version of this
docstring that claimed otherwise): `sentinel_ok` runs entirely inside
`cli.corpus_main`, BEFORE `generate_corpus` is ever called, and its own
candidates are never threaded into `GenerationResult.targets` or
`.dropped` -- `render_report`'s printed "candidates proposed/tried/kept"
and `.seconds` cover only `generate_corpus`'s own loop, not this search.
A sentinel attempt that happened to break something is not recorded as a
keep anywhere a reader of the corpus report would see it, and is not
counted toward any target's `proposed`/`tried` either."""


def _sentinel_via_search(repo: Path, runner: Runner) -> bool:
    """The general case: works for ANY repo, not just robigo's own. Draws
    real candidates from `repo`'s OWN source via task 1's `candidates()`
    (`_source_files`, in sorted order) and tries each in turn -- apply,
    run, restore -- until one produces breakage that resolves inside the
    clone. That first breaking candidate directly proves the harness sees
    a SOURCE mutation imported through `repo`'s own package, which is
    exactly what invariant 7 needs proven.

    A hand-written failing TEST FILE would not prove this, and is
    deliberately not what this function tries: pytest collects test files
    straight out of the clone regardless of whether `PYTHONPATH` resolves
    `import <package>` there at all, so a test-file sentinel would read
    as working against a completely blind harness. `_source_files`
    excludes test-shaped paths for exactly this reason -- the sentinel
    has to mutate code the tests IMPORT, not code the tests ARE.

    Bounded to `_SENTINEL_SEARCH_LIMIT` attempts. Returns `False`, not
    raises, when the bound is exhausted with no breakage seen -- which is
    genuinely ambiguous between "the harness is blind" (invariant 4) and
    "none of the first `_SENTINEL_SEARCH_LIMIT` candidates from this
    repo's real source happen to be caught by anything" (a poor corpus
    mine, task 4's target-selection concern). This function cannot tell
    the two apart from a bool alone and does not guess; a caller that
    needs to tell them apart runs a keep-rate check across more of the
    repo's candidates, which task 4 needs regardless of this function."""
    attempts = 0
    for source_path in _source_files(repo):
        try:
            source = (repo / source_path).read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for mutant in candidates(source, source_path):
            if attempts >= _SENTINEL_SEARCH_LIMIT:
                return False
            attempts += 1
            text = _apply_and_run(repo, mutant, runner)
            if _detects_breakage(repo, text):
                return True
    return False


_EXCLUDED_DIR_NAMES = frozenset({"__pycache__", "venv", "build", "dist", "node_modules"})


def _excluded_dir(name: str) -> bool:
    """A directory component the sentinel search never descends into for
    candidates: hidden/VCS/tooling directories (anything starting with
    `.`, catching `.git`, `.venv`, `.tox`, `.eggs`, `.pytest_cache` in one
    check), common build/dependency output, and anything shaped like an
    installed-package metadata directory."""
    return name.startswith(".") or name in _EXCLUDED_DIR_NAMES or name.endswith(".egg-info")


def _is_test_path(relative: Path) -> bool:
    """Whether `relative` is shaped like a test file or lives under a
    tests directory. Excluded from the sentinel search on purpose -- see
    `_sentinel_via_search`'s docstring: a mutation to a TEST file would
    not prove invariant 7 at all, because pytest collects test files
    straight out of the clone independent of whether `import <package>`
    resolves there."""
    if any(part in ("tests", "test") for part in relative.parts):
        return True
    return relative.name.startswith("test_") or relative.name.endswith("_test.py")


def _source_files(repo: Path) -> tuple[Path, ...]:
    """Every `.py` file under `repo` that plausibly holds real, imported
    package source, as paths relative to `repo` (the same convention
    `verify` requires of `Mutant.path`), sorted for determinism -- same
    repo in, same search order out, every time.

    Prefers `repo / "src"` when it exists (this project's own layout, and
    a common `src`-layout convention other repos share), which already
    excludes a flat repo's top-level tests directory for free. Without a
    `src` directory, walks the whole tree instead, applying `_excluded_
    dir` to every directory component and `_is_test_path` to the result --
    a heuristic, not a parse of the target's own packaging config, and
    documented as such (see the report's concerns) rather than assumed
    exhaustive for every possible repo layout."""
    root = repo / "src" if (repo / "src").is_dir() else repo
    found: list[Path] = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(repo)
        if any(_excluded_dir(part) for part in relative.parts[:-1]):
            continue
        if _is_test_path(relative):
            continue
        found.append(relative)
    return tuple(sorted(found))


def _primary_package(repo: Path) -> str:
    """The package name to check invariant 7's marker against when no
    specific mutant is in scope -- `baseline`'s case, which measures the
    whole unmodified repo rather than one mutant's file. Derived from the
    top-level package of the FIRST real source file `_source_files` finds
    (sorted, so this is deterministic for a given `repo`), via the same
    `_package_name` every mutant-specific caller uses -- so a single-
    package repo (robigo's own, or a typical foreign target) gets
    `baseline` and every later `verify`/`sentinel_ok` call agreeing on the
    same package name without either side re-deriving it differently.

    Raises `ValueError` if `_source_files(repo)` is empty -- there is no
    real source to derive a package from at all, and refusing is the
    honest answer (this module's whole verification standard: guessing
    would be worse). A repo with more than one top-level package picks
    the alphabetically-first source file's package; documented as a
    heuristic, not exhaustive for every possible repo layout, matching
    `_source_files`'s own documented limitation."""
    files = _source_files(repo)
    if not files:
        raise ValueError(
            f"{repo} offers no real source file (via _source_files) to "
            f"derive a package from -- cannot check invariant 7 without one"
        )
    return _package_name(files[0])


def _find_line(source: str, text: str) -> int:
    """The 1-based line number of the single line in `source` that reads
    exactly `text`. Raises if there is not exactly one -- zero means the
    sentinel's target has moved or changed shape since this module was
    written, and more than one would make "the" line ambiguous; either
    way, guessing would be worse than refusing. Used only by
    `_sentinel_fast_path`, which treats this raising as "doesn't apply
    here" (`None`), not as a fatal error."""
    lines = source.splitlines(keepends=True)
    matches = [i + 1 for i, line in enumerate(lines) if line == text]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one line reading {text!r}, found "
            f"{len(matches)} -- the sentinel target has moved or changed "
            f"shape and this module's hardcoded sentinel needs updating"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# The baseline
# ---------------------------------------------------------------------------


def baseline(repo: Path, runner: Runner) -> Baseline:
    """Measures `repo` completely unmodified: one `runner` call, timed by
    this function (not parsed out of the runner's text, so `seconds` is
    honest even against a runner that says nothing about timing). The
    package `runner` is asked to check is derived from `repo`'s own
    source (`_primary_package`, since there is no specific `Mutant` in
    scope here) -- never a fixed name, so this works identically for
    robigo's own repo and for whatever foreign repo `--repo` points at.

    Raises `WrongTreeError` if the runner's own report does not resolve
    inside `repo` (invariant 7) -- there is no safe fallback `Baseline` to
    return for a measurement that might describe the wrong tree; every
    `verify()` call downstream trusts `.broken` unconditionally, so a
    silently-wrong baseline would poison every mutant judged against it.
    Also propagates `_primary_package`'s `ValueError` if `repo` offers no
    real source to derive a package from at all.

    Does NOT assume `.broken` is zero, and does not special-case it either
    -- it is whatever `_broken_count` finds in `runner`'s report. Measured
    2026-08-10: a `git archive` copy of this project's own repo (no
    `.git`) baselined at 6, because the git-dependent tests fail without
    a real `.git` directory."""
    package = _primary_package(repo)
    start = time.monotonic()
    text = runner(repo, package)
    elapsed = time.monotonic() - start
    _assert_in_clone(text, repo)
    return Baseline(
        broken=_broken_count(text), executed=_executed_total(text), seconds=elapsed
    )


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def verify(mutant: Mutant, repo: Path, baseline: Baseline, runner: Runner) -> Verdict:
    """Applies `mutant` to its file inside `repo`, runs `runner`, restores
    the file (always -- even if `runner` raises, even if the verdict is
    "wrong tree"), and judges the result.

    `mutant.path` must be relative to `repo` -- see `_resolve_in_clone`;
    an absolute path is rejected before anything is written, not silently
    misinterpreted by pathlib's `/` operator.

    Invariant 7 first: if the runner's own report does not resolve inside
    `repo`, the result is rejected outright -- `kept=False`, `reason`
    names the wrong tree -- regardless of what the raw counts say. This is
    deliberately checked before the "survived" / "kept" logic below, so a
    wrong-tree run that happens to LOOK like a clean survivor (measured
    2026-08-10: without `PYTHONPATH` forced, every one of 8 real mutants
    read this way) is never reported as an ordinary, trustworthy
    "survived" verdict, and a wrong-tree run that happens to look like
    exactly one new failure is never reported as an ordinary "kept" one.

    Invariant 5: `broken = failures + errors` for this run (`Verdict.
    failures`), and the mutant is judged on `broken - baseline.broken`,
    never on `broken` alone -- a nonzero baseline is not assumed away.

    Invariant 6: `kept=True` only when that net figure is exactly 1 AND
    the runner's report lets the one new failure be isolated by id. The
    id is isolated by counting how many broken-test ids the report names
    in total (`_broken_ids`): when the baseline itself is 0, "exactly one
    net new failure" and "exactly one id in the report" are the same fact,
    and that one id is `test_id`. When the baseline is NOT 0 (measured:
    6, in a `git archive` copy), a real runner's report still names every
    CURRENTLY broken test, not a delta -- so a net-new-failure-of-1
    against a nonzero baseline produces `baseline.broken + 1` ids in the
    report, not 1, and this function correctly refuses to guess which one
    is new: `kept=False`, `test_id=None`, `reason` says so. `Baseline`
    carries a count, not the identities of what was already broken, so
    there is no way to attribute a new failure by id under a nonzero
    baseline without guessing -- and invariant 6 is explicit that a kept
    mutant without a verified id is not acceptable, so this function
    declines rather than guessing.

    Four checks added by the whole-branch review (2026-08-10), all before
    "exactly one" is ever asked, and all producing `kept=False` with a
    `reason` naming which one fired:

    C1/C2 -- `_run_did_not_complete`: a collection error (a mutant that
    breaks a module's IMPORT, e.g. `swapped_args` on a module-level
    `re.compile(pattern, flags)`) makes pytest exit 2 and print
    "Interrupted: N errors during collection" -- never 0 or 1 broken tests
    reported as an ordinary failure, and rejected here before `broken` is
    even computed for the "exactly one" arithmetic. Then the executed-test
    total (`_executed_total`, `passed + broken`) must equal `baseline.
    executed` -- measured 2026-08-10: `PYTEST_ADDOPTS=-x` ambient in the
    shell makes a mutant that broke 2 tests report only 1 (pytest stops at
    the first failure), which passes the exit-code check (a clean `-x`
    exit is still code 1) but shrinks the executed total below the
    baseline's, which THIS check catches.

    I1 -- a keep additionally requires `baseline.broken == 0`: the
    reference patch's own cleanliness (spec 5.1's other half) must be
    verified, not merely assumed transitively from `baseline()` having
    run once. A dirty baseline (measured 2026-08-10: 6, in a `git archive`
    copy) makes EVERY candidate against it unkeepable, regardless of net
    count -- checked before the net computation, not folded into the
    "cannot isolate an id" side effect the old code relied on implicitly.

    C2(c) -- the isolated id must itself look like a pytest node id
    (`"::"` present). A bare file path (`ERROR src/pkg/mod.py`, no `::`)
    names a collection error, not one specific test, and must never become
    a kept record's `test_id` even in the (now unreachable via the checks
    above, but not proven unreachable by construction) case where it is
    the only broken id in the report."""
    text = _apply_and_run(repo, mutant, runner)

    try:
        _assert_in_clone(text, repo)
    except WrongTreeError as exc:
        return Verdict(kept=False, failures=_broken_count(text), test_id=None, reason=str(exc))

    if baseline.broken != 0:
        return Verdict(
            kept=False,
            failures=_broken_count(text),
            test_id=None,
            reason=(
                f"baseline is not clean ({baseline.broken} pre-existing broken "
                f"test(s)) -- a kept mutant requires baseline.broken == 0 so "
                f"the reference patch it certifies is verified clean, not "
                f"merely inferred (I1, whole-branch review 2026-08-10)"
            ),
        )

    incomplete = _run_did_not_complete(text)
    if incomplete is not None:
        return Verdict(
            kept=False, failures=_broken_count(text), test_id=None, reason=incomplete
        )

    executed = _executed_total(text)
    if executed != baseline.executed:
        return Verdict(
            kept=False,
            failures=_broken_count(text),
            test_id=None,
            reason=(
                f"executed {executed} test(s), baseline executed "
                f"{baseline.executed} -- not the same suite ran (a collection "
                f"error or an early exit such as PYTEST_ADDOPTS=-x shrinks "
                f"this number without necessarily changing failures+errors "
                f"alone; whole-branch review C1/C2, 2026-08-10)"
            ),
        )

    broken = _broken_count(text)
    net = broken - baseline.broken
    if net != 1:
        if net <= 0:
            reason = f"survived: {broken} broken vs baseline {baseline.broken} (net {net})"
        else:
            reason = (
                f"broke too many: {broken} broken vs baseline {baseline.broken} "
                f"(net {net}, want exactly 1)"
            )
        return Verdict(kept=False, failures=broken, test_id=None, reason=reason)

    ids = _broken_ids(text)
    if len(ids) != 1:
        return Verdict(
            kept=False,
            failures=broken,
            test_id=None,
            reason=(
                f"exactly one net new failure ({broken} broken vs baseline "
                f"{baseline.broken}) but the runner's report names "
                f"{len(ids)} broken test ids, not 1 -- cannot isolate which "
                f"one is new without guessing, so no diagnostic test id can "
                f"be recorded"
            ),
        )
    if "::" not in ids[0]:
        return Verdict(
            kept=False,
            failures=broken,
            test_id=None,
            reason=(
                f"the one broken id the runner reported ({ids[0]!r}) is not a "
                f"pytest node id (no '::') -- a bare file/module path names a "
                f"collection error, not one specific test (C2(c), whole-branch "
                f"review 2026-08-10)"
            ),
        )
    return Verdict(
        kept=True, failures=broken, test_id=ids[0], reason="exactly one net new failure"
    )


# ---------------------------------------------------------------------------
# The real runner
# ---------------------------------------------------------------------------

_IMPORT_CHECK_TIMEOUT = 30
_PYTEST_TIMEOUT = 300


def pytest_runner(repo: Path, package: str) -> str:
    """The one real `Runner`: shells out to pytest inside `repo`. Never
    touches the network, a model daemon, or port 8081 -- this runs pytest
    and one `python -c` import check, both fully local subprocesses.

    Forces `PYTHONPATH` to `repo`'s own `src`, and `PYTHONDONTWRITEBYTECODE
    =1` -- stale `.pyc` files have confused two implementers on this
    project, and without `PYTHONPATH` forced a subprocess started inside a
    copied tree can still import a REAL install's source instead of the
    clone's (measured 2026-08-10 for robigo's own editable install:
    `robigo.__file__` resolved to the real repo by default and into the
    copy only with `PYTHONPATH` set), which silently inverts a mutant's
    result to "survived".

    Prepends a `MODULE_UNDER_TEST=<path>` marker line reporting exactly
    what `import <package>` resolved to in that SAME environment -- run as
    its own quick subprocess, before pytest, under identical `cwd`/`env`
    -- which is what `_assert_in_clone` checks (invariant 7) before any
    caller trusts a result. `package` is never hardcoded here: it names
    whatever top-level package the caller wants checked (robigo's own, or
    a foreign repo's own package -- `_apply_and_run`/`baseline` derive it
    and pass it down). Coordinator review (2026-08-10) caught an
    incomplete first fix: generalising the SENTINEL's target without also
    generalising this marker still asked "does robigo resolve inside the
    clone" for every repo, which rejects every foreign repo for the wrong
    reason (robigo was never the code under test there, so of course it
    resolves elsewhere) rather than for a real problem.

    If the import check fails for ANY reason -- `package` unimportable in
    that environment, not a valid identifier, a broken clone -- the
    marker is omitted entirely rather than guessed, and every caller's
    `_assert_in_clone` correctly rejects the run for having no marker at
    all. An unimportable target is "cannot certify", never silently
    "certified" by falling back to some other answer.

    Runs with `--tb=no -rfE`: no tracebacks (keeps output bounded and
    deterministic), but the short summary info section that names every
    broken test's id, which `_broken_ids` depends on.

    `PYTEST_ADDOPTS` and `PYTEST_PLUGINS` are dropped from the copied
    environment before either subprocess runs (whole-branch review C2,
    ruled 2026-08-10) -- both are ordinary things to have exported in a
    real operator's shell, and both change what pytest actually does
    without appearing anywhere in `env`'s own values a caller might think
    to check: measured directly, `PYTEST_ADDOPTS=-x` ambient in the
    launching shell turned a mutant that broke 2 tests into a report of
    exactly 1 (pytest stopped at the first failure), which the OLD
    `verify()` scored as "exactly one net new failure" -- a false keep,
    manufactured by a shell variable this function never even looked at.
    `os.environ.copy()` still inherits every OTHER ambient variable (this
    is deliberately not a full sanitisation) -- only these two, plugin-
    loading and pytest-option-injecting by design, are the ones proven to
    change a VERDICT rather than merely cosmetic output.

    Prepends an `EXIT_CODE=<n>` marker naming pytest's own real return
    code -- previously computed and thrown away entirely. A collection
    error exits 2 (`"Interrupted: N errors during collection"`), never 0
    or 1; `verify`'s `_run_did_not_complete` rejects on this before ever
    asking "exactly one" (whole-branch review C1)."""
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(repo / "src")

    import_check = subprocess.run(
        [sys.executable, "-c", f"import {package}; print({package}.__file__)"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=_IMPORT_CHECK_TIMEOUT,
    )
    marker = (
        f"MODULE_UNDER_TEST={import_check.stdout.strip()}\n"
        if import_check.returncode == 0
        else ""
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "-rfE"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=_PYTEST_TIMEOUT,
    )
    return f"{marker}EXIT_CODE={result.returncode}\n{result.stdout}{result.stderr}"
