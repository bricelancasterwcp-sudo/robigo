# src/robigo/profile/verify.py
"""The verifier: decides whether a `Mutant` (task 1) becomes a corpus
record. Everything downstream -- stage 4's number, and the project's 40%
kill criterion -- inherits its trustworthiness from this module, so every
function here is built to REFUSE to certify a result it cannot back up,
rather than to report the friendliest-looking answer.

`runner: Callable[[Path], str]` is injected everywhere real work happens
(`sentinel_ok`, `baseline`, `verify`), so this module's own tests are
offline and instant -- no daemon, no GPU, no network, no real pytest
subprocess. `pytest_runner`, below, is the one real implementation: it
shells out to pytest inside `repo` and is what a production caller (task
4's CLI) passes in place of a test's canned callable.

Four invariants, each one closing a false result this project actually
produced while measuring plan 04 (2026-08-10):

  4. `sentinel_ok` must prove the harness can SEE a break before any
     survival is believed -- a blind harness scored 8 of 8 mutants as
     survivors, a perfect false negative.
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
     mutant appears to survive.

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

from robigo.profile.corpus import Mutant, _apply

Runner = Callable[[Path], str]
"""A runner takes the repo root it should test and returns the raw text of
what it did there. The real runner (`pytest_runner`) returns the pytest
short-test-summary output, prefixed with a `ROBIGO_MODULE=<path>` marker
line naming what `import robigo` resolved to in that same subprocess
environment -- the fact invariant 7 checks before anything else is
believed. A test's canned runner constructs that same text by hand, which
is what makes every test in this module able to run with no daemon, no
GPU, no network, and no real pytest subprocess."""


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
    runner whose text says nothing about timing at all."""

    broken: int
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

_MODULE_MARKER = re.compile(r"^ROBIGO_MODULE=(.+)$", re.MULTILINE)
"""The marker `pytest_runner` prepends, naming what `import robigo`
resolved to in the SAME subprocess environment pytest itself ran in. A
canned test runner that doesn't care about invariant 7 still includes a
well-formed marker (via this test file's `_output` helper) pointing inside
the fixture repo -- only the tests that specifically exercise invariant 7
give it something else."""


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


def _module_path(text: str) -> Path | None:
    """The resolved path the `ROBIGO_MODULE=` marker names, or `None` if
    the runner's report carries no such marker at all -- treated
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
    trusting any broken-test count -- robigo's editable install means a
    subprocess started in a copied tree still imports the real repo's
    source unless `PYTHONPATH` was forced, and when that happens every
    mutant appears to survive (measured 2026-08-10: 8 of 8)."""
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
matter, not merely fail to parse."""


def sentinel_ok(repo: Path, runner: Runner) -> bool:
    """Invariant 4: applies the known-fatal `estimate_tokens` change to
    `repo`'s own copy, runs `runner`, and returns whether the harness
    reported ANY breakage -- not `True` unless it did. Restores the file
    to its original content before returning, in either direction, and
    even if `runner` raises.

    Returns `False` (not raises) for either way a result can fail to be
    trusted: the runner reported zero breakage (the harness is blind --
    measured 2026-08-10, a blind harness reported 8 of 8 mutants
    surviving), or the runner reported breakage but on the wrong tree
    (invariant 7 -- a "broken" result on the real repo's source proves
    nothing about this clone). Both are legitimate "not proven" outcomes a
    caller should abort on the same way.

    Raises if the sentinel itself cannot be constructed -- `repo`'s copy
    of `_SENTINEL_PATH` is missing, or does not contain
    `_SENTINEL_ORIGINAL` at exactly one line. That is not "the harness
    looks blind"; it is a bug in this module or a repo-shape drift, and
    hiding it behind a bare `False` would be exactly the kind of quiet
    non-certification this project's whole verification standard exists
    to prevent."""
    target = _resolve_in_clone(repo, _SENTINEL_PATH)
    source = target.read_text()
    line = _find_line(source, _SENTINEL_ORIGINAL)
    mutant = Mutant(target, line, _SENTINEL_ORIGINAL, _SENTINEL_MUTATED, "sentinel")
    target.write_text(_apply(source, mutant))
    try:
        text = runner(repo)
    finally:
        target.write_text(source)

    try:
        _assert_in_clone(text, repo)
    except WrongTreeError:
        return False
    return _broken_count(text) > 0


def _find_line(source: str, text: str) -> int:
    """The 1-based line number of the single line in `source` that reads
    exactly `text`. Raises if there is not exactly one -- zero means the
    sentinel's target has moved or changed shape since this module was
    written, and more than one would make "the" line ambiguous; either
    way, guessing would be worse than refusing."""
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
    honest even against a runner that says nothing about timing).

    Raises `WrongTreeError` if the runner's own report does not resolve
    inside `repo` (invariant 7) -- there is no safe fallback `Baseline` to
    return for a measurement that might describe the wrong tree; every
    `verify()` call downstream trusts `.broken` unconditionally, so a
    silently-wrong baseline would poison every mutant judged against it.

    Does NOT assume `.broken` is zero, and does not special-case it either
    -- it is whatever `_broken_count` finds in `runner`'s report. Measured
    2026-08-10: a `git archive` copy of this project's own repo (no
    `.git`) baselined at 6, because the git-dependent tests fail without
    a real `.git` directory."""
    start = time.monotonic()
    text = runner(repo)
    elapsed = time.monotonic() - start
    _assert_in_clone(text, repo)
    return Baseline(broken=_broken_count(text), seconds=elapsed)


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
    declines rather than guessing."""
    target = _resolve_in_clone(repo, mutant.path)
    source = target.read_text()
    mutated = _apply(source, mutant)
    target.write_text(mutated)
    try:
        text = runner(repo)
    finally:
        target.write_text(source)

    try:
        _assert_in_clone(text, repo)
    except WrongTreeError as exc:
        return Verdict(kept=False, failures=_broken_count(text), test_id=None, reason=str(exc))

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
    return Verdict(
        kept=True, failures=broken, test_id=ids[0], reason="exactly one net new failure"
    )


# ---------------------------------------------------------------------------
# The real runner
# ---------------------------------------------------------------------------

_IMPORT_CHECK_TIMEOUT = 30
_PYTEST_TIMEOUT = 300


def pytest_runner(repo: Path) -> str:
    """The one real `Runner`: shells out to pytest inside `repo`. Never
    touches the network, a model daemon, or port 8081 -- this runs pytest
    and one `python -c` import check, both fully local subprocesses.

    Forces `PYTHONPATH` to `repo`'s own `src`, and `PYTHONDONTWRITEBYTECODE
    =1` -- stale `.pyc` files have confused two implementers on this
    project, and without `PYTHONPATH` forced a subprocess started inside a
    copied tree still imports the real repo through robigo's editable
    install (measured 2026-08-10: `robigo.__file__` resolves to the real
    repo by default and into the copy only with `PYTHONPATH` set), which
    silently inverts every mutant's result to "survived".

    Prepends a `ROBIGO_MODULE=<path>` marker line reporting exactly what
    `import robigo` resolved to in that SAME environment -- run as its own
    quick subprocess, before pytest, under identical `cwd`/`env` -- which
    is what `_assert_in_clone` checks (invariant 7) before any caller
    trusts a result. If that import check fails (a broken clone, an
    uninstalled package), the marker is omitted entirely rather than
    guessed, and every caller's `_assert_in_clone` correctly rejects the
    run for having no marker at all.

    Runs with `--tb=no -rfE`: no tracebacks (keeps output bounded and
    deterministic), but the short summary info section that names every
    broken test's id, which `_broken_ids` depends on."""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(repo / "src")

    import_check = subprocess.run(
        [sys.executable, "-c", "import robigo; print(robigo.__file__)"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=_IMPORT_CHECK_TIMEOUT,
    )
    marker = (
        f"ROBIGO_MODULE={import_check.stdout.strip()}\n" if import_check.returncode == 0 else ""
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "-rfE"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=_PYTEST_TIMEOUT,
    )
    return marker + result.stdout + result.stderr
