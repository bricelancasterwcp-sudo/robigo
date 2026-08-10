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
    here would measure a tool nobody has.

    This is deliberately checked against `task_for`'s OWN return value, not
    against the rendered prompt `robigo.context.render.render` eventually
    sends the model (fix round 1: flagged as undocumented). The rendered
    prompt legitimately contains `record.path` and its source text -- scope
    resolution (`robigo.context.scope.resolve`) reads the pytest failure
    `robigo.adapters.python_.PythonAdapter` reports and shows the model the
    file *that failure points at*, which is exactly how a user of the real
    tool would see it too. Testing the rendered prompt for the path's
    absence would therefore be testing a property that is FALSE for the
    real, correctly-working tool, not a property that failing this test
    would mean something is wrong. `task_for`'s return value -- the literal
    task STRING handed to `robigo.loop.run` as its `task` argument, never
    augmented with ground truth -- is the one surface spec 4.2 actually
    constrains: it is the only input that could smuggle in `line`/`fixed`/
    an explicit `path` the model would otherwise have to discover for
    itself by running the suite, exactly as a real user's model must."""
    return f"the test {record.test_id} fails; make it pass"


def _anchor_path(record: CorpusRecord, repo: Path) -> Path:
    """The test file the anchor hash guards -- the file part of the pytest
    node id, resolved inside the clone."""
    return _resolve_in_clone(repo, Path(record.test_id.split("::")[0]))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class _Pristine:
    """`repo`'s own branch name and HEAD sha, exactly as they stood before
    the FIRST attempt this process ever ran against this clone touched
    anything. `reset_clone` restores to this, not to "whatever branch/
    commit the tree happens to be on right now" (Critical 2, fix round 1)."""

    branch: str
    sha: str


_PRISTINE_CACHE: dict[Path, _Pristine] = {}
"""Captured ONCE per (process, repo), never re-derived per attempt.

Task 5/7's loop reuses ONE clone across every record and seed -- roughly
940 `attempt_repair` calls in a single process, per the plan. Re-deriving
"pristine" from `repo`'s CURRENT branch/HEAD on every call is exactly the
bug this cache exists to prevent: measured directly (fix round 1,
Critical 2), a mutated `run()` that lands `use_git=True`'s loop on a
`robigo/*` branch with a real commit -- including, in the worst case, one
that commits the STAGED DEFECT itself, because `use_git=True`'s own
snapshot-before-first-patch step runs `git add -A` unconditionally --
means the naive `git checkout -- . && git clean -fdq` this module used to
run restores to THAT branch's index, never back to the tree the record
was actually supposed to start from, and never returns to the original
branch at all. Every later record in the same run then inherits record
1's still-committed defect: `state.broken == 0` can never hold again, no
matter how correctly the model repairs ITS OWN record, and the project's
40% kill criterion would fire on this bug, not on the model. Capturing
pristine once, before anything has run, and reusing that same captured
value for every later `reset_clone` call breaks that chain: attempt 2's
`reset_clone` restores to the ORIGINAL branch and the ORIGINAL sha,
regardless of where attempt 1 left the tree.

Keyed by the resolved repo path, not a bare module-level scalar, so two
distinct clones used within the same process (two `--repo` runs in one
Python session, or two test fixtures in the same pytest session) never
share state -- each gets its own pristine value, captured independently
the first time `reset_clone` sees that particular path."""


def _pristine(repo: Path) -> _Pristine:
    """`repo`'s captured pristine state, computing and caching it on the
    first call for this `repo` and returning the cached value on every
    later call -- see `_PRISTINE_CACHE`'s docstring for why re-deriving
    this on every call would be wrong. Deliberately reads `repo`'s CURRENT
    branch/HEAD only when nothing is cached yet; a caller that needs this
    read at some OTHER moment (mid-run, after corruption) has already lost
    the information this function exists to preserve."""
    key = repo.resolve()
    cached = _PRISTINE_CACHE.get(key)
    if cached is not None:
        return cached
    branch = _git_text(repo, "branch", "--show-current")
    sha = _git_text(repo, "rev-parse", "HEAD")
    captured = _Pristine(branch, sha)
    _PRISTINE_CACHE[key] = captured
    return captured


def _git_text(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


_ROBIGO_BRANCH_GLOB = "robigo/*"
"""The exact pattern `robigo.apply.safety.start_branch` names its own
branches with (`f"robigo/{slug}-{number}"`) -- matched here, not
reimplemented, so a rename of that convention only needs to change in one
place plus this literal, not a parallel regex that could silently stop
matching."""


def _delete_stray_branches(repo: Path) -> None:
    """Deletes every local branch matching `_ROBIGO_BRANCH_GLOB` that a
    prior attempt's `use_git=True` loop run may have created and left
    behind (spec 4.3.3). Across ~940 attempts against one repeatedly-
    reused clone, an undeleted branch from attempt 1 would still be
    sitting there at attempt 940 -- harmless by itself once `reset_clone`
    always checks out the pristine branch first (this function runs AFTER
    that checkout, so it never tries to delete the branch currently
    checked out, which git refuses), but an unbounded, ever-growing branch
    list is exactly the kind of state leak spec 4.3.3 asks to be closed,
    not merely worked around."""
    listing = _git_text(
        repo, "branch", "--list", _ROBIGO_BRANCH_GLOB, "--format=%(refname:short)"
    ).split()
    if listing:
        subprocess.run(["git", "branch", "-D", *listing], cwd=repo,
                       check=True, capture_output=True)


def reset_clone(repo: Path) -> None:
    """Discard everything every PRIOR attempt did, all the way back to the
    clone's own pristine state -- not merely `git checkout -- .` against
    whatever the tree currently reads, which only restores TRACKED files
    to the CURRENT commit's content and does nothing about a wrong branch
    or a commit `use_git=True`'s own loop made (Critical 2, fix round 1;
    see `_PRISTINE_CACHE`'s docstring for the measured failure this
    replaces). Runs before EVERY attempt, not once per record (spec
    4.3.3): `git checkout -f <pristine.branch>` (forced, so a prior
    attempt's uncommitted leftovers on some OTHER branch never block the
    switch), `git reset --hard <pristine.sha>` (discards every commit made
    since, on whatever branch is now checked out), `git clean -fdq`
    (removes untracked stray files), then `_delete_stray_branches` (so
    `robigo/*` branches do not accumulate across the whole run)."""
    pristine = _pristine(repo)
    subprocess.run(["git", "checkout", "-f", pristine.branch], cwd=repo,
                   check=True, capture_output=True)
    subprocess.run(["git", "reset", "--hard", pristine.sha], cwd=repo,
                   check=True, capture_output=True)
    subprocess.run(["git", "clean", "-fdq"], cwd=repo, check=True,
                   capture_output=True)
    _delete_stray_branches(repo)


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
    """One (record, seed) repair attempt through the SHIPPED tool
    (`use_git=True`, the real `turn_cap`, the real `codec`,
    `allow_test_edits=False` -- spec 4.3.1: a defect on any of those paths
    counts against the tool, because a user meets it), judged strictly.

    Task 5 calls this in a tight loop, once per (record, seed) pair,
    reusing the SAME `repo` clone across roughly 940 calls in a single
    process. Every side effect this function has on `repo` -- staging the
    defect, and everything `use_git=True`'s own loop run does to the tree
    (branches, commits) -- must be fully undone before the NEXT call's
    `reset_clone` runs, or one record's run corrupts every later record's
    measurement; `reset_clone`/`_PRISTINE_CACHE` exist specifically to
    guarantee that (Critical 2, fix round 1).

    The judgment below runs in a SPECIFIC, load-bearing order -- getting
    it wrong is invisible in a spot check (an excluded, unrescuable
    `stalled` attempt and a genuinely failed one both read `passed=False`
    in isolation; only a rate computed over many attempts, or a check of
    `.excluded` specifically, exposes the difference), which is exactly
    how this shipped with the wrong order the first time:

    1. `_INFRA_OUTCOMES` first, immediately after `run()` returns -- a
       daemon that never answered proves nothing about the MODEL, in
       either direction, and must never become `passed=True` (a false
       positive) or an ordinary scored failure (a false negative against
       a tool the model never actually got to use).
    2. `result.outcome != "pass"` SECOND, before any suite reading at all.
       A `stalled`, `refused`, or `budget_exhausted` outcome is already a
       definitive, unrescuable failure -- no suite reading can turn a
       turn-cap exhaustion into a repair, so nothing below this line may
       ever promote one of those outcomes to `excluded` on the strength of
       what the suite happens to look like afterward. (Critical 1, fix
       round 1: measured directly, moving this check BELOW the
       `incomplete`/`executed`-mismatch/exception guards let a model that
       stalled after leaving a syntax error, or after deleting a
       neighbouring test, or after breaking the target's own import, read
       as `excluded` instead of failed through three separate doors --
       silently removing exactly the failure mode a turn-capped local
       model produces most often from BOTH the numerator and the
       denominator of every rate downstream, biasing the measured repair
       rate upward, concentrated precisely where a weak model fails most.)
    3. Only once `outcome == "pass"` is confirmed does suite-state reading
       begin: `incomplete`, then the `executed` total against `base`, then
       the anchor hash. These exist ONLY to catch a `"pass"` outcome that
       is actually a false positive (a truncated suite run, a defect that
       silently escaped detection, a rewritten anchor test) -- they must
       never manufacture an exclusion for an outcome that was already a
       certain failure by step 2.

    Wrapped in a last-resort `except Exception` (Important 3, fix round
    1): staging a defect for a `test_id` whose file does not exist, or
    `run()` itself raising -- `robigo.loop.run` deliberately RE-RAISES any
    escaping exception after recording it, on the assumption that
    `cli.main` is the catcher; `attempt_repair` is a SECOND caller with no
    such handler upstream of it -- must produce one `excluded` `Attempt`
    and let the other ~939 attempts continue, never abort a run that can
    take on the order of 12 hours."""
    try:
        return _judge(record, repo, client, seed=seed, codec=codec, base=base,
                      turn_cap=turn_cap, runner=runner)
    except Exception as exc:
        return Attempt(record.name, seed, False, "", 0, 0,
                       f"unexpected error: {exc!r}")


def _judge(
    record: CorpusRecord,
    repo: Path,
    client,
    *,
    seed: int,
    codec: str,
    base: Baseline,
    turn_cap: int,
    runner: Runner,
) -> Attempt:
    """The unguarded body `attempt_repair` wraps in its last-resort safety
    net -- split out so every ANTICIPATED failure point keeps its own
    specific, diagnostic `excluded` reason (via the `excluded` closure
    below), while `attempt_repair` itself only has to catch whatever this
    function did not anticipate. See `attempt_repair`'s docstring for the
    load-bearing ORDER these checks run in."""
    def excluded(why: str, outcome: str = "", turns: int = 0) -> Attempt:
        return Attempt(record.name, seed, False, outcome, turns, 0, why)

    try:
        reset_clone(repo)
        break_it(record, repo)
        anchor = _anchor_path(record, repo)
        before = _sha(anchor)
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError) as exc:
        return excluded(f"could not stage the defect: {exc}")

    try:
        # The SHIPPED tool: real turn cap, real codec, git on, test edits
        # off (spec 4.3.1). A defect on any of those paths counts against
        # the tool, because a user meets it.
        result: RunResult = run(
            task_for(record), repo, client, PythonAdapter(),
            codec=codec, turn_cap=turn_cap, allow_test_edits=False, use_git=True,
        )
    except Exception as exc:
        # loop.run's own contract: it re-raises whatever escapes _execute
        # after recording it, assuming cli.main is the catcher. It is not,
        # here -- this is the second caller Important 3 (fix round 1)
        # named, and one record's internal error must not abort the run.
        return excluded(f"the loop raised: {exc!r}")

    if result.outcome in _INFRA_OUTCOMES:
        return excluded(f"loop infrastructure: {result.detail}",
                        result.outcome, result.turns)

    if result.outcome != "pass":
        # A definitive, unrescuable model failure (Critical 1, fix round
        # 1) -- stalled/refused/budget_exhausted, never excluded, and
        # never subjected to the suite-reading checks below, which exist
        # only to catch a "pass" that was actually a false positive.
        return Attempt(record.name, seed, False, result.outcome, result.turns, 0, None)

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

    try:
        anchor_intact = _sha(anchor) == before
    except OSError as exc:
        return excluded(f"could not verify the anchor after the run: {exc}",
                        result.outcome, result.turns)

    passed = (
        state.broken == 0
        # `record.test_id not in state.broken_ids` cannot currently flip
        # this verdict on its own: `state.broken == 0` already implies
        # `state.broken_ids == ()` (SuiteState's own invariant), so the
        # left conjunct alone already forces this term true whenever it is
        # even evaluated. Kept as defensive redundancy, not because it is
        # load-bearing today -- a future change to what `broken` counts
        # (e.g., a broken-but-not-failing state) is exactly the kind of
        # drift this line would catch for free.
        and record.test_id not in state.broken_ids
        and anchor_intact
    )
    return Attempt(record.name, seed, passed, result.outcome, result.turns,
                   0, None)
