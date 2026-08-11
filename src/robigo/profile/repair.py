# src/robigo/profile/repair.py
from __future__ import annotations

import functools
import hashlib
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from robigo.adapters.python_ import PythonAdapter
from robigo.loop import RunResult, run
from robigo.profile.corpus_io import CorpusRecord
from robigo.profile.verify import (
    Baseline, Runner, SuiteState, _package_name, _resolve_in_clone,
    baseline as measure_baseline, pytest_runner, suite_state,
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
    NEITHER the numerator nor the denominator of any rate.

    `repeats` is `result.repeats` (`robigo.loop.RunResult.repeats`) carried
    through verbatim -- how many turns of THIS attempt re-emitted an
    (action, reply) pair the attempt had already tried, a TOTAL rather
    than a consecutive streak (see that field's own docstring for why the
    distinction matters). It stays `0` for an attempt that never reached
    `robigo.loop.run` at all -- a staging failure (a missing anchor file,
    a corrupted clone) or the loop itself raising -- for the same reason
    `turns` stays `0` there: no turn ever happened, so there is nothing
    to have repeated. Task 6's `robigo.profile.discipline.stage5_discipline`
    is this field's consumer; an `excluded` attempt's `repeats` (like its
    `turns`) is kept for the record but contributes to neither of that
    stage's metrics, exactly as `excluded is not None` already excludes it
    from `Stage4.rate`."""

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


class CorruptedCloneError(RuntimeError):
    """Raised by `_pristine` when `repo` is ALREADY checked out on a
    `robigo/*` branch before any attempt in this process has touched it
    (Important B, fix round 2). `robigo.loop.run` never checks back out
    after an attempt -- it only hands back an undo recipe (`UndoInfo`),
    never applies it -- so a clone left mid-run by an earlier process, or
    a resumed run against a reused clone directory, sits on exactly this
    kind of branch. Capturing THAT as "pristine" is the same class of
    error re-deriving pristine from a dirty tree would be (see
    `_PRISTINE_CACHE`'s docstring): every one of the ~940 attempts in the
    run would then silently restore to a corrupted starting point instead
    of the real one, all of them scored against ground truth that was
    never actually there.

    Deliberately NOT caught by `attempt_repair`'s own last-resort safety
    net (`except Exception`) -- unlike an unanticipated PER-RECORD
    surprise, which affects only that one record, this is a property of
    the shared `repo` itself, identical for every attempt in the run.
    Swallowing it into one `excluded` `Attempt` and continuing would just
    repeat the identical exclusion ~940 times, silently burning the whole
    run for nothing instead of failing loudly, immediately, and once, the
    way a `repo` set up wrong deserves to fail."""


class InterpreterMismatchError(RuntimeError):
    """Raised by `stage4_repair`, ONCE, before the `(record, seed)` grid
    starts (fix round 2, 2026-08-10 review) -- when `python` does not
    reproduce the corpus's own recorded `Baseline.executed` against a
    fresh, pristine run of `repo`'s suite.

    Fix round 1 gave the loop and the judge ONE shared interpreter instead
    of two independently-defaulted ones, but that alone does not stop a
    caller from naming a WRONG one explicitly: measured live, `--python
    /usr/bin/python3` (a real interpreter, genuinely importable, but with
    no `pytest` installed) reproduces the EXACT failure shape fix round 1
    removed -- every attempt excluded, `repair_rate: None`, the whole
    grid's ~12h burned for nothing -- just under a different message
    (`"... cannot import pytest"` from `PythonAdapter._preflight`, on
    attempt 1 of ~940, discovered no sooner than a working interpreter's
    OWN first attempt would have discovered it).

    Checking "can this interpreter import pytest at all" is necessary but
    not sufficient (fix round 2's review, verbatim): a `--python` that HAS
    `pytest` but a DIFFERENT dependency set than whatever measured `base`
    can still diverge on `state.executed` for reasons that have nothing to
    do with the model (different package versions changing collection,
    skipped tests, etc.) -- exactly the property `_judge`'s own
    `state.executed != base.executed` exclusion rule depends on, and the
    one this check reproduces directly, once, up front, via the SAME
    `verify.baseline` call `robigo corpus` used to measure `base` in the
    first place (never a second, independently-written comparison that
    could drift from `_judge`'s own).

    Deliberately NOT caught by `attempt_repair`'s own last-resort safety
    net -- it never reaches `attempt_repair` at all, by construction (it
    is raised by `stage4_repair` before the loop that calls
    `attempt_repair` even starts) -- and deliberately not swallowed by any
    broad `except Exception` in `run_profile`/`cli.profile_main` either,
    for the identical reason `CorruptedCloneError` is not: this is a
    property of `python` itself, identical for every attempt the grid
    would otherwise run, and failing loudly once, before spending 12h
    discovering it 940 times, is the whole point."""


@dataclass(frozen=True)
class _Pristine:
    """`repo`'s own branch name and HEAD sha, exactly as they stood before
    the FIRST attempt this process ever ran against this clone touched
    anything. `reset_clone` restores to this, not to "whatever branch/
    commit the tree happens to be on right now" (Critical 2, fix round 1).

    `branch == ""` means the clone was in DETACHED HEAD at capture time --
    `git branch --show-current` prints nothing there (Important A, fix
    round 2). This is not an edge case: spec 4.1 says stage 4 runs
    "against a clone ... checked out at its recorded `source_sha`", and a
    plain `git checkout <sha>` produces EXACTLY a detached HEAD. Measured
    directly: treating `""` as an ordinary branch name made `reset_clone`
    run `git checkout -f ''`, which exits 128, excluding EVERY attempt in
    the run -- the old, pre-Critical-2 naive `reset_clone` handled
    detached HEAD fine, so this was a regression introduced BY the
    Critical-2 fix, not a pre-existing gap. `checkout_target` is what
    `reset_clone` actually passes to `git checkout -f`: the branch name
    when attached, the sha itself when detached -- both correctly restore
    the exact starting point either way."""

    branch: str
    sha: str

    @property
    def checkout_target(self) -> str:
        return self.branch if self.branch else self.sha


_PRISTINE_CACHE: dict[tuple[Path, str], _Pristine] = {}
"""Captured ONCE per (process, repo IDENTITY), never re-derived per
attempt.

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

Keyed by `_repo_identity(repo)` -- `(resolved path, root-commit sha)`, NOT
the resolved path alone (Important C, fix round 2). Path alone has a
silent wrong-pristine mode: measured directly, a DIFFERENT repo re-cloned
at the SAME filesystem path after an earlier run silently reused the
earlier repo's cached `_Pristine` and `reset_clone` applied it without
complaint (`passed=True, excluded=None`, at a cached sha the ACTUAL
clone's history did not even contain). The root commit -- invariant to
whichever commit HEAD happens to be checked out at right now -- identifies
WHICH repository's history this is, so a different repo reusing the same
path cannot collide with a stale entry. `_pristine` additionally validates
the cached sha still resolves (`_sha_exists`) before trusting it, belt-
and-suspenders alongside the compound key."""


def _repo_identity(repo: Path) -> tuple[Path, str]:
    """`(resolved path, root-commit sha)` -- see `_PRISTINE_CACHE`'s
    docstring for why path alone is not enough. The root commit is the
    very FIRST commit in `repo`'s history, found once and cheaply via
    `git rev-list --max-parents=0 HEAD`; if history somehow has more than
    one root (unrelated histories merged), the first one listed is used --
    good enough to distinguish "this repository" from "some other
    repository", which is all this key needs to do."""
    root = _git_text(repo, "rev-list", "--max-parents=0", "HEAD")
    return repo.resolve(), root.split()[0]


def _sha_exists(repo: Path, sha: str) -> bool:
    """Whether `sha` still resolves to a real commit object inside
    `repo` -- validated before trusting a cached `_Pristine` (Important
    C, fix round 2), belt-and-suspenders alongside the compound cache key
    above: a `git gc`/prune, or any other way a commit could vanish from
    a long-lived clone across a ~12 h run, must not silently hand back a
    sha `git reset --hard` can no longer reach. One extra `git cat-file`
    call per attempt is nothing against a run measured in tens of seconds
    per attempt."""
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=repo,
        capture_output=True,
    )
    return result.returncode == 0


def _pristine(repo: Path) -> _Pristine:
    """`repo`'s captured pristine state, computing and caching it on the
    first call for this `repo` (keyed by `_repo_identity`, not path alone
    -- Important C) and returning the cached value on every later call,
    once the cached sha is confirmed to still resolve (`_sha_exists` --
    Important C's second half). See `_PRISTINE_CACHE`'s docstring for why
    re-deriving this on every call, instead of caching, would be wrong.

    Raises `CorruptedCloneError` if `repo` is ALREADY on a `robigo/*`
    branch the first time this runs for a given identity (Important B) --
    see that exception's own docstring for why this is a hard failure,
    not an ordinary excluded attempt."""
    key = _repo_identity(repo)
    cached = _PRISTINE_CACHE.get(key)
    if cached is not None and _sha_exists(repo, cached.sha):
        return cached
    branch = _git_text(repo, "branch", "--show-current")
    if branch.startswith("robigo/"):
        raise CorruptedCloneError(
            f"{repo} is already checked out on {branch!r} before any "
            f"attempt in this run has touched it. This looks like a "
            f"clone left mid-run by an earlier process (robigo.loop.run "
            f"never checks back out after an attempt -- it only hands "
            f"back an undo recipe) or a resumed run against a reused "
            f"clone directory. Restore {repo} to its real original "
            f"branch/commit (its recorded source_sha) before retrying -- "
            f"capturing this branch as \"pristine\" would apply the same "
            f"corruption to every attempt in the run."
        )
    sha = _git_text(repo, "rev-parse", "HEAD")
    captured = _Pristine(branch, sha)
    _PRISTINE_CACHE[key] = captured
    return captured


def clear_pristine_cache() -> None:
    """Test-only escape hatch: drops every cached `_Pristine` value.
    Production code never calls this -- Task 5/7's driver runs once per
    process, against one `repo`, for the process's whole life, which is
    exactly what `_PRISTINE_CACHE` is FOR. Most tests get natural
    isolation for free (`tmp_path` gives each one a unique path, and each
    fresh `git init` gives it a unique root-commit sha, so `_repo_identity`
    never collides across tests) -- this function exists for a test that
    specifically wants to exercise cache/re-derivation or cross-repo-
    identity behaviour directly, without relying on that natural
    uniqueness."""
    _PRISTINE_CACHE.clear()


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
    behind (spec 4.3.3), EXCLUDING whatever branch is currently checked
    out -- checked directly here via a fresh `git branch --show-current`,
    not merely assumed safe because `reset_clone` already switched away
    first (Important B, fix round 2: `_pristine` can now only ever
    capture a NON-`robigo/*` branch as pristine, since it raises
    `CorruptedCloneError` instead of accepting one -- but this function no
    longer trusts that invariant blindly from upstream; it re-derives
    "what's current" itself, so the guarantee below is true by
    construction, not by an assumption this function cannot see broken).
    `git branch -D <currently-checked-out-branch>` exits 1, which -- if
    that invariant were ever violated by a future change here or upstream
    -- would turn EVERY subsequent attempt in the run into an exclusion,
    identically, one per attempt, for the life of the run. Across ~940
    attempts against one repeatedly-reused clone, an undeleted stray
    branch from attempt 1 would otherwise still be sitting there at
    attempt 940; an unbounded, ever-growing branch list is exactly the
    kind of state leak spec 4.3.3 asks to be closed, not merely worked
    around."""
    current = _git_text(repo, "branch", "--show-current")
    listing = _git_text(
        repo, "branch", "--list", _ROBIGO_BRANCH_GLOB, "--format=%(refname:short)"
    ).split()
    doomed = [branch for branch in listing if branch != current]
    if doomed:
        subprocess.run(["git", "branch", "-D", *doomed], cwd=repo,
                       check=True, capture_output=True)


def reset_clone(repo: Path) -> None:
    """Discard everything every PRIOR attempt did, all the way back to the
    clone's own pristine state -- not merely `git checkout -- .` against
    whatever the tree currently reads, which only restores TRACKED files
    to the CURRENT commit's content and does nothing about a wrong branch
    or a commit `use_git=True`'s own loop made (Critical 2, fix round 1;
    see `_PRISTINE_CACHE`'s docstring for the measured failure this
    replaces). Runs before EVERY attempt, not once per record (spec
    4.3.3): `git checkout -f <pristine.checkout_target>` (forced, so a
    prior attempt's uncommitted leftovers on some OTHER branch never
    block the switch; the target is the pristine BRANCH when attached, or
    the pristine SHA itself when the clone started in detached HEAD --
    Important A, fix round 2), `git reset --hard <pristine.sha>`
    (discards every commit made since, on whatever ref is now checked
    out), `git clean -fdq` (removes untracked stray files), then
    `_delete_stray_branches` (so `robigo/*` branches do not accumulate
    across the whole run).

    `repo` MUST be a throwaway clone, never a working tree the operator
    cares about -- this function runs `git reset --hard` and `git branch
    -D` unconditionally, both of which discard real history with no undo.
    `robigo.profile.generate.generate_corpus` states the identical
    requirement for its own destructive path ("it is the caller's job ...
    to make sure `repo` is a throwaway clone, never the working tree");
    the same is true here, and it is `attempt_repair`'s caller's job
    (Task 5's driver) to guarantee it, exactly as `cli.corpus_main`
    already does for stage-3 generation."""
    pristine = _pristine(repo)
    subprocess.run(["git", "checkout", "-f", pristine.checkout_target],
                   cwd=repo, check=True, capture_output=True)
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


def _effective_runner(runner: Runner | None, python: str) -> Runner:
    """The one interpreter-binding rule `attempt_repair` and
    `stage4_repair` both need, shared so it is written -- and can drift --
    in exactly one place: a caller-supplied `runner` (a test's canned,
    `python`-unaware fake -- the only shape any test in this project uses
    via that parameter) is used AS GIVEN, and `python` is bound only onto
    the DEFAULT (`pytest_runner` itself, via `functools.partial`), never
    forced as an unexpected keyword onto a replacement that never declared
    it (see `attempt_repair`'s own docstring for the full reasoning)."""
    return runner if runner is not None else functools.partial(pytest_runner, python=python)


def attempt_repair(
    record: CorpusRecord,
    repo: Path,
    client,
    *,
    seed: int,
    codec: str,
    base: Baseline,
    turn_cap: int = 8,
    python: str = sys.executable,
    runner: Runner | None = None,
) -> Attempt:
    """One (record, seed) repair attempt through the SHIPPED tool
    (`use_git=True`, the real `turn_cap`, the real `codec`,
    `allow_test_edits=False` -- spec 4.3.1: a defect on any of those paths
    counts against the tool, because a user meets it), judged strictly.

    `repo` MUST be a throwaway clone, never a working tree the operator
    cares about -- `reset_clone`, which this function calls before every
    attempt, runs `git reset --hard` and `git branch -D` unconditionally.
    It is the CALLER's job (Task 5's driver, exactly as `cli.corpus_main`
    already guarantees it for stage-3 generation) to make sure of that.

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
    take on the order of 12 hours.

    `CorruptedCloneError` is the ONE exception this safety net does NOT
    catch (Important B, fix round 2) -- it propagates and ends the run
    immediately, on purpose, because it names a defect in the shared
    `repo` itself, not in one record; see that exception's own docstring.

    `python` (task 8, fix round 1 -- confirmed live, "python cannot import
    pytest" excluded EVERY attempt against a real third-party clone)
    governs BOTH halves of this attempt, and that agreement is the whole
    point: the LOOP side (`PythonAdapter(python=python)`, below -- the
    model's edits and the loop's own scope discovery run under it) and the
    JUDGE side (`suite_state`'s `runner`, which reads the post-attempt
    suite state) used to be two INDEPENDENT interpreter choices within one
    attempt -- `PythonAdapter()`'s own default resolution (`.venv/bin/
    python` -> `venv/bin/python` -> bare `"python"`, searched relative to
    `repo`) versus `pytest_runner`'s formerly-hardcoded `sys.executable`.
    Confirmed live against a fresh `boltons` clone with no venv of its
    own: `PythonAdapter._preflight` ran bare `"python" -m pytest
    --version`, found no `pytest` on `PATH`, and raised `AdapterError` at
    the very first line of `loop._execute` -- before the model client was
    ever touched, `result.outcome == "infrastructure"`, and every single
    attempt in a `--full` run would fail this identical way, for a reason
    that has nothing to do with the model being profiled. The JUDGE side's
    own independent choice was never even reached in that failure (the
    LOOP side fails first, unconditionally), but it is an equally real
    second hazard: even a `repo` whose PATH-resolved `"python"` genuinely
    has `pytest` need not be the SAME interpreter `sys.executable`
    resolves to, and `suite_state`'s executed-total comparison against
    `base.executed` (the frozen corpus's own recorded `Baseline`, spec 4's
    exclusion rule) is only meaningful if the interpreter that produced
    `base.executed` also produces the number being compared against it.

    Defaults to `sys.executable` -- deliberately NOT `PythonAdapter`'s own
    `.venv`/`venv`/`PATH` search -- because `sys.executable` is what
    `robigo corpus` was itself running under when IT measured the corpus's
    `Baseline.executed` in the first place (both are invoked through the
    same `robigo` entry point, hence the same interpreter, by
    construction). Threading anything else through by default would make
    every attempt's executed-total comparison meaningless from the start,
    not just occasionally wrong.

    `runner`, when the CALLER explicitly overrides it (a test's canned,
    `python`-unaware fake; the only shape any test in this project actually
    uses today), is passed through AS GIVEN -- `python` is bound only onto
    the DEFAULT (`pytest_runner` itself, via `functools.partial` below),
    never forced onto a caller-supplied replacement, so a caller who has
    already chosen to replace `pytest_runner` entirely keeps full control
    of what it does and is not asked to also accept a `python=` keyword it
    may not even declare."""
    effective_runner = _effective_runner(runner, python)
    try:
        return _judge(record, repo, client, seed=seed, codec=codec, base=base,
                      turn_cap=turn_cap, python=python, runner=effective_runner)
    except CorruptedCloneError:
        raise
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
    python: str,
    runner: Runner,
) -> Attempt:
    """The unguarded body `attempt_repair` wraps in its last-resort safety
    net -- split out so every ANTICIPATED failure point keeps its own
    specific, diagnostic `excluded` reason (via the `excluded` closure
    below), while `attempt_repair` itself only has to catch whatever this
    function did not anticipate. See `attempt_repair`'s docstring for the
    load-bearing ORDER these checks run in.

    `python` is the SAME value `attempt_repair` already used to build
    `runner` (already bound onto `pytest_runner` there, if `runner` was
    not itself overridden) -- passed down here separately only because
    this is where `PythonAdapter` is actually constructed, not because it
    is a second independent choice.

    The post-run section (everything from `_INFRA_OUTCOMES` onward) is
    itself wrapped in one more `try`/`except` beyond its own specific
    inner handlers (Minor, fix round 2): `result` -- and therefore
    `result.outcome`/`result.turns` -- is only ever in scope HERE, never
    in `attempt_repair`'s outer wrapper, which sees nothing but the bare
    exception once it has unwound past this function. Anything that slips
    past every specific handler below still gets `excluded` with the real
    outcome/turns preserved, rather than falling through to the outer net
    and reporting `outcome=""` for an attempt that, in truth, DID run and
    DID have a known outcome."""
    def excluded(
        why: str, outcome: str = "", turns: int = 0, repeats: int = 0
    ) -> Attempt:
        return Attempt(record.name, seed, False, outcome, turns, repeats, why)

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
            task_for(record), repo, client, PythonAdapter(python=python),
            codec=codec, turn_cap=turn_cap, allow_test_edits=False, use_git=True,
        )
    except Exception as exc:
        # loop.run's own contract: it re-raises whatever escapes _execute
        # after recording it, assuming cli.main is the catcher. It is not,
        # here -- this is the second caller Important 3 (fix round 1)
        # named, and one record's internal error must not abort the run.
        return excluded(f"the loop raised: {exc!r}")

    try:
        if result.outcome in _INFRA_OUTCOMES:
            return excluded(f"loop infrastructure: {result.detail}",
                            result.outcome, result.turns, result.repeats)

        if result.outcome != "pass":
            # A definitive, unrescuable model failure (Critical 1, fix
            # round 1) -- stalled/refused/budget_exhausted, never
            # excluded, and never subjected to the suite-reading checks
            # below, which exist only to catch a "pass" that was actually
            # a false positive.
            return Attempt(record.name, seed, False, result.outcome,
                           result.turns, result.repeats, None)

        try:
            state: SuiteState = suite_state(repo, runner, _package_name(record.path))
        except Exception as exc:
            return excluded(f"suite did not run: {exc}", result.outcome,
                            result.turns, result.repeats)

        if state.incomplete is not None:
            return excluded(f"suite run incomplete: {state.incomplete}",
                            result.outcome, result.turns, result.repeats)
        if state.executed != base.executed:
            return excluded(
                f"executed total {state.executed} != baseline {base.executed}",
                result.outcome, result.turns, result.repeats)

        try:
            anchor_intact = _sha(anchor) == before
        except OSError as exc:
            return excluded(f"could not verify the anchor after the run: {exc}",
                            result.outcome, result.turns, result.repeats)

        passed = (
            state.broken == 0
            # `record.test_id not in state.broken_ids` cannot currently
            # flip this verdict on its own: `state.broken == 0` implies
            # `state.broken_ids == ()` in practice, given how a runner's
            # report is CURRENTLY parsed (`_broken_count` and
            # `_broken_ids` are independent regex passes over the SAME
            # text, verify.py:269-282) -- this is NOT an invariant
            # `SuiteState` itself validates or enforces (it validates
            # nothing at all; it is a plain frozen dataclass), so the
            # left conjunct alone already forces this term true whenever
            # it is even evaluated, TODAY. Kept as defensive redundancy,
            # not because it is load-bearing today -- a future change to
            # how those two fields are derived is exactly the kind of
            # drift this line would catch for free.
            and record.test_id not in state.broken_ids
            and anchor_intact
        )
        return Attempt(record.name, seed, passed, result.outcome, result.turns,
                       result.repeats, None)
    except Exception as exc:
        # A backstop for the post-run section specifically (Minor, fix
        # round 2) -- see this function's own docstring for why this
        # differs from attempt_repair's outer net: `result` is in scope
        # here, so outcome/turns are preserved rather than lost.
        return excluded(f"unexpected error after the loop ran: {exc!r}",
                        result.outcome, result.turns, result.repeats)


@dataclass(frozen=True)
class Stage4:
    """Spec 4's stage 4, reduced to the one number the project's kill
    criterion reads (spec 0.2/1.4: below 33.3%, `robigo` ships as a
    benchmark repo rather than a tool) -- so every field here exists to
    make that number checkable, not merely computable.

    `rate` is `float | None`, never a bare `float` defaulting to 0.0, for
    the exact reason spec 4.4 states about `codecs` and spec 4.4's next
    paragraph restates for this field specifically: "`repair_rate: None`
    and `repair_rate: 0.0` are different facts and must stay
    distinguishable -- 'never measured' versus 'measured and nothing was
    repaired'." A family that never reached stage 4 (upstream gated it,
    or `records` was empty) reporting `0.0` here would read, to anyone
    comparing against 33.3%, as "measured and definitively below
    threshold" -- indistinguishable from a family that WAS measured and
    genuinely failed every attempt. `docs/CARRIED-DEBT.md`'s plan-03
    section already carries the identical gap one layer up
    (`verdict_for` cannot express "stage 0 never ran" apart from "stage 0
    ran and found nothing", both reading UNUSABLE via
    `envelope_fidelity=0.0`) -- this dataclass is the one place in the
    profiler where that specific collapse must not recur, because unlike
    `verdict_for`'s three-way string it feeds a numeric threshold
    comparison with no textual hedge attached.

    `rate` is attempt-level (passes / attempts, spec 4.5) -- the quantity
    spec 1.4's attempts-to-success arithmetic, and therefore the 33.3%
    threshold itself, is actually about. The record-level 95% confidence
    interval spec 6.1 reports beside it is a SEPARATE computation, on
    purpose: seeds within one record share the same defect, the same
    file, and the same starting tree, so they are correlated in exactly
    the way independent Bernoulli trials are not, and an attempt-level
    interval would claim roughly sqrt(10) more precision than ten
    correlated seeds per record actually buys (spec 4.5, spec 6.1's
    worked table). `per_record` is what makes that interval computable
    at all -- without the per-record (passes, scored) breakdown, only the
    attempt-level rate survives this stage, and Task 7's gate could
    report nothing but an overclaiming number. Kept even though `rate`
    itself is attempt-level: the two are deliberately different views of
    the same 940 attempts, not a redundant pair.

    `attempts` and `records` count only what actually got a fair chance
    (spec 4.3.4): `attempts` is `sum(scored)` across `per_record`, never
    the raw count of `attempt_repair` calls, and `records` is the number
    of distinct records with at least one scored attempt -- a record
    every seed of which excluded (a broken clone, a suite that would not
    run) contributes to neither, so it never inflates `records` while
    contributing zero real signal. `dropped` names every excluded attempt
    by record and seed, so a reader can tell "94 records measured" from
    "94 records attempted, several silently dropped" without re-running
    anything.

    `all_attempts` carries every `Attempt` this stage produced, scored or
    excluded, verbatim and unfiltered. Stage 4 itself has no use for the
    unscored ones beyond `dropped`'s summary, but Task 6's
    `stage5_discipline` (turns-to-green, repeat rate) is derived from the
    identical attempt list, filtering by `excluded is None` itself rather
    than trusting a second reduction of the same data -- reshaping this
    dataclass to add the field later would mean every caller of stage 4
    changes too, so it is here from the start even though this stage does
    not read it back."""

    rate: float | None
    attempts: int
    records: int
    per_record: dict[str, tuple[int, int]]
    dropped: tuple[str, ...]
    all_attempts: tuple[Attempt, ...]


def stage4_repair(
    records: Sequence[CorpusRecord],
    repo: Path,
    client,
    *,
    seeds: int,
    codec: str,
    base: Baseline,
    turn_cap: int = 8,
    python: str = sys.executable,
    runner: Runner | None = None,
) -> Stage4:
    """Every corpus record against every seed (spec 4.1: `seeds` fixed at
    10 by the `--full` contract in real runs, the frozen 94-record corpus
    times that is spec 6.1's ~940-attempt, ~12h `N`), reusing the SAME
    `repo` clone throughout -- `attempt_repair`'s own `reset_clone` is
    what makes that reuse safe, restoring `repo` to its captured pristine
    state before every single attempt (see `_PRISTINE_CACHE`'s
    docstring). This function's only job is the reduction: turn ~940
    `Attempt`s into the numbers spec 4.5 and Task 7's gate need, without
    quietly turning a harness fault into a data point.

    `attempt_repair` is called with NO exception handling around it
    beyond what this function's own body does naturally (nothing) --
    deliberately, and this is the single most important property of this
    loop. `attempt_repair` already contains its own last-resort `except
    Exception` internally and turns every ANTICIPATED failure into an
    `excluded` `Attempt` on its own; the ONE exception it deliberately
    lets escape is `CorruptedCloneError`, raised when `repo` starts this
    process already sitting on a `robigo/*` branch -- a defect in the
    shared clone itself, identical for every attempt in the run, not a
    per-record surprise (see that exception's own docstring). A
    `try/except Exception` wrapped around the call here -- the shape this
    driver does NOT have -- would silently convert that one loud,
    immediate abort into the same `excluded` `Attempt` manufactured
    roughly 940 times over a ~12h run, one per attempt, scoring nothing
    and reporting a corpus-shaped `dropped` list instead of failing where
    the actual problem is: `repo`. Two independent readers (implementer
    and reviewer) flagged this exact hazard in the task that built
    `attempt_repair`; the contract this loop honours is to add nothing
    of its own that could reintroduce it. If `CorruptedCloneError`
    propagates, it propagates all the way out of `stage4_repair` too --
    intentionally uncaught here, exactly as intentionally uncaught inside
    `attempt_repair`.

    An attempt whose `excluded` is non-`None` is recorded in `dropped`
    (spec 4.3.4: it never gave the model a fair chance) and skipped
    before it can touch `per_record` -- neither the numerator nor the
    denominator of any rate below ever sees it. Every other attempt
    updates `per_record[record.name]` as `(passes + int(a.passed), scored
    + 1)`; `records`/`attempts`/`rate` are all derived from `per_record`
    afterward, not accumulated in parallel, so there is exactly one place
    an attempt could be double-counted or miscounted, not two that could
    drift apart.

    `python`/`runner` (task 8, fix round 1) are passed straight through to
    every `attempt_repair` call unchanged -- this function makes no
    interpreter decision of its own, it only relays the ONE choice its own
    caller (`report.run_profile`) made, so every attempt in this ~940-call
    loop resolves the identical interpreter, not just the identical
    default. See `attempt_repair`'s own docstring for why that agreement
    -- LOOP side and JUDGE side alike -- is load-bearing, not cosmetic.

    **`python` is validated ONCE, before the grid, when `records` is
    non-empty** (fix round 2, 2026-08-10 review): `reset_clone(repo)` (the
    SAME reset every attempt already runs -- also where `CorruptedCloneError`
    would fire, surfaced here before ~940 calls instead of on the first
    one) followed by one `measure_baseline(repo, effective_runner)` call
    against the now-pristine tree, compared against `base.executed`.
    Raises `InterpreterMismatchError` on any disagreement -- see that
    exception's own docstring for why "can `python` import pytest at all"
    is necessary but not sufficient, and why this specific comparison is
    the one `_judge`'s own exclusion rule actually depends on. Skipped
    entirely when `records` is empty: there is no grid to protect, and
    nothing here would ever be a fair thing to validate `python` against."""
    if records:
        effective_runner = _effective_runner(runner, python)
        reset_clone(repo)
        check = measure_baseline(repo, effective_runner)
        if check.executed != base.executed:
            raise InterpreterMismatchError(
                f"--python {python} does not reproduce this corpus's own "
                f"baseline: a suite run against the pristine clone just "
                f"now executed {check.executed} test(s), but the corpus "
                f"was mined with executed={base.executed} -- refusing to "
                f"spend up to {len(records) * seeds} attempts that would "
                f"all be excluded the same way, one at a time, ~12h later. "
                f"Check that --python has the same test dependencies "
                f"installed as whatever measured this corpus (it need not "
                f"even lack pytest entirely -- a different dependency set "
                f"changing which tests collect is enough)."
            )

    per_record: dict[str, tuple[int, int]] = {}
    dropped: list[str] = []
    attempts: list[Attempt] = []

    for record in records:
        for seed in range(seeds):
            attempt = attempt_repair(
                record, repo, client, seed=seed, codec=codec, base=base,
                turn_cap=turn_cap, python=python, runner=runner,
            )
            attempts.append(attempt)
            if attempt.excluded is not None:
                dropped.append(
                    f"{attempt.record} seed {attempt.seed}: {attempt.excluded}")
                continue
            passes, scored = per_record.get(attempt.record, (0, 0))
            per_record[attempt.record] = (passes + int(attempt.passed), scored + 1)

    scored_total = sum(scored for _, scored in per_record.values())
    passes_total = sum(passes for passes, _ in per_record.values())
    return Stage4(
        rate=(passes_total / scored_total) if scored_total else None,
        attempts=scored_total,
        records=len(per_record),
        per_record=per_record,
        dropped=tuple(dropped),
        all_attempts=tuple(attempts),
    )
