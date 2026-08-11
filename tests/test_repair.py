import shutil
import subprocess
from pathlib import Path
import pytest
from robigo.loop import RunResult
from robigo.profile.corpus_io import CorpusRecord
from robigo.profile.verify import Baseline, SuiteState
from robigo.profile import repair as R


def _record(**over):
    base = dict(
        name="off_by_one", path=Path("src/pkg/mod.py"), line=2,
        broken="    return len(items) - 1\n", fixed="    return len(items)\n",
        test_id="tests/test_mod.py::test_len", diagnostic="exactly one",
        operator="arith", source_repo="/tmp/src", source_sha="deadbeef",
    )
    base.update(over)
    return CorpusRecord(**base)


def _repo(tmp_path):
    """A real git repo, not just a directory tree. `attempt_repair`'s
    `reset_clone` shells out to real `git checkout -- .` / `git clean -fdq`
    unconditionally (spec 4.3.3: it runs before EVERY attempt, and Task 5's
    actual caller always hands it a real clone) -- verified directly: with
    no `git init` here, `git checkout -- .` exits 128 ("not a git
    repository"), which `attempt_repair` catches as `excluded("could not
    stage the defect: ...")`, and every test asserting `excluded is None`
    or checking a SPECIFIC exclusion reason failed for the wrong reason,
    never reaching the behaviour it claimed to test. The brief's fixture
    omitted this; committing a clean initial tree is required for
    `reset_clone`'s no-op case (nothing to discard yet) to succeed."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "pkg" / "mod.py").write_text(
        "def n(items):\n    return len(items)\n", encoding="utf-8")
    (tmp_path / "tests" / "test_mod.py").write_text(
        "from pkg.mod import n\ndef test_len():\n    assert n([1]) == 1\n",
        encoding="utf-8")
    run = lambda *argv: subprocess.run(
        ["git", *argv], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "Test")
    run("add", "-A")
    run("commit", "-q", "-m", "initial")
    return tmp_path


def test_the_task_string_never_leaks_the_defect_location():
    """Falsification test for spec 4.2. One token away from silently
    inflating the headline figure."""
    r = _record()
    task = R.task_for(r)
    assert r.test_id in task
    assert str(r.path) not in task
    assert "2" not in task.replace(r.test_id, "")   # no line number
    assert r.fixed.strip() not in task
    assert r.broken.strip() not in task


def test_a_clean_repair_passes(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="pass", turns=2, exit_code=0, branch=None, detail="tests pass"))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=0, executed=1, broken_ids=(), incomplete=None))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is True and a.excluded is None and a.turns == 2


def test_a_clean_repairs_repeats_comes_from_the_run_result(tmp_path, monkeypatch):
    """`Attempt.repeats` was hardcoded to `0` at every construction site
    until Task 6 wired it to `result.repeats` -- Stage 5's repeat-rate
    would silently read zero for every attempt regardless of how many
    times the model actually repeated itself. Threaded through the
    `passed` path here; the not-`passed` path is covered separately
    below, since that is a different `Attempt(...)` construction."""
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="pass", turns=3, exit_code=0, branch=None, detail="tests pass",
        repeats=2))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=0, executed=1, broken_ids=(), incomplete=None))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is True and a.repeats == 2


def test_greening_the_target_while_breaking_a_neighbour_fails(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=2, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="pass", turns=1, exit_code=0, branch=None, detail="tests pass"))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=1, executed=2, broken_ids=("tests/test_other.py::test_y",),
        incomplete=None))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False and a.excluded is None


def test_editing_the_anchor_test_file_fails_even_if_the_suite_is_green(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)

    def sneaky(*a, **k):
        (repo / "tests" / "test_mod.py").write_text(
            "def test_len():\n    assert True\n", encoding="utf-8")
        return RunResult(outcome="pass", turns=1, exit_code=0,
                         branch=None, detail="tests pass")

    monkeypatch.setattr(R, "run", sneaky)
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=0, executed=1, broken_ids=(), incomplete=None))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False


def test_an_incomplete_suite_run_is_excluded_not_failed(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="pass", turns=1, exit_code=0, branch=None, detail="tests pass"))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=1, executed=0, broken_ids=(), incomplete="Interrupted: collection error"))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False
    assert a.excluded is not None and "Interrupted" in a.excluded


def test_an_executed_total_mismatch_is_excluded_not_failed(tmp_path, monkeypatch):
    """A mutant that breaks a module's import makes pytest report `1 error`
    while three tests never ran. Plan 04 lesson 2."""
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=120, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="pass", turns=1, exit_code=0, branch=None, detail="tests pass"))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=0, executed=117, broken_ids=(), incomplete=None))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.excluded is not None


def test_a_loop_infrastructure_outcome_is_excluded_not_failed(tmp_path, monkeypatch):
    """`suite_state` is deliberately mocked GREEN here, not left unmocked.
    Left unmocked, the real `suite_state`/`pytest_runner` runs against
    `_repo`'s bare fixture tree, whose `pkg` directory has no `__init__.py`
    and resolves as a namespace package with `__file__ is None` -- the
    `MODULE_UNDER_TEST=None` marker that produces makes `_assert_in_clone`
    raise `WrongTreeError` for an UNRELATED reason, which `attempt_repair`'s
    broad `except Exception` around `suite_state` also turns into an
    exclusion. That accidentally satisfies `excluded is not None` even with
    `_INFRA_OUTCOMES` mutated empty (verified directly), so it proves
    nothing about whether the infrastructure short-circuit itself ran --
    exactly the "breaks something incidental" trap. Mocking a clean, green
    `SuiteState` removes that confound: if `_INFRA_OUTCOMES` is mutated
    away, execution falls through to this green state, `excluded` stays
    `None`, and the assertion below correctly catches it."""
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="infrastructure", turns=0, exit_code=4, branch=None,
        detail="daemon unreachable"))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=0, executed=1, broken_ids=(), incomplete=None))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False and a.excluded is not None


def test_a_stalled_run_is_a_real_model_failure_not_an_exclusion(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="stalled", turns=8, exit_code=1, branch=None,
        detail="turn cap 8 reached"))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=1, executed=1, broken_ids=("tests/test_mod.py::test_len",),
        incomplete=None))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False and a.excluded is None


def test_a_stalled_runs_repeats_is_also_preserved(tmp_path, monkeypatch):
    """The not-`passed` `Attempt(...)` construction (`result.outcome !=
    "pass"`) is a SEPARATE call site from the passing one covered above --
    it must independently thread `result.repeats` through rather than
    keep the old hardcoded `0`."""
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="stalled", turns=8, exit_code=1, branch=None,
        detail="turn cap 8 reached", repeats=5))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False and a.excluded is None and a.repeats == 5


# --------------------------------------------------------------------------
# Fix round 1 -- Critical 1: outcome != "pass" must be scored a real
# failure BEFORE any suite reading, through all three doors the review
# found. Each test below mocks `suite_state` to return EXACTLY the value
# that, under the old (wrong) ordering, would have produced `excluded`
# instead of `passed=False, excluded=None` -- so a regression back to the
# old ordering fails these specifically, not incidentally.
# --------------------------------------------------------------------------


def test_a_stalled_run_with_an_incomplete_suite_is_a_failure_not_an_exclusion(
    tmp_path, monkeypatch
):
    """Door 1: a model that stalls having also left a syntax error behind
    (e.g. a half-written patch applied by a bare `run` action) makes the
    suite read `incomplete`. That must not promote an already-certain
    `stalled` failure to an exclusion."""
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="stalled", turns=8, exit_code=1, branch=None,
        detail="turn cap 8 reached"))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=1, executed=0, broken_ids=(),
        incomplete="pytest did not complete normally (exit code 2)"))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False and a.excluded is None and a.outcome == "stalled"


def test_a_stalled_run_with_an_executed_mismatch_is_a_failure_not_an_exclusion(
    tmp_path, monkeypatch
):
    """Door 2: a model that stalls having deleted a neighbouring test
    shrinks `executed` below baseline. Same rule: already a certain
    failure, must not become an exclusion."""
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=2, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="stalled", turns=8, exit_code=1, branch=None,
        detail="turn cap 8 reached"))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=0, executed=1, broken_ids=(), incomplete=None))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False and a.excluded is None


def test_a_stalled_run_never_calls_suite_state_at_all(tmp_path, monkeypatch):
    """Door 3: a model that stalls having broken the target's own import
    would make a REAL `suite_state` raise (`WrongTreeError`, via a missing
    `MODULE_UNDER_TEST=` marker). `suite_state` is stubbed here to raise
    unconditionally -- the only way this test can pass is if `outcome !=
    "pass"` short-circuits BEFORE `suite_state` is ever called at all, not
    merely before its result is trusted."""
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="stalled", turns=8, exit_code=1, branch=None,
        detail="turn cap 8 reached"))

    def never(*a, **k):
        raise AssertionError("suite_state must not run for a non-pass outcome")

    monkeypatch.setattr(R, "suite_state", never)
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False and a.excluded is None


# --------------------------------------------------------------------------
# Fix round 1 -- Important 3: an exception `attempt_repair` did not
# anticipate must produce `excluded`, never escape and abort the whole
# ~940-attempt run.
# --------------------------------------------------------------------------


def test_run_raising_produces_an_exclusion_not_a_crash(tmp_path, monkeypatch):
    """`robigo.loop.run` deliberately RE-RAISES any exception that escapes
    its internal `_execute`, on the assumption that `cli.main` is the
    catcher. `attempt_repair` is a second caller with no handler upstream
    of it -- an escaping exception here would abort the whole run over one
    record's surprise."""
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)

    def boom(*a, **k):
        raise RuntimeError("model client blew up")

    monkeypatch.setattr(R, "run", boom)
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False
    assert a.excluded is not None and "blew up" in a.excluded


def test_a_missing_anchor_file_produces_an_exclusion_not_a_crash(tmp_path, monkeypatch):
    """A corpus record whose `test_id` names a file absent from this
    clone (e.g. provenance drift between the corpus and `--repo`) must not
    raise `FileNotFoundError` out of `attempt_repair`. `run` is stubbed to
    raise if called at all, proving staging fails BEFORE the loop ever
    runs."""
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)

    def never(*a, **k):
        raise AssertionError("run() must not be called when the anchor is missing")

    monkeypatch.setattr(R, "run", never)
    bad_record = _record(test_id="tests/test_missing.py::test_nope")
    a = R.attempt_repair(bad_record, repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False
    assert a.excluded is not None and "could not stage the defect" in a.excluded


def test_an_unanticipated_exception_while_staging_still_produces_an_exclusion(
    tmp_path, monkeypatch
):
    """`_judge`'s staging `try` only catches `(OSError, CalledProcessError,
    IndexError, ValueError)` -- deliberately narrow, so the anticipated
    failure modes there keep a specific "could not stage the defect"
    message. Something OUTSIDE that tuple (a bare `RuntimeError`, injected
    here by replacing `_anchor_path` itself) is NOT caught by that inner
    handler -- this test only passes because `attempt_repair`'s own outer
    safety net catches it instead, proving that net is not redundant with
    the per-step handlers above it."""
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)

    def boom(*a, **k):
        raise RuntimeError("unexpected internal error")

    monkeypatch.setattr(R, "_anchor_path", boom)
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False
    assert a.excluded is not None and "unexpected error" in a.excluded


# --------------------------------------------------------------------------
# Fix round 1 -- Critical 2: the falsification test spec 4.3.3 named and
# nobody wrote until this fix round.
# --------------------------------------------------------------------------


def test_reset_clone_restores_the_pristine_tree_and_branch_across_attempts(
    tmp_path, monkeypatch
):
    """Task 5/7's loop reuses ONE `repo` clone across ~940 `attempt_repair`
    calls in a single process -- `reset_clone` must restore the ORIGINAL
    branch and HEAD, not just `checkout -- .` / `clean -fdq` against
    whatever branch/commit `use_git=True`'s own loop left the tree on.
    Measured directly before this fix: a stray committed defect on a
    `robigo/*` branch survived the old `reset_clone` entirely, and every
    later record inherited it (`state.broken == 0` could never hold
    again). `dirty_run` simulates exactly that: a `robigo/*` branch, a
    stray unrelated file, and a real commit, all left behind by an attempt
    that then stalls."""
    repo = _repo(tmp_path)
    original_branch = R._git_text(repo, "branch", "--show-current")
    base = Baseline(broken=0, executed=1, seconds=0.1)

    def dirty_run(*a, **k):
        subprocess.run(["git", "checkout", "-b", "robigo/the-test-fails-1"],
                       cwd=repo, check=True, capture_output=True)
        (repo / "src" / "pkg" / "other.py").write_text(
            "stray = True\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True,
                       capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "robigo: stray edit"],
                       cwd=repo, check=True, capture_output=True)
        return RunResult(outcome="stalled", turns=8, exit_code=1,
                         branch="robigo/the-test-fails-1",
                         detail="turn cap 8 reached")

    observed: dict[str, object] = {}

    def observing_run(*a, **k):
        observed["branch"] = R._git_text(repo, "branch", "--show-current")
        observed["stray_exists"] = (repo / "src" / "pkg" / "other.py").exists()
        observed["mod_py"] = (repo / "src" / "pkg" / "mod.py").read_text(
            encoding="utf-8")
        observed["robigo_branches"] = R._git_text(
            repo, "branch", "--list", "robigo/*", "--format=%(refname:short)")
        return RunResult(outcome="pass", turns=1, exit_code=0, branch=None,
                         detail="tests pass")

    monkeypatch.setattr(R, "run", dirty_run)
    R.attempt_repair(_record(), repo, client=object(), seed=0,
                     codec="search_replace", base=base)

    monkeypatch.setattr(R, "run", observing_run)
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=0, executed=1, broken_ids=(), incomplete=None))
    R.attempt_repair(_record(), repo, client=object(), seed=1,
                     codec="search_replace", base=base)

    assert observed["branch"] == original_branch
    assert observed["stray_exists"] is False
    # break_it re-staged the SAME defect for attempt 2 -- confirms the
    # tree really was pristine before break_it ran, not merely "a" tree.
    assert observed["mod_py"] == "def n(items):\n    return len(items) - 1\n"
    assert observed["robigo_branches"] == ""


# --------------------------------------------------------------------------
# Fix round 2 -- Important A: detached HEAD is what spec 4.1 actually
# produces ("checked out at its recorded source_sha"), and it must not
# exclude every attempt.
# --------------------------------------------------------------------------


def test_reset_clone_handles_a_detached_head_pristine_state(tmp_path, monkeypatch):
    """`git branch --show-current` prints nothing on a detached HEAD, so
    `_pristine` used to capture `branch=""` and `reset_clone` ran `git
    checkout -f ''`, exit 128, excluding EVERY attempt. Spec 4.1 says
    stage 4 runs "against a clone ... checked out at its recorded
    source_sha" -- a plain `git checkout <sha>` produces exactly this
    state, so Task 5's driver can trip it on the very first attempt of a
    real run."""
    R.clear_pristine_cache()
    repo = _repo(tmp_path)
    head_sha = R._git_text(repo, "rev-parse", "HEAD")
    subprocess.run(["git", "checkout", head_sha], cwd=repo, check=True,
                   capture_output=True)
    # Confirm the fixture really is detached before trusting the rest.
    assert R._git_text(repo, "branch", "--show-current") == ""

    base = Baseline(broken=0, executed=1, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="pass", turns=1, exit_code=0, branch=None, detail="tests pass"))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=0, executed=1, broken_ids=(), incomplete=None))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is True and a.excluded is None


# --------------------------------------------------------------------------
# Fix round 2 -- Important B: a clone already sitting on a robigo/* branch
# before any attempt has run is a corrupted starting state, not an
# ordinary excluded attempt.
# --------------------------------------------------------------------------


def test_a_repo_already_on_a_robigo_branch_fails_loudly(tmp_path, monkeypatch):
    """`robigo.loop.run` never checks back out after an attempt -- it
    only hands back an undo recipe, never applies it -- so a clone left
    mid-run by an earlier process, or a resumed run against a reused
    clone directory, can start already on a `robigo/*` branch. Capturing
    that as "pristine" would silently apply the corruption to every
    attempt; this must raise `CorruptedCloneError` immediately and must
    NOT be swallowed by `attempt_repair`'s own last-resort safety net."""
    R.clear_pristine_cache()
    repo = _repo(tmp_path)
    subprocess.run(["git", "checkout", "-b", "robigo/leftover-3"], cwd=repo,
                   check=True, capture_output=True)

    base = Baseline(broken=0, executed=1, seconds=0.1)
    with pytest.raises(R.CorruptedCloneError, match="robigo/leftover-3"):
        R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)


def test_delete_stray_branches_never_targets_the_current_branch(tmp_path):
    """Defensive half of Important B: `_delete_stray_branches` re-derives
    "what's current" itself rather than trusting an upstream guarantee it
    cannot see. Simulated directly at the unit level: checked out on a
    `robigo/*` branch (bypassing `_pristine`'s new guard, which is tested
    separately above), confirm the deletion skips exactly that one and
    removes every OTHER `robigo/*` branch."""
    repo = _repo(tmp_path)
    subprocess.run(["git", "checkout", "-b", "robigo/stray-a"], cwd=repo,
                   check=True, capture_output=True)
    subprocess.run(["git", "branch", "robigo/stray-b"], cwd=repo,
                   check=True, capture_output=True)

    R._delete_stray_branches(repo)

    remaining = R._git_text(
        repo, "branch", "--list", "robigo/*", "--format=%(refname:short)"
    ).split()
    assert remaining == ["robigo/stray-a"]
    assert R._git_text(repo, "branch", "--show-current") == "robigo/stray-a"


# --------------------------------------------------------------------------
# Fix round 2 -- Important C: the pristine cache must not silently apply
# to the wrong repository, and must not trust a sha that no longer
# resolves.
# --------------------------------------------------------------------------


def test_pristine_cache_keys_on_repo_identity_not_path_alone(tmp_path):
    """Measured directly before this fix: a DIFFERENT repo re-cloned at
    the SAME filesystem path after an earlier run silently reused the
    earlier repo's cached `_Pristine` (`passed=True, excluded=None` at a
    sha the actual clone's history did not even contain). Simulated here
    with an ORPHAN branch, not a wholesale `.git` replacement -- an orphan
    commit has a genuinely different ROOT (a different `_repo_identity`
    key), but git keeps the ORIGINAL branch's commit fully reachable in
    the SAME object database, so `_sha_exists(repo, first.sha)` stays
    TRUE throughout. A wholesale `.git` replacement would make the old sha
    stop resolving too, which would make `_sha_exists` alone -- tested
    separately below -- already force a re-capture, confounding which of
    the two Important-C mechanisms actually caught the problem. This
    construction isolates the compound KEY specifically: only it, not
    sha-existence, can explain why `second` must differ from `first`
    here."""
    R.clear_pristine_cache()
    repo = _repo(tmp_path)
    first = R._pristine(repo)
    assert first.sha == R._git_text(repo, "rev-parse", "HEAD")

    subprocess.run(["git", "checkout", "-q", "--orphan", "unrelated"],
                   cwd=repo, check=True, capture_output=True)
    (repo / "unrelated.txt").write_text(
        "nothing to do with the original repo\n", encoding="utf-8")
    git = lambda *argv: subprocess.run(
        ["git", *argv], cwd=repo, check=True, capture_output=True)
    git("add", "-A")
    git("commit", "-q", "-m", "unrelated root")
    second_sha = R._git_text(repo, "rev-parse", "HEAD")
    assert second_sha != first.sha  # sanity: genuinely different commit

    # The confound this test specifically rules out: the OLD sha is still
    # a perfectly resolvable object in this same repo (reachable via the
    # original branch's ref), so `_sha_exists` alone would NOT have forced
    # a re-capture here.
    assert R._sha_exists(repo, first.sha) is True

    second = R._pristine(repo)
    assert second.sha == second_sha
    assert second.sha != first.sha


def test_sha_exists_true_for_head_false_for_garbage(tmp_path):
    repo = _repo(tmp_path)
    head = R._git_text(repo, "rev-parse", "HEAD")
    assert R._sha_exists(repo, head) is True
    assert R._sha_exists(repo, "0" * 40) is False


# --------------------------------------------------------------------------
# Task 5 -- the stage-4 aggregate. `stage4_repair` reduces every
# (record, seed) `Attempt` `attempt_repair` produces into the one number
# the project's 33.3% kill criterion reads. `attempt_repair` itself is
# stubbed throughout: these tests are about the REDUCTION (exclusion
# bookkeeping, None-vs-0.0, per-record retention, and the
# CorruptedCloneError carve-out), not about repair judgement, which
# `attempt_repair`'s own tests above already cover.
# --------------------------------------------------------------------------


def test_rate_is_attempt_level_and_per_record_is_kept(monkeypatch, tmp_path):
    calls = []

    def fake_attempt(record, repo, client, *, seed, **kw):
        calls.append((record.name, seed))
        # record "a" passes on even seeds; record "b" never passes
        ok = record.name == "a" and seed % 2 == 0
        return R.Attempt(record.name, seed, ok, "pass" if ok else "stalled",
                         1, 0, None)

    monkeypatch.setattr(R, "attempt_repair", fake_attempt)
    recs = [_record(name="a"), _record(name="b")]
    s = R.stage4_repair(recs, tmp_path, object(), seeds=4,
                        codec="search_replace",
                        base=Baseline(broken=0, executed=1, seconds=0.1))
    assert len(calls) == 8
    assert s.attempts == 8 and s.records == 2
    assert s.rate == pytest.approx(2 / 8)
    assert s.per_record == {"a": (2, 4), "b": (0, 4)}
    assert s.dropped == ()
    # all_attempts carries every Attempt produced, scored or not, for
    # Task 6's stage5_discipline to derive its own metrics from.
    assert len(s.all_attempts) == 8
    assert all(isinstance(a, R.Attempt) for a in s.all_attempts)


def test_excluded_attempts_leave_both_sides_of_the_rate(monkeypatch, tmp_path):
    def fake_attempt(record, repo, client, *, seed, **kw):
        if seed == 0:
            return R.Attempt(record.name, seed, False, "", 0, 0, "daemon died")
        return R.Attempt(record.name, seed, True, "pass", 1, 0, None)

    monkeypatch.setattr(R, "attempt_repair", fake_attempt)
    s = R.stage4_repair([_record(name="a")], tmp_path, object(), seeds=3,
                        codec="search_replace",
                        base=Baseline(broken=0, executed=1, seconds=0.1))
    assert s.attempts == 2            # not 3
    assert s.rate == pytest.approx(1.0)   # 2/2, NOT 2/3
    assert any("daemon died" in d for d in s.dropped)
    assert len(s.dropped) == 1
    assert len(s.all_attempts) == 3   # the excluded one is still recorded


def test_a_record_with_every_attempt_excluded_is_not_counted(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "attempt_repair", lambda record, repo, client, *, seed, **kw:
                        R.Attempt(record.name, seed, False, "", 0, 0, "clone broken"))
    s = R.stage4_repair([_record(name="a")], tmp_path, object(), seeds=2,
                        codec="search_replace",
                        base=Baseline(broken=0, executed=1, seconds=0.1))
    assert s.attempts == 0 and s.records == 0
    assert s.rate is None       # not 0.0 -- nothing was measured
    assert len(s.dropped) == 2


def test_no_records_at_all_means_rate_is_none_not_zero(tmp_path):
    """The empty-corpus edge of the same invariant: nothing to iterate at
    all must read identically to "everything excluded" -- both are
    "never measured", not "measured zero"."""
    s = R.stage4_repair([], tmp_path, object(), seeds=5,
                        codec="search_replace",
                        base=Baseline(broken=0, executed=1, seconds=0.1))
    assert s.rate is None
    assert s.attempts == 0 and s.records == 0
    assert s.dropped == () and s.all_attempts == ()


def test_corrupted_clone_error_propagates_through_the_driver_loop(
    monkeypatch, tmp_path
):
    """Carry-forward hazard from the task that built `attempt_repair`,
    flagged independently by both its implementer and its reviewer:
    `CorruptedCloneError` names a defect in the shared `repo` itself
    (already checked out on a `robigo/*` branch before this process
    touched it), not in one record, and `attempt_repair` deliberately
    lets it escape rather than swallowing it into an `excluded` Attempt.
    `stage4_repair` must not wrap the call in a broad `except Exception`
    -- doing so would convert one loud, immediate abort into the
    identical exclusion manufactured for every remaining attempt in a
    ~940-attempt run, scoring nothing while looking like it ran.

    Asserts BOTH that the exception reaches the caller of `stage4_repair`
    AND that the loop stopped at the very first attempt (`calls == [0]`)
    -- the second assertion is what a broad `except Exception` would
    break even though the first, alone, would still superficially look
    like it passed (a `pytest.raises` around a loop that swallowed the
    error 1 out of 6 times would still fail overall, but for a confusing
    reason; asserting the call count makes the failure mode legible)."""
    calls = []

    def fake_attempt(record, repo, client, *, seed, **kw):
        calls.append(seed)
        raise R.CorruptedCloneError(
            f"{repo} is already checked out on 'robigo/leftover-3'")

    monkeypatch.setattr(R, "attempt_repair", fake_attempt)
    with pytest.raises(R.CorruptedCloneError, match="robigo/leftover-3"):
        R.stage4_repair([_record(name="a")], tmp_path, object(), seeds=3,
                        codec="search_replace",
                        base=Baseline(broken=0, executed=1, seconds=0.1))
    assert calls == [0]   # aborted on the FIRST attempt, never swallowed


def test_pristine_recaptures_if_the_cached_sha_no_longer_resolves(tmp_path):
    """The second half of Important C: even with the identity-keyed cache
    fix, a cached sha that no longer resolves (e.g. pruned in a long-lived
    clone) must not be trusted blindly -- `_pristine` re-derives instead
    of hand back a sha `git reset --hard` can no longer reach."""
    R.clear_pristine_cache()
    repo = _repo(tmp_path)
    real = R._pristine(repo)
    key = R._repo_identity(repo)
    # Corrupt the cache directly: same identity, a sha that cannot
    # possibly resolve in this repo.
    R._PRISTINE_CACHE[key] = R._Pristine(real.branch, "f" * 40)

    recaptured = R._pristine(repo)

    assert recaptured.sha == real.sha
    assert recaptured.sha != "f" * 40
