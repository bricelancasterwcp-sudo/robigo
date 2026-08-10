import subprocess
from pathlib import Path
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
