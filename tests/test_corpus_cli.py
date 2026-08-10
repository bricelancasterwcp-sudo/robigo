# tests/test_corpus_cli.py
from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import pytest

import robigo.cli as cli_module
from robigo.loop import OUTCOMES
from robigo.model.geometry import WindowPlan
from robigo.profile.corpus_io import read_corpus
from robigo.profile.generate import GenerationResult, TargetOutcome
from robigo.profile.verify import Baseline

# ---------------------------------------------------------------------------
# Offline guarantee: this whole module never runs a real pytest subprocess
# -- `pytest_runner` is always monkeypatched away before `corpus_main` is
# invoked. A couple of dedicated tests DO shell out to real `git` (local,
# no network, milliseconds for a two-file repo) to prove `_clone_repo`/
# `_source_sha` actually work; every other test stubs cloning entirely.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("socket.socket.connect must never be called in this test suite")

    monkeypatch.setattr(socket.socket, "connect", _blocked)


# ---------------------------------------------------------------------------
# Dispatch: "corpus" as a leading argv element, same shape as "profile"
# ---------------------------------------------------------------------------


def test_leading_corpus_argument_dispatches_to_corpus_main(monkeypatch: pytest.MonkeyPatch):
    # Fails if `main` still routes a leading "corpus" argv element into the
    # ordinary flat task parser instead of into corpus_main.
    captured = {}

    def fake_corpus_main(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(cli_module, "corpus_main", fake_corpus_main)
    code = cli_module.main(["corpus", "--repo", "x", "--out", "y"])
    assert code == 0
    assert captured["argv"] == ["--repo", "x", "--out", "y"]


def test_a_flat_task_containing_the_word_corpus_still_reaches_the_ordinary_parser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # The flat parser stays intact for everything that isn't literally
    # "corpus" (or "profile") as argv[0] -- `robigo "<task>"` keeps
    # working, even for a task whose own text happens to contain the word.
    from robigo.loop import RunResult

    def fake_run(task, root, client, adapter, *, codec, turn_cap,
                 allow_test_edits, use_git, scope_paths, recorder):
        return RunResult("pass", 1, 0, None, "ok")

    def stub_plan_window(*args, **kwargs) -> WindowPlan:
        return WindowPlan(8192, "user_cap", None, 1024, 8 * 1024**3, 256 * 1024**2)

    monkeypatch.setattr(cli_module, "run", fake_run)
    monkeypatch.setattr(cli_module, "plan_window", stub_plan_window)
    code = cli_module.main(["fix the corpus of bugs", "--root", str(tmp_path),
                            "--model", "m"])
    assert code == 0


# ---------------------------------------------------------------------------
# _resolve_targets — the one integration seam the brief calls out: verify()
# requires a path relative to the clone; candidates()/CorpusRecord.path do
# not enforce it, and a --target the user pastes might be absolute.
# ---------------------------------------------------------------------------


def test_resolve_targets_passes_a_relative_target_through_unchanged(tmp_path: Path):
    result = cli_module._resolve_targets(tmp_path, tmp_path, [Path("src/a.py")])
    assert result == (Path("src/a.py"),)


def test_resolve_targets_converts_an_absolute_target_inside_repo_to_relative(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    absolute = repo / "src" / "a.py"
    result = cli_module._resolve_targets(repo, repo, [absolute])
    assert result == (Path("src/a.py"),)


def test_resolve_targets_raises_for_an_absolute_target_outside_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    elsewhere = tmp_path / "elsewhere" / "a.py"
    with pytest.raises(ValueError, match="not inside --repo"):
        cli_module._resolve_targets(repo, repo, [elsewhere])


def test_resolve_targets_discovers_source_files_when_none_are_given(tmp_path: Path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "a.py").write_text("x = 1\n")
    result = cli_module._resolve_targets(tmp_path, tmp_path, None)
    assert result == (Path("src/pkg/a.py"),)


def test_resolve_targets_mixes_absolute_and_relative_targets_in_one_call(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    result = cli_module._resolve_targets(
        repo, repo, [Path("src/b.py"), repo / "src" / "a.py"]
    )
    assert result == (Path("src/b.py"), Path("src/a.py"))


# ---------------------------------------------------------------------------
# I2 (whole-branch review 2026-08-10) -- an explicit --target is checked
# against `_is_test_path` too, not only the no-target discovery path
# ---------------------------------------------------------------------------


def test_resolve_targets_refuses_a_relative_target_that_is_test_shaped(tmp_path: Path):
    # Before I2, `_resolve_targets` applied `_is_test_path` only via
    # `discover_source_files` (the no-`--target` path) -- an EXPLICIT
    # `--target tests/test_foo.py` sailed straight through.
    with pytest.raises(ValueError, match="test-shaped"):
        cli_module._resolve_targets(tmp_path, tmp_path, [Path("tests/test_foo.py")])


def test_resolve_targets_refuses_an_absolute_target_that_is_test_shaped(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    absolute = repo / "tests" / "test_foo.py"
    with pytest.raises(ValueError, match="test-shaped"):
        cli_module._resolve_targets(repo, repo, [absolute])


def test_resolve_targets_refuses_a_target_named_like_a_test_file_without_a_tests_dir(
    tmp_path: Path,
):
    # `_is_test_path` also catches the filename convention alone
    # (`test_*.py` / `*_test.py`), not only a `tests/`/`test/` directory.
    with pytest.raises(ValueError, match="test-shaped"):
        cli_module._resolve_targets(tmp_path, tmp_path, [Path("src/pkg/test_helpers.py")])


def test_resolve_targets_still_accepts_an_ordinary_source_target_alongside_the_check(
    tmp_path: Path,
):
    # The control: I2 must not become an over-broad refusal of every
    # explicit target -- an ordinary source path still passes through.
    result = cli_module._resolve_targets(tmp_path, tmp_path, [Path("src/pkg/core.py")])
    assert result == (Path("src/pkg/core.py"),)


# ---------------------------------------------------------------------------
# _clone_repo / _source_sha — the one place this module shells out to a
# REAL (local, no-network) git subprocess. Dedicated, isolated tests.
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    _git("init", "-q", cwd=source)
    _git("config", "user.email", "test@example.com", cwd=source)
    _git("config", "user.name", "Test", cwd=source)
    (source / "src").mkdir()
    (source / "src" / "a.py").write_text("def f():\n    return 1\n")
    _git("add", ".", cwd=source)
    _git("commit", "-q", "-m", "init", cwd=source)
    return source


def test_clone_repo_produces_a_real_independent_working_tree_with_git_history(
    git_repo: Path, tmp_path: Path
):
    dest = tmp_path / "clone"
    cli_module._clone_repo(git_repo, dest)
    assert (dest / "src" / "a.py").read_text() == "def f():\n    return 1\n"
    assert (dest / ".git").is_dir()


def test_clone_repo_never_hardlinks_git_objects(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path, tmp_path: Path
):
    # Measured directly, running this for real (invariant 14): plain
    # `--local` against a temp destination on a different filesystem than
    # --repo raised "fatal: ... Invalid cross-device link" -- `dest` is a
    # `tempfile` directory with no guarantee of sharing a device with
    # `repo`. Pins the flag that fixes it stays present.
    seen_cmds: list[list[str]] = []
    real_run = subprocess.run

    def spy_run(cmd, **kwargs):
        seen_cmds.append(list(cmd))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy_run)
    cli_module._clone_repo(git_repo, tmp_path / "clone")
    assert "--no-hardlinks" in seen_cmds[0]


def test_mutating_a_file_in_the_clone_never_touches_the_original_repo(
    git_repo: Path, tmp_path: Path
):
    # The property "never mutate the working tree" actually depends on --
    # a git-clone working-tree file is never hardlinked back to the
    # source's own working tree.
    dest = tmp_path / "clone"
    cli_module._clone_repo(git_repo, dest)
    (dest / "src" / "a.py").write_text("def f():\n    return 999\n")
    assert (git_repo / "src" / "a.py").read_text() == "def f():\n    return 1\n"


def test_source_sha_reports_the_clones_real_head_commit(git_repo: Path, tmp_path: Path):
    dest = tmp_path / "clone"
    cli_module._clone_repo(git_repo, dest)
    expected = subprocess.run(
        ["git", "-C", str(git_repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert cli_module._source_sha(dest) == expected
    assert len(expected) == 40  # a real sha, not a placeholder


def test_clone_repo_raises_for_a_repo_that_is_not_a_git_repository(tmp_path: Path):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    (not_a_repo / "a.py").write_text("x = 1\n")
    with pytest.raises(subprocess.CalledProcessError):
        cli_module._clone_repo(not_a_repo, tmp_path / "dest")


# ---------------------------------------------------------------------------
# corpus_main — argument wiring, with cloning/sentinel/baseline/generation
# all stubbed so these tests are pure wiring proofs, not re-tests of
# generate.py's own logic (already covered by test_corpus_generate.py).
# ---------------------------------------------------------------------------


def _stub_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sentinel: bool = True,
    base: Baseline | None = None,
    targets: tuple[Path, ...] = (Path("src/a.py"),),
    result: GenerationResult | None = None,
):
    monkeypatch.setattr(cli_module, "_clone_repo", lambda repo, dest: None)
    monkeypatch.setattr(cli_module, "_source_sha", lambda clone: "deadbeef" * 5)
    monkeypatch.setattr(cli_module, "sentinel_ok", lambda clone, runner: sentinel)
    default_base = Baseline(broken=0, executed=1, seconds=1.0)
    monkeypatch.setattr(
        cli_module, "measure_baseline",
        lambda clone, runner: base if base is not None else default_base,
    )
    monkeypatch.setattr(cli_module, "_resolve_targets", lambda repo, clone, given: targets)
    captured: dict[str, object] = {}

    def fake_generate_corpus(clone, tgts, base_arg, runner, **kw):
        captured["clone"] = clone
        captured["targets"] = tgts
        captured["base"] = base_arg
        captured["kwargs"] = kw
        return result if result is not None else GenerationResult((), (), (), 0.0)

    monkeypatch.setattr(cli_module, "generate_corpus", fake_generate_corpus)
    return captured


def test_repo_that_is_not_a_directory_is_refused_before_cloning_anything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    called = []
    monkeypatch.setattr(cli_module, "_clone_repo", lambda repo, dest: called.append(1))
    code = cli_module.corpus_main([
        "--repo", str(tmp_path / "does-not-exist"), "--out", str(tmp_path / "out.json"),
    ])
    assert code == OUTCOMES["refused"]
    assert called == []


def test_a_sentinel_failure_aborts_before_any_generation_and_writes_no_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    # "Done when": a sentinel-blind harness aborts instead of reporting
    # survivors -- pinned at the CLI boundary too, not just in verify.py.
    captured = _stub_pipeline(monkeypatch, sentinel=False)
    out = tmp_path / "out.json"
    code = cli_module.corpus_main(["--repo", str(tmp_path), "--out", str(out)])
    assert code == OUTCOMES["infrastructure"]
    assert not out.exists()
    assert "kwargs" not in captured  # generate_corpus must never even be called
    assert "sentinel" in capsys.readouterr().out.lower()


def test_a_sentinel_timeout_aborts_as_infrastructure_not_an_unhandled_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # A hung test suite must produce a clean CLI error, not an unhandled
    # subprocess.TimeoutExpired traceback -- pytest_runner's own timeouts
    # (verify.py) are real and finite, so this is a real, reachable path.
    monkeypatch.setattr(cli_module, "_clone_repo", lambda repo, dest: None)

    def timing_out(clone, runner):
        raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=300)

    monkeypatch.setattr(cli_module, "sentinel_ok", timing_out)
    code = cli_module.corpus_main([
        "--repo", str(tmp_path), "--out", str(tmp_path / "out.json"),
    ])
    assert code == OUTCOMES["infrastructure"]


def test_a_baseline_wrong_tree_error_aborts_as_infrastructure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from robigo.profile.verify import WrongTreeError

    _stub_pipeline(monkeypatch)

    def raising_baseline(clone, runner):
        raise WrongTreeError("boom")

    monkeypatch.setattr(cli_module, "measure_baseline", raising_baseline)
    code = cli_module.corpus_main([
        "--repo", str(tmp_path), "--out", str(tmp_path / "out.json"),
    ])
    assert code == OUTCOMES["infrastructure"]


def test_no_targets_found_is_refused_rather_than_writing_an_empty_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _stub_pipeline(monkeypatch, targets=())
    out = tmp_path / "out.json"
    code = cli_module.corpus_main(["--repo", str(tmp_path), "--out", str(out)])
    assert code == OUTCOMES["refused"]
    assert not out.exists()


def test_max_records_and_time_budget_flags_reach_generate_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured = _stub_pipeline(monkeypatch)
    cli_module.corpus_main([
        "--repo", str(tmp_path), "--out", str(tmp_path / "out.json"),
        "--max-records", "7", "--time-budget", "42.5",
    ])
    assert captured["kwargs"]["max_records"] == 7
    assert captured["kwargs"]["time_budget"] == 42.5


def test_flags_default_to_sane_values_when_not_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured = _stub_pipeline(monkeypatch)
    cli_module.corpus_main(["--repo", str(tmp_path), "--out", str(tmp_path / "out.json")])
    assert captured["kwargs"]["max_records"] == cli_module._DEFAULT_MAX_RECORDS
    assert captured["kwargs"]["time_budget"] == cli_module._DEFAULT_TIME_BUDGET


def test_source_repo_and_source_sha_are_derived_not_hardcoded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Invariant 9, at the call site: fails if corpus_main ever passes a
    # fixed placeholder instead of --repo's own string and the clone's
    # real HEAD commit.
    captured = _stub_pipeline(monkeypatch)
    cli_module.corpus_main(["--repo", str(tmp_path), "--out", str(tmp_path / "out.json")])
    assert captured["kwargs"]["source_repo"] == str(tmp_path)
    assert captured["kwargs"]["source_sha"] == "deadbeef" * 5


def test_target_flag_is_resolved_and_forwarded_to_generate_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    seen = {}
    monkeypatch.setattr(cli_module, "_clone_repo", lambda repo, dest: None)
    monkeypatch.setattr(cli_module, "_source_sha", lambda clone: "s" * 40)
    monkeypatch.setattr(cli_module, "sentinel_ok", lambda clone, runner: True)
    monkeypatch.setattr(
        cli_module, "measure_baseline",
        lambda clone, runner: Baseline(broken=0, executed=1, seconds=1.0),
    )

    def spy_resolve_targets(repo, clone, given):
        seen["given"] = given
        return (Path("src/only.py"),)

    monkeypatch.setattr(cli_module, "_resolve_targets", spy_resolve_targets)
    monkeypatch.setattr(
        cli_module, "generate_corpus",
        lambda *a, **k: GenerationResult((), (), (), 0.0),
    )
    cli_module.corpus_main([
        "--repo", str(tmp_path), "--out", str(tmp_path / "out.json"),
        "--target", "src/only.py",
    ])
    assert seen["given"] == [Path("src/only.py")]


def test_generation_result_is_written_to_the_out_path_with_the_derived_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from robigo.profile.corpus_io import CorpusRecord

    record = CorpusRecord(
        name="a-off_by_one-2", path=Path("src/a.py"), line=2,
        broken="    return 2\n", fixed="    return 1\n",
        test_id="tests/test_a.py::test_f", diagnostic="exactly one net new failure",
        operator="off_by_one", source_repo=str(tmp_path), source_sha="s" * 40,
    )
    result = GenerationResult(
        records=(record,), dropped=("some target: barren",),
        targets=(TargetOutcome(Path("src/a.py"), 1, 1, 1, False),), seconds=3.2,
    )
    _stub_pipeline(monkeypatch, result=result)

    repo_dir = tmp_path / "my-repo"
    repo_dir.mkdir()
    out = tmp_path / "out" / "corpus.json"
    code = cli_module.corpus_main(["--repo", str(repo_dir), "--out", str(out)])

    assert code == 0
    assert out.exists()
    name, records, dropped = read_corpus(out)
    assert name == "my-repo-v1"
    assert records == (record,)
    assert dropped == ("some target: barren",)


def test_the_report_printed_states_the_keep_rate_content_not_merely_something(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    result = GenerationResult(
        records=(), dropped=(),
        targets=(TargetOutcome(Path("src/barren.py"), 7, 7, 0, False),), seconds=1.5,
    )
    _stub_pipeline(monkeypatch, result=result)
    cli_module.corpus_main(["--repo", str(tmp_path), "--out", str(tmp_path / "out.json")])
    out = capsys.readouterr().out
    assert "target src/barren.py  0/7 kept" in out
    assert "written to" in out


def test_usage_error_returns_ex_usage_not_a_run_outcome_code(tmp_path: Path):
    # argparse exits 2 on a missing required flag -- must not alias
    # budget_exhausted (also 2 in the run-outcome contract).
    code = cli_module.corpus_main(["--repo", str(tmp_path)])  # --out missing
    assert code == cli_module._EX_USAGE


def test_absolute_target_outside_repo_is_a_usage_error_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(cli_module, "_clone_repo", lambda repo, dest: None)
    monkeypatch.setattr(cli_module, "sentinel_ok", lambda clone, runner: True)
    monkeypatch.setattr(
        cli_module, "measure_baseline",
        lambda clone, runner: Baseline(broken=0, executed=1, seconds=1.0),
    )
    elsewhere = tmp_path.parent / "definitely-elsewhere.py"
    code = cli_module.corpus_main([
        "--repo", str(tmp_path), "--out", str(tmp_path / "out.json"),
        "--target", str(elsewhere),
    ])
    assert code == cli_module._EX_USAGE


# ---------------------------------------------------------------------------
# End-to-end: a real git clone, a real (but injected, non-pytest) runner,
# real mutation/verification through the full stack. `pytest_runner` itself
# is monkeypatched -- no real pytest subprocess anywhere in this test.
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_target_repo(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "src" / "mylib").mkdir(parents=True)
    _git("init", "-q", cwd=source)
    _git("config", "user.email", "test@example.com", cwd=source)
    _git("config", "user.name", "Test", cwd=source)
    (source / "src" / "mylib" / "__init__.py").write_text("")
    (source / "src" / "mylib" / "calc.py").write_text("def bump(n):\n    return n + 1\n")
    (source / "src" / "mylib" / "other.py").write_text("def double(n):\n    return n * 2\n")
    _git("add", ".", cwd=source)
    _git("commit", "-q", "-m", "init", cwd=source)
    return source


def _end_to_end_runner(calc_broken_content: str):
    """Reports exactly one new failure when calc.py's on-disk content is
    THE off_by_one mutation, and clean for everything else -- matching
    both `sentinel_ok`'s own general search (which must find this same
    breakage to certify the harness) and the later real generation pass."""

    def runner(repo: Path, package: str) -> str:
        marker = f"MODULE_UNDER_TEST={repo / 'src' / 'mylib' / '__init__.py'}\nEXIT_CODE=0\n"
        calc = (repo / "src" / "mylib" / "calc.py").read_text()
        if calc == calc_broken_content:
            return marker + (
                "FAILED tests/test_calc.py::test_bump - AssertionError\n"
                "1 failed, 1 passed in 0.01s\n"
            )
        return marker + "0 failed, 2 passed in 0.01s\n"

    return runner


def test_a_real_end_to_end_run_produces_a_usable_corpus_file(
    monkeypatch: pytest.MonkeyPatch, two_target_repo: Path, tmp_path: Path
):
    from robigo.profile.corpus import candidates

    calc_source = "def bump(n):\n    return n + 1\n"
    off_by_one = next(
        m for m in candidates(calc_source, Path("src/mylib/calc.py"))
        if m.operator == "off_by_one"
    )
    lines = calc_source.splitlines(keepends=True)
    lines[off_by_one.line - 1] = off_by_one.mutated
    calc_broken_content = "".join(lines)

    monkeypatch.setattr(cli_module, "pytest_runner", _end_to_end_runner(calc_broken_content))

    out = tmp_path / "corpus.json"
    code = cli_module.corpus_main([
        "--repo", str(two_target_repo), "--out", str(out),
        "--target", "src/mylib/calc.py", "src/mylib/other.py",
    ])
    assert code == 0

    name, records, dropped = read_corpus(out)
    assert name == "source-v1"
    kept_operators = {r.operator for r in records}
    assert "off_by_one" in kept_operators
    calc_record = next(r for r in records if r.path == Path("src/mylib/calc.py"))
    assert calc_record.broken == off_by_one.mutated
    assert calc_record.fixed == off_by_one.original
    assert calc_record.test_id == "tests/test_calc.py::test_bump"
    assert calc_record.source_repo == str(two_target_repo)
    assert len(calc_record.source_sha) == 40
    # other.py offers real candidates too, none of which this runner ever
    # reports breakage for -- every one of them must be named in dropped,
    # never silently absent.
    assert any("src/mylib/other.py" in d for d in dropped)
    # The original repo's own working tree must be completely untouched.
    assert (two_target_repo / "src" / "mylib" / "calc.py").read_text() == calc_source


def test_a_real_run_against_a_repo_with_no_sentinel_target_available_aborts(
    monkeypatch: pytest.MonkeyPatch, two_target_repo: Path, tmp_path: Path
):
    # A blind runner (never reports breakage for anything, ever) must make
    # sentinel_ok fail against this real clone, and corpus_main must abort
    # rather than proceed to "generate" an all-survivors corpus.
    def blind_runner(repo: Path, package: str) -> str:
        marker = f"MODULE_UNDER_TEST={repo / 'src' / 'mylib' / '__init__.py'}\nEXIT_CODE=0\n"
        return marker + "0 failed, 2 passed\n"

    monkeypatch.setattr(cli_module, "pytest_runner", blind_runner)
    out = tmp_path / "corpus.json"
    code = cli_module.corpus_main(["--repo", str(two_target_repo), "--out", str(out)])
    assert code == OUTCOMES["infrastructure"]
    assert not out.exists()
