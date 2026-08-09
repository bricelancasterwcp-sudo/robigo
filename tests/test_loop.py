# tests/test_loop.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robigo.adapters.python_ import PythonAdapter
from robigo.loop import OUTCOMES, run
from robigo.model.client import Generation


class _ScriptedClient:
    """A model whose replies are fixed. The loop must be testable with no
    GPU, or it cannot be tested at all."""

    def __init__(self, *replies: str, truncated: bool = False) -> None:
        self.replies = list(replies)
        self.truncated = truncated
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, seed: int) -> Generation:
        self.prompts.append(prompt)
        text = self.replies.pop(0) if self.replies else "done nothing left"
        return Generation(text, 10, 5, self.truncated)


FIX = """patch src/fog.py
```python
<<<<<<< SEARCH
    return t
=======
    return t * 2
>>>>>>> REPLACE
```
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "fog.py").write_text("def radius(t):\n    return t\n")
    (tmp_path / "tests" / "test_fog.py").write_text(
        "import sys\nsys.path.insert(0, 'src')\nfrom fog import radius\n\n"
        "def test_radius():\n    assert radius(2) == 4\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def test_a_correct_patch_reaches_pass(repo: Path):
    result = run("make the failing test pass", repo,
                 _ScriptedClient(FIX), PythonAdapter(python=sys.executable),
                 codec="search_replace")
    assert (result.outcome, result.exit_code) == ("pass", 0)
    assert result.turns == 1
    assert (repo / "src" / "fog.py").read_text().endswith("return t * 2\n")


def test_a_truncated_generation_is_never_applied(repo: Path):
    before = (repo / "src" / "fog.py").read_text()
    client = _ScriptedClient(FIX, FIX, FIX, truncated=True)
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace", turn_cap=3)
    assert result.outcome == "stalled"
    assert (repo / "src" / "fog.py").read_text() == before


def test_a_payload_containing_a_verb_at_column_zero_is_applied(repo: Path):
    # `done = False` at column 0 is exactly what the old "\ndone " stop
    # sequence cut, mid-payload, with finish_reason "stop".
    payload = """patch src/fog.py
```python
<<<<<<< SEARCH
def radius(t):
    return t
=======
done = False


def radius(t):
    return t * 2
>>>>>>> REPLACE
```
"""
    result = run("fix", repo, _ScriptedClient(payload),
                 PythonAdapter(python=sys.executable), codec="search_replace")
    assert result.outcome == "pass"
    assert "done = False" in (repo / "src" / "fog.py").read_text()


def test_a_parse_failure_is_fed_back_and_costs_a_turn(repo: Path):
    client = _ScriptedClient("edit src/fog.py", FIX)
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace")
    assert result.outcome == "pass"
    assert result.turns == 2
    # The diagnostic must reach the model, or it repairs blind.
    assert "not a verb" in client.prompts[1]


def test_repeating_an_identical_failing_patch_stalls(repo: Path):
    miss = FIX.replace("    return t\n", "    return t;\n")
    result = run("fix", repo, _ScriptedClient(miss, miss, miss, miss),
                 PythonAdapter(python=sys.executable),
                 codec="search_replace", stall_cap=3)
    assert (result.outcome, result.exit_code) == ("stalled", 1)


def test_the_stall_cap_ends_the_run_on_the_expected_turn(repo: Path):
    # Both existing stall tests end `stalled` whether the threshold is
    # `stall_cap - 1` or `stall_cap`, so the arithmetic itself was unguarded.
    # Only the turn number distinguishes them.
    miss = FIX.replace("    return t\n", "    return t;\n")
    result = run("fix", repo, _ScriptedClient(miss, miss, miss, miss, miss),
                 PythonAdapter(python=sys.executable),
                 codec="search_replace", stall_cap=3, turn_cap=8)
    assert result.outcome == "stalled"
    assert result.turns == 3


def test_a_mid_loop_scope_failure_keeps_the_previous_scope(repo: Path):
    # An unanchorable diagnostic (a timeout, or a failure the adapter could
    # not place in the repo) makes the mid-loop re-resolve raise. Nothing
    # pinned the guard, so removing it -- which now escapes `run` outright --
    # left all 122 tests green.
    from robigo.adapters.base import Diagnostic

    adapter = PythonAdapter(python=sys.executable)
    real_run, calls = adapter.run, {"n": 0}

    def anchorless(root: Path, filt: str | None) -> Diagnostic:
        calls["n"] += 1
        if calls["n"] > 1:
            return Diagnostic(False, None, None, "tests failed", "raw tail")
        return real_run(root, filt)

    adapter.run = anchorless  # type: ignore[method-assign]
    result = run("fix", repo, _ScriptedClient(FIX, "read src/fog.py"), adapter,
                 codec="search_replace", turn_cap=2)
    assert (result.outcome, result.turns) == ("stalled", 2)
    assert calls["n"] > 1


def test_the_turn_cap_ends_the_run(repo: Path):
    result = run("fix", repo, _ScriptedClient("run", "run", "run", "run"),
                 PythonAdapter(python=sys.executable),
                 codec="search_replace", turn_cap=2)
    assert result.turns == 2
    assert result.outcome == "stalled"


def test_a_passing_suite_refuses_before_any_generation(repo: Path):
    (repo / "src" / "fog.py").write_text("def radius(t):\n    return t * 2\n")
    client = _ScriptedClient(FIX)
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace")
    assert (result.outcome, result.exit_code) == ("refused", OUTCOMES["refused"])
    assert "failing test" in result.detail
    assert client.prompts == []


def test_a_refused_run_still_leaves_a_meta_json(repo: Path):
    # The wrapper's whole purpose, previously unguarded by any test.
    (repo / "src" / "fog.py").write_text("def radius(t):\n    return t * 2\n")
    result = run("fix", repo, _ScriptedClient(FIX),
                 PythonAdapter(python=sys.executable), codec="search_replace")
    assert result.outcome == "refused"
    metas = list((repo / ".robigo" / "runs").glob("*/meta.json"))
    assert len(metas) == 1
    assert json.loads(metas[0].read_text())["outcome"] == "refused"


def test_overflow_with_evidence_is_budget_exhausted_not_infrastructure(repo: Path):
    # Law 3: with at least one attempt already made, running out of window
    # is a SESSION RESULT with the work preserved -- not an abort. Mapping
    # it to infrastructure would discard real evidence and misreport a
    # model-side limit as a broken daemon.
    from robigo.model.client import ServerContextOverflowError

    class _OverflowsOnTurnTwo(_ScriptedClient):
        def generate(self, prompt: str, *, seed: int) -> Generation:
            if seed > 1:
                raise ServerContextOverflowError("prompt exceeds the window")
            return super().generate(prompt, seed=seed)

    result = run("fix", repo, _OverflowsOnTurnTwo("read src/fog.py"),
                 PythonAdapter(python=sys.executable), codec="search_replace")
    assert (result.outcome, result.exit_code) == ("budget_exhausted", 2)
    assert result.turns == 2


def test_overflow_with_no_evidence_is_refused(repo: Path):
    # Zero attempts submitted: nothing to preserve, so it is a loud
    # refusal rather than a fabricated result (law 3, other branch).
    from robigo.model.client import ServerContextOverflowError

    class _OverflowsImmediately(_ScriptedClient):
        def generate(self, prompt: str, *, seed: int) -> Generation:
            raise ServerContextOverflowError("prompt exceeds the window")

    result = run("fix", repo, _OverflowsImmediately(),
                 PythonAdapter(python=sys.executable), codec="search_replace")
    assert (result.outcome, result.exit_code) == ("refused", 3)


def test_a_run_is_branch_scoped_and_snapshots_first(repo: Path):
    result = run("fix", repo, _ScriptedClient(FIX),
                 PythonAdapter(python=sys.executable), codec="search_replace")
    assert result.branch is not None and result.branch.startswith("robigo/")
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo,
                         capture_output=True, text=True).stdout
    assert "snapshot" in log


def test_snapshot_never_commits_an_ignored_path(repo: Path):
    # Mutating `snapshot` to `git add -f -A` left all 122 tests green,
    # because nothing drove snapshot end to end. Force-adding is exactly how
    # a secret, a build artefact, or robigo's own transcripts reach the
    # user's history -- and an ignored file so committed has no pre-image.
    (repo / ".gitignore").write_text("secret.txt\n")
    (repo / "secret.txt").write_text("token\n")
    result = run("fix", repo, _ScriptedClient(FIX),
                 PythonAdapter(python=sys.executable), codec="search_replace")
    assert result.outcome == "pass"
    tracked = subprocess.run(["git", "ls-files"], cwd=repo,
                             capture_output=True, text=True).stdout.split()
    assert "secret.txt" not in tracked
    assert not any(name.startswith(".robigo/") for name in tracked)


def test_a_git_failure_is_infrastructure_not_a_crash(repo: Path, monkeypatch):
    import robigo.loop as loop_module

    def boom(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(loop_module, "start_branch", boom)
    result = run("fix", repo, _ScriptedClient(FIX),
                 PythonAdapter(python=sys.executable), codec="search_replace")
    assert (result.outcome, result.exit_code) == ("infrastructure", 4)


def test_only_the_patched_file_is_committed(repo: Path):
    (repo / "unrelated.py").write_text("y = 999\n")
    run("fix", repo, _ScriptedClient(FIX), PythonAdapter(python=sys.executable),
        codec="search_replace")
    changed = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"],
        cwd=repo, capture_output=True, text=True,
    ).stdout.split()
    assert changed == ["src/fog.py"]


def test_a_bare_read_does_not_crash_the_run(repo: Path):
    result = run("fix", repo, _ScriptedClient("read", FIX),
                 PythonAdapter(python=sys.executable), codec="search_replace")
    assert result.outcome == "pass"


def test_reading_a_binary_file_does_not_crash_the_run(repo: Path):
    (repo / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
    result = run("fix", repo, _ScriptedClient("read logo.png", FIX),
                 PythonAdapter(python=sys.executable), codec="search_replace")
    assert result.outcome == "pass"


def test_a_bare_find_does_not_crash_the_run(repo: Path):
    result = run("fix", repo, _ScriptedClient("find", FIX),
                 PythonAdapter(python=sys.executable), codec="search_replace")
    assert result.outcome == "pass"


def test_an_ignored_scope_file_is_refused_before_any_branch_exists(repo: Path):
    (repo / ".gitignore").write_text("src/fog.py\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "ignore"], cwd=repo, check=True)
    result = run("fix", repo, _ScriptedClient(FIX),
                 PythonAdapter(python=sys.executable), codec="search_replace")
    assert (result.outcome, result.exit_code) == ("refused", 3)
    branches = subprocess.run(
        ["git", "branch", "--format=%(refname:short)"], cwd=repo,
        capture_output=True, text=True,
    ).stdout.split()
    assert not any(name.startswith("robigo/") for name in branches)


def test_a_failure_after_branching_reports_the_branch(repo: Path, monkeypatch):
    import robigo.loop as loop_module

    def boom(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(loop_module, "snapshot", boom)
    result = run("fix", repo, _ScriptedClient(FIX),
                 PythonAdapter(python=sys.executable), codec="search_replace")
    assert result.outcome == "infrastructure"
    assert result.branch is not None and result.branch.startswith("robigo/")


def test_scope_paths_use_explicit_scope_instead_of_import_tracing(repo: Path, monkeypatch):
    import robigo.loop as loop_module

    calls: list = []

    def fake_explicit(diag, root, paths):
        calls.append(paths)
        return loop_module.Scope(
            anchor=repo / "tests" / "test_fog.py",
            full=(repo / "tests" / "test_fog.py", repo / "src" / "fog.py"),
            signatures=(),
        )

    def fail_resolve(*args, **kwargs):
        raise AssertionError("resolve must not run when scope_paths is given")

    monkeypatch.setattr(loop_module, "explicit", fake_explicit)
    monkeypatch.setattr(loop_module, "resolve", fail_resolve)
    given = [Path("src/fog.py")]
    result = run("fix", repo, _ScriptedClient(FIX),
                 PythonAdapter(python=sys.executable), codec="search_replace",
                 scope_paths=given)
    assert result.outcome == "pass"
    assert calls == [given]


def test_no_scope_paths_falls_back_to_import_traced_scope(repo: Path, monkeypatch):
    import robigo.loop as loop_module

    def fail_explicit(*args, **kwargs):
        raise AssertionError("explicit must not run when scope_paths is absent")

    monkeypatch.setattr(loop_module, "explicit", fail_explicit)
    result = run("fix", repo, _ScriptedClient(FIX),
                 PythonAdapter(python=sys.executable), codec="search_replace")
    assert result.outcome == "pass"


def test_a_nul_in_a_patch_path_is_a_model_facing_refusal(repo: Path):
    # Verified escape route before I6: `resolve()` raises ValueError on an
    # embedded NUL, which left the loop as a traceback and wrote no record.
    client = _ScriptedClient(FIX.replace("src/fog.py", "a\x00b.py"), FIX)
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace")
    assert result.outcome == "pass"
    assert "PATCH REJECTED" in client.prompts[1]


def test_find_skips_a_vendored_directory_inside_the_repo(tmp_path: Path):
    from robigo.loop import _find

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "fog.py").write_text("def computeRadius():\n    pass\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "vendored.py").write_text("computeRadius = 1\n")
    assert _find(tmp_path, "computeRadius") == "src/fog.py:1"


def test_find_works_in_a_repo_whose_own_path_contains_a_skip_name(tmp_path: Path):
    # Verified: identical content under `.../venv/myproject` found nothing,
    # because the skip list was matched against the absolute path. Every
    # file in the repo was skipped and `find` answered "not found" for
    # every symbol.
    from robigo.loop import _find

    root = tmp_path / "venv" / "myproject"
    (root / "src").mkdir(parents=True)
    (root / "src" / "fog.py").write_text("def computeRadius():\n    pass\n")
    assert _find(root, "computeRadius") == "src/fog.py:1"


def test_a_re_resolve_that_pulls_in_an_ignored_file_ends_the_run(repo: Path):
    # Task 7's ruling, reopened mid-loop: the setup check ran once, and every
    # applied turn replaced the scope with no re-check. An ignored `.py` in
    # the new scope would be patched with no pre-image, and reported as
    # `infrastructure: git failed` when `git add` refused to stage it.
    (repo / ".gitignore").write_text("src/secret.py\n")
    (repo / "src" / "secret.py").write_text("VALUE = 1\n")
    breaks = """patch src/fog.py
```python
<<<<<<< SEARCH
def radius(t):
    return t
=======
import secret


def radius(t):
    raise ValueError("boom " + str(secret.VALUE))
>>>>>>> REPLACE
```
"""
    result = run("fix", repo, _ScriptedClient(breaks, FIX),
                 PythonAdapter(python=sys.executable), codec="search_replace")
    assert (result.outcome, result.exit_code) == ("refused", 3)
    assert result.turns == 1        # mid-loop, not the setup check
    assert "ignored by git" in result.detail
    assert (repo / "src" / "secret.py").read_text() == "VALUE = 1\n"


def test_a_mid_loop_adapter_failure_is_infrastructure(repo: Path):
    from robigo.adapters.base import AdapterError

    adapter = PythonAdapter(python=sys.executable)
    real_run, calls = adapter.run, {"n": 0}

    def flaky(root, filt):
        calls["n"] += 1
        if calls["n"] > 1:
            raise AdapterError("pytest vanished")
        return real_run(root, filt)

    adapter.run = flaky  # type: ignore[method-assign]
    result = run("fix", repo, _ScriptedClient(FIX), adapter, codec="search_replace")
    assert (result.outcome, result.exit_code) == ("infrastructure", 4)
    # It says what actually failed. "git failed: pytest vanished" sent every
    # reader of the record looking at git.
    assert "pytest vanished" in result.detail
    assert "git failed" not in result.detail
