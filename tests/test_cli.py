# tests/test_cli.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robigo.cli import build_client, main
from robigo.model.client import Generation, LlamaCppClient, OllamaClient


def test_backend_selection_and_window_pass_through():
    ollama = build_client(_args(backend="ollama", model="m", window=4096))
    llama = build_client(_args(backend="llamacpp", model="m", window=8192))
    assert isinstance(ollama, OllamaClient) and ollama.window == 4096
    assert isinstance(llama, LlamaCppClient) and llama.window == 8192


def test_no_stop_sequence_can_cut_a_payload():
    # Every stop sequence robigo used to send ("\nread ", "\nfind ",
    # "\nrun\n", "\ndone ") matches ordinary Python at column 0, and a cut
    # payload arrives as an unclosed fence with finish_reason "stop" -- so
    # the truncation veto never fires.
    for backend in ("ollama", "llamacpp"):
        assert build_client(_args(backend=backend, model="m")).stop == []


def _args(**kw):
    import argparse

    defaults = dict(backend="ollama", model="m", window=8192, host=None,
                    num_predict=1024)
    return argparse.Namespace(**{**defaults, **kw})


def test_exit_code_is_3_when_the_suite_already_passes(tmp_path: Path, capsys):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert 1\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    code = main(["--root", str(tmp_path), "--python", sys.executable,
                 "--model", "m", "fix it"])
    assert code == 3
    assert "failing test" in capsys.readouterr().out


def test_exit_code_is_3_outside_a_git_repo(tmp_path: Path):
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert 0\n")
    code = main(["--root", str(tmp_path), "--python", sys.executable,
                 "--model", "m", "fix it"])
    assert code == 3


def test_scope_flag_is_parsed_and_forwarded_to_run(tmp_path: Path, monkeypatch):
    import robigo.cli as cli_module
    from robigo.loop import RunResult

    captured = {}

    def fake_run(task, root, client, adapter, *, codec, turn_cap,
                 allow_test_edits, use_git, scope_paths, recorder):
        captured["scope_paths"] = scope_paths
        return RunResult("pass", 1, 0, None, "ok")

    monkeypatch.setattr(cli_module, "run", fake_run)
    code = cli_module.main([
        "fix it", "--root", str(tmp_path), "--python", sys.executable,
        "--model", "m", "--scope", "src", "tests/test_x.py",
    ])
    assert code == 0
    assert captured["scope_paths"] == [Path("src"), Path("tests/test_x.py")]


def test_scope_flag_defaults_to_none(tmp_path: Path, monkeypatch):
    import robigo.cli as cli_module
    from robigo.loop import RunResult

    captured = {}

    def fake_run(task, root, client, adapter, *, codec, turn_cap,
                 allow_test_edits, use_git, scope_paths, recorder):
        captured["scope_paths"] = scope_paths
        return RunResult("pass", 1, 0, None, "ok")

    monkeypatch.setattr(cli_module, "run", fake_run)
    cli_module.main(["--root", str(tmp_path), "--python", sys.executable,
                     "--model", "m", "fix it"])
    assert captured["scope_paths"] is None


_FIX = """patch src/fog.py
```python
<<<<<<< SEARCH
    return t
=======
    return t * 2
>>>>>>> REPLACE
```
"""


class _Scripted:
    """A model whose replies are fixed, so the CLI can be driven end to end
    with no daemon."""

    model = "scripted"
    window = 8192

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    def generate(self, prompt: str, *, seed: int) -> Generation:
        text = self.replies.pop(0) if self.replies else "done nothing left"
        return Generation(text, 10, 5, False)


def _git(root: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *argv], cwd=root, capture_output=True,
                          text=True)


def _tree(root: Path) -> dict[str, bytes]:
    """Every file's bytes, minus what is not part of the user's work."""
    skip = (".git", ".robigo", "__pycache__")
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not any(part in skip for part in path.relative_to(root).parts)
    }


def test_the_printed_undo_recipe_really_restores_the_pre_run_tree(
    tmp_path: Path, capsys, monkeypatch
):
    """The spec's headline invariant, executed rather than asserted about.

    The recipe is read back out of stdout and run with real git, because the
    previous instruction (`git checkout -`) was printed for twelve reviews
    and never once performed: it depends on a reflog another shell clobbers,
    and it drops the user's uncommitted work out of the working tree.
    """
    import robigo.cli as cli_module

    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    (tmp_path / "src" / "fog.py").write_text("def radius(t):\n    return t\n")
    (tmp_path / "tests" / "test_fog.py").write_text(
        "import sys\nsys.path.insert(0, 'src')\nfrom fog import radius\n\n"
        "def test_radius():\n    assert radius(2) == 4\n"
    )
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "init")
    # A dirty tree: one hand-edit to a tracked file, one untracked file.
    (tmp_path / "src" / "fog.py").write_text(
        "def radius(t):\n    return t\n\n\n# work in progress\n"
    )
    (tmp_path / "scratch.py").write_text("WIP = 1\n")
    before = _tree(tmp_path)
    original = _git(tmp_path, "symbolic-ref", "--short", "HEAD").stdout.strip()

    monkeypatch.setattr(cli_module, "build_client", lambda args: _Scripted(_FIX))
    code = main(["--root", str(tmp_path), "--python", sys.executable,
                 "--model", "m", "fix it"])
    assert code == 0
    assert _tree(tmp_path) != before          # the run really changed the tree

    printed = capsys.readouterr().out
    assert "`git checkout -`" not in printed   # the reflog recipe is gone
    recipe = [
        line[line.index("git "):].split("#")[0].split()
        for line in printed.splitlines() if "git checkout" in line
    ]
    # Two steps, because one is not enough: returning to the branch is what
    # `git checkout -` meant to do, and restoring the snapshot is what it
    # silently skipped.
    assert len(recipe) == 2, printed
    assert recipe[0] == ["git", "checkout", original]
    assert recipe[1][1:2] == ["checkout"] and recipe[1][3:] == ["--", "."]

    back = _git(tmp_path, *recipe[0][1:])
    assert back.returncode == 0, (back.stdout, back.stderr)
    # Step 1 alone is the old behaviour, and it is not enough: the hand-edit
    # and the untracked file are now stranded in the snapshot commit.
    assert _tree(tmp_path) != before

    restore = _git(tmp_path, *recipe[1][1:])
    assert restore.returncode == 0, (restore.stdout, restore.stderr)
    assert _tree(tmp_path) == before


def test_a_raising_adapter_is_exit_4_not_a_traceback(tmp_path: Path, capsys,
                                                     monkeypatch):
    # An escaping exception is the "never conflated" law broken in the more
    # damaging direction: a traceback exits 1, which the contract defines as
    # `stalled` -- a model result.
    import robigo.cli as cli_module

    class _Exploding:
        name = "python"
        test_command = "pytest"

        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, root: Path, filt: str | None) -> None:
            raise RuntimeError("adapter blew up")

        def imports(self, path: Path, root: Path) -> list[Path]:
            return []

        def syntax_ok(self, text: str) -> bool:
            return True

    monkeypatch.setattr(cli_module, "PythonAdapter", _Exploding)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    code = main(["--root", str(tmp_path), "--python", sys.executable,
                 "--model", "m", "fix it"])
    assert code == 4
    assert "internal error" in capsys.readouterr().out
    # The record must exist even though the loop never returned.
    metas = list((tmp_path / ".robigo" / "runs").glob("*/meta.json"))
    assert len(metas) == 1
    assert json.loads(metas[0].read_text())["outcome"] == "infrastructure"


def test_a_usage_error_never_aliases_a_contract_exit_code(capsys):
    # --scope is greedy, so this swallows the task and argparse exits 2 --
    # which is `budget_exhausted`. It must not be reported as one.
    code = main(["--model", "m", "--scope", "src", "fix the test"])
    assert code == 64
    assert "usage" in capsys.readouterr().err.lower()


def test_help_exits_zero(capsys):
    assert main(["--help"]) == 0
    assert "--scope" in capsys.readouterr().out


@pytest.mark.live
def test_live_one_real_repair(tmp_path: Path):
    """One real generation end to end. Asserts the plumbing works, NOT
    that the model succeeds -- a failure here is a valid result.

    Uses the 0.5B deliberately. This machine shares its GPU with another
    long-running experiment, so a 7B's 8.1GB will not fit; and since the
    assertion tolerates exit 1, model capability is irrelevant to what
    this test checks. It must skip cleanly rather than fail when the
    daemon is down or the model is not pulled.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("def double(x):\n    return x\n")
    (tmp_path / "test_m.py").write_text(
        "import sys\nsys.path.insert(0, 'src')\nfrom m import double\n\n"
        "def test_double():\n    assert double(2) == 4\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=10)
    except (urllib.error.URLError, OSError):
        pytest.skip("ollama daemon not reachable")

    code = main([
        "--root", str(tmp_path), "--python", sys.executable,
        "--model", "qwen2.5-coder:0.5b-instruct-q8_0",
        "--window", "4096", "make the failing test pass",
    ])
    # 0 pass, 1 stalled, 4 infrastructure (model not pulled). Capability is
    # not under test; an uncaught traceback would be.
    assert code in (0, 1, 4)
