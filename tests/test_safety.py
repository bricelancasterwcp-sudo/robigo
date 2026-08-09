# tests/test_safety.py
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from robigo.apply.safety import (
    RefusedError,
    check_target,
    commit_all,
    ensure_repo,
    snapshot,
    start_branch,
)
from robigo.context.scope import Scope


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src.py").write_text("x = 1\n")
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert 0\n")
    (tmp_path / "outside.py").write_text("y = 1\n")
    for argv in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", *argv], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], cwd=tmp_path, check=True)
    return tmp_path


def _scope(repo: Path) -> Scope:
    return Scope(repo / "test_x.py", (repo / "test_x.py", repo / "src.py"), ())


def test_a_file_in_scope_is_allowed(repo: Path):
    assert check_target("src.py", repo, _scope(repo)) == (repo / "src.py").resolve()


def test_the_anchor_test_is_read_only_by_default(repo: Path):
    with pytest.raises(RefusedError) as e:
        check_target("test_x.py", repo, _scope(repo))
    assert "failing test" in str(e.value)


def test_the_anchor_test_can_be_opted_into(repo: Path):
    assert check_target("test_x.py", repo, _scope(repo), allow_test_edits=True)


def test_a_file_outside_scope_is_refused(repo: Path):
    with pytest.raises(RefusedError) as e:
        check_target("outside.py", repo, _scope(repo))
    assert "scope" in str(e.value)


@pytest.mark.parametrize("bad", ["../escape.py", "/etc/passwd"])
def test_paths_leaving_the_repo_are_refused(repo: Path, bad: str):
    with pytest.raises(RefusedError):
        check_target(bad, repo, _scope(repo))


def test_ensure_repo_refuses_a_non_repo(tmp_path: Path):
    with pytest.raises(RefusedError) as e:
        ensure_repo(tmp_path)
    assert "--no-git" in str(e.value)


def test_snapshot_commits_a_dirty_tree_so_nothing_is_lost(repo: Path):
    (repo / "src.py").write_text("x = 999\n")
    branch = start_branch(repo, "fog")
    snapshot(repo, "robigo: snapshot before first patch")
    assert branch.startswith("robigo/fog-")
    out = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                         capture_output=True, text=True).stdout
    assert out.strip() == ""


def test_commit_all_records_each_applied_patch(repo: Path):
    start_branch(repo, "fog")
    (repo / "src.py").write_text("x = 2\n")
    commit_all(repo, "robigo: patch src.py", [repo / "src.py"])
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo,
                         capture_output=True, text=True).stdout
    assert "robigo: patch src.py" in log


def test_commit_all_commits_only_the_named_path(repo: Path):
    start_branch(repo, "fog")
    snapshot(repo, "robigo: snapshot")
    (repo / "src.py").write_text("x = 2\n")
    (repo / "outside.py").write_text("y = 999\n")
    commit_all(repo, "robigo: patch src.py", [repo / "src.py"])
    changed = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"],
        cwd=repo, capture_output=True, text=True,
    ).stdout.split()
    assert changed == ["src.py"]


def test_a_no_op_patch_commits_without_raising(repo: Path):
    start_branch(repo, "fog")
    snapshot(repo, "robigo: snapshot")
    commit_all(repo, "robigo: patch src.py", [repo / "src.py"])


def test_an_ignored_scope_file_is_refused(repo: Path):
    (repo / ".gitignore").write_text("secret.py\n")
    (repo / "secret.py").write_text("x = 1\n")
    start_branch(repo, "fog")
    with pytest.raises(RefusedError) as e:
        snapshot(repo, "robigo: snapshot", [repo / "secret.py"])
    assert "ignored by git" in str(e.value)


def test_snapshot_with_no_ignored_scope_files_proceeds(repo: Path):
    start_branch(repo, "fog")
    snapshot(repo, "robigo: snapshot", [repo / "src.py"])
    assert subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo,
        capture_output=True, text=True,
    ).stdout.strip() == ""
