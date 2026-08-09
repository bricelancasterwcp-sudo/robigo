# src/robigo/apply/safety.py
from __future__ import annotations

import subprocess
from pathlib import Path

from robigo.context.scope import Scope

_GIT_ID = ("-c", "user.email=robigo@localhost", "-c", "user.name=robigo")


class RefusedError(Exception):
    """A refusal, not a failure. Raised before anything is written."""


def check_target(
    arg: str, root: Path, scope: Scope, allow_test_edits: bool = False
) -> Path:
    target = (root / arg).resolve()
    root = root.resolve()
    if not target.is_relative_to(root):
        raise RefusedError(
            f"'{arg}' resolves outside the repository. Patch only files "
            f"you were shown."
        )
    if target == scope.anchor.resolve() and not allow_test_edits:
        raise RefusedError(
            f"'{arg}' is the failing test itself and is read-only. Fix the "
            f"code under test, not the test. (--allow-test-edits overrides.)"
        )
    if target not in {p.resolve() for p in scope.full}:
        raise RefusedError(
            f"'{arg}' is outside the current scope. Use `read {arg}` first, "
            f"or re-run with --scope to widen it."
        )
    return target


def ensure_repo(root: Path) -> None:
    if not (root / ".git").is_dir():
        raise RefusedError(
            f"{root} is not a git repository, so a run could not be undone. "
            f"Run `git init`, or pass --no-git to accept unreversible edits."
        )


def start_branch(root: Path, slug: str) -> str:
    """The first UNUSED name, not a count. Counting collides as soon as any
    earlier branch is deleted — two branches minus one deleted still counts
    1, and `git checkout -b` on a name that already exists aborts the run
    under check=True."""
    existing = set(
        subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.split()
    )
    number = 1
    while f"robigo/{slug}-{number}" in existing:
        number += 1
    branch = f"robigo/{slug}-{number}"
    subprocess.run(["git", "checkout", "-q", "-b", branch], cwd=root, check=True)
    return branch


def snapshot(root: Path, message: str) -> None:
    """Commit whatever is in the tree BEFORE the first patch, dirty or
    not, so a pre-existing uncommitted change can never be lost."""
    _commit(root, message, allow_empty=True)


def commit_all(root: Path, message: str) -> None:
    _commit(root, message, allow_empty=False)


def _commit(root: Path, message: str, allow_empty: bool) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    argv = ["git", *_GIT_ID, "commit", "-qm", message]
    if allow_empty:
        argv.append("--allow-empty")
    subprocess.run(argv, cwd=root, check=True)
