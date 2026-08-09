# src/robigo/apply/safety.py
from __future__ import annotations

import subprocess
from collections.abc import Sequence
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
            f"Run `git init`, or pass --no-git to accept irreversible edits."
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


def snapshot(root: Path, message: str, scope_files: Sequence[Path] = ()) -> None:
    """Commit whatever is in the tree BEFORE the first patch, dirty or not,
    so a pre-existing uncommitted change in a non-ignored path cannot be
    lost.

    The guarantee has one hole, and it is refused rather than papered over:
    git will not stage an ignored file, so an ignored file in scope would be
    patched with no recoverable pre-image.
    """
    ignored = _ignored(root, scope_files)
    if ignored:
        raise RefusedError(
            f"{', '.join(ignored)} is in scope but ignored by git, so robigo "
            f"cannot snapshot its pre-patch state and could not undo a change "
            f"to it. Un-ignore it, or narrow --scope to exclude it."
        )
    _commit(root, message, ["-A"])


def commit_all(root: Path, message: str, paths: Sequence[Path]) -> None:
    """Commit exactly the paths named. Staging the whole tree would fold a
    concurrent hand-edit into a commit titled as the model's patch, and
    `git checkout -` would then strand that edit on this branch."""
    _commit(root, message, [str(path) for path in paths])


def _ignored(root: Path, paths: Sequence[Path]) -> list[str]:
    """Which of these paths git ignores. `check-ignore` exits 1 when none
    match, which is not an error — so no check=True here."""
    if not paths:
        return []
    proc = subprocess.run(
        ["git", "check-ignore", "--", *(str(path) for path in paths)],
        cwd=root, capture_output=True, text=True,
    )
    return proc.stdout.split()


def _commit(root: Path, message: str, pathspec: list[str]) -> None:
    """--allow-empty unconditionally. Probing whether anything is staged is
    fiddly on an unborn HEAD, and an empty commit titled after a patch is
    honest — it records that the patch was a no-op. It also cannot raise,
    which a bare `git commit` on a clean tree does."""
    subprocess.run(["git", "add", *pathspec], cwd=root, check=True)
    subprocess.run(
        ["git", *_GIT_ID, "commit", "-q", "--allow-empty", "-m", message],
        cwd=root, check=True,
    )
