# src/robigo/paths.py
from __future__ import annotations

from pathlib import Path


class OutsideRepo(Exception):
    """A supplied path that does not resolve to a location inside the repo."""


def contain(root: Path, arg: str | Path) -> Path:
    """Resolve `arg` against `root` and return it, or raise `OutsideRepo`.

    One implementation, because there were five and only one of them guarded
    `ValueError` from `resolve()` — which is why a path containing an
    embedded NUL crashed the run.
    """
    root = root.resolve()
    try:
        resolved = (root / Path(arg)).resolve()
    except (ValueError, OSError) as exc:
        raise OutsideRepo(f"{arg!r} is not a usable path: {exc}") from exc
    if not resolved.is_relative_to(root):
        raise OutsideRepo(f"{arg!r} resolves outside the repository")
    return resolved
