# src/robigo/context/scope.py
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from robigo.adapters.base import Adapter, Diagnostic


class ScopeError(Exception):
    """Scope cannot be resolved. Refused before any generation, because
    a session with no anchor can only fabricate a result (spec 3, step 5)."""


@dataclass(frozen=True)
class Scope:
    anchor: Path
    full: tuple[Path, ...]
    signatures: tuple[Path, ...]


def resolve(
    diag: Diagnostic, adapter: Adapter, root: Path, hops: int = 2
) -> Scope:
    if not diag.file:
        raise ScopeError(
            "the test output named no file, so there is no anchor to scope "
            "from. Run with --scope to supply one explicitly."
        )
    anchor = (root / diag.file).resolve()
    if not anchor.is_relative_to(root.resolve()):
        raise ScopeError(
            f"anchor {diag.file} resolves outside {root}. Scope never leaves "
            f"the repository; pass --scope to set it explicitly."
        )
    if not anchor.is_file():
        raise ScopeError(
            f"anchor {diag.file} does not exist under {root}. Check the path "
            f"is repo-relative and spelled correctly, or pass --scope to name "
            f"the files to work in explicitly."
        )

    full: list[Path] = [anchor]
    for path in adapter.imports(anchor, root):
        if path not in full:
            full.append(path)
    signatures: list[Path] = []
    if hops >= 2:
        for parent in full[1:]:
            for path in adapter.imports(parent, root):
                if path not in full and path not in signatures:
                    signatures.append(path)
    return Scope(anchor=anchor, full=tuple(full), signatures=tuple(signatures))


def explicit(diag: Diagnostic, root: Path, paths: Sequence[Path]) -> Scope:
    """Scope drawn by the user rather than traced. The anchor still comes
    from the diagnostic — the failing test is what the run is about — but
    nothing is inferred beyond the paths given."""
    if not diag.file:
        raise ScopeError(
            "--scope needs a failing test to anchor on, and the test output "
            "named no file."
        )
    anchor = (root / diag.file).resolve()
    full: list[Path] = [anchor]
    for given in paths:
        target = (root / given).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ScopeError(f"--scope path {given} is outside {root}")
        found = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for path in found:
            if path.is_file() and path not in full:
                full.append(path)
    return Scope(anchor=anchor, full=tuple(full), signatures=())


def signatures_of(text: str) -> str:
    """Definition lines only, in source order, decorators included. Hop-2
    files are for orientation, and an outline whose order does not match
    the file is worse than no outline."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    lines = text.split("\n")
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    out: list[str] = []
    for node in sorted(nodes, key=lambda item: item.lineno):
        start = node.lineno
        if node.decorator_list:
            start = min(decorator.lineno for decorator in node.decorator_list)
        out.extend(lines[number - 1].rstrip() for number in range(start, node.lineno + 1))
    return "\n".join(out) + ("\n" if out else "")
