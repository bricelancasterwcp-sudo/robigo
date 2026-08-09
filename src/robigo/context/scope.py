# src/robigo/context/scope.py
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

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
    if not anchor.is_file():
        raise ScopeError(f"anchor {diag.file} does not exist under {root}")

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


def signatures_of(text: str) -> str:
    """Definition lines only. Hop-2 files are for orientation, not
    reading, and their bodies are the single largest avoidable cost in a
    small window."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    lines = text.split("\n")
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(lines[node.lineno - 1].rstrip())
    return "\n".join(out) + ("\n" if out else "")
