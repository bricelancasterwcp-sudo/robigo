# src/robigo/context/scope.py
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from robigo.adapters.base import Adapter, Diagnostic
from robigo.paths import OutsideRepo, contain


class ScopeError(Exception):
    """Scope cannot be resolved. Refused before any generation, because
    a session with no anchor can only fabricate a result (spec 3, step 5)."""


@dataclass(frozen=True)
class Scope:
    anchor: Path
    full: tuple[Path, ...]
    signatures: tuple[Path, ...]
    anchor_window: tuple[int, int] | None = None
    anchor_line: int | None = None
    """The diagnostic's line, 1-indexed, carried here rather than recomputed:
    `degrade` has no access to the `Diagnostic` that built this `Scope`, and
    windowing must centre on the failing line, not the file's midpoint
    (amendment 2026-08-09, invariant 1). Defaults to None so every existing
    construction -- including every one in tests/test_*.py that predates this
    field -- stays valid, and so a diagnostic with no line degrades to the
    old midpoint behaviour instead of raising."""

    def degrade(self, step: int) -> Scope:
        """One step down a FIXED ladder (spec section 3). Fixed rather
        than heuristic so the result is reproducible and testable without
        a model. `step` must be 1-4: `fit` never asks for more than that
        (rungs 1-4, then refusal), and a caller that miscomputes a step
        beyond 4 gets a ValueError rather than rung 4's scope handed back
        silently as if it were something further down the ladder
        (amendment 2, ruled 2026-08-09)."""
        if step <= 1:
            return self
        if step == 2:
            return Scope(self.anchor, self.full, (), self.anchor_window, self.anchor_line)
        if step == 3:
            return Scope(self.anchor, (self.anchor,), self.full[1:], None, self.anchor_line)
        if step == 4:
            return Scope(
                self.anchor, (self.anchor,), self.full[1:], (-60, 60), self.anchor_line
            )
        raise ValueError(f"degrade() step must be between 1 and 4, got {step}")


def _anchor(diag_file: str, root: Path) -> Path:
    """The anchor, contained. Shared by both entry points so neither can
    reach the working tree with an unchecked path."""
    try:
        return contain(root, diag_file)
    except OutsideRepo as exc:
        raise ScopeError(
            f"anchor {diag_file} resolves outside {root}. Scope never leaves "
            f"the repository; pass --scope to set it explicitly."
        ) from exc


def resolve(
    diag: Diagnostic, adapter: Adapter, root: Path, hops: int = 2
) -> Scope:
    if not diag.file:
        raise ScopeError(
            "the test output named no file, so there is no anchor to scope "
            "from. Run with --scope to supply one explicitly."
        )
    anchor = _anchor(diag.file, root)
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
    return Scope(
        anchor=anchor, full=tuple(full), signatures=tuple(signatures),
        anchor_line=diag.line,
    )


def explicit(diag: Diagnostic, root: Path, paths: Sequence[Path]) -> Scope:
    """Scope drawn by the user rather than traced. The anchor still comes
    from the diagnostic — the failing test is what the run is about — but
    nothing is inferred beyond the paths given."""
    if not diag.file:
        raise ScopeError(
            "--scope needs a failing test to anchor on, and the test output "
            "named no file."
        )
    anchor = _anchor(diag.file, root)
    full: list[Path] = [anchor]
    for given in paths:
        try:
            target = contain(root, given)
        except OutsideRepo as exc:
            raise ScopeError(f"--scope path {given} is outside {root}") from exc
        found = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for path in found:
            if path.is_file() and path not in full:
                full.append(path)
    return Scope(anchor=anchor, full=tuple(full), signatures=(), anchor_line=diag.line)


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
