# src/robigo/context/budget.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robigo.context.scope import Scope, signatures_of

CHARS_PER_TOKEN = 3.3
"""Calibrated for code. Deliberately crude: it exists to catch an overrun
early, and the server's real tokenizer always outranks it (spec 3.3)."""

SYSTEM_TOKENS = 350
DIAGNOSTIC_TOKENS = 600
MAX_STEP = 5


class BudgetExhausted(Exception):
    """Scope cannot fit after every degradation step. Raised BEFORE any
    generation: with no evidence yet, a refusal is honest and a truncated
    attempt is not (spec section 3, step 5)."""


@dataclass(frozen=True)
class Budget:
    window: int
    reserve_out: int
    system: int = SYSTEM_TOKENS
    diagnostic: int = DIAGNOSTIC_TOKENS
    history: int = 200

    @property
    def scope_budget(self) -> int:
        return (
            self.window - self.reserve_out - self.system
            - self.diagnostic - self.history
        )


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def reserve_for(codec: str, file_tokens: int) -> int:
    if codec == "whole_file":
        return int(file_tokens * 1.15)
    return 512 if codec == "search_replace" else 384


def fit(scope: Scope, budget: Budget, root: Path) -> tuple[Scope, int]:
    """The first degradation step whose rendered scope fits, or refuse."""
    for step in range(1, MAX_STEP):
        candidate = scope.degrade(step)
        if _cost(candidate) <= budget.scope_budget:
            return candidate, step
    raise BudgetExhausted(
        f"scope cannot fit the window after all {MAX_STEP - 1} degradation "
        f"steps.\n"
        f"  window {budget.window}   system {budget.system}   "
        f"reserve {budget.reserve_out}   diagnostic {budget.diagnostic}\n"
        f"  available for scope {budget.scope_budget}   "
        f"smallest scope {_cost(scope.degrade(MAX_STEP - 1))}\n"
        f"Narrow it with --scope, or use a model with a larger window."
    )


def _cost(scope: Scope) -> int:
    total = 0
    for path in scope.full:
        text = path.read_text(encoding="utf-8")
        if path == scope.anchor and scope.anchor_window:
            text = _window(text, scope.anchor_window, scope.anchor_line)
        total += estimate_tokens(text)
    for path in scope.signatures:
        total += estimate_tokens(signatures_of(path.read_text(encoding="utf-8")))
    return total


def _window(text: str, span: tuple[int, int], anchor_line: int | None) -> str:
    """Imported from render rather than reimplemented: two copies of this
    would drift, and the drift shows up as an estimate that says a prompt
    fits when the server disagrees. `anchor_line` is threaded through
    unchanged from `scope.anchor_line` -- never recomputed here -- so the
    centre this computes and the centre `render` prints are the same value
    by construction, not by coincidence (amendment 2026-08-09, invariant 2)."""
    from robigo.context.render import _window_text

    return _window_text(text, span, anchor_line)
