# src/robigo/context/budget.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robigo.context.scope import Scope

CHARS_PER_TOKEN = 3.3
"""Calibrated for code. Deliberately crude: it exists to catch an overrun
early, and the server's real tokenizer always outranks it (spec 3.3)."""

SYSTEM_TOKENS = 350
DIAGNOSTIC_TOKENS = 600
MAX_STEP = 5


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _default_history_tokens() -> int:
    """The worst case `loop.py` can actually hand `render` as history, not a
    round number a second module can silently contradict (ruled
    2026-08-09): a fixed `history=200` measured against `loop.py`'s real
    `_READ_CAP` (4000 chars) and `_HISTORY_TURNS` (2) under-reserved by
    12x -- two capped `read` results render to ~2449 tokens, so `fit()`
    accepted prompts that then overflowed the window.

    Reproduces the exact shape `render` prints for a turn
    (`f"you: {action}\\nresult: {result}"`) and the exact suffix `loop.py`'s
    `_read` appends to a capped result, so this tracks both constants for
    real rather than restating them. Rounded UP to the nearest 100 -- the
    over-reserving direction the invariant requires -- rather than kept as
    the precise, fragile-looking token count.

    Imported from `robigo.loop` inside this function, not at module level:
    `context` is the lower layer here (`robigo.loop` is the orchestrator
    that will eventually call `fit()`), and a deferred import keeps that
    direction from becoming a hard dependency every time `context.budget`
    is imported, the same reason `_section` below imports `render` lazily
    rather than at the top of this file."""
    from robigo.loop import _HISTORY_TURNS, _READ_CAP

    capped_read = "x" * _READ_CAP + "\n<truncated>\n"
    turn_text = f"you: read <path>\nresult: {capped_read}"
    worst_case = estimate_tokens("\n".join([turn_text] * _HISTORY_TURNS))
    return -(-worst_case // 100) * 100


DEFAULT_HISTORY_TOKENS = _default_history_tokens()


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
    history: int = DEFAULT_HISTORY_TOKENS

    @property
    def scope_budget(self) -> int:
        return (
            self.window - self.reserve_out - self.system
            - self.diagnostic - self.history
        )


def reserve_for(codec: str, file_tokens: int) -> int:
    if codec == "whole_file":
        return int(file_tokens * 1.15)
    return 512 if codec == "search_replace" else 384


def fit(scope: Scope, budget: Budget, root: Path) -> tuple[Scope, int]:
    """The first degradation step whose rendered scope fits, or refuse."""
    for step in range(1, MAX_STEP):
        candidate = scope.degrade(step)
        if _cost(candidate, root) <= budget.scope_budget:
            return candidate, step
    raise BudgetExhausted(
        f"scope cannot fit the window after all {MAX_STEP - 1} degradation "
        f"steps.\n"
        f"  window {budget.window}   system {budget.system}   "
        f"reserve {budget.reserve_out}   diagnostic {budget.diagnostic}   "
        f"history {budget.history}\n"
        f"  available for scope {budget.scope_budget}   "
        f"smallest scope {_cost(scope.degrade(MAX_STEP - 1), root)}\n"
        f"Narrow it with --scope, or use a model with a larger window."
    )


def _cost(scope: Scope, root: Path) -> int:
    """The scope's cost, computed from the exact text `render` would emit
    for it -- never from a second, independently-summed idea of the same
    files (amendment 2, ruled 2026-08-09). `diag`, `history`, and `codec`
    are deliberately NOT threaded in here: those are the caller's fixed
    costs and are already seated in `Budget`; only the scope's own text has
    to agree with what render prints for it."""
    return estimate_tokens(_section(scope, root))


def _section(scope: Scope, root: Path) -> str:
    """Imported from render rather than reimplemented: two copies of the
    scope's own text were the root cause of amendment 2's three findings --
    a cost that omitted the header/label tokens the prompt actually
    contains, a cost that raised where render degrades cleanly to a
    placeholder, and an honest-fallback label with no test at all. This is
    the `_window_text`/`_window` delegation pattern from invariant 2,
    widened to cover the whole scope section rather than only the windowed
    span."""
    from robigo.context.render import _scope_section

    return _scope_section(scope, root)
