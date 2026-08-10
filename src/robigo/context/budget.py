# src/robigo/context/budget.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from robigo.context.scope import Scope

if TYPE_CHECKING:
    from robigo.adapters.base import Diagnostic
    from robigo.context.render import Turn

CHARS_PER_TOKEN = 3.3
"""Calibrated for code. Deliberately crude: it exists to catch an overrun
early, and the server's real tokenizer always outranks it (spec 3.3)."""

SYSTEM_TOKENS = 350
"""A conservative guess, not a measurement (task 1, invariant 3). Now that
the real cost is measurable -- `measure()` derives it from the exact
preamble text `render` emits, ~211-233 tokens depending on codec -- this
stops being the number `Budget` is built with. It survives only as
`Budget.system`'s dataclass default: the fallback for a caller building a
`Budget` directly with no diag/history in hand to measure against.

The "~211-233" figure is codec alone holding everything else fixed; it is
NOT the whole story. `measure()`'s seated `system` also shifts by +/-1
token with the *scope* passed in, because `estimate_tokens` is
`int(len/CHARS_PER_TOKEN) + 1` -- not additive -- and `_fixed_costs` seats
`system` as a delta between two prefixes that both include the scope's own
text (task 2, invariant 4's amendment). Do not cache this per codec alone,
and do not read the range above as the full domain it is measured over."""
DIAGNOSTIC_TOKENS = 600
"""Same status as `SYSTEM_TOKENS`, and for the same reason: a guess, kept
only as `Budget.diagnostic`'s fallback default. The real cost is a single
`--tb=line` summary line, not the full `DIAGNOSTIC_CHAR_CAP` -- observed
~16-80 tokens rather than 600. `measure()` derives it for real from a
`Diagnostic` in hand."""
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

    Used as `Budget.history`'s `default_factory` (ruled 2026-08-09, fix
    wave round 2), NOT called at module import time: a `DEFAULT_HISTORY_
    TOKENS = _default_history_tokens()` module-level call was only
    SYNTACTICALLY deferred -- `robigo.loop` was still imported while
    `context.budget` itself was mid-import, so the first thing in `src/`
    that imports both (the still-to-come `loop.py` wiring of `fit()`,
    which is the whole reason `Budget`/`fit` exist) would hit `ImportError:
    cannot import name '_HISTORY_TURNS' from partially initialized module
    'robigo.loop'`. `default_factory` runs this at `Budget()` CONSTRUCTION
    time instead, by which point both modules have finished importing, so
    the coupling to `robigo.loop` is a value dependency, not an
    import-order one."""
    from robigo.loop import _HISTORY_TURNS, _READ_CAP

    capped_read = "x" * _READ_CAP + "\n<truncated>\n"
    turn_text = f"you: read <path>\nresult: {capped_read}"
    worst_case = estimate_tokens("\n".join([turn_text] * _HISTORY_TURNS))
    return -(-worst_case // 100) * 100


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
    history: int = field(default_factory=_default_history_tokens)

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


def _fixed_costs(
    scope: Scope,
    diag: Diagnostic,
    history: tuple[Turn, ...],
    codec: str,
    root: Path,
) -> tuple[int, int, int]:
    """Measures (system, diagnostic, history) as marginal deltas between
    successive EXACT prefixes of the text `render` actually builds, so that

        system + diagnostic + history + _cost(scope, root)
            == estimate_tokens(render(scope, diag, history, codec, root))

    holds exactly (task 1, invariant 1) rather than approximately.
    `estimate_tokens` is `int(len(text) / CHARS_PER_TOKEN) + 1` -- NOT
    additive over concatenation, since both the floor and the trailing
    `+1` apply once per call. Four independently-estimated pieces (system
    alone, diagnostic alone, history alone, scope alone) do not in general
    sum to the estimate of the whole joined text. The only way four
    buckets can reconstruct one non-additive total exactly is for each to
    be the DIFFERENCE between two running totals over the same string, so
    that summing them telescopes back to the outermost total algebraically
    -- regardless of where each individual rounding happens to land. That
    is why this measures cumulative prefixes (`upto_scope`, `upto_diag`,
    `upto_history`) rather than estimating each section's own text in
    isolation.

    Built from the same helpers `render` calls -- `_preamble`,
    `_scope_section` (via `_section`), `_diagnostic_section`,
    `_history_section` -- never a second copy of the preamble, the
    diagnostic line format, or the history block format (invariant 2).

    Where the trailer and the top-level joining newlines are seated: in
    `history`, via `_history_section` (see its docstring). `render`'s
    trailer sits immediately after history, present even with zero turns,
    and `Budget` has no bucket of its own for it."""
    from robigo.context.render import _diagnostic_section, _history_section, _preamble

    preamble = _preamble(codec)
    scope_text = _section(scope, root)
    diagnostic_text = _diagnostic_section(diag)
    history_text = _history_section(history)

    scope_cost = estimate_tokens(scope_text)
    upto_scope = estimate_tokens("\n".join([preamble, scope_text]))
    upto_diag = estimate_tokens("\n".join([preamble, scope_text, diagnostic_text]))
    upto_history = estimate_tokens(
        "\n".join([preamble, scope_text, diagnostic_text, history_text])
    )

    system = upto_scope - scope_cost
    diagnostic = upto_diag - upto_scope
    history_tokens = upto_history - upto_diag
    return system, diagnostic, history_tokens


def measure(
    scope: Scope,
    diag: Diagnostic,
    history: tuple[Turn, ...],
    codec: str,
    root: Path,
    window: int,
    reserve_out: int,
) -> Budget:
    """Build a `Budget` whose system/diagnostic/history are MEASURED from
    the exact text `render` would emit for this call, rather than the
    conservative constants `SYSTEM_TOKENS`/`DIAGNOSTIC_TOKENS` stand in for
    when no diag/history is available (task 1, invariant 3). Needs a real
    `diag` and `history` to measure against -- a caller with neither in
    hand builds `Budget(...)` directly and takes its named fallbacks
    instead."""
    system, diagnostic, history_tokens = _fixed_costs(scope, diag, history, codec, root)
    return Budget(
        window=window, reserve_out=reserve_out, system=system,
        diagnostic=diagnostic, history=history_tokens,
    )
