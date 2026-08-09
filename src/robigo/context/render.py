# src/robigo/context/render.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robigo.adapters.base import Diagnostic
from robigo.context.scope import Scope, signatures_of

SYSTEM = """You are repairing one failing test in an existing codebase.

Emit exactly ONE action per reply, as a line of its own, then stop:

  read <path>        show a file you have not been given
  find <symbol>      locate a symbol elsewhere in the repo
  patch <path>       change a file (needs a fenced payload)
  run                re-run the tests
  done <summary>     the test passes and you are finished

Rules:
- One action per reply. Never two.
- Only `patch` takes a fenced payload.
- You may not edit the failing test itself.
- Do not explain at length; the action is what matters.
"""

_CODEC_HELP = {
    "search_replace": (
        "For `patch`, the payload is one or more blocks:\n"
        "<<<<<<< SEARCH\n<exact existing lines>\n=======\n"
        "<replacement lines>\n>>>>>>> REPLACE\n"
        "The SEARCH lines must match the file byte-for-byte."
    ),
    "whole_file": (
        "For `patch`, the payload is the complete new file, top to "
        "bottom. Do not abbreviate or elide any part of it."
    ),
}


_UNREADABLE = "<unreadable or not valid UTF-8; not shown>\n"


def _read(path: Path) -> str | None:
    """None when the file cannot be read. Callers substitute a marker: a
    file vanishing or being binary must not end the run."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


@dataclass(frozen=True)
class Turn:
    action: str
    result: str


def render(
    scope: Scope,
    diag: Diagnostic,
    history: tuple[Turn, ...],
    codec: str,
    root: Path,
) -> str:
    parts = [SYSTEM, _CODEC_HELP[codec], "", _scope_section(scope, root)]
    if diag.file and diag.line:
        where = f"{diag.file}:{diag.line}"
    elif diag.file:
        where = diag.file
    else:
        where = "(location unknown)"
    parts += ["--- failing test ---", f"{where}  {diag.message}", ""]
    for turn in history:
        parts.append(f"you: {turn.action}\nresult: {turn.result}")
    parts.append("Your action:")
    return "\n".join(parts)


def _scope_section(scope: Scope, root: Path) -> str:
    """The scope's own file text, exactly as it enters the prompt: headers,
    windowing and its label, and the unreadable-file substitution all
    included. `render` and `budget._cost` both call this rather than each
    building their own idea of "the scope's text" (amendment 2, ruled
    2026-08-09) -- two separate implementations were the root cause of
    three findings in review: an estimate that omitted the header/label
    tokens the prompt actually contains, an estimate that raised where this
    degrades cleanly to a placeholder, and an honest-fallback label with no
    test of its own. Delegating keeps agreement structural: there is
    exactly one function that decides what the scope's text is, so the two
    callers cannot drift apart by one of them changing without the other."""
    parts: list[str] = []
    for path in scope.full:
        text = _read(path)
        label = _rel(path, root)
        if text is not None and path == scope.anchor and scope.anchor_window:
            windowed = _window_text(text, scope.anchor_window, scope.anchor_line)
            if windowed != text:
                # Only claim a window when it actually removed something
                # (invariant 3): a span wider than the file is a no-op, and
                # labelling a no-op as "windowed" is as false as the label
                # this replaced, which claimed the failure was included
                # while centring on the file's midpoint instead of it.
                text = windowed
                label += (
                    f" (windowed around line {scope.anchor_line})"
                    if scope.anchor_line is not None
                    else " (windowed around the file's midpoint; failing "
                    "line unknown)"
                )
        parts.append(f"--- {label} ---")
        parts.append(text if text is not None else _UNREADABLE)
    for path in scope.signatures:
        text = _read(path)
        parts.append(f"--- {_rel(path, root)} (signatures only) ---")
        parts.append(signatures_of(text) if text is not None else _UNREADABLE)
    return "\n".join(parts)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _window_text(text: str, span: tuple[int, int], anchor_line: int | None) -> str:
    """The single implementation of anchor windowing, called only from
    `_scope_section` above -- which is itself the single implementation
    `budget._cost` delegates to (via `budget._section`), so windowing
    reaches the cost estimate through the same one path the prompt does,
    never a second one that could diverge (amendment 2, ruled 2026-08-09).

    Centres on `anchor_line` (1-indexed, carried on the `Scope` -- never on
    a `Diagnostic` passed in separately, even by a caller that has one:
    `budget._cost` has no `Diagnostic` at all, so if this centred on one,
    the two sites could disagree about where the window sits whenever they
    were called with a stale or different diagnostic than the one the
    `Scope` was built from. Deriving the centre from the `Scope` alone is
    what keeps agreement structural rather than a matter of callers
    behaving.) Falls back to the file's midpoint only when `anchor_line` is
    None -- a diagnostic with no line -- which is the pre-amendment
    behaviour and the only case left where it is honest."""
    lines = text.split("\n")
    if anchor_line is not None:
        center = max(0, min(anchor_line - 1, len(lines) - 1))
    else:
        center = len(lines) // 2
    return "\n".join(lines[max(0, center + span[0]) : center + span[1]])
