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
    parts = [SYSTEM, _CODEC_HELP[codec], ""]
    for path in scope.full:
        parts.append(f"--- {_rel(path, root)} ---")
        parts.append(path.read_text(encoding="utf-8"))
    for path in scope.signatures:
        parts.append(f"--- {_rel(path, root)} (signatures only) ---")
        parts.append(signatures_of(path.read_text(encoding="utf-8")))
    where = f"{diag.file}:{diag.line}" if diag.line else str(diag.file)
    parts += ["--- failing test ---", f"{where}  {diag.message}", ""]
    for turn in history:
        parts.append(f"you: {turn.action}\nresult: {turn.result}")
    parts.append("Your action:")
    return "\n".join(parts)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
