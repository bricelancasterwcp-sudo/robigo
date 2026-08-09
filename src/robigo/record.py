# src/robigo/record.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robigo.loop import RunResult

_IGNORE_ALL = "*\n"


def next_run_id(root: Path, slug: str) -> str:
    """The first unused id, not a count — counting collides as soon as any
    earlier run directory is deleted, and would then overwrite it."""
    runs = root / ".robigo" / "runs"
    number = 1
    while (runs / f"{slug}-{number}").exists():
        number += 1
    return f"{slug}-{number}"


class RunRecorder:
    """Prompts, raw replies, and adapter output, verbatim.

    Verbatim is the point: trailing whitespace and line endings are exactly
    what break a SEARCH block, so normalising here would erase the evidence.
    These records are also corpus candidates.

    Nothing is created until something is written, so a run refused before it
    starts leaves no directory behind — the same law `safety.py` applies to
    branches. Once `error` is set, no further file is written, including
    `meta.json`.
    """

    def __init__(self, root: Path, run_id: str) -> None:
        self.dir = root / ".robigo" / "runs" / run_id
        self.error: str | None = None
        self._turns = 0
        self._ready = False

    def turn(self, prompt: str, reply: str, adapter_raw: str) -> None:
        self._turns += 1
        stem = f"turn-{self._turns:02d}"
        self._write(f"{stem}-prompt.txt", prompt)
        self._write(f"{stem}-reply.txt", reply)
        self._write(f"{stem}-adapter.txt", adapter_raw)

    def finish(
        self, result: RunResult, model: str, window: int, codec: str
    ) -> None:
        self._write("meta.json", json.dumps({
            "outcome": result.outcome, "turns": result.turns,
            "exit_code": result.exit_code, "branch": result.branch,
            "detail": result.detail, "model": model, "window": window,
            "codec": codec,
        }, indent=2, sort_keys=True))

    def _ensure(self) -> bool:
        """Create the tree on first write, and make `.robigo/` ignore itself
        so `snapshot`'s `git add -A` can never commit robigo's transcripts
        into the user's repository."""
        if self._ready or self.error is not None:
            return self.error is None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            marker = self.dir.parent.parent / ".gitignore"
            if not marker.exists():
                marker.write_text(_IGNORE_ALL, encoding="utf-8")
        except OSError as exc:
            self.error = f"cannot create {self.dir}: {exc}"
            return False
        self._ready = True
        return True

    def _write(self, name: str, text: str) -> None:
        """A write failure is remembered, not raised: losing the transcript
        is bad, failing a completed repair because the transcript could not
        be saved is worse. `UnicodeError` is caught alongside `OSError`
        because a lone surrogate in a model reply raises
        `UnicodeEncodeError` — a `ValueError`, not an `OSError` — which
        would otherwise escape `run()` *after* the repair had landed."""
        if not self._ensure():
            return
        try:
            (self.dir / name).write_text(text, encoding="utf-8", newline="")
        except (OSError, UnicodeError) as exc:
            self.error = f"cannot write {name}: {exc}"


def slug(task: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:24] or "run"


def new_recorder(root: Path, task: str) -> RunRecorder:
    """The single place a run id is named, so `cli` and `loop` cannot drift."""
    return RunRecorder(root, next_run_id(root, slug(task)))
