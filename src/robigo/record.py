# src/robigo/record.py
from __future__ import annotations

import json
from pathlib import Path


def next_run_id(root: Path, slug: str) -> str:
    """The first unused id, not a count — counting collides as soon as any
    earlier run directory is deleted, and would then overwrite it."""
    runs = root / ".robigo" / "runs"
    number = 1
    while (runs / f"{slug}-{number}").exists():
        number += 1
    return f"{slug}-{number}"


class RunRecorder:
    """Prompts, raw replies, and adapter output, verbatim. Verbatim is the
    point: trailing whitespace and line endings are exactly what breaks a
    SEARCH block, so normalising here would erase the evidence. These
    records are also corpus candidates (spec 5.1).

    A record is a diagnostic, not the product: a read-only `.robigo/`, a
    full disk, or a permission error must not turn a passing repair into a
    crash. The first write failure is remembered on `.error` (not raised)
    and disables every write that follows, so the reason is not lost either.
    """

    def __init__(self, root: Path, run_id: str) -> None:
        self.dir = root / ".robigo" / "runs" / run_id
        self.error: str | None = None
        self._turns = 0
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.error = f"cannot create {self.dir}: {exc}"

    def turn(self, prompt: str, reply: str, adapter_raw: str) -> None:
        self._turns += 1
        stem = f"turn-{self._turns:02d}"
        self._write(f"{stem}-prompt.txt", prompt)
        self._write(f"{stem}-reply.txt", reply)
        self._write(f"{stem}-adapter.txt", adapter_raw)

    def finish(
        self, result, model: str, window: int, codec: str
    ) -> None:
        self._write("meta.json", json.dumps({
            "outcome": result.outcome, "turns": result.turns,
            "exit_code": result.exit_code, "branch": result.branch,
            "detail": result.detail, "model": model, "window": window,
            "codec": codec,
        }, indent=2, sort_keys=True))

    def _write(self, name: str, text: str) -> None:
        """Records are diagnostics. A write failure is remembered, not
        raised: losing the transcript is bad, failing a completed repair
        because the transcript could not be saved is worse."""
        if self.error is not None:
            return
        try:
            (self.dir / name).write_text(text, encoding="utf-8", newline="")
        except OSError as exc:
            self.error = f"cannot write {name}: {exc}"
