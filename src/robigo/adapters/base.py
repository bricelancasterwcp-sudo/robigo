# src/robigo/adapters/base.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DIAGNOSTIC_CHAR_CAP = 2400
"""Roughly 600 tokens. A single bad turn must not be able to eat the
window (spec section 3)."""


@dataclass(frozen=True)
class Diagnostic:
    passed: bool
    file: str | None
    line: int | None
    message: str
    raw: str


class Adapter(Protocol):
    name: str
    test_command: str

    def run(self, root: Path, filt: str | None) -> Diagnostic: ...
    def imports(self, path: Path, root: Path) -> list[Path]: ...
    def syntax_ok(self, text: str) -> bool: ...
