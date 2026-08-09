# src/robigo/adapters/python_.py
from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path

from robigo.adapters.base import AdapterError, DIAGNOSTIC_CHAR_CAP, Diagnostic
from robigo.paths import OutsideRepo, contain

_FAIL_LINE = re.compile(r"^(?P<file>[^\s:][^:]*\.py):(?P<line>\d+):\s*(?P<msg>.*)$")
_ERROR_LINE = re.compile(r"^E\s+(?P<msg>\S.*)$")
_EXCLUDED = ("site-packages", "dist-packages", "/.venv/", "/venv/")
_FRAME_TAIL = re.compile(r"^in\s")
_TIMEOUT_S = 300


class PythonAdapter:
    name = "python"
    test_command = "pytest --tb=line -q --no-header"

    def __init__(self, python: str | None = None) -> None:
        self._python = python

    def _interpreter(self, root: Path) -> str:
        """The project's interpreter, not robigo's. Checked in order, so a
        repo with its own venv needs nothing activated."""
        if self._python:
            return self._python
        for candidate in (root / ".venv/bin/python", root / "venv/bin/python"):
            if candidate.is_file():
                return str(candidate)
        return "python"

    def _env(self) -> dict[str, str]:
        # No bytecode: a .pyc written during our own test run would be
        # committed by snapshot and then rewritten by the next run, and
        # `git checkout <branch>` would abort on it — breaking the undo
        # recipe the CLI prints. Do not "optimise" this away by restoring
        # the bytecode cache.
        return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    def _preflight(self, python: str) -> None:
        """Refuse loudly rather than fail per-run with a confusing
        ModuleNotFoundError from inside a subprocess."""
        try:
            proc = subprocess.run(
                [python, "-m", "pytest", "--version"],
                capture_output=True, text=True, timeout=60,
                env=self._env(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AdapterError(f"cannot execute {python!r}: {exc}") from exc
        if proc.returncode != 0:
            raise AdapterError(
                f"{python} cannot import pytest. Activate the project's "
                f"virtualenv, or pass --python <path> to name the "
                f"interpreter that has the project's test dependencies."
            )

    def run(self, root: Path, filt: str | None) -> Diagnostic:
        python = self._interpreter(root)
        self._preflight(python)
        argv = [python, "-m", "pytest", "--tb=line", "-q", "--no-header", "-p",
                "no:cacheprovider"]
        if filt:
            argv += ["-k", filt]
        try:
            proc = subprocess.run(
                argv, cwd=root, capture_output=True, text=True,
                timeout=_TIMEOUT_S, env=self._env(),
            )
        except subprocess.TimeoutExpired:
            return Diagnostic(
                False, None, None,
                f"tests timed out after {_TIMEOUT_S}s — the last patch may "
                f"not terminate", "",
            )
        raw = (proc.stdout + proc.stderr)[-DIAGNOSTIC_CHAR_CAP:]
        if proc.returncode == 0:
            return Diagnostic(True, None, None, "all tests passed", raw)
        return self._first_failure(raw, root)

    def _first_failure(self, raw: str, root: Path) -> Diagnostic:
        root = root.resolve()
        lines = raw.split("\n")
        anchor = self._anchor(lines, root)
        if anchor is None:
            summary = self._error_summary(lines, 0)
            return Diagnostic(False, None, None, summary or "tests failed", raw)
        index, rel, number, tail = anchor
        if tail and not _FRAME_TAIL.match(tail):
            message = tail
        else:
            message = self._error_summary(lines, index) or tail or "tests failed"
        return Diagnostic(False, rel, number, message, raw)

    def _anchor(self, lines: list[str], root: Path) -> tuple[int, str, int, str] | None:
        for index, line in enumerate(lines):
            match = _FAIL_LINE.match(line.strip())
            if not match:
                continue
            number = int(match.group("line"))
            rel = self._in_repo(match.group("file"), root, number)
            if rel is not None:
                return index, rel, number, match.group("msg")
        return None

    def _in_repo(self, candidate: str, root: Path, number: int) -> str | None:
        """Repo-relative path for a location that could plausibly be a real
        failure site, or None.

        An anchor the model cannot edit is worse than none, and a location
        the model merely PRINTED is not a failure site — so the file must
        exist and the line must fall inside it.

        Residual, accepted: a model that prints "src/real.py:12: ..." —
        naming a real file at a plausible line — can still misdirect the
        anchor. Bounding captured output by layout was tried twice and
        failed twice; the cost here is one wasted turn, and Task 5 refuses
        an anchor that is not a real file.
        """
        if any(fragment in candidate for fragment in _EXCLUDED):
            return None
        try:
            resolved = contain(root, candidate)
        except OutsideRepo:
            return None
        try:
            if not resolved.is_file():
                return None
            body = resolved.read_text(encoding="utf-8", errors="replace")
            if number < 1 or number > len(body.splitlines()):
                return None
        except OSError:
            return None
        return str(resolved.relative_to(root.resolve()))

    def _error_summary(self, lines: list[str], start: int) -> str | None:
        """pytest's `E   <Type>: <message>`, searched FORWARD from the
        anchor, so a message can never be attached to a different
        failure's location."""
        for line in lines[start:]:
            match = _ERROR_LINE.match(line.rstrip())
            if match:
                return match.group("msg")
        return None

    def imports(self, path: Path, root: Path) -> list[Path]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return []
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        found: list[Path] = []
        for name in names:
            for candidate in self._candidates(name, root):
                if candidate.is_file() and candidate not in found:
                    found.append(candidate)
                    break
        return found

    def _candidates(self, name: str, root: Path) -> list[Path]:
        parts = name.split(".")
        return [
            root / "src" / Path(*parts).with_suffix(".py"),
            root / Path(*parts).with_suffix(".py"),
            root / "src" / Path(*parts) / "__init__.py",
            root / Path(*parts) / "__init__.py",
        ]

    def syntax_ok(self, text: str) -> bool:
        try:
            ast.parse(text)
        except SyntaxError:
            return False
        return True
