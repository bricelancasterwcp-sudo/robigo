# src/robigo/adapters/python_.py
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

from robigo.adapters.base import AdapterError, DIAGNOSTIC_CHAR_CAP, Diagnostic

_FAIL_LINE = re.compile(r"^(?P<file>[^\s:][^:]*\.py):(?P<line>\d+):\s*(?P<msg>.*)$")
_ERROR_LINE = re.compile(r"^E\s+(?P<msg>\S.*)$")
_EXCLUDED = ("site-packages", "dist-packages", "/.venv/", "/venv/")
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

    def _preflight(self, python: str) -> None:
        """Refuse loudly rather than fail per-run with a confusing
        ModuleNotFoundError from inside a subprocess."""
        try:
            proc = subprocess.run(
                [python, "-m", "pytest", "--version"],
                capture_output=True, text=True, timeout=60,
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
                timeout=_TIMEOUT_S,
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
        summary = self._error_summary(raw)
        for line in raw.split("\n"):
            match = _FAIL_LINE.match(line.strip())
            if not match:
                continue
            rel = self._in_repo(match.group("file"), root)
            if rel is None:
                continue
            return Diagnostic(
                False, rel, int(match.group("line")),
                summary or match.group("msg"), raw,
            )
        return Diagnostic(False, None, None, summary or "tests failed", raw)

    def _in_repo(self, candidate: str, root: Path) -> str | None:
        """Repo-relative path, or None when the location is outside the
        project. An anchor the model cannot edit is worse than none."""
        if any(fragment in candidate for fragment in _EXCLUDED):
            return None
        path = Path(candidate)
        resolved = path if path.is_absolute() else root / path
        try:
            return str(resolved.resolve().relative_to(root))
        except (ValueError, OSError):
            return None

    def _error_summary(self, raw: str) -> str | None:
        """pytest's own `E   <Type>: <message>` line, which names the real
        failure when a traceback replaces --tb=line's one-liner."""
        for line in raw.split("\n"):
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
