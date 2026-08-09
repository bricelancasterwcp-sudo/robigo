# src/robigo/adapters/python_.py
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

from robigo.adapters.base import DIAGNOSTIC_CHAR_CAP, Diagnostic

_FAIL_LINE = re.compile(r"^(?P<file>/[^:]+\.py):(?P<line>\d+): (?P<msg>.+)$")
_TIMEOUT_S = 300


class PythonAdapter:
    name = "python"
    test_command = "pytest --tb=line -q --no-header"

    def run(self, root: Path, filt: str | None) -> Diagnostic:
        argv = ["python", "-m", "pytest", "--tb=line", "-q", "--no-header", "-p",
                "no:cacheprovider"]
        if filt:
            argv += ["-k", filt]
        proc = subprocess.run(
            argv, cwd=root, capture_output=True, text=True, timeout=_TIMEOUT_S
        )
        raw = (proc.stdout + proc.stderr)[-DIAGNOSTIC_CHAR_CAP:]
        if proc.returncode == 0:
            return Diagnostic(True, None, None, "all tests passed", raw)
        return self._first_failure(raw, root)

    def _first_failure(self, raw: str, root: Path) -> Diagnostic:
        for line in raw.split("\n"):
            match = _FAIL_LINE.match(line.strip())
            if not match:
                continue
            path = Path(match.group("file"))
            try:
                rel = str(path.relative_to(root))
            except ValueError:
                rel = str(path)
            return Diagnostic(
                False, rel, int(match.group("line")), match.group("msg"), raw
            )
        return Diagnostic(False, None, None, "tests failed", raw)

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
