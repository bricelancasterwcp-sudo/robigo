# tests/test_adapter_python.py
from __future__ import annotations

from pathlib import Path

import pytest

from robigo.adapters.python_ import DIAGNOSTIC_CHAR_CAP, PythonAdapter


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "fog.py").write_text("def radius(t):\n    return t\n")
    (tmp_path / "src" / "grid.py").write_text("CELL = 5\n")
    (tmp_path / "tests" / "test_fog.py").write_text(
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        "from fog import radius\n\n"
        "def test_radius():\n"
        "    assert radius(2) == 4\n"
    )
    return tmp_path


def test_run_reports_a_failure_with_file_and_line(repo: Path):
    diag = PythonAdapter().run(repo, None)
    assert diag.passed is False
    assert diag.file == "tests/test_fog.py"
    assert diag.line == 6
    assert "assert" in diag.message.lower()


def test_run_reports_a_pass(repo: Path):
    (repo / "src" / "fog.py").write_text("def radius(t):\n    return t * 2\n")
    diag = PythonAdapter().run(repo, None)
    assert diag.passed is True
    assert diag.file is None


def test_raw_output_is_capped(repo: Path):
    diag = PythonAdapter().run(repo, None)
    assert len(diag.raw) <= DIAGNOSTIC_CHAR_CAP


def test_imports_resolves_local_modules_only(repo: Path):
    found = PythonAdapter().imports(repo / "tests" / "test_fog.py", repo)
    # `fog` resolves inside the repo; `sys` is stdlib and must not appear.
    assert found == [repo / "src" / "fog.py"]


def test_syntax_ok_distinguishes_valid_from_broken():
    adapter = PythonAdapter()
    assert adapter.syntax_ok("x = 1\n") is True
    assert adapter.syntax_ok("def f(\n") is False
