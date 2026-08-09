# tests/test_adapter_python.py
from __future__ import annotations

import sys
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
    diag = PythonAdapter(python=sys.executable).run(repo, None)
    assert diag.passed is False
    assert diag.file == "tests/test_fog.py"
    assert diag.line == 6
    assert "assert" in diag.message.lower()


def test_run_reports_a_pass(repo: Path):
    (repo / "src" / "fog.py").write_text("def radius(t):\n    return t * 2\n")
    diag = PythonAdapter(python=sys.executable).run(repo, None)
    assert diag.passed is True
    assert diag.file is None


def test_raw_output_is_capped(repo: Path):
    diag = PythonAdapter(python=sys.executable).run(repo, None)
    assert len(diag.raw) <= DIAGNOSTIC_CHAR_CAP


def test_imports_resolves_local_modules_only(repo: Path):
    found = PythonAdapter(python=sys.executable).imports(repo / "tests" / "test_fog.py", repo)
    # `fog` resolves inside the repo; `sys` is stdlib and must not appear.
    assert found == [repo / "src" / "fog.py"]


def test_syntax_ok_distinguishes_valid_from_broken():
    adapter = PythonAdapter(python=sys.executable)
    assert adapter.syntax_ok("x = 1\n") is True
    assert adapter.syntax_ok("def f(\n") is False


def test_a_project_venv_is_preferred_over_path(tmp_path: Path):
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    assert PythonAdapter()._interpreter(tmp_path) == str(venv_python)


def test_an_interpreter_without_pytest_is_refused_loudly(tmp_path: Path):
    from robigo.adapters.base import AdapterError

    fake = tmp_path / "fake-python"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    with pytest.raises(AdapterError) as e:
        PythonAdapter(python=str(fake)).run(tmp_path, None)
    assert "--python" in str(e.value)
