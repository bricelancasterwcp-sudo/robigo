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


def test_a_broken_import_anchors_in_the_repo_not_in_importlib(repo: Path):
    (repo / "tests" / "test_bad.py").write_text("import nonexistent_xyz\n")
    diag = PythonAdapter(python=sys.executable).run(repo, None)
    assert diag.passed is False
    assert diag.file == "tests/test_bad.py"
    assert "nonexistent_xyz" in diag.message


def test_a_syntax_error_anchors_in_the_repo_not_in_pytest(repo: Path):
    (repo / "src" / "fog.py").write_text("def radius(t:\n")
    diag = PythonAdapter(python=sys.executable).run(repo, None)
    assert diag.file is not None
    assert "site-packages" not in diag.file
    assert diag.file.startswith(("src/", "tests/"))


def test_a_hanging_suite_is_a_model_result_not_a_crash(repo: Path, monkeypatch):
    monkeypatch.setattr("robigo.adapters.python_._TIMEOUT_S", 3)
    (repo / "tests" / "test_fog.py").write_text(
        "def test_spin():\n    while True:\n        pass\n"
    )
    diag = PythonAdapter(python=sys.executable).run(repo, None)
    assert diag.passed is False
    assert "timed out" in diag.message


def test_model_authored_stdout_cannot_hijack_the_anchor(repo: Path):
    (repo / "tests" / "test_fog.py").write_text(
        "def test_x():\n"
        "    print('nonexistent_config.py:999: totally unrelated fake path')\n"
        "    assert 1 == 2, 'the real failure'\n"
    )
    diag = PythonAdapter(python=sys.executable).run(repo, None)
    assert diag.file == "tests/test_fog.py"
    assert "nonexistent_config" not in (diag.file or "")


def test_a_path_that_does_not_exist_is_not_an_anchor(tmp_path: Path):
    assert PythonAdapter()._in_repo("nonexistent_config.py", tmp_path, 1) is None


def test_a_real_file_at_an_impossible_line_is_not_an_anchor(tmp_path: Path):
    (tmp_path / "short.py").write_text("x = 1\n")
    adapter = PythonAdapter()
    assert adapter._in_repo("short.py", tmp_path, 1) == "short.py"
    assert adapter._in_repo("short.py", tmp_path, 999) is None


def test_the_summary_is_paired_with_the_anchor_not_the_first_failure():
    lines = [
        "E   AssertionError: an earlier unrelated failure",
        "tests/test_b.py:2: AssertionError",
        "E   AssertionError: the failure that belongs here",
    ]
    assert PythonAdapter()._error_summary(lines, 1) == (
        "AssertionError: the failure that belongs here"
    )
