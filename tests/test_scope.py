# tests/test_scope.py
from __future__ import annotations

from pathlib import Path

import pytest

from robigo.adapters.base import Diagnostic
from robigo.adapters.python_ import PythonAdapter
from robigo.context.scope import Scope, ScopeError, resolve, signatures_of


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "grid.py").write_text("CELL = 5\n")
    (tmp_path / "src" / "fog.py").write_text("import grid\n\ndef radius(t):\n    return t\n")
    (tmp_path / "tests" / "test_fog.py").write_text("import fog\n\ndef test_r():\n    assert 0\n")
    return tmp_path


def _diag(file: str) -> Diagnostic:
    return Diagnostic(False, file, 4, "AssertionError", "raw")


def test_anchor_is_the_diagnostic_file_and_hop_one_is_full_text(repo: Path):
    scope = resolve(_diag("tests/test_fog.py"), PythonAdapter(), repo, hops=1)
    assert scope.anchor == repo / "tests" / "test_fog.py"
    assert scope.full == (repo / "tests" / "test_fog.py", repo / "src" / "fog.py")
    assert scope.signatures == ()


def test_hop_two_arrives_as_signatures_only(repo: Path):
    scope = resolve(_diag("tests/test_fog.py"), PythonAdapter(), repo, hops=2)
    assert scope.signatures == (repo / "src" / "grid.py",)
    # grid.py must NOT also be in full -- paying for it twice is the bug.
    assert repo / "src" / "grid.py" not in scope.full


def test_repo_size_cannot_affect_scope(repo: Path):
    for i in range(200):
        (repo / "src" / f"noise{i}.py").write_text("x = 1\n")
    scope = resolve(_diag("tests/test_fog.py"), PythonAdapter(), repo, hops=2)
    assert len(scope.full) + len(scope.signatures) == 3


def test_a_diagnostic_with_no_file_is_refused(repo: Path):
    with pytest.raises(ScopeError) as e:
        resolve(Diagnostic(False, None, None, "tests failed", "raw"), PythonAdapter(), repo)
    assert "anchor" in str(e.value)


def test_signatures_of_keeps_definitions_and_drops_bodies():
    out = signatures_of("import os\n\ndef f(a, b):\n    return a\n\nclass K:\n    pass\n")
    assert "def f(a, b):" in out
    assert "class K:" in out
    assert "return a" not in out
