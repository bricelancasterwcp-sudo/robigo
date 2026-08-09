# tests/test_apply_patch.py
from __future__ import annotations

from pathlib import Path

import pytest

from robigo.action.codec import PatchError
from robigo.action.verbs import Action
from robigo.adapters.python_ import PythonAdapter
from robigo.apply.patch import apply_patch, write_atomic
from robigo.context.scope import Scope

ORIGINAL = "def f():\n    return 1\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text(ORIGINAL)
    (tmp_path / "test_a.py").write_text("def test_a():\n    assert 0\n")
    return tmp_path


def _scope(repo: Path) -> Scope:
    return Scope(repo / "test_a.py", (repo / "test_a.py", repo / "a.py"), ())


def _patch(payload: str) -> Action:
    return Action("patch", "a.py", payload, "python")


def test_a_valid_patch_is_written(repo: Path):
    payload = "<<<<<<< SEARCH\n    return 1\n=======\n    return 2\n>>>>>>> REPLACE\n"
    apply_patch(_patch(payload), repo, _scope(repo), PythonAdapter(), "search_replace")
    assert (repo / "a.py").read_text() == "def f():\n    return 2\n"


def test_a_patch_producing_broken_syntax_is_rejected_and_nothing_is_written(repo: Path):
    payload = "<<<<<<< SEARCH\n    return 1\n=======\n    return (\n>>>>>>> REPLACE\n"
    with pytest.raises(PatchError) as e:
        apply_patch(_patch(payload), repo, _scope(repo), PythonAdapter(), "search_replace")
    assert "syntax" in str(e.value).lower()
    assert (repo / "a.py").read_text() == ORIGINAL


def test_write_atomic_leaves_no_partial_file_and_no_temp_files(repo: Path):
    write_atomic(repo / "a.py", "x = 1\n")
    assert (repo / "a.py").read_text() == "x = 1\n"
    assert list(repo.glob("*.tmp*")) == []


def test_a_patch_to_the_anchor_test_is_refused_before_any_write(repo: Path):
    from robigo.apply.safety import RefusedError

    before = (repo / "test_a.py").read_text()
    payload = "<<<<<<< SEARCH\n    assert 0\n=======\n    assert 1\n>>>>>>> REPLACE\n"
    with pytest.raises(RefusedError):
        apply_patch(Action("patch", "test_a.py", payload, None), repo,
                    _scope(repo), PythonAdapter(), "search_replace")
    assert (repo / "test_a.py").read_text() == before


def test_the_original_permission_bits_survive_a_patch(repo: Path):
    (repo / "a.py").chmod(0o755)
    payload = "<<<<<<< SEARCH\n    return 1\n=======\n    return 2\n>>>>>>> REPLACE\n"
    apply_patch(_patch(payload), repo, _scope(repo), PythonAdapter(), "search_replace")
    assert (repo / "a.py").stat().st_mode & 0o777 == 0o755


def test_an_unreadable_target_is_a_patch_error_not_a_crash(repo: Path):
    (repo / "a.py").unlink()
    payload = "<<<<<<< SEARCH\n    return 1\n=======\n    return 2\n>>>>>>> REPLACE\n"
    with pytest.raises(PatchError) as e:
        apply_patch(_patch(payload), repo, _scope(repo), PythonAdapter(), "search_replace")
    assert "could not be read" in str(e.value)
