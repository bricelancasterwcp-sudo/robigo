# tests/test_paths.py
from __future__ import annotations

from pathlib import Path

import pytest

from robigo.paths import OutsideRepo, contain


def test_a_path_inside_the_repo_comes_back_resolved(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "fog.py").write_text("x = 1\n")
    assert contain(tmp_path, "src/fog.py") == (tmp_path / "src" / "fog.py").resolve()


def test_an_embedded_nul_is_a_refusal_not_a_valueerror(tmp_path: Path):
    # The whole reason this helper exists: resolve() raises ValueError here,
    # and four of the five original call sites let it escape the run.
    with pytest.raises(OutsideRepo) as e:
        contain(tmp_path, "a\x00b.py")
    assert "not a usable path" in str(e.value)


def test_a_dot_dot_escape_is_refused(tmp_path: Path):
    (tmp_path / "repo").mkdir()
    (tmp_path / "outside.py").write_text("x = 1\n")
    with pytest.raises(OutsideRepo):
        contain(tmp_path / "repo", "../outside.py")


def test_an_absolute_path_outside_the_repo_is_refused(tmp_path: Path):
    with pytest.raises(OutsideRepo):
        contain(tmp_path, "/etc/passwd")


def test_a_symlink_pointing_out_of_the_repo_is_refused(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "secret.py").write_text("token = 1\n")
    (repo / "link.py").symlink_to(tmp_path / "secret.py")
    with pytest.raises(OutsideRepo):
        contain(repo, "link.py")
