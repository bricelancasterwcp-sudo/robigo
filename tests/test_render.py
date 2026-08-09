# tests/test_render.py
from __future__ import annotations

from pathlib import Path

from robigo.adapters.base import Diagnostic
from robigo.context.render import SYSTEM, Turn, render
from robigo.context.scope import Scope


def _scope(root: Path) -> Scope:
    (root / "a.py").write_text("def f():\n    return 1\n")
    (root / "b.py").write_text("def g(x):\n    return x\n")
    return Scope(root / "a.py", (root / "a.py",), (root / "b.py",))


def test_prompt_contains_the_verbs_the_scope_and_the_diagnostic(tmp_path: Path):
    diag = Diagnostic(False, "a.py", 2, "AssertionError: 1 != 2", "raw tail")
    out = render(_scope(tmp_path), diag, (), "search_replace", tmp_path)
    for verb in ("read", "find", "patch", "run", "done"):
        assert verb in out
    assert "def f():" in out                 # full text for hop 0/1
    assert "def g(x):" in out                # signature for hop 2
    assert "return x" not in out             # but not its body
    assert "AssertionError: 1 != 2" in out
    assert "a.py:2" in out


def test_only_the_selected_codec_is_described(tmp_path: Path):
    diag = Diagnostic(False, "a.py", 2, "boom", "raw")
    out = render(_scope(tmp_path), diag, (), "whole_file", tmp_path)
    # Describing codecs the profile did not select wastes window and
    # invites the model to use the wrong one.
    assert "SEARCH" not in out
    assert "complete new file" in out


def test_history_is_included_oldest_first(tmp_path: Path):
    diag = Diagnostic(False, "a.py", 2, "boom", "raw")
    history = (Turn("patch a.py", "SEARCH not found"), Turn("run", "still failing"))
    out = render(_scope(tmp_path), diag, history, "search_replace", tmp_path)
    assert out.index("SEARCH not found") < out.index("still failing")


def test_system_prompt_stays_small():
    # The window is the scarce resource; the fixed cost must stay bounded.
    assert len(SYSTEM) < 1800


def test_an_unreadable_file_is_reported_in_place_not_raised(tmp_path: Path):
    scope = _scope(tmp_path)
    (tmp_path / "a.py").write_bytes(b"\xff\xfe not utf-8\n")
    diag = Diagnostic(False, "a.py", 2, "boom", "raw")
    out = render(scope, diag, (), "search_replace", tmp_path)
    assert "unreadable" in out


def test_a_missing_signature_file_is_reported_in_place(tmp_path: Path):
    scope = _scope(tmp_path)
    (tmp_path / "b.py").unlink()
    diag = Diagnostic(False, "a.py", 2, "boom", "raw")
    out = render(scope, diag, (), "search_replace", tmp_path)
    assert "unreadable" in out


def test_a_diagnostic_with_no_file_never_renders_the_word_None(tmp_path: Path):
    # Reachable: the adapter returns file=None for a timed-out suite and
    # for a failure it could not anchor in the repo.
    diag = Diagnostic(False, None, None, "tests timed out after 300s", "")
    out = render(_scope(tmp_path), diag, (), "search_replace", tmp_path)
    assert "location unknown" in out
    assert "None:" not in out
