# tests/test_budget.py
from __future__ import annotations

from pathlib import Path

import pytest

from robigo.adapters.base import Diagnostic
from robigo.context.budget import (
    Budget,
    BudgetExhausted,
    estimate_tokens,
    fit,
    reserve_for,
)
from robigo.context.render import render
from robigo.context.scope import Scope

# Fixture per amendment (ruled 2026-08-09, "the ladder's rung 4 is dead, and
# its label lies"): the anchor must be long enough that a +/-60-line window
# actually removes most of it (81 lines was the original bug -- a window
# that wide is the whole file), and hop2 needs real signature content so
# rung 2's band is wide enough to test rather than 3 tokens wide.
_FAIL_LINE = 350  # 1-indexed; the line degrade(4) must keep visible


def _anchor_text() -> str:
    body = ["    assert 0"] * 399
    body[_FAIL_LINE - 2] = "    assert FAIL_HERE == 0"  # line 1 is "def test_x():"
    return "def test_x():\n" + "\n".join(body) + "\n"


def _hop2_text() -> str:
    lines: list[str] = []
    for i in range(40):
        lines.append(f"def g{i}():")
        lines.append(f"    y = {i}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def scope(tmp_path: Path) -> Scope:
    (tmp_path / "anchor.py").write_text(_anchor_text())
    (tmp_path / "hop1.py").write_text("def f():\n" + "    x = 1\n" * 80)
    (tmp_path / "hop2.py").write_text(_hop2_text())
    return Scope(
        tmp_path / "anchor.py",
        (tmp_path / "anchor.py", tmp_path / "hop1.py"),
        (tmp_path / "hop2.py",),
        anchor_line=_FAIL_LINE,
    )


@pytest.fixture
def diag() -> Diagnostic:
    return Diagnostic(False, "anchor.py", _FAIL_LINE, "AssertionError", "raw")


# Measured against the fixture above, reserve_out=128 system=60 diagnostic=60
# history=0 (248 tokens fixed cost), AFTER invariant 1 (centre on anchor_line
# rather than the file's midpoint):
#
#   rung | cost | saves | window band that lands fit() here | width
#   -----|------|-------|-------------------------------------|------
#   1    | 1958 |   --  | >= 2206                             |  --
#   2    | 1827 |  131  | 2075 - 2205                          | 131
#   3    | 1584 |  243  | 1832 - 2074                          | 243
#   4    |  445 | 1139  | 693  - 1831                          | 1139
#
# Each rung test below picks its window from the MIDDLE of its band, not an
# edge (amendment: "a budget picked at a band edge is one estimator tweak
# away from testing the neighbouring rung").


def test_scope_budget_is_the_window_minus_the_fixed_costs():
    budget = Budget(window=4096, reserve_out=512, system=350, diagnostic=600,
                    history=200)
    assert budget.scope_budget == 4096 - 512 - 350 - 600 - 200


def test_reserve_for_whole_file_covers_the_file_plus_margin():
    # whole_file must reserve the entire file: this is why weak families
    # are least able to afford the codec easiest for them (spec 3.3).
    assert reserve_for("whole_file", file_tokens=1000) == 1150
    assert reserve_for("search_replace", file_tokens=1000) == 512
    assert reserve_for("udiff", file_tokens=1000) == 384


def test_estimate_tokens_is_conservative_for_code():
    # Deliberately crude and deliberately NOT authoritative: the server's
    # tokenizer always outranks it (spec 3.3).
    assert estimate_tokens("x" * 36) == 11


def test_a_generous_window_keeps_the_full_scope(scope: Scope, tmp_path: Path):
    fitted, step = fit(scope, Budget(32768, 512, 350, 600, 200), tmp_path)
    assert step == 1
    assert fitted.full == scope.full and fitted.signatures == scope.signatures


def test_dropping_hop_two_signatures_is_what_step_two_saves(scope: Scope, tmp_path: Path):
    # Rung 2 in isolation: hop-2's signatures (40 `def` lines) are gone but
    # hop-1 is still full text. This is the rung the original test never
    # pinned -- its own band (2075-2205) is what makes it distinguishable
    # from rung 3 at all.
    fitted, step = fit(scope, Budget(2140, 128, 60, 60, 0), tmp_path)
    assert step == 2
    assert fitted.full == scope.full
    assert fitted.signatures == ()


def test_a_tight_window_reduces_hop_one_to_signatures(scope: Scope, tmp_path: Path):
    # Corrected per amendment: at step 3, hop-1 has become a signature, not
    # vanished. `signatures == ()` was rung 2's shape, not rung 3's.
    fitted, step = fit(scope, Budget(1953, 128, 60, 60, 0), tmp_path)
    assert step == 3
    assert fitted.full == (scope.anchor,)
    assert fitted.signatures == (scope.full[1],)


def test_windowing_the_anchor_is_the_last_rung_before_refusal(scope: Scope, tmp_path: Path):
    # Rung 4 in isolation, from the middle of its own band (693-1831) --
    # pre-amendment this band was empty (cost(4) == cost(3)) so no window
    # could ever land fit() here at all.
    fitted, step = fit(scope, Budget(1263, 128, 60, 60, 0), tmp_path)
    assert step == 4
    assert fitted.anchor_window is not None


def test_an_impossible_window_refuses_and_prints_the_arithmetic(scope: Scope, tmp_path: Path):
    with pytest.raises(BudgetExhausted) as e:
        fit(scope, Budget(200, 128, 60, 60, 0), tmp_path)
    message = str(e.value)
    for token in ("window 200", "--scope", "reserve"):
        assert token in message


def test_degrade_step_three_reduces_hop_one_to_signatures(scope: Scope):
    reduced = scope.degrade(3)
    assert reduced.full == (scope.anchor,)
    assert scope.full[1] in reduced.signatures


def test_degrade_step_four_windows_the_anchor(scope: Scope):
    assert scope.degrade(4).anchor_window is not None


def test_degrade_preserves_the_anchor_line_at_every_step(scope: Scope):
    # anchor_line has no default that would silently reappear here -- if a
    # branch of degrade() forgot to thread it through, this fails at that
    # branch specifically.
    for step in (2, 3, 4):
        assert scope.degrade(step).anchor_line == scope.anchor_line


def test_windowing_the_anchor_is_what_shrinks_step_four(scope: Scope, diag: Diagnostic, tmp_path: Path):
    # As written pre-amendment this compared degrade(4) against the
    # UNDEGRADED scope, so it passed on the length drop from rung 3's
    # hop-1 collapse alone -- proven by mutation: a `_window_text` that
    # returns its input unchanged left that version green. degrade(3) and
    # degrade(4) differ ONLY in anchor_window, so nothing else can explain
    # a difference here.
    at_3 = render(scope.degrade(3), diag, (), "search_replace", tmp_path)
    at_4 = render(scope.degrade(4), diag, (), "search_replace", tmp_path)
    assert len(at_4) < len(at_3)


def test_the_window_includes_the_failing_line(scope: Scope, diag: Diagnostic, tmp_path: Path):
    # Invariant 1: the window must contain the failing line, not the
    # file's midpoint. Line 350 sits well outside the old file-midpoint
    # window on this 400-line fixture, so a regression to len(lines)//2
    # centring fails this immediately.
    out = render(scope.degrade(4), diag, (), "search_replace", tmp_path)
    assert "FAIL_HERE" in out


def test_the_window_label_names_the_line_it_centred_on(scope: Scope, diag: Diagnostic, tmp_path: Path):
    # Invariant 3 (positive case): a window that DID shrink the file must
    # say so, and say around what -- not the generic, now-false
    # "(windowed around the failure)" the pre-amendment code always printed.
    out = render(scope.degrade(4), diag, (), "search_replace", tmp_path)
    assert f"windowed around line {_FAIL_LINE}" in out


def test_a_window_that_would_not_shrink_the_file_is_not_labelled_as_one(tmp_path: Path):
    # Invariant 3 (negative case): on a file no wider than the window span,
    # windowing is a no-op and must not claim otherwise. This is exactly
    # the pre-amendment rung-4-is-dead scenario (81-line file, +/-60 window
    # covers all of it) -- it must now render honestly, not silently.
    (tmp_path / "small.py").write_text("def test_x():\n" + "    assert 0\n" * 80)
    small = Scope(
        tmp_path / "small.py", (tmp_path / "small.py",), (), (-60, 60), 40,
    )
    small_diag = Diagnostic(False, "small.py", 40, "AssertionError", "raw")
    out = render(small, small_diag, (), "search_replace", tmp_path)
    assert "windowed" not in out


def test_the_cost_estimate_and_the_render_window_use_the_identical_text(scope: Scope):
    # Invariant 2, direct: budget._window must delegate to
    # render._window_text rather than reimplement it, and both must derive
    # the centre from the Scope's own anchor_line -- never from a
    # Diagnostic passed in separately, since budget._cost never receives
    # one at all and a second centring rule could disagree with render's.
    from robigo.context.budget import _window
    from robigo.context.render import _window_text

    text = scope.anchor.read_text(encoding="utf-8")
    span = (-60, 60)
    assert _window(text, span, scope.anchor_line) == _window_text(text, span, scope.anchor_line)
