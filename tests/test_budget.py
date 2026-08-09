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
from robigo.context.render import Turn, render
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
# history=0 (248 tokens fixed cost), AFTER amendment 2 (cost includes the
# "--- {label} ---" headers and windowing suffix, matching exactly what
# render emits rather than file contents alone):
#
#   rung | cost | saves | window band that lands fit() here | width
#   -----|------|-------|-------------------------------------|------
#   1    | 1977 |   --  | >= 2225                             |  --
#   2    | 1836 |  141  | 2084 - 2224                          | 141
#   3    | 1599 |  237  | 1847 - 2083                          | 237
#   4    |  469 | 1130  | 717  - 1846                          | 1130
#
# (Amendment 1's pre-amendment-2 table read 1958/1827/1584/445 -- every rung
# rose by the header/label tokens amendment 2 added to the estimate; the
# *shape* -- rung 2 barely above rung 1, rung 4 far below rung 3 -- is
# unchanged, as expected since the same files still degrade the same way.)
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


def test_the_default_history_reserve_covers_two_real_capped_reads(
    scope: Scope, diag: Diagnostic, tmp_path: Path
):
    # Item 5 (Important, ruled 2026-08-09): `history=200` under-reserved by
    # 12x against loop.py's real shape -- `_READ_CAP` (4000 chars) capped
    # reads, `_HISTORY_TURNS` (2) kept -- so `fit()` accepted prompts that
    # then overflowed the window by over a thousand tokens at a comfortable
    # 4096-token window. This reproduces the real shape end to end: an
    # actual over-cap file, read through loop.py's own `_read` (so the
    # exact truncation suffix is not re-typed here), rendered twice via the
    # real `render()` to measure the REAL marginal cost history adds to a
    # prompt, and checks the DEFAULT reserves at least that much.
    from robigo.loop import _HISTORY_TURNS, _READ_CAP, _read

    big = tmp_path / "big.py"
    big.write_text("x" * (_READ_CAP + 500))
    capped = _read(tmp_path, "big.py")
    assert len(capped) > _READ_CAP  # sanity: this really was a capped read

    turn = Turn("read big.py", capped)
    history = (turn,) * _HISTORY_TURNS

    bare = render(scope, diag, (), "search_replace", tmp_path)
    with_history = render(scope, diag, history, "search_replace", tmp_path)
    real_history_cost = estimate_tokens(with_history) - estimate_tokens(bare)

    default_history = Budget(window=1, reserve_out=0).history
    assert default_history >= real_history_cost
    # And it does not do so by reverting to the OLD, already-refuted 200:
    # that number is what this test exists to fail against.
    assert default_history > 200


def test_importing_budget_does_not_require_loop_to_finish_importing():
    # Item 9 (round 2, ruled 2026-08-09): `_default_history_tokens`'s
    # `from robigo.loop import ...` was only SYNTACTICALLY deferred by
    # living inside a function -- a module-level `DEFAULT_HISTORY_TOKENS =
    # _default_history_tokens()` call still ran it while `context.budget`
    # itself was mid-import. The wiring slice this module exists for
    # (`loop.py` importing `Budget`/`fit`, its natural first line) hit
    # `ImportError: cannot import name '_HISTORY_TURNS' from partially
    # initialized module 'robigo.loop'`, reproduced directly against the
    # round-1 code before this fix. `field(default_factory=...)` moves the
    # `robigo.loop` import to `Budget()` CONSTRUCTION time, well after both
    # modules have finished importing in any real call order.
    #
    # Run in a FRESH subprocess, not in-process: other tests in this same
    # pytest session (anything that imports `robigo.cli`) have almost
    # certainly already imported `robigo.loop`, which would make an
    # in-process `"robigo.loop" not in sys.modules` check order-dependent
    # and unable to tell a fix from a regression.
    import subprocess
    import sys

    script = (
        "import sys\n"
        "import robigo.context.budget as b\n"
        "assert 'robigo.loop' not in sys.modules, ("
        "'importing context.budget must not import robigo.loop as a side "
        "effect')\n"
        "budget = b.Budget(window=1, reserve_out=0)\n"
        "assert budget.history > 0\n"
        "print('OK', budget.history)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_estimate_tokens_does_not_undercount_a_real_functions_lexical_tokens():
    # Whole-branch review, item 7 (ruled 2026-08-09): the old version of
    # this test asserted `estimate_tokens("x" * 36) == 11`, which is just
    # `int(36 / CHARS_PER_TOKEN) + 1` restated -- it would still pass if
    # CHARS_PER_TOKEN were recalibrated to something that genuinely
    # undercounts real code, as long as the magic number 11 were updated to
    # match, because nothing in it is compared to anything outside the
    # formula itself. It verified the arithmetic, never the "conservative
    # for code" the name claimed.
    #
    # This instead compares against Python's own stdlib `tokenize` module:
    # a real, independent lexical token count for a real function's source
    # (this file's own `fit`, imported already above). A genuinely
    # conservative code-token estimate must never fall below it -- a
    # BPE/subword tokenizer never needs FEWER tokens than the lexical
    # count, since every identifier, operator, and literal is at least one
    # subword token, often several. Fails today's `estimate_tokens` (239
    # for `fit`'s 135 lexical tokens) if CHARS_PER_TOKEN is ever loosened
    # to 6 or higher -- verified by mutation, see the fix-wave report.
    import inspect
    import io
    import tokenize

    source = inspect.getsource(fit)
    skip = {
        tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NEWLINE,
        tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT,
    }
    lexical_tokens = sum(
        1
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type not in skip
    )
    assert estimate_tokens(source) >= lexical_tokens


def test_a_generous_window_keeps_the_full_scope(scope: Scope, tmp_path: Path):
    fitted, step = fit(scope, Budget(32768, 512, 350, 600, 200), tmp_path)
    assert step == 1
    assert fitted.full == scope.full and fitted.signatures == scope.signatures


def test_dropping_hop_two_signatures_is_what_step_two_saves(scope: Scope, tmp_path: Path):
    # Rung 2 in isolation: hop-2's signatures (40 `def` lines) are gone but
    # hop-1 is still full text. This is the rung the original test never
    # pinned -- its own band is what makes it distinguishable from rung 3
    # at all.
    fitted, step = fit(scope, Budget(2154, 128, 60, 60, 0), tmp_path)
    assert step == 2
    assert fitted.full == scope.full
    assert fitted.signatures == ()


def test_a_tight_window_reduces_hop_one_to_signatures(scope: Scope, tmp_path: Path):
    # Corrected per amendment 1: at step 3, hop-1 has become a signature,
    # not vanished. `signatures == ()` was rung 2's shape, not rung 3's.
    fitted, step = fit(scope, Budget(1965, 128, 60, 60, 0), tmp_path)
    assert step == 3
    assert fitted.full == (scope.anchor,)
    assert fitted.signatures == (scope.full[1],)


def test_windowing_the_anchor_is_the_last_rung_before_refusal(scope: Scope, tmp_path: Path):
    # Rung 4 in isolation, from the middle of its own band -- pre-amendment
    # this band was empty (cost(4) == cost(3)) so no window could ever land
    # fit() here at all.
    fitted, step = fit(scope, Budget(1282, 128, 60, 60, 0), tmp_path)
    assert step == 4
    assert fitted.anchor_window is not None


def test_an_impossible_window_refuses_and_prints_the_arithmetic(scope: Scope, tmp_path: Path):
    with pytest.raises(BudgetExhausted) as e:
        fit(scope, Budget(200, 128, 60, 60, 0), tmp_path)
    message = str(e.value)
    for token in ("window 200", "--scope", "reserve"):
        assert token in message


def test_the_refusals_printed_terms_reconcile_to_the_printed_total(
    scope: Scope, tmp_path: Path
):
    # Item 3 (Important, ruled 2026-08-09): `history` was subtracted from
    # `scope_budget` but never printed, so the printed terms did not sum to
    # the printed total -- "window 800 / system 60 / reserve 128 /
    # diagnostic 60" against "available for scope 352" is short by exactly
    # the unprinted history=200. The pre-fix test for this message passed
    # history=0, so the missing term contributed nothing either way and the
    # bug was invisible to it. This one uses a non-zero history and checks
    # the arithmetic actually reconciles, not just that "history" appears
    # as a substring.
    budget = Budget(window=800, reserve_out=128, system=60, diagnostic=60,
                    history=200)
    with pytest.raises(BudgetExhausted) as e:
        fit(scope, budget, tmp_path)
    message = str(e.value)
    assert "history 200" in message
    assert f"available for scope {budget.scope_budget}" in message
    assert budget.scope_budget == 800 - 60 - 128 - 60 - 200 == 352
    # Every subtracted term named in the message must actually reconcile to
    # the printed total, not merely co-occur with it as separate strings.
    terms = {
        "window": budget.window, "system": budget.system,
        "reserve": budget.reserve_out, "diagnostic": budget.diagnostic,
        "history": budget.history,
    }
    for label, value in terms.items():
        assert f"{label} {value}" in message
    reconciled = terms["window"] - terms["system"] - terms["reserve"] \
        - terms["diagnostic"] - terms["history"]
    assert reconciled == budget.scope_budget


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


def test_degrade_rejects_a_step_beyond_the_ladder(scope: Scope):
    # Amendment 2: step >= 5 used to fall through to rung 4's scope
    # silently. `fit` never asks for one, but a caller that miscomputes a
    # step must get a ValueError, not a plausible-looking scope one rung
    # short of what it asked for.
    with pytest.raises(ValueError):
        scope.degrade(5)


def test_degrade_rejects_a_step_of_zero(scope: Scope):
    # Whole-branch review, item 7 (ruled 2026-08-09): `step <= 1` used to
    # return rung 1's scope for 0 too -- the same plausible-looking wrong
    # answer amendment 2 refused to hand back above the ladder, just below
    # it instead. There is no rung 0.
    with pytest.raises(ValueError):
        scope.degrade(0)


def test_degrade_rejects_a_negative_step(scope: Scope):
    with pytest.raises(ValueError):
        scope.degrade(-1)


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


def test_a_windowed_scope_with_no_known_line_is_labelled_honestly(tmp_path: Path):
    # Amendment 2, finding 3: anchor_window set but anchor_line None (an
    # adapter that could not determine a line) had NO test at all -- a
    # regression that collapsed the fallback branch to print the literal
    # "windowed around line None" would have left the pre-amendment-2
    # suite fully green. Needs a file long enough that the window still
    # shrinks it (a no-op window would suppress the label entirely and
    # never reach this branch), so this reuses the 400-line anchor text.
    (tmp_path / "big.py").write_text(_anchor_text())
    unknown_line = Scope(
        tmp_path / "big.py", (tmp_path / "big.py",), (), (-60, 60), None,
    )
    unknown_diag = Diagnostic(False, "big.py", None, "AssertionError", "raw")
    out = render(unknown_line, unknown_diag, (), "search_replace", tmp_path)
    assert "None" not in out
    assert "windowed around the file's midpoint" in out


def test_the_estimate_equals_the_cost_of_what_render_actually_emits(
    scope: Scope, diag: Diagnostic, tmp_path: Path
):
    # Item 4 (Important, ruled 2026-08-09): the old version of this test
    # asserted `_cost(scope, root) == estimate_tokens(_scope_section(scope,
    # root))`, but `_cost` IS that exact expression (see `budget._cost`'s
    # own body) -- both sides are one call, and `render`, the function that
    # actually builds the prompt a model sees, was never invoked. Proven by
    # mutation: giving `render` a private copy of `_scope_section` that
    # differs only in the header delimiter ("=== label ===" instead of
    # "--- label ---") left the whole suite, including the old version of
    # this test, green (see the fix-wave report for the mutation run).
    #
    # This version calls `render` for real and asserts the text `_cost`
    # charges for is verbatim inside what `render` actually emits, so a
    # `render` that diverges from `_scope_section` -- however slightly --
    # fails this test where the old one could not. Exercises all four
    # rungs, since each has a different shape (whole files, dropped
    # signatures, hop-one-as-signature, windowed anchor).
    from robigo.context.budget import _cost
    from robigo.context.render import _scope_section

    for step in (1, 2, 3, 4):
        candidate = scope.degrade(step)
        section = _scope_section(candidate, tmp_path)
        prompt = render(candidate, diag, (), "search_replace", tmp_path)
        assert section in prompt
        assert _cost(candidate, tmp_path) == estimate_tokens(section)


def test_an_unreadable_file_is_costed_the_way_it_is_rendered(scope: Scope, tmp_path: Path):
    # render substitutes a placeholder for an unreadable file; the estimate
    # must cost that same placeholder rather than raising. A crash here is
    # worse than a bad number: fit's one guarantee is fit-or-refuse-with-
    # arithmetic, never a crash instead of either.
    from robigo.context.budget import _cost
    from robigo.context.render import _scope_section

    (tmp_path / "hop1.py").write_bytes(b"\xff\xfe not utf-8")
    cost = _cost(scope, tmp_path)  # must not raise
    assert cost == estimate_tokens(_scope_section(scope, tmp_path))
