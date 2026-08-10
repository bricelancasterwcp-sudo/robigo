# tests/test_corpus_candidates.py
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from robigo.profile.corpus import (
    OPERATORS,
    Mutant,
    _apply,
    _NOT_GAP,
    _RETURN_GAP,
    candidates,
    reverse,
)

SCOPE_PY = Path(__file__).resolve().parent.parent / "src" / "robigo" / "context" / "scope.py"

# A source file whose comments and docstring quote every operator's trigger
# text ("not ready", "return 1", "max(a, b)") verbatim, and whose real code
# includes both the shapes each operator targets AND the two paren-glued
# forms ("return(x)", "not(x)") that a naive keyword-plus-space substitution
# would corrupt. Line numbers below are load-bearing for several tests and
# are pinned by `test_trap_source_has_the_expected_shape` so a future edit
# to this constant fails loudly instead of silently misdirecting a test at
# the wrong line.
TRAP_SOURCE = (
    '"""Module docstring: not ready, return 1, max(a, b) -- bait, never mutated."""\n'
    "from __future__ import annotations\n"
    "\n"
    "\n"
    "def gate(ready, values):\n"
    "    # comment bait: not ready, return 1, max(a, b) -- never mutated\n"
    "    if not ready:\n"
    "        return max(values[0], values[1])\n"
    "    total = 0\n"
    "    for v in values:\n"
    "        total = total + v\n"
    "    if total <= 10:\n"
    "        return total\n"
    "    return total - 1\n"
    "\n"
    "\n"
    "def odd_calls(a, b, c):\n"
    "    nested = max(min(a, b), c)\n"
    '    quoted = pair("a, b", c)\n'
    "    if not(nested):\n"
    "        pass\n"
    "    return(nested)\n"
)
TRAP_DOCSTRING_LINE = 1
TRAP_COMMENT_LINE = 6
TRAP_BLANK_LINES = {3, 4, 15, 16}
TRAP_NOT_PAREN_LINE = 20  # "    if not(nested):\n"
TRAP_RETURN_PAREN_LINE = 22  # "    return(nested)\n"


def test_trap_source_has_the_expected_shape():
    # Pins the line numbers every other test using TRAP_SOURCE depends on.
    # Fails if TRAP_SOURCE is ever edited without updating the constants
    # above, which would otherwise silently point later assertions at the
    # wrong line.
    assert ast.parse(TRAP_SOURCE)  # must be valid Python to begin with
    lines = TRAP_SOURCE.splitlines()
    assert lines[TRAP_DOCSTRING_LINE - 1].startswith('"""Module docstring')
    assert lines[TRAP_COMMENT_LINE - 1].strip().startswith("# comment bait")
    for n in TRAP_BLANK_LINES:
        assert lines[n - 1] == ""
    assert lines[TRAP_NOT_PAREN_LINE - 1] == "    if not(nested):"
    assert lines[TRAP_RETURN_PAREN_LINE - 1] == "    return(nested)"


# ---------------------------------------------------------------------------
# OPERATORS
# ---------------------------------------------------------------------------


def test_operators_lists_exactly_the_five_shapes_fixtures_v1_hand_wrote():
    # Fails if a shape is added, removed, renamed, or reordered without this
    # test being updated to match -- OPERATORS is public interface other
    # code (and later tasks) can rely on being stable.
    assert OPERATORS == (
        "off_by_one",
        "flipped_comparison",
        "swapped_args",
        "dropped_return",
        "inverted_condition",
    )


# ---------------------------------------------------------------------------
# Per-operator behaviour, each pinned to an exact expected mutated line --
# not merely "a mutation happened" (verification standard item 5: the
# operator field must be shown to vary, not always read the same string).
# ---------------------------------------------------------------------------


def test_off_by_one_increments_the_integer_literal_by_exactly_one():
    # Fails for an operator that changes the literal by any amount other
    # than +1 (e.g. -1, or +2), or that targets the wrong literal.
    source = "def f():\n    return 5\n"
    off = [m for m in candidates(source, Path("f.py")) if m.operator == "off_by_one"]
    assert len(off) == 1
    assert off[0].line == 2
    assert off[0].original == "    return 5\n"
    assert off[0].mutated == "    return 6\n"


def test_off_by_one_skips_bool_even_though_it_is_an_int_subclass():
    # Fails if `isinstance(node.value, bool)` is not excluded -- Python's
    # `isinstance(True, int)` is True, so a naive int check would produce
    # `True -> 2`, a shape this operator does not claim to be.
    source = "def f():\n    return True\n"
    assert not any(m.operator == "off_by_one" for m in candidates(source, Path("f.py")))


def test_flipped_comparison_negates_the_relation():
    # Fails for a flip table entry that swaps direction instead of negating
    # (e.g. "<" -> ">" rather than ">=") or that flips the wrong operator.
    source = "def f(x):\n    return x < 3\n"
    flips = [m for m in candidates(source, Path("f.py")) if m.operator == "flipped_comparison"]
    assert len(flips) == 1
    assert flips[0].mutated == "    return x >= 3\n"


def test_flipped_comparison_skips_chained_comparisons():
    # Fails for an implementation that flips the first operator of a
    # multi-op Compare (`a < b < c`), which is ambiguous about what defect
    # is being represented.
    source = "def f(a, b, c):\n    return a < b < c\n"
    assert not any(
        m.operator == "flipped_comparison" for m in candidates(source, Path("f.py"))
    )


def test_swapped_args_swaps_the_first_two_positional_arguments():
    # Fails for an implementation that swaps the wrong pair, or that
    # produces a no-op (identical) line.
    source = "def f(a, b):\n    return max(a, b)\n"
    swaps = [m for m in candidates(source, Path("f.py")) if m.operator == "swapped_args"]
    assert len(swaps) == 1
    assert swaps[0].mutated == "    return max(b, a)\n"


def test_swapped_args_ignores_keyword_arguments():
    # Fails for an implementation that swaps a keyword argument's value
    # into a positional slot, silently discarding the keyword's name.
    source = "def f(a, b):\n    return dict(key=a, value=b)\n"
    assert not any(m.operator == "swapped_args" for m in candidates(source, Path("f.py")))


def test_swapped_args_handles_a_nested_call_a_naive_comma_split_would_break():
    """Invariant 2's pin case. A naive swap that splits a call's argument
    text on the first two top-level-looking ", " substrings -- ignorant of
    quotes and nesting -- corrupts `pair("a, b", c)` into invalid Python:
    confirmed directly below by actually running the naive approach and
    asserting `ast.parse` raises on ITS output. This operator instead reads
    each argument's exact AST span (`col_offset`/`end_col_offset`), so its
    real candidate for the same line swaps both arguments whole and still
    parses -- proven on the nested nested `max(min(a, b), c)` line too.
    """

    def naive_swap(line: str) -> str:
        start = line.index("(") + 1
        end = line.rindex(")")
        inner = line[start:end]
        parts = inner.split(", ")
        swapped = ", ".join([parts[1], parts[0]] + parts[2:])
        return line[:start] + swapped + line[end:]

    lines = TRAP_SOURCE.splitlines(keepends=True)
    target = '    quoted = pair("a, b", c)\n'
    idx = lines.index(target)
    naive_lines = list(lines)
    naive_lines[idx] = naive_swap(target)
    with pytest.raises(SyntaxError):
        ast.parse("".join(naive_lines))

    muts = candidates(TRAP_SOURCE, Path("trap.py"))
    real = [m for m in muts if m.operator == "swapped_args" and m.original == target]
    assert len(real) == 1
    assert real[0].mutated == '    quoted = pair(c, "a, b")\n'
    ast.parse(_apply(TRAP_SOURCE, real[0]))  # must not raise

    nested_target = "    nested = max(min(a, b), c)\n"
    nested_swap = next(
        m
        for m in muts
        if m.operator == "swapped_args"
        and m.original == nested_target
        and m.mutated == "    nested = max(c, min(a, b))\n"
    )
    ast.parse(_apply(TRAP_SOURCE, nested_swap))  # must not raise


def test_dropped_return_removes_the_keyword_leaving_the_bare_expression():
    # Fails for an implementation that removes too many or too few
    # characters (e.g. leaves a leading space, or eats part of the value).
    source = "def f():\n    return 1 + 2\n"
    drops = [m for m in candidates(source, Path("f.py")) if m.operator == "dropped_return"]
    assert len(drops) == 1
    assert drops[0].mutated == "    1 + 2\n"


def test_dropped_return_skips_a_bare_return_with_no_value():
    # Fails for an implementation that deletes a bare `return`, leaving an
    # empty line where a statement is syntactically required.
    source = "def f(flag):\n    if flag:\n        return\n    return flag\n"
    drops = {m.line: m for m in candidates(source, Path("f.py")) if m.operator == "dropped_return"}
    assert 3 not in drops
    assert 4 in drops
    assert drops[4].mutated == "    flag\n"


def test_dropped_return_skips_a_paren_glued_return():
    # Fails for an implementation that assumes the gap is always exactly
    # the 6 characters "return" -- `return(x)` has no space before the
    # parenthesis, so blindly removing 6 chars there would leave `(x)`,
    # right by accident rather than by a check that actually looked.
    source = "def f(x):\n    return(x)\n"
    assert not any(m.operator == "dropped_return" for m in candidates(source, Path("f.py")))


def test_return_gap_regex_requires_whitespace_after_the_keyword():
    # Direct pin on the site-level guard itself, independent of whether the
    # end-to-end candidates() pipeline's final ast.parse safety net would
    # also happen to reject a bad candidate downstream for some particular
    # input (it does for "return(x)", but that is a property of THIS
    # input, not of the guard -- see the report's mutation-testing notes
    # for the input where a loosened version of this regex still slips a
    # syntactically-valid-but-wrong candidate past that net).
    assert _RETURN_GAP.match("return ")
    assert _RETURN_GAP.match("return   ")
    assert not _RETURN_GAP.match("return(")
    assert not _RETURN_GAP.match("return")


def test_inverted_condition_removes_not_leaving_the_bare_condition():
    # Fails for an implementation that also disturbs the following suite,
    # or that removes the wrong span.
    source = "def f(ready):\n    if not ready:\n        pass\n"
    inv = [m for m in candidates(source, Path("f.py")) if m.operator == "inverted_condition"]
    assert len(inv) == 1
    assert inv[0].mutated == "    if ready:\n"
    assert inv[0].original == "    if not ready:\n"


def test_inverted_condition_skips_a_paren_glued_not():
    # Fails for an implementation matching the bare substring "not "
    # (with a required trailing space) against the raw line instead of the
    # AST-located gap -- `not(x)` has no space, so a naive scan would
    # simply fail to match rather than prove there was nothing safe to cut,
    # and a *looser* naive scan (e.g. dropping the space requirement) would
    # cut "not(" and leave a dangling "x)".
    source = "def f(x):\n    if not(x):\n        pass\n"
    assert not any(m.operator == "inverted_condition" for m in candidates(source, Path("f.py")))


def test_not_gap_regex_requires_whitespace_after_the_keyword():
    # Direct pin on the site-level guard itself -- see the note on
    # test_return_gap_regex_requires_whitespace_after_the_keyword above.
    assert _NOT_GAP.match("not ")
    assert _NOT_GAP.match("not   ")
    assert not _NOT_GAP.match("not(")
    assert not _NOT_GAP.match("not")


# ---------------------------------------------------------------------------
# Invariant 3 -- comments, docstrings, blank lines
# ---------------------------------------------------------------------------


def test_comments_docstrings_and_blank_lines_are_never_mutated():
    # Fails for an implementation that scans raw line text for a trigger
    # pattern ("not ", "return ", a bare digit, "max(a, b)") instead of an
    # AST node's exact location -- TRAP_SOURCE's docstring and comment
    # contain every operator's trigger text on purpose, and this test would
    # catch a candidate landing on any of them.
    muts = candidates(TRAP_SOURCE, Path("trap.py"))
    assert len(muts) > 0  # otherwise "never touched" is vacuously true
    hit_lines = {m.line for m in muts}
    assert TRAP_DOCSTRING_LINE not in hit_lines
    assert TRAP_COMMENT_LINE not in hit_lines
    assert not (TRAP_BLANK_LINES & hit_lines)


def test_paren_glued_forms_in_a_real_file_are_left_alone_by_their_operators():
    # The same two traps as the unit tests above, exercised inside a full
    # file with other real candidates around them, rather than in
    # isolation -- fails if either operator's guard only works on a
    # single-statement file.
    muts = candidates(TRAP_SOURCE, Path("trap.py"))
    assert not any(
        m.line == TRAP_NOT_PAREN_LINE and m.operator == "inverted_condition" for m in muts
    )
    assert not any(
        m.line == TRAP_RETURN_PAREN_LINE and m.operator == "dropped_return" for m in muts
    )


def test_all_five_operators_are_exercised_across_real_sources():
    # Verification standard item 5: an operator field that always reads the
    # same string would fail this test, since it requires all five distinct
    # names to actually appear.
    scope_source = SCOPE_PY.read_text()
    muts = candidates(scope_source, SCOPE_PY) + candidates(TRAP_SOURCE, Path("trap.py"))
    assert {m.operator for m in muts} == set(OPERATORS)


# ---------------------------------------------------------------------------
# Invariant 2 -- never changes what the file means to Python
# ---------------------------------------------------------------------------


def test_a_source_that_does_not_parse_yields_no_candidates():
    # Fails for an implementation that tries to mutate a broken file rather
    # than refusing to touch it at all.
    assert candidates("def f(:\n", Path("broken.py")) == ()


def test_a_proposal_that_does_not_parse_is_rejected_even_if_a_site_finder_proposes_it(
    monkeypatch: pytest.MonkeyPatch,
):
    """Direct test of invariant 2's rejection path, independent of whether
    any of the five real operators currently happen to trigger it on the
    sources this test file uses elsewhere (they do not -- see the report).
    A site-finder that proposes a syntactically broken replacement line
    must never let that proposal reach the returned tuple."""
    import robigo.profile.corpus as corpus_module

    def _broken_site_finder(node: ast.AST, lines: list[str]) -> list[tuple[int, str, str]]:
        if isinstance(node, ast.Module):
            return [(1, "def f(:\n", "off_by_one")]
        return []

    monkeypatch.setattr(corpus_module, "_off_by_one", _broken_site_finder)
    source = "def f():\n    return 1\n"
    muts = candidates(source, Path("f.py"))
    assert not any(m.mutated == "def f(:\n" for m in muts)


def test_every_candidate_from_a_real_source_file_parses_as_python():
    # Verification standard item 4: the >0 assertion runs BEFORE the parse
    # check, so this test cannot pass vacuously on a file that produces
    # zero candidates.
    source = SCOPE_PY.read_text()
    muts = candidates(source, SCOPE_PY)
    assert len(muts) > 0
    for m in muts:
        ast.parse(_apply(source, m))  # must not raise for any candidate


# ---------------------------------------------------------------------------
# Invariant 1 -- exactly one line, exactly reversible
# ---------------------------------------------------------------------------


def test_reverse_swaps_original_and_mutated_and_keeps_everything_else():
    m = Mutant(Path("f.py"), 2, "old\n", "new\n", "off_by_one")
    r = reverse(m)
    assert r.original == "new\n"
    assert r.mutated == "old\n"
    assert r.path == m.path
    assert r.line == m.line
    assert r.operator == m.operator


def test_apply_raises_if_the_mutants_original_does_not_match_the_source():
    # Fails for an implementation that blindly trusts `mutant.line` and
    # `mutant.mutated` without checking the source it is handed actually
    # still reads what the mutant recorded as `original` -- silently
    # editing the wrong content otherwise.
    m = Mutant(Path("f.py"), 1, "wrong line\n", "new\n", "off_by_one")
    with pytest.raises(ValueError):
        _apply("actual line\n", m)


def test_every_candidate_from_a_real_source_file_round_trips_byte_for_byte():
    """Invariant 1, tested over a real, non-trivial file rather than a
    hand-made two-line string (verification standard item 3). Fails for
    any operator whose reverse does not restore the file exactly -- e.g.
    one that also strips trailing whitespace, normalises quote style, or
    otherwise "cleans up" the line on the way back -- because the
    comparison is against the raw bytes read from disk, not a
    str-vs-str comparison that a decoding/encoding round trip could
    quietly paper over.
    """
    original_bytes = SCOPE_PY.read_bytes()
    source = original_bytes.decode("utf-8")
    muts = candidates(source, SCOPE_PY)
    assert len(muts) > 0
    for m in muts:
        applied = _apply(source, m)
        assert applied.encode("utf-8") != original_bytes
        restored = _apply(applied, reverse(m))
        assert restored.encode("utf-8") == original_bytes
