# src/robigo/profile/corpus.py
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

OPERATORS: tuple[str, ...] = (
    "off_by_one",
    "flipped_comparison",
    "swapped_args",
    "dropped_return",
    "inverted_condition",
)
"""The five shapes `fixtures-v1` hand-wrote (`robigo.profile.fixtures`),
generalised to apply to any real source file rather than one hand-picked
line each. Every entry here is the exact string a `Mutant.operator` this
module produces can hold -- nothing in `candidates` emits a name outside
this tuple."""


@dataclass(frozen=True)
class Mutant:
    """One candidate single-line, single-defect edit to `path`, cut from
    real source rather than hand-written -- this is what replaces plan 03's
    `fixtures-v1` (spec: "Generate single-defect repair tasks by mutating
    real code").

    `original` and `mutated` are the COMPLETE text of line `line`
    (1-indexed, counted the way `str.splitlines` counts), each including
    its own line ending exactly as it appeared in the source `candidates`
    was called with. Applying the mutation is replacing that one line with
    `mutated`; applying the REVERSE -- replacing `mutated` back with
    `original` -- is the corpus's ground truth (plan 04, task 3: "the
    reverse patch is stored, not derived at read time"). Storing both full
    line strings, rather than a column span or a diff fragment, makes that
    reverse exact and requires no re-derivation: a later reader (or a
    verifier running in a completely separate clone, per task 2) recovers
    the original file by one line-for-line substitution, never by
    re-running this module's own logic.
    """

    path: Path
    line: int
    original: str
    mutated: str
    operator: str


def _apply(source: str, mutant: Mutant) -> str:
    """Replace line `mutant.line` of `source` with `mutant.mutated`,
    leaving every other line byte-identical. Used both by `candidates`
    itself (to check invariant 2 -- does the mutated FILE still parse --
    before a candidate is ever returned) and, deliberately, by this
    module's own tests: a round trip is `_apply(_apply(source, m),
    reverse(m))`, and there is exactly one implementation of "replace one
    line" for both directions to share, rather than the test re-deriving
    its own copy that could drift from what `candidates` actually checked.

    Raises `ValueError` if `mutant.line` does not name a line of `source`,
    or if that line's current text does not match `mutant.original` --
    either means `mutant` was not cut from `source`, and applying it would
    silently edit the wrong line.
    """
    lines = source.splitlines(keepends=True)
    index = mutant.line - 1
    if not 0 <= index < len(lines):
        raise ValueError(f"line {mutant.line} is outside a {len(lines)}-line source")
    if lines[index] != mutant.original:
        raise ValueError(
            f"line {mutant.line} reads {lines[index]!r}, not this mutant's "
            f"recorded original {mutant.original!r} -- this mutant was not "
            f"cut from this source"
        )
    lines[index] = mutant.mutated
    return "".join(lines)


def reverse(mutant: Mutant) -> Mutant:
    """The undo: `original` and `mutated` swapped, everything else kept.
    `_apply(_apply(source, m), reverse(m)) == source` is invariant 1, and
    this is the one place that pairing is written down -- every caller
    that needs the reverse patch (a verifier restoring a clone, a test
    proving the round trip) uses this rather than constructing the swapped
    `Mutant` by hand, which is exactly the kind of parallel copy free to
    drift that this project's `CARRIED-DEBT.md` repeatedly names as a
    defect source."""
    return Mutant(mutant.path, mutant.line, mutant.mutated, mutant.original, mutant.operator)


def candidates(source: str, path: Path) -> tuple[Mutant, ...]:
    """Every mutation this module's five operators can propose against
    `source`, filtered down to the ones invariant 2 allows to exist at all.

    `source` that does not parse produces no candidates (there is nothing
    here to mutate -- a broken input file is not this function's problem
    to diagnose). Otherwise each operator walks the real AST looking for
    its own shape (an int literal, a single-op comparison, a call with two
    or more positional arguments, a `return <expr>`, a `not <expr>`) and
    proposes a one-line text substitution for each site it finds, located
    by the site's own `col_offset`/`end_col_offset` -- never by scanning
    the line's raw text for a matching substring. That distinction is load
    -bearing, not stylistic: a substring scan for e.g. `"not "` matches
    inside `"is not"`/`"not in"`, inside a string literal that happens to
    contain the word, or inside an identifier like `annotations`, and a
    scan for `"return "` matches `return(x)` (no space before the paren)
    and would leave a dangling `(x)` if it blindly deleted six characters
    instead of the exact gap between the keyword and the value. Reading
    the gap between two AST-located spans and requiring it to be exactly
    what the shape expects (`_RETURN_GAP`, `_NOT_GAP`) rejects those cases
    by construction, before the final parse check ever runs.

    Every proposal is still verified before being kept, never trusted on
    the strength of its own site-finder: the candidate line is spliced
    into a copy of `source` and the WHOLE resulting file is re-parsed with
    `ast.parse` (invariant 2 -- "a mutant that does not parse is not a
    defect, it is a broken file"). A proposal identical to the original
    line (impossible for these five operators as written, since each one
    always changes a value, but checked rather than assumed) is also
    dropped, since a no-op mutation is not a mutation.

    Every operator locates its site as a node whose relevant span sits on
    ONE physical line (checked via `lineno == end_lineno`/`lineno`-
    equality between adjacent nodes before any substitution is built), so
    every kept `Mutant` changes exactly one line of `source` and nothing
    else -- invariant 1's other half, "exactly one line".

    Comments do not exist in the AST at all, so no operator can ever
    locate a site inside one. Docstrings are `ast.Constant` nodes holding
    a `str`, and none of these five operators ever targets a `str`
    constant, a `Compare`'s left/right text, or anything else that could
    sit inside one -- so a docstring is never a candidate's line either
    (invariant 3). Blank lines hold no AST node and are never proposed for
    the same structural reason.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()

    lines = source.splitlines(keepends=True)
    proposals: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        proposals.extend(_off_by_one(node, lines))
        proposals.extend(_flipped_comparison(node, lines))
        proposals.extend(_swapped_args(node, lines))
        proposals.extend(_dropped_return(node, lines))
        proposals.extend(_inverted_condition(node, lines))

    kept: list[Mutant] = []
    for line_no, new_line, operator in proposals:
        original_line = lines[line_no - 1]
        if new_line == original_line:
            continue
        mutated_lines = list(lines)
        mutated_lines[line_no - 1] = new_line
        try:
            ast.parse("".join(mutated_lines))
        except SyntaxError:
            continue
        kept.append(Mutant(path, line_no, original_line, new_line, operator))
    return tuple(kept)


def _off_by_one(node: ast.AST, lines: list[str]) -> list[tuple[int, str, str]]:
    """An integer literal, incremented by one -- the shape `fixtures-v1`'s
    `off_by_one` fixture drew from (`len(items) - 1` vs `len(items)`, read
    as "the boundary is one value off"). `bool` is excluded even
    though `isinstance(True, int)` is `True` in Python: a mutated `True`
    would not be an off-by-one defect, it would be a different shape this
    operator does not claim to produce."""
    if not isinstance(node, ast.Constant):
        return []
    if not isinstance(node.value, int) or isinstance(node.value, bool):
        return []
    if node.lineno != node.end_lineno:
        return []
    line = lines[node.lineno - 1]
    new_line = (
        line[: node.col_offset] + str(node.value + 1) + line[node.end_col_offset :]
    )
    return [(node.lineno, new_line, "off_by_one")]


_COMPARISON_FLIP: dict[type, str] = {
    ast.Lt: ">=",
    ast.LtE: ">",
    ast.Gt: "<=",
    ast.GtE: "<",
    ast.Eq: "!=",
    ast.NotEq: "==",
}
_COMPARISON_SYMBOL = re.compile(r"<=|>=|==|!=|<|>")


def _flipped_comparison(node: ast.AST, lines: list[str]) -> list[tuple[int, str, str]]:
    """A single relational comparison (`<`, `<=`, `>`, `>=`, `==`, `!=`),
    negated to the opposite relation -- the shape `fixtures-v1`'s
    `wrong_operator` fixture drew from. Chained comparisons (`a < b < c`,
    `len(node.ops) > 1`) are skipped: which of two operators a repair task
    should target would be ambiguous, and `Is`/`IsNot`/`In`/`NotIn` are
    skipped because they are words, not symbols, and are not in the flip
    table at all.

    The operator's own text is never in the AST -- `cmpop` nodes (`Lt`,
    `Eq`, ...) carry no `lineno`/`col_offset` of their own in CPython's
    `ast` module. It is found by searching the GAP between `left`'s end
    and the comparator's start, a span Python's grammar guarantees holds
    only the operator symbol and optional whitespace for a single-op
    Compare -- nothing else can lexically appear there without a syntax
    error already having been raised by `ast.parse` before this function
    ever runs. That is a bounded, grammar-guaranteed search window, not a
    scan of the whole line, so a `<` appearing anywhere else on the same
    physical line (inside a string, inside a different expression) is
    never in scope."""
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return []
    replacement = _COMPARISON_FLIP.get(type(node.ops[0]))
    if replacement is None:
        return []
    left = node.left
    right = node.comparators[0]
    if left.end_lineno != right.lineno:
        return []
    line = lines[left.end_lineno - 1]
    gap = line[left.end_col_offset : right.col_offset]
    match = _COMPARISON_SYMBOL.search(gap)
    if match is None:
        return []
    start = left.end_col_offset + match.start()
    end = left.end_col_offset + match.end()
    new_line = line[:start] + replacement + line[end:]
    return [(left.end_lineno, new_line, "flipped_comparison")]


def _swapped_args(node: ast.AST, lines: list[str]) -> list[tuple[int, str, str]]:
    """The first two POSITIONAL arguments of a call, swapped -- the shape
    `fixtures-v1`'s `swapped_args` fixture drew from (`max(high, min(low,
    value))` vs `max(low, min(high, value))`). Keyword arguments and
    `*args` unpacking are left alone (only `node.args`, and only when
    neither of the first two is `ast.Starred`) -- a keyword's name is part
    of the call's meaning in a way a swap should not silently reorder past.

    Each argument's own span (`col_offset`/`end_col_offset`) is read
    directly from the AST and swapped as opaque text, never split out by
    scanning the line for commas. That distinction matters concretely for
    a call like `max(min(a, b), c)`: a comma-counting substitution that
    does not track paren depth finds three top-level-looking commas
    instead of one and produces mismatched parentheses; reading each
    argument's exact AST span keeps `min(a, b)` intact as one swapped
    unit regardless of the comma nested inside it. `test_swapped_args_
    handles_a_nested_call_a_naive_comma_split_would_break` in this
    module's test file demonstrates the naive version failing on exactly
    this input and this operator's real output surviving `ast.parse`.
    """
    if not isinstance(node, ast.Call) or len(node.args) < 2:
        return []
    first, second = node.args[0], node.args[1]
    if isinstance(first, ast.Starred) or isinstance(second, ast.Starred):
        return []
    if first.lineno != first.end_lineno or second.lineno != second.end_lineno:
        return []
    if first.lineno != second.lineno or first.end_col_offset > second.col_offset:
        return []
    line = lines[first.lineno - 1]
    between = line[first.end_col_offset : second.col_offset]
    first_text = line[first.col_offset : first.end_col_offset]
    second_text = line[second.col_offset : second.end_col_offset]
    new_line = (
        line[: first.col_offset]
        + second_text
        + between
        + first_text
        + line[second.end_col_offset :]
    )
    return [(first.lineno, new_line, "swapped_args")]


_RETURN_GAP = re.compile(r"^return\s+$")


def _dropped_return(node: ast.AST, lines: list[str]) -> list[tuple[int, str, str]]:
    """`return <expr>` with the `return` keyword removed, leaving a bare
    expression statement -- the shape `fixtures-v1`'s `missing_return`
    fixture drew from (`sum(values)` vs `return sum(values)`). A bare
    `return` with no value (`node.value is None`) is skipped: there is no
    expression left to turn into a statement, and dropping just the word
    would leave an empty line where a statement is required.

    The text removed is not simply "the first six characters after
    `col_offset`" -- it is the exact gap between the `Return` node's own
    start and its `value`'s start, and that gap must match `return`
    followed by REQUIRED whitespace and nothing else. `return(x)` (no
    space before the parenthesis) fails that check and is skipped, because
    the six characters `return` are not the whole gap there -- deleting
    just them would leave `(x)`, which happens to still parse, but for the
    wrong reason: the real gap is `return(`, and blindly assuming it is
    always exactly the keyword is the same class of naive-substitution risk
    invariant 2 warns against, just not one that happens to produce a
    `SyntaxError` on this particular input. Requiring the gap to fully
    match `_RETURN_GAP` keeps the operator honest about what it is
    actually looking at rather than getting lucky."""
    if not isinstance(node, ast.Return) or node.value is None:
        return []
    value = node.value
    if node.lineno != value.lineno:
        return []
    line = lines[node.lineno - 1]
    gap = line[node.col_offset : value.col_offset]
    if not _RETURN_GAP.match(gap):
        return []
    new_line = line[: node.col_offset] + line[value.col_offset :]
    return [(node.lineno, new_line, "dropped_return")]


_NOT_GAP = re.compile(r"^not\s+$")


def _inverted_condition(node: ast.AST, lines: list[str]) -> list[tuple[int, str, str]]:
    """`not <expr>` with the `not` removed -- the shape `fixtures-v1`'s
    `inverted_test` fixture drew from (`if not ready:` vs `if ready:`).
    This is the exact shape the brief calls out by name: a naive
    substitution that deletes the substring `"not "` wherever it finds it
    on the line, ignorant of whether the line is a compound-statement
    header, is the class of bug that shipped in `fixtures-v1`'s consumer
    (`stages.fixture_body`, documented there) -- one column away from
    shipping here too if this function scanned the raw line instead of
    reading the exact gap between the `UnaryOp`'s own start and its
    `operand`'s start. That gap is required to match `_NOT_GAP` (the word
    `not` plus required whitespace, nothing else), so `not(x)` -- no space
    before the parenthesis, where the naive six-characters-of-"not " guess
    would again land on the wrong span -- is skipped rather than mutated.
    Whether the resulting line is itself a header (`if ready:`) or a
    plain boolean expression makes no difference here: this function only
    ever changes the ONE line the `not` keyword sits on, and the file's
    other lines -- including any suite the header introduces -- are never
    touched, so there is no version of `fixtures-v1`'s isolated-fragment
    failure mode to reproduce against a real, complete source file."""
    if not isinstance(node, ast.UnaryOp) or not isinstance(node.op, ast.Not):
        return []
    operand = node.operand
    if node.lineno != operand.lineno:
        return []
    line = lines[node.lineno - 1]
    gap = line[node.col_offset : operand.col_offset]
    if not _NOT_GAP.match(gap):
        return []
    new_line = line[: node.col_offset] + line[operand.col_offset :]
    return [(node.lineno, new_line, "inverted_condition")]
