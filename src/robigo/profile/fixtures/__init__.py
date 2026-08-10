# src/robigo/profile/fixtures/__init__.py
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from robigo.profile.corpus_io import CorpusRecord


@dataclass(frozen=True)
class Fixture:
    """One single-defect edit with a known-correct target. A stopgap for
    stage 2 until plan 04's mutation generator replaces it; the interface
    it presents to stages.py does not change when that happens.

    `original` is the exact line(s) as they exist before the fix; `expect`
    is the line(s) after. Both are wrapped into a tiny, self-contained
    function body by `stages.fixture_body` before being shown to a model
    -- neither field is a complete file on its own."""

    name: str
    filename: str
    original: str
    expect: str


FIXTURES: tuple[Fixture, ...] = (
    Fixture("off_by_one", "src/counter.py",
            "    return len(items) - 1\n", "    return len(items)\n"),
    Fixture("wrong_operator", "src/scale.py",
            "    return value + factor\n", "    return value * factor\n"),
    Fixture("swapped_args", "src/clamp.py",
            "    return max(high, min(low, value))\n",
            "    return max(low, min(high, value))\n"),
    Fixture("missing_return", "src/total.py",
            "    sum(values)\n", "    return sum(values)\n"),
    Fixture("inverted_test", "src/gate.py",
            "    if not ready:\n", "    if ready:\n"),
)

CORPUS_NAME = "fixtures-v1"
"""This bundle's identity when it is presented as a corpus (task 4,
invariant 12: "the bundled fixtures are expressible as a corpus file").
Defined once, here, beside `FIXTURES` itself -- `cli.profile_main` imports
this rather than typing the string a second time, which is exactly the
carried debt this plan exists to close (`run_profile`'s old `corpus: str =
"fixtures-v1"` kwarg default named nothing that actually tied the string
to this data)."""

_OPERATOR_NAMES: dict[str, str] = {
    "off_by_one": "off_by_one",
    "wrong_operator": "flipped_comparison",
    "swapped_args": "swapped_args",
    "missing_return": "dropped_return",
    "inverted_test": "inverted_condition",
}
"""`Fixture.name` predates `robigo.profile.corpus.OPERATORS` (plan 04, task
1) and used its own, slightly different names for the same five shapes
(`wrong_operator` vs `flipped_comparison`, `missing_return` vs
`dropped_return`, `inverted_test` vs `inverted_condition`) -- this is the
one place that maps a fixture's own name to the operator name a real
generated `Mutant.operator` would carry, so `as_corpus_records` reports an
`operator` a reader can compare against the corpus schema's own vocabulary,
not a fixtures-only synonym."""

_FIXTURE_LINE = 2
"""Every `Fixture.original`/`.expect` is wrapped by `stages.fixture_body`
as the SECOND line of a tiny function -- one fixed one-line header
(`_FUNCTION_HEADER`), then the fixture's own text -- for every one of the
five, so `line=2` is the honest answer for all of them, not a guess."""


def as_corpus_records() -> tuple[CorpusRecord, ...]:
    """`FIXTURES`, expressed as `CorpusRecord`s -- proves the bundled,
    hand-written set fits the SAME schema a mined corpus does, honouring
    "fixtures-v1 becomes one of them [corpus files]" without pretending
    these five were ever mined or verified the way a real corpus record is.

    Honestly labelled, not dressed up as verifier output: these five were
    hand-picked (plan 03) to LOOK like real defects, never proposed by
    `robigo.profile.corpus.candidates()` and never run through `robigo.
    profile.verify.verify()` against a real test suite. `test_id`,
    `diagnostic`, and `source_sha` say so explicitly rather than inventing
    a plausible-looking pytest node id or commit hash for data that was
    never measured that way -- this project's own review has repeatedly
    found and named exactly that failure mode (a fabricated-looking value
    standing in for a real measurement)."""
    return tuple(
        CorpusRecord(
            name=fixture.name,
            path=Path(fixture.filename),
            line=_FIXTURE_LINE,
            broken=fixture.original,
            fixed=fixture.expect,
            test_id="n/a -- hand-authored, never run through verify()",
            diagnostic="hand-authored fixture (plan 03); not mined or "
                       "verified against a real test suite",
            operator=_OPERATOR_NAMES[fixture.name],
            source_repo="robigo (plan 03, hand-authored -- no source repo)",
            source_sha="n/a",
        )
        for fixture in FIXTURES
    )


@dataclass(frozen=True)
class FixturesFromCorpus:
    """`fixtures_from_corpus`'s result: the usable `Fixture`s, and every
    record dropped at conversion time because its wrapped body does not
    parse (I4, whole-branch review 2026-08-10) -- stated, never silently
    absent, the same "anything dropped is stated as dropped" rule this
    plan follows everywhere else (`GenerationResult.dropped`,
    `corpus_io`'s own `"dropped"` key)."""

    fixtures: tuple[Fixture, ...]
    dropped: tuple[str, ...]


def fixtures_from_corpus(records: Sequence[CorpusRecord]) -> FixturesFromCorpus:
    """The reverse of `as_corpus_records`: any generated corpus's records,
    expressed as `Fixture`s `stage2_codecs` can actually iterate. Closes
    the loop `as_corpus_records` alone left open (coordinator review,
    2026-08-10): proving fixtures-v1 is EXPRESSIBLE as a corpus file is
    only half of "stage 2 consumes a corpus file" -- nothing read a
    generated one back until this existed.

    The field mapping is the exact inverse of `as_corpus_records`'s own:
    `Fixture.original` (the DEFECTIVE line a model is shown) is `record.
    broken`, and `Fixture.expect` (the corrected line) is `record.fixed`
    -- both dataclasses agree on which of their two line fields is the
    broken one and which is the fixed one, so this is a direct field
    rename, not a re-derivation. `Fixture.filename` is `str(record.path)`
    (`Fixture` predates `Path`-typed fields; `CorpusRecord.path` is a
    `Path`). `Fixture.name` is `record.name` verbatim.

    Every one of `CorpusRecord`'s four fields `Fixture` needs (name, path,
    broken, fixed) is REQUIRED with no default (invariant 9) and directly
    derivable here -- there is no fifth field `Fixture` needs that a
    generated record could fail to carry.

    I4 (whole-branch review, 2026-08-10): each candidate `Fixture` IS now
    checked here for whether `stages.fixture_body` wraps it into something
    `ast.parse` accepts -- a record whose body does not parse is dropped
    and named in `.dropped`, rather than passed through to `stage2_codecs`
    where it would score as "result does not parse as Python", the SAME
    outcome a genuinely broken model reply produces. Measured across all
    986 real candidates from `src/robigo`: 206 (20.9%) failed this check
    before `stages.fixture_body` itself was also fixed to re-indent a
    mutant's line to the wrapper's own assumed level (that fix alone
    recovers most of them); this check is the backstop for the rest (91,
    9.2%) -- lines cut from a multi-line expression (a list comprehension's
    own `if` clause, an unclosed call spanning several physical lines) that
    cannot form a complete statement in isolation at ANY indent, a
    genuinely different defect than the indent mismatch `fixture_body`'s
    own fix addresses. `stages.fixture_body` is imported HERE, inside the
    function, not at module level -- `stages.py` already imports `Fixture`/
    `FIXTURES` from this module at ITS OWN top level, so a top-level import
    the other way would be a real cycle; deferred to call time, both
    modules are already fully loaded and the cycle never bites."""
    from robigo.profile.stages import fixture_body

    kept: list[Fixture] = []
    dropped: list[str] = []
    for record in records:
        fixture = Fixture(
            name=record.name,
            filename=str(record.path),
            original=record.broken,
            expect=record.fixed,
        )
        try:
            ast.parse(fixture_body(fixture))
        except SyntaxError as exc:
            dropped.append(
                f"{record.name} ({record.path}:{record.line}): wrapped body "
                f"does not parse as Python ({exc}), dropped at conversion "
                f"time (I4)"
            )
            continue
        kept.append(fixture)
    return FixturesFromCorpus(fixtures=tuple(kept), dropped=tuple(dropped))
