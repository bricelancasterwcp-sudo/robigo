# src/robigo/profile/fixtures/__init__.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
