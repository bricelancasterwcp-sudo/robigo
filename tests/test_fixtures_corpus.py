# tests/test_fixtures_corpus.py
"""Invariant 12 ("stage 2 consumes a corpus file, and fixtures-v1 becomes
one of them"): proves the bundled, hand-written FIXTURES fit the SAME
corpus schema a mined corpus does -- `as_corpus_records()` converts them,
and this file proves that conversion round-trips through the real
`write_corpus`/`read_corpus` (task 3) exactly like any other corpus."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from robigo.profile.corpus import candidates
from robigo.profile.corpus_io import CorpusRecord, read_corpus, write_corpus
from robigo.profile.fixtures import (
    CORPUS_NAME,
    FIXTURES,
    Fixture,
    as_corpus_records,
    fixtures_from_corpus,
)
from robigo.profile.verify import Baseline

_BASE = Baseline(broken=0, executed=430, seconds=12.3)
"""A stand-in `Baseline` (I1, whole-branch review 2026-08-10) -- every
`write_corpus` call in this module now requires one."""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("socket.socket.connect must never be called in this test suite")

    monkeypatch.setattr(socket.socket, "connect", _blocked)


def test_every_fixture_becomes_exactly_one_corpus_record():
    records = as_corpus_records()
    assert len(records) == len(FIXTURES) == 5


def test_broken_and_fixed_match_each_fixtures_original_and_expect_lines():
    # Fails if broken/fixed are ever swapped: `broken` must be the
    # DEFECTIVE line a reader is shown (Fixture.original), `fixed` the
    # corrected one (Fixture.expect) -- corpus_io.py's own invariant 8.
    by_name = {r.name: r for r in as_corpus_records()}
    for fixture in FIXTURES:
        record = by_name[fixture.name]
        assert record.broken == fixture.original
        assert record.fixed == fixture.expect
        assert record.path == Path(fixture.filename)
        assert record.broken != record.fixed


def test_operator_names_match_the_real_corpus_vocabulary_not_fixture_only_synonyms():
    # Fixture.name predates robigo.profile.corpus.OPERATORS and used
    # different words for two of the five shapes -- fails if a reader of
    # this bundled corpus would see an operator name a generated corpus
    # never uses.
    from robigo.profile.corpus import OPERATORS

    for record in as_corpus_records():
        assert record.operator in OPERATORS


def test_provenance_is_honest_not_a_fabricated_looking_measurement():
    # These five were hand-authored (plan 03), never mined by candidates()
    # or verified by verify() -- test_id/source_sha must say so, not
    # imitate what a real pytest node id or commit sha looks like.
    for record in as_corpus_records():
        assert "hand-authored" in record.diagnostic
        assert record.source_sha == "n/a"
        # A real sha this project's own verify.py produces is 40 hex
        # characters; this must not accidentally look like one.
        assert record.source_sha != "0" * 40


def test_as_corpus_records_round_trips_through_a_real_corpus_file(tmp_path: Path):
    out = tmp_path / "fixtures-v1.json"
    write_corpus(as_corpus_records(), out, name=CORPUS_NAME, dropped=(), baseline=_BASE)
    name, records, dropped = read_corpus(out)
    assert name == CORPUS_NAME == "fixtures-v1"
    assert records == as_corpus_records()
    assert dropped == ()


def test_corpus_name_is_defined_once_beside_fixtures_not_a_second_literal():
    # A structural guard against the exact carried debt this fixes: import
    # this from robigo.cli too and assert it is the SAME object/value the
    # CLI actually uses, rather than two modules independently typing
    # "fixtures-v1".
    import robigo.cli as cli_module

    assert cli_module.CORPUS_NAME is CORPUS_NAME


# ---------------------------------------------------------------------------
# fixtures_from_corpus — the OTHER direction: a generated corpus, read back
# as the Fixture shape stage2_codecs actually iterates (coordinator review,
# 2026-08-10: as_corpus_records alone only proved fixtures-v1 was
# EXPRESSIBLE as a corpus file; nothing read a generated one back).
# ---------------------------------------------------------------------------


def _record(**kw: object) -> CorpusRecord:
    defaults: dict[str, object] = dict(
        name="calc-dropped_return-2",
        path=Path("src/mylib/calc.py"),
        line=2,
        broken="    sum(values)\n",
        fixed="    return sum(values)\n",
        test_id="tests/test_calc.py::test_total_sums",
        diagnostic="exactly one net new failure",
        operator="dropped_return",
        source_repo="/home/user/mylib",
        source_sha="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
    )
    return CorpusRecord(**{**defaults, **kw})  # type: ignore[arg-type]


def test_fixtures_from_corpus_is_the_exact_inverse_of_as_corpus_records():
    # The strongest possible pin: converting fixtures-v1 to corpus records
    # and back must reproduce FIXTURES exactly, field for field.
    result = fixtures_from_corpus(as_corpus_records())
    assert result.fixtures == FIXTURES
    # I4 (whole-branch review 2026-08-10): none of the five bundled
    # fixtures should ever be dropped at conversion time -- they are the
    # exact shapes `stages.fixture_body`'s own wrapper was designed around.
    assert result.dropped == ()


def test_original_and_expect_map_from_broken_and_fixed_not_swapped():
    # Grounded in a REAL Mutant from candidates() (matching this project's
    # own convention, e.g. test_corpus_io.py's _dropped_return_mutant),
    # not two hand-typed strings chosen to merely look plausible. Fails if
    # original/expect are ever swapped relative to broken/fixed.
    source = "def total(values):\n    return sum(values)\n"
    mutant = next(
        m for m in candidates(source, Path("calc.py")) if m.operator == "dropped_return"
    )
    record = _record(
        path=mutant.path, line=mutant.line,
        broken=mutant.mutated, fixed=mutant.original, operator=mutant.operator,
    )
    fixture = fixtures_from_corpus([record]).fixtures[0]
    assert fixture.original == mutant.mutated == record.broken
    assert fixture.expect == mutant.original == record.fixed
    assert fixture.original != fixture.expect  # guards against a vacuous pass
    assert fixture.name == record.name
    assert fixture.filename == "calc.py"


def test_fixtures_from_corpus_preserves_order():
    records = (
        _record(name="a", path=Path("a.py")),
        _record(name="b", path=Path("b.py")),
        _record(name="c", path=Path("c.py")),
    )
    fixtures = fixtures_from_corpus(records).fixtures
    assert [f.name for f in fixtures] == ["a", "b", "c"]


def test_fixtures_from_corpus_on_no_records_returns_an_empty_tuple():
    result = fixtures_from_corpus(())
    assert result.fixtures == ()
    assert result.dropped == ()


def test_fixture_filename_is_a_string_not_a_path_object():
    # Fixture.filename predates Path-typed fields (CorpusRecord.path IS a
    # Path) -- fails if the conversion leaks a Path where stages.py's own
    # `landing_prompt`/`_Good`-style test fakes expect a plain string to
    # substring-match against a prompt.
    fixture = fixtures_from_corpus([_record()]).fixtures[0]
    assert isinstance(fixture.filename, str)
    assert fixture.filename == "src/mylib/calc.py"


def test_fixtures_from_corpus_round_trips_through_a_written_corpus_file(tmp_path: Path):
    # The full loop: write a corpus file (as robigo corpus would), read it
    # back, convert -- no in-memory record ever reused.
    records = (_record(),)
    out = tmp_path / "generated.json"
    write_corpus(records, out, name="mylib-v1", dropped=(), baseline=_BASE)
    _, read_records, _ = read_corpus(out)
    fixtures = fixtures_from_corpus(read_records).fixtures
    assert fixtures == (
        Fixture(
            name="calc-dropped_return-2", filename="src/mylib/calc.py",
            original="    sum(values)\n", expect="    return sum(values)\n",
        ),
    )


# ---------------------------------------------------------------------------
# I4 (whole-branch review 2026-08-10) — a record whose wrapped body does
# not parse is dropped at conversion time, and the drop is stated
# ---------------------------------------------------------------------------


def test_a_record_whose_wrapped_body_does_not_parse_is_dropped_not_passed_through():
    # A line cut from inside a list comprehension's own `if` clause is a
    # real shape `candidates()` can produce (a real example: `action/
    # codec.py:49`'s `if not any(...)` inside `_dangling_search_markers`)
    # -- not a complete statement in isolation at ANY indent, so no filler
    # or re-indentation can make `stages.fixture_body`'s wrapper parse it.
    # Before I4, this reached `stage2_codecs` and scored as "result does
    # not parse as Python" -- indistinguishable from the model's own
    # failure.
    record = _record(
        name="dangling-inverted_condition-49",
        broken="        if any(start <= match.start() < end for start, end in spans)\n",
        fixed="        if not any(start <= match.start() < end for start, end in spans)\n",
    )
    result = fixtures_from_corpus([record])
    assert result.fixtures == ()
    assert len(result.dropped) == 1
    assert "dangling-inverted_condition-49" in result.dropped[0]
    assert "does not parse" in result.dropped[0]


def test_a_parsing_record_is_kept_alongside_a_dropped_one_in_the_same_call():
    good = _record(name="calc-dropped_return-2")
    bad = _record(
        name="dangling-inverted_condition-49",
        broken="        if any(start <= match.start() < end for start, end in spans)\n",
        fixed="        if not any(start <= match.start() < end for start, end in spans)\n",
    )
    result = fixtures_from_corpus([good, bad])
    assert [f.name for f in result.fixtures] == ["calc-dropped_return-2"]
    assert len(result.dropped) == 1
    assert "dangling-inverted_condition-49" in result.dropped[0]
