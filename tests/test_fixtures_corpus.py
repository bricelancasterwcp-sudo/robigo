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
    write_corpus(as_corpus_records(), out, name=CORPUS_NAME, dropped=())
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
    assert fixtures_from_corpus(as_corpus_records()) == FIXTURES


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
    fixture = fixtures_from_corpus([record])[0]
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
    fixtures = fixtures_from_corpus(records)
    assert [f.name for f in fixtures] == ["a", "b", "c"]


def test_fixtures_from_corpus_on_no_records_returns_an_empty_tuple():
    assert fixtures_from_corpus(()) == ()


def test_fixture_filename_is_a_string_not_a_path_object():
    # Fixture.filename predates Path-typed fields (CorpusRecord.path IS a
    # Path) -- fails if the conversion leaks a Path where stages.py's own
    # `landing_prompt`/`_Good`-style test fakes expect a plain string to
    # substring-match against a prompt.
    fixture = fixtures_from_corpus([_record()])[0]
    assert isinstance(fixture.filename, str)
    assert fixture.filename == "src/mylib/calc.py"


def test_fixtures_from_corpus_round_trips_through_a_written_corpus_file(tmp_path: Path):
    # The full loop: write a corpus file (as robigo corpus would), read it
    # back, convert -- no in-memory record ever reused.
    records = (_record(),)
    out = tmp_path / "generated.json"
    write_corpus(records, out, name="mylib-v1", dropped=())
    _, read_records, _ = read_corpus(out)
    fixtures = fixtures_from_corpus(read_records)
    assert fixtures == (
        Fixture(
            name="calc-dropped_return-2", filename="src/mylib/calc.py",
            original="    sum(values)\n", expect="    return sum(values)\n",
        ),
    )
