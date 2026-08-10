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

from robigo.profile.corpus_io import read_corpus, write_corpus
from robigo.profile.fixtures import CORPUS_NAME, FIXTURES, as_corpus_records


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
