# tests/test_corpus_io.py
from __future__ import annotations

import dataclasses
import inspect
import json
import socket
from pathlib import Path

import pytest

from robigo.profile.corpus import candidates
from robigo.profile.corpus_io import CorpusRecord, read_corpus, read_corpus_baseline, write_corpus
from robigo.profile.verify import Baseline

# ---------------------------------------------------------------------------
# Offline guarantee: no test in this module ever needs a real socket --
# corpus_io.py touches only json and the filesystem. Blocking the real
# connect call makes that a structural guarantee, matching the convention
# `tests/test_corpus_verify.py` already established for this plan, rather
# than an assumption nothing here happens to violate today.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("socket.socket.connect must never be called in this test suite")

    monkeypatch.setattr(socket.socket, "connect", _blocked)


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


_BASE = Baseline(broken=0, executed=430, seconds=12.3)
"""A stand-in `Baseline` (I1, whole-branch review 2026-08-10) for every
test in this module that doesn't care about its specific values -- only
that `write_corpus` now requires one and `read_corpus_baseline` reads it
back unchanged."""


# ---------------------------------------------------------------------------
# CorpusRecord — frozen, and every field required (invariant 9)
# ---------------------------------------------------------------------------


def test_corpus_record_is_frozen():
    # Fails if `frozen=True` is dropped from the dataclass decorator.
    record = _record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.test_id = "tests/test_other.py::test_x"  # type: ignore[misc]


def test_no_corpus_record_field_carries_a_default():
    # Invariant 9, pinned at the dataclass level directly: fails for an
    # implementation that gives ANY of the ten fields a default -- not
    # just source_repo/source_sha -- which is the exact shape plan 03
    # shipped for `Profile.corpus = "fixtures-v1"` and is named in this
    # plan as debt not to repeat.
    for field in dataclasses.fields(CorpusRecord):
        assert field.default is dataclasses.MISSING, f"{field.name} has a default value"
        assert field.default_factory is dataclasses.MISSING, (
            f"{field.name} has a default_factory"
        )


def test_constructing_without_source_repo_and_source_sha_raises():
    # The behavioural half of invariant 9: a caller that forgets
    # provenance is refused outright by Python's own dataclass __init__,
    # not handed back a record that silently reads as from nowhere.
    with pytest.raises(TypeError):
        CorpusRecord(  # type: ignore[call-arg]
            name="r", path=Path("f.py"), line=1, broken="x\n", fixed="y\n",
            test_id="tests/test_f.py::test_g", diagnostic="reason",
            operator="off_by_one",
        )


def test_write_corpus_name_has_no_default():
    sig = inspect.signature(write_corpus)
    assert sig.parameters["name"].default is inspect.Parameter.empty


def test_write_corpus_dropped_has_no_default():
    sig = inspect.signature(write_corpus)
    assert sig.parameters["dropped"].default is inspect.Parameter.empty


def test_write_corpus_baseline_has_no_default():
    # I1 (whole-branch review 2026-08-10): the same "state it, don't let a
    # reader assume it" rule `name`/`dropped` already follow.
    sig = inspect.signature(write_corpus)
    assert sig.parameters["baseline"].default is inspect.Parameter.empty


def test_calling_write_corpus_without_name_or_dropped_raises(tmp_path: Path):
    # Fails if either keyword silently defaults instead of being required
    # -- proven behaviourally, not merely via the signature reflection
    # above (which could pass while the function body still tolerated a
    # missing call some other way).
    with pytest.raises(TypeError):
        write_corpus((), tmp_path / "corpus.json")  # type: ignore[call-arg]


def test_calling_write_corpus_without_baseline_raises(tmp_path: Path):
    with pytest.raises(TypeError):
        write_corpus(  # type: ignore[call-arg]
            (), tmp_path / "corpus.json", name="mylib-v1", dropped=()
        )


# ---------------------------------------------------------------------------
# Invariant 8 — the reverse patch is stored, not derived at read time
# ---------------------------------------------------------------------------


def _dropped_return_mutant():
    # The measured real-world shape from the task brief: "a dropped_return
    # on line 2" (e.g. tests/test_calc.py::test_total_sums).
    source = "def total(values):\n    return sum(values)\n"
    return next(
        m for m in candidates(source, Path("calc.py")) if m.operator == "dropped_return"
    )


def test_fixed_carries_the_mutants_real_original_line_grounded_in_a_real_mutant():
    """Builds `broken`/`fixed` from an actual `Mutant` that `robigo.
    profile.corpus.candidates()` produces, rather than two hand-written
    strings chosen to merely look plausible. Fails if `fixed`/`broken` are
    ever swapped, or if `fixed` is anything other than the mutant's own
    `original` byte-for-byte."""
    mutant = _dropped_return_mutant()
    assert mutant.line == 2
    record = _record(
        path=mutant.path, line=mutant.line,
        broken=mutant.mutated, fixed=mutant.original, operator=mutant.operator,
    )
    assert record.fixed == mutant.original == "    return sum(values)\n"
    assert record.broken == mutant.mutated == "    sum(values)\n"
    assert record.fixed != record.broken  # guards against a vacuous equal-strings test


def test_ground_truth_survives_the_full_round_trip_with_no_source_repo_involved(
    tmp_path: Path,
):
    # Invariant 8's actual claim, made concrete: a consumer checks ground
    # truth "without the source repo present". This test writes the
    # record, deletes every local reference to the mutant/source that
    # produced it, and re-derives the fixed line from disk alone.
    mutant = _dropped_return_mutant()
    record = _record(
        path=mutant.path, line=mutant.line,
        broken=mutant.mutated, fixed=mutant.original, operator=mutant.operator,
    )
    out = tmp_path / "corpus.json"
    write_corpus([record], out, name="mylib-v1", dropped=(), baseline=_BASE)
    del mutant, record  # simulate "without the source repo, or the mutant, present"

    _, records, _ = read_corpus(out)
    assert records[0].fixed == "    return sum(values)\n"
    assert records[0].broken == "    sum(values)\n"


# ---------------------------------------------------------------------------
# Invariant 10 — the round trip is checked structurally AND by equality
# ---------------------------------------------------------------------------


def test_round_trips_through_json_by_equality():
    original = _record()
    assert CorpusRecord.from_json(json.loads(original.to_json())) == original


def _collect_keys(obj: object) -> set[str]:
    """Every dict key present anywhere in a JSON-like structure, at any
    nesting depth -- same shape and purpose as `tests/test_profile_schema.
    py`'s own `_collect_keys`, kept as a private copy per this project's
    convention of each test file owning its own fixtures rather than a
    shared test-only import."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.add(key)
            keys |= _collect_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _collect_keys(item)
    return keys


def test_every_corpus_record_field_is_wired_into_the_json_payload():
    # The other half of invariant 10: catches a field that was never
    # serialised at all, which whole-object equality above cannot -- see
    # this module's docstring and the task report for the acceptance
    # demonstration (a field added to CorpusRecord with a default, wired
    # into neither to_json nor from_json, turns THIS test red while the
    # equality test above stays green).
    payload = json.loads(_record().to_json())
    keys = _collect_keys(payload)
    missing = [f.name for f in dataclasses.fields(CorpusRecord) if f.name not in keys]
    assert not missing, f"CorpusRecord field(s) missing from the JSON payload: {missing}"


def test_from_json_raises_for_a_payload_missing_a_required_field():
    # Fails for an implementation that fills a missing key with a default
    # or with None instead of refusing outright.
    payload = json.loads(_record().to_json())
    del payload["source_sha"]
    with pytest.raises(KeyError):
        CorpusRecord.from_json(payload)


def test_to_json_round_trips_every_field_with_distinct_non_default_looking_values():
    # Uses values that could not coincidentally match some hidden internal
    # default (unlike, say, an empty string or 0) -- fails for an
    # implementation that hardcodes or mismaps any one of the ten fields.
    record = CorpusRecord(
        name="zzz-unique-name",
        path=Path("src/widget/core.py"),
        line=42,
        broken="    return n * 3\n",
        fixed="    return n * 2\n",
        test_id="tests/test_core.py::test_double_not_triple",
        diagnostic="exactly one net new failure",
        operator="off_by_one",
        source_repo="git@example.com:someone/widget.git",
        source_sha="deadbeef" * 5,
    )
    reloaded = CorpusRecord.from_json(json.loads(record.to_json()))
    assert reloaded == record
    for field in dataclasses.fields(CorpusRecord):
        assert getattr(reloaded, field.name) == getattr(record, field.name)


# ---------------------------------------------------------------------------
# Invariant 11 — the corpus states what was dropped
# ---------------------------------------------------------------------------


def test_write_corpus_records_dropped_reasons_verbatim_and_in_order(tmp_path: Path):
    dropped = (
        "context/scope.py: target abandoned, keep rate 0/7",
        "loop.py:88 flipped_comparison: broke 10 tests, not exactly one",
        "context/scope.py: target abandoned, keep rate 0/7",  # duplicate, on purpose
    )
    out = tmp_path / "corpus.json"
    write_corpus([_record()], out, name="mylib-v1", dropped=dropped, baseline=_BASE)
    _, _, read_dropped = read_corpus(out)
    assert read_dropped == dropped  # order AND duplicates preserved -- not deduped/sorted


def test_an_empty_dropped_tuple_is_still_an_explicit_key_in_the_file(tmp_path: Path):
    # Falsifies "invariant 11 holds only when something WAS dropped" --
    # "nothing dropped" must be a stated fact in the file (an explicit
    # empty list), never an absent key a reader could mistake for "the
    # generator forgot to report what it dropped".
    out = tmp_path / "corpus.json"
    write_corpus([_record()], out, name="mylib-v1", dropped=(), baseline=_BASE)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "dropped" in payload
    assert payload["dropped"] == []
    assert read_corpus(out)[2] == ()


# ---------------------------------------------------------------------------
# Filesystem round trip — the file is the artifact (verification standard 4)
# ---------------------------------------------------------------------------


def test_write_then_read_corpus_round_trips_name_records_and_dropped(tmp_path: Path):
    records = (
        _record(name="calc-dropped_return-2", source_repo="/repo/a", source_sha="a" * 40),
        _record(
            name="other-off_by_one-5", path=Path("src/mylib/other.py"), line=5,
            broken="    return 6\n", fixed="    return 5\n", operator="off_by_one",
            test_id="tests/test_other.py::test_boundary",
            diagnostic="exactly one net new failure",
            source_repo="/repo/a", source_sha="a" * 40,
        ),
    )
    dropped = ("candidate at other.py:9: broke 3 tests, not exactly one",)
    out = tmp_path / "nested" / "corpus.json"  # parent does not exist yet

    write_corpus(records, out, name="mylib-v1", dropped=dropped, baseline=_BASE)
    name, read_records, read_dropped = read_corpus(out)

    assert name == "mylib-v1"
    assert read_records == records  # order preserved, dataclass equality
    assert read_dropped == dropped
    assert isinstance(read_records[0].path, Path)  # not left as a bare str
    assert read_records[0].path == records[0].path
    assert read_records[1].operator == "off_by_one"  # operator genuinely varies


def test_write_corpus_creates_missing_parent_directories(tmp_path: Path):
    out = tmp_path / "a" / "b" / "c" / "corpus.json"
    write_corpus([_record()], out, name="mylib-v1", dropped=(), baseline=_BASE)
    assert out.is_file()


def test_read_corpus_on_zero_records_returns_an_empty_tuple_not_none(tmp_path: Path):
    out = tmp_path / "corpus.json"
    write_corpus([], out, name="empty-v1", dropped=("target x: barren, keep rate 0/12",), baseline=_BASE)
    name, records, dropped = read_corpus(out)
    assert name == "empty-v1"
    assert records == ()
    assert dropped == ("target x: barren, keep rate 0/12",)


def test_write_corpus_never_touches_a_path_other_than_its_own_output(tmp_path: Path):
    # Constraints: "this task writes only corpus files, never source" --
    # fails if write_corpus is ever changed to touch anything in the
    # directory it writes into beyond creating it and its own output file.
    sentinel = tmp_path / "do-not-touch.py"
    sentinel.write_text("ORIGINAL\n", encoding="utf-8")
    write_corpus([_record()], tmp_path / "corpus.json", name="mylib-v1", dropped=(), baseline=_BASE)
    assert sentinel.read_text(encoding="utf-8") == "ORIGINAL\n"


def test_written_file_is_valid_json_with_the_four_top_level_keys(tmp_path: Path):
    out = tmp_path / "corpus.json"
    write_corpus([_record()], out, name="mylib-v1", dropped=("x",), baseline=_BASE)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) == {"name", "records", "dropped", "baseline"}
    assert payload["name"] == "mylib-v1"
    assert isinstance(payload["records"], list) and len(payload["records"]) == 1


# ---------------------------------------------------------------------------
# I1 (whole-branch review 2026-08-10) — the baseline is written into the
# corpus file, and readable back without re-running the generator
# ---------------------------------------------------------------------------


def test_baseline_is_wired_into_the_json_payload_with_all_three_fields(tmp_path: Path):
    # The structural half: a reader inspecting the raw JSON file (never
    # re-running the generator, never trusting its process memory) can see
    # broken/executed/seconds directly.
    out = tmp_path / "corpus.json"
    write_corpus([_record()], out, name="mylib-v1", dropped=(), baseline=_BASE)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["baseline"] == {"broken": 0, "executed": 430, "seconds": 12.3}


def test_read_corpus_baseline_round_trips_the_exact_baseline_written(tmp_path: Path):
    out = tmp_path / "corpus.json"
    distinct = Baseline(broken=0, executed=987, seconds=42.5)  # values that could not
    # coincidentally match some hidden internal default (unlike 0/0.0).
    write_corpus([_record()], out, name="mylib-v1", dropped=(), baseline=distinct)
    assert read_corpus_baseline(out) == distinct


def test_read_corpus_baseline_raises_for_a_file_written_before_this_field_existed(
    tmp_path: Path,
):
    # No default is fabricated for a fact an older file never recorded --
    # refusing is the honest answer, same principle as `from_json` raising
    # `KeyError` for a payload missing a required `CorpusRecord` field.
    out = tmp_path / "old-corpus.json"
    out.write_text(
        json.dumps({"name": "mylib-v1", "records": [], "dropped": []}), encoding="utf-8"
    )
    with pytest.raises(KeyError):
        read_corpus_baseline(out)
