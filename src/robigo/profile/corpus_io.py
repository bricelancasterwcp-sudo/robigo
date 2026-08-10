# src/robigo/profile/corpus_io.py
"""The corpus record: the on-disk artifact `robigo corpus` (task 4) will
emit and stage 4 will read. Everything upstream of this module -- task 1's
`candidates()`, task 2's `verify()` -- exists only to produce the values a
`CorpusRecord` carries; this module is the last place those values are
still trustworthy before they become a file some completely different
process, quite possibly a completely different machine, has to take on
faith. Stage 4's number, and this project's 40% kill criterion, is read
off runs against these records -- a record that misrepresents itself
misrepresents the whole result.

Four invariants, all about what a record and a corpus FILE must state
rather than let a reader assume:

  8.  `fixed` carries the reverse patch -- the mutant's own ORIGINAL line
      -- so a consumer checks ground truth by string comparison alone,
      never by re-running `robigo.profile.corpus.reverse` and never with
      the source repo present. `broken` is the paired half: the mutant's
      own MUTATED line, the form the corpus's "broken" file actually
      contains at `line`. Both are stored verbatim, in full (including
      each line's own line ending, the same convention `Mutant.original`/
      `Mutant.mutated` already use) -- nothing here is a diff fragment or
      a column span a reader would have to re-apply against source that
      might not even be on disk anymore.
  9.  `source_repo`/`source_sha` are required fields with no default --
      every `CorpusRecord` names exactly which repo, at exactly which
      commit, the mutant was cut from. Plan 03 shipped `Profile.corpus` as
      a kwarg DEFAULT (`"fixtures-v1"`, `robigo/profile/report.py`) that
      would have mislabelled every profile it produced once a real corpus
      replaced the fixtures -- carried debt this plan exists partly to
      stop repeating. Neither `CorpusRecord`'s ten fields nor
      `write_corpus`'s `name`/`dropped`/`baseline` keywords carry a
      default anywhere in this module.
  10. The round trip is checked two ways, because the two catch different
      defects. `from_json(json.loads(to_json())) == original` (whole-
      object equality) catches a VALUE that changed shape in transit. It
      CANNOT catch a field that was never wired into the JSON at all: a
      field added later with a default lets both the original and the
      reloaded copy receive that same default and compare equal, even
      though the file itself never mentioned it anywhere -- plan 03
      proved this concretely by adding exactly such a field to `Profile`
      and watching all 18 of that module's tests stay green. The second
      check walks the actual serialised payload and demands every
      dataclass field name appear in it somewhere, independent of value.
  11. `write_corpus` takes `dropped` as a required keyword (no default),
      and every corpus FILE on disk carries it under a `"dropped"` key
      next to `"records"` -- what a generator rejected, what target it
      abandoned as barren, what a time budget cut short: all part of the
      corpus's own shape on disk, not a fact that lived only in a log the
      file's own later reader will never see. `dropped=()` -- "nothing was
      dropped" -- is a real, permitted answer, but it must be SAID (an
      explicit empty list in the file), never omitted by leaving the
      keyword unpassed, which is exactly why it has no default either.

`CorpusRecord` is a flat, frozen value. It does not import or construct
`robigo.profile.corpus.Mutant` or `robigo.profile.verify.Verdict` itself --
deciding WHEN a candidate has earned a place in the corpus, and wiring a
kept `Mutant`'s `mutated`/`original`/`line`/`operator` and that mutant's
`Verdict.test_id`/`Verdict.reason` into a record's fields, is the
generator's job (task 4's `robigo corpus`). This module only ever decides
how a record that already exists is written, read, and proven to have
survived the trip -- and it writes only corpus files, never source: no
function here ever opens a path that isn't the corpus file itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from robigo.profile.verify import Baseline


@dataclass(frozen=True)
class CorpusRecord:
    """One verified mutant, ground truth and provenance both included.
    Every field is required -- none of the ten carries a default -- so a
    caller can never construct a record that is silently missing its
    provenance the way plan 03's `Profile.corpus` kwarg default could
    (invariant 9); omitting any of them at construction raises `TypeError`
    rather than filling a plausible-looking placeholder in.

    `broken` and `fixed` are the mutant's two complete line texts, each
    including its own line ending exactly as `robigo.profile.corpus.
    Mutant` produced it (`splitlines(keepends=True)`'s convention).
    `broken` is the MUTATED line -- what the corpus's presented, defective
    file actually reads at `line` -- and `fixed` is the ORIGINAL line,
    stored so a consumer can check a proposed repair by direct string
    comparison, without re-deriving it from `broken` and without the
    source repo present at all (invariant 8).

    `test_id` and `diagnostic` come from the `Verdict` (task 2) that kept
    this mutant: `test_id` is the one test `verify()` isolated by id, and
    `diagnostic` is `Verdict.reason` -- the verifier's own account of why
    this candidate was kept (`"exactly one net new failure"`). That is the
    only descriptive text task 2's interface actually produces:
    `pytest_runner` runs with `--tb=no`, so there is no captured traceback
    text anywhere upstream of this record to carry instead.

    `source_repo`/`source_sha` pin the exact code this mutant was cut from
    (invariant 9): `source_repo` is whatever identifies the repo to the
    caller (a filesystem path or a URL; this module does not constrain the
    shape), and `source_sha` is the commit the clone was checked out at
    when `line` was read from it.
    """

    name: str
    path: Path
    line: int
    broken: str
    fixed: str
    test_id: str
    diagnostic: str
    operator: str
    source_repo: str
    source_sha: str

    def to_json(self) -> str:
        """The record's own JSON text. `json.loads(to_json())` is exactly
        what `from_json` expects back -- invariant 10's equality half --
        and `_payload` is the single mapping from field to JSON key this
        method shares with `write_corpus`, so a corpus file's `"records"`
        entries and a lone record's own `to_json()` can never describe one
        field two different ways."""
        return json.dumps(_payload(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: dict) -> CorpusRecord:
        """The inverse of `_payload` (via `to_json`'s `json.dumps`).
        Raises `KeyError` if `payload` is missing any of the ten fields --
        this never fills a gap with a guessed or default value, which is
        exactly the failure mode invariant 10's structural test exists to
        catch before a silent default could paper over it here."""
        return cls(
            name=payload["name"],
            path=Path(payload["path"]),
            line=payload["line"],
            broken=payload["broken"],
            fixed=payload["fixed"],
            test_id=payload["test_id"],
            diagnostic=payload["diagnostic"],
            operator=payload["operator"],
            source_repo=payload["source_repo"],
            source_sha=payload["source_sha"],
        )


def _payload(record: CorpusRecord) -> dict[str, object]:
    """The one place a `CorpusRecord`'s ten fields become JSON keys.
    `to_json` wraps this in `json.dumps` for a single record; `write_corpus`
    embeds it directly inside a corpus file's `"records"` array for many --
    so a multi-record file is never built by parsing each record's own
    `to_json()` string back out again, and there is exactly one
    field-to-key mapping for a structural test to walk (invariant 10)."""
    return {
        "name": record.name,
        "path": str(record.path),
        "line": record.line,
        "broken": record.broken,
        "fixed": record.fixed,
        "test_id": record.test_id,
        "diagnostic": record.diagnostic,
        "operator": record.operator,
        "source_repo": record.source_repo,
        "source_sha": record.source_sha,
    }


def _baseline_payload(baseline: Baseline) -> dict[str, object]:
    """`Baseline`'s three fields as JSON -- the one mapping `write_corpus`
    and `read_corpus_baseline` both use, so the two can never describe the
    shape two different ways."""
    return {
        "broken": baseline.broken,
        "executed": baseline.executed,
        "seconds": baseline.seconds,
    }


def write_corpus(
    records: Sequence[CorpusRecord],
    path: Path,
    *,
    name: str,
    dropped: Sequence[str],
    baseline: Baseline,
) -> None:
    """Writes one corpus file to `path`: `name` (the corpus's own identity
    -- a required keyword with no default, so this is the one call site
    that actually produces a corpus's name and it cannot repeat plan 03's
    `corpus="fixtures-v1"` mistake), every record in `records` (each via
    `_payload`, in the order given), `dropped` (invariant 11 -- every
    candidate this corpus's generator rejected, every target it abandoned,
    every record a time budget cut, as free-form strings; also a required
    keyword with no default, so "nothing was dropped" must be said with an
    explicit `dropped=()` rather than by leaving the argument out), and
    `baseline` (I1, whole-branch review 2026-08-10: every kept record's
    ground truth is a mutant that broke a `baseline.broken == 0` run --
    `verify()` now REQUIRES this before a keep, but until this field
    existed a reader of the corpus file had no way to check that half of
    spec 5.1's rule was ever satisfied, only the generator's own process
    memory did. Also a required keyword with no default -- the same
    "state it, don't let a reader assume it" reasoning `dropped` already
    follows).

    Creates `path`'s parent directories if needed, but touches nothing
    else on disk -- this writes a corpus file, never source (constraints:
    "this task writes only corpus files, never source")."""
    payload = {
        "name": name,
        "records": [_payload(record) for record in records],
        "dropped": list(dropped),
        "baseline": _baseline_payload(baseline),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_corpus(path: Path) -> tuple[str, tuple[CorpusRecord, ...], tuple[str, ...]]:
    """The inverse of `write_corpus`, for the three fields every existing
    caller already unpacks: the corpus's name, every record (via
    `CorpusRecord.from_json`, in the file's own order), and everything
    `"dropped"` named -- all three read back off the filesystem, not
    reconstructed from an in-memory value (verification standard item 4).

    Does NOT also return `baseline` -- growing this function's return
    tuple to four elements would break every existing 3-tuple unpacking
    call site over a fact (I1's baseline) that has nothing to do with any
    of them. `read_corpus_baseline`, below, is the dedicated accessor."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = tuple(CorpusRecord.from_json(entry) for entry in payload["records"])
    return payload["name"], records, tuple(payload["dropped"])


def read_corpus_baseline(path: Path) -> Baseline:
    """The `Baseline` `write_corpus` recorded alongside this corpus's
    records (I1, whole-branch review 2026-08-10) -- a reader can now check
    that a kept record's reference patch was verified against a clean
    (`broken == 0`) run without re-running the generator or trusting its
    process memory, the exact gap I1 closes. Raises `KeyError` if `path`
    was written before this field existed (no default is fabricated for a
    fact this old a file never recorded)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["baseline"]
    return Baseline(
        broken=entry["broken"], executed=entry["executed"], seconds=entry["seconds"]
    )
