# src/robigo/profile/generate.py
"""The corpus generator's orchestration: turns a repo's real source into a
`CorpusRecord` sequence by walking `robigo.profile.corpus.candidates()` per
target and deciding each one with `robigo.profile.verify.verify()` (plan 04,
task 4). Everything upstream (tasks 1-3) already decided HOW a candidate is
proposed and HOW a verdict is judged; this module decides WHICH candidates
get tried, in what order, and when to stop -- the only remaining question
plan 04 leaves open.

`runner` is injected, exactly like `robigo.profile.verify`'s own functions
-- this module never shells out itself, and its own tests are offline and
instant. `cli.corpus_main` is the one production caller: it clones the
target repo, proves the harness with `sentinel_ok`, measures a `Baseline`,
and passes the real `pytest_runner` in here.

Three stopping rules, all of which end the run with something USABLE
rather than an unbounded grind (invariant 13, invariant 14):

  - **A barren target is abandoned**, not exhausted candidate-by-candidate.
    Measured 2026-08-10: mutating `context/scope.py` kept 0 of 7 -- a
    small file that happened to be fully explorable in 7 tries. A hot
    file can offer far more candidates than that (`robigo.profile.verify`'s
    own `_SENTINEL_SEARCH_LIMIT` docstring cites `loop.py` alone yielding
    125), so `_TARGET_ABANDON_AFTER` bounds how many UNPRODUCTIVE
    (`kept == 0`) attempts one target gets before this module moves on,
    rather than grinding through all 125 for a target that clearly is not
    producing.
  - **A global record cap** (`max_records`) stops generation the moment
    enough kept records exist, mid-target if need be -- a corpus does not
    need to be exhaustive to be useful, and every verification costs a
    real subprocess run (measured: ~15s on robigo's own suite).
  - **A global wall-clock budget** (`time_budget`) stops generation the
    moment it is exceeded, wherever it happens to be -- the one hard
    guarantee that a barren repo cannot turn this into an unbounded run.

Whatever gets cut short by any of the three is named in `GenerationResult.
dropped`, never silently absent (this plan's own "Global Constraints":
"Anything dropped is stated as dropped")."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from robigo.profile.corpus import Mutant, candidates
from robigo.profile.corpus_io import CorpusRecord
from robigo.profile.verify import Baseline, Runner, verify

_TARGET_ABANDON_AFTER = 10
"""How many UNPRODUCTIVE (net zero kept) attempts one target gets before
this module abandons it as barren and moves on (invariant 13). Chosen
above the measured `context/scope.py` sample size (7) so that exact,
already-measured result -- "0 of 7", every one of the seven genuinely
tried, none cut short -- stays reproducible rather than getting truncated
by this module's own abandonment logic before it ever gets there."""


@dataclass(frozen=True)
class TargetOutcome:
    """One target's own keep rate (invariant 13: "the keep rate is
    reported per target"). `proposed` is `len(candidates(source, path))`
    -- every candidate this target COULD have offered, computed once
    regardless of how many were actually tried. `tried` is how many were
    actually run through `verify()`, which can be less than `proposed`
    (abandoned as barren, or generation stopped globally on records/time).
    `abandoned` is True only for the EARLY-STOP case (`_TARGET_ABANDON_
    AFTER` unproductive attempts hit before `proposed` was exhausted) --
    a target tried to completion at 0 kept (like the measured `context/
    scope.py` result) is NOT `abandoned`; it was fully explored and turned
    out barren, a different fact from being cut off early."""

    path: Path
    proposed: int
    tried: int
    kept: int
    abandoned: bool


@dataclass(frozen=True)
class GenerationResult:
    """The whole run's result: every kept record, every dropped reason
    (candidates that failed `verify()`'s exactly-one rule, targets
    abandoned as barren, and a note naming whichever global stop --
    `max_records` or `time_budget` -- cut generation short, if either
    did), the per-target breakdown, and the real wall-clock this module
    itself measured (never parsed out of a runner's own text, the same
    honesty rule `robigo.profile.verify.Baseline.seconds` follows)."""

    records: tuple[CorpusRecord, ...]
    dropped: tuple[str, ...]
    targets: tuple[TargetOutcome, ...]
    seconds: float


def _record_name(target: Path, mutant: Mutant) -> str:
    """A readable, mostly-unique identifier for one kept record --
    `<file-stem>-<operator>-<line>` (e.g. `scope-off_by_one-42`). Not
    guaranteed globally unique (two different lines of the same file
    could share an operator and, in principle, a stem collision across
    directories is possible), but `path`+`line` together already are, and
    `name` exists for a human skimming a corpus file, not as a key
    anything in this plan looks up by."""
    return f"{target.stem}-{mutant.operator}-{mutant.line}"


def generate_corpus(
    repo: Path,
    targets: Sequence[Path],
    base: Baseline,
    runner: Runner,
    *,
    max_records: int,
    time_budget: float,
    source_repo: str,
    source_sha: str,
) -> GenerationResult:
    """Walks every target in order, proposing candidates via `candidates()`
    and deciding each with `verify()`, until every target is exhausted or
    one of the three stopping rules (this module's docstring) fires.

    `repo` must be the ISOLATED CLONE `verify()` itself requires -- this
    function applies real mutations to real files under `repo` (through
    `verify`) and restores them afterward, but never touches anything
    outside `repo`; it is the caller's job (`cli.corpus_main`) to make
    sure `repo` is a throwaway clone, never the working tree.

    Every `Path` in `targets` must already be relative to `repo` -- the
    same requirement `verify()` itself enforces (raises `ValueError`
    otherwise, via `verify`'s own `_resolve_in_clone`). Converting a
    user-supplied `--target` that might be absolute is the CALLER's job
    (`cli._resolve_targets`) -- this function assumes that conversion has
    already happened, exactly once, rather than re-doing it here and
    risking two implementations of "relative to what" drifting apart."""
    start = time.monotonic()
    records: list[CorpusRecord] = []
    dropped: list[str] = []
    outcomes: list[TargetOutcome] = []
    stopped_for_records = False
    stopped_for_time = False

    for target in targets:
        if stopped_for_records or stopped_for_time:
            break
        try:
            source = (repo / target).read_text()
        except (OSError, UnicodeDecodeError) as exc:
            dropped.append(f"{target}: could not read source ({exc}), skipped")
            outcomes.append(TargetOutcome(target, 0, 0, 0, False))
            continue

        proposed = candidates(source, target)
        tried = 0
        kept = 0
        abandoned = False

        for mutant in proposed:
            if len(records) >= max_records:
                stopped_for_records = True
                break
            if time.monotonic() - start >= time_budget:
                stopped_for_time = True
                break

            verdict = verify(mutant, repo, base, runner)
            tried += 1
            if verdict.kept:
                kept += 1
                records.append(
                    CorpusRecord(
                        name=_record_name(target, mutant),
                        path=mutant.path,
                        line=mutant.line,
                        broken=mutant.mutated,
                        fixed=mutant.original,
                        test_id=verdict.test_id or "",
                        diagnostic=verdict.reason,
                        operator=mutant.operator,
                        source_repo=source_repo,
                        source_sha=source_sha,
                    )
                )
            else:
                dropped.append(f"{target}:{mutant.line} {mutant.operator}: {verdict.reason}")

            if kept == 0 and tried >= _TARGET_ABANDON_AFTER:
                abandoned = True
                break

        if abandoned:
            dropped.append(
                f"{target}: abandoned as barren after {tried} unproductive "
                f"candidates (kept 0 of {tried}); {len(proposed) - tried} "
                f"candidate(s) not attempted"
            )
        outcomes.append(TargetOutcome(target, len(proposed), tried, kept, abandoned))

    if stopped_for_records:
        dropped.append(
            f"max_records ({max_records}) reached; remaining targets and "
            f"candidates not attempted"
        )
    if stopped_for_time:
        dropped.append(
            f"time budget ({time_budget:.0f}s) exceeded; remaining targets "
            f"and candidates not attempted"
        )

    return GenerationResult(
        records=tuple(records),
        dropped=tuple(dropped),
        targets=tuple(outcomes),
        seconds=time.monotonic() - start,
    )


def render_report(result: GenerationResult, *, name: str) -> str:
    """The human-readable account invariant 13/14 require: candidates
    proposed/tried/kept/rejected, the real wall-clock, the keep rate PER
    TARGET (never pooled into one number -- a 4-of-5 target and a 0-of-7
    target averaged together would hide the exact finding invariant 13
    exists to surface), and every dropped reason verbatim."""
    total_proposed = sum(t.proposed for t in result.targets)
    total_tried = sum(t.tried for t in result.targets)
    total_kept = sum(t.kept for t in result.targets)
    lines = [
        f"corpus {name}",
        f"  candidates    proposed {total_proposed}  tried {total_tried}  "
        f"kept {total_kept}  rejected {total_tried - total_kept}",
        f"  wall-clock    {result.seconds:.1f}s",
    ]
    for outcome in result.targets:
        note = "  (abandoned as barren)" if outcome.abandoned else ""
        lines.append(
            f"  target {outcome.path}  {outcome.kept}/{outcome.tried} kept{note}"
        )
    for note in result.dropped:
        lines.append(f"  dropped       {note}")
    return "\n".join(lines)
