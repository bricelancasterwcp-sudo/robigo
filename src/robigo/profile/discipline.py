# src/robigo/profile/discipline.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from robigo.profile.repair import Attempt


@dataclass(frozen=True)
class Stage5:
    """Spec 5's stage 5: two diagnostics derived ENTIRELY from stage 4's
    already-run attempts, at zero additional model calls (invariant 7.1 --
    see `stage5_discipline`'s own docstring for how that guarantee is made
    true by construction rather than by convention). Where stage 4 answers
    "how often did the tool repair the defect", stage 5 answers a
    different question about the SAME ~940 attempts: when it repaired,
    how long did that take, and when it did not, how much of the failure
    was the model spinning on a patch it had already tried and already
    seen rejected -- a pattern the loop's own transcript can diagnose
    without another token spent.

    That distinction is the whole point of this stage. A bare repair rate
    below the project's 33.3% kill criterion (spec 0.2/1.4) is consistent
    with two very different failures: a model that genuinely cannot
    produce a correct edit for the defect it is shown (a CAPABILITY
    result -- no amount of loop tuning fixes it), or a model that
    routinely produces one, sees it rejected, and re-emits it, or a close
    variant, instead of trying something else (an INSTRUMENT result --
    fixable by tuning the loop's feedback, stall handling, or prompt,
    without touching the model at all). `turns_to_green_median` and
    `repeat_rate` are the two numbers that tell those apart: a low median
    with a low repeat rate says the model mostly gets there fast when it
    gets there, and failures are elsewhere; a high repeat rate says a
    meaningful share of the FAILING attempts were the model stuck
    re-trying its own already-rejected patch, which is exactly the shape
    `robigo.loop.RunResult.repeats` (Task 6) was added to measure, because
    the loop's pre-existing `stalls` counter -- a CONSECUTIVE streak that
    resets to 0 on any non-repeat -- cannot answer it (a run cycling
    through several distinct wrong patches, repeating each one, never
    trips `stall_cap` and looks identical to a run that never repeated at
    all, unless every repeat across the whole run is counted, not just an
    unbroken run of them).

    Both fields are `float | None`, never a bare `float` defaulting to
    `0.0`, when nothing was observed (invariant 7.2). This is the same
    collapse `robigo.profile.repair.Stage4.rate` already guards against
    for the identical reason (see that field's own docstring), and the
    same collapse this codebase has ALREADY shipped once and recorded as
    a defect: `robigo.profile.schema.CodecResult.max_file_tokens` conflated
    "no landed attempt was large enough to measure a ceiling from" with
    "the ceiling measured is zero", and a reader comparing either field
    against a threshold cannot tell "never measured" from "measured and
    it was zero" unless the type itself keeps them apart. A repair run
    with zero passing attempts has NO passing attempt's turn count to take
    a median of -- `turns_to_green_median: 0.0` there would read as "the
    model reaches green in zero turns", the opposite of the truth, not as
    "there is nothing to report". The same applies to `repeat_rate` when
    zero turns were ever scored at all (an empty attempt list, or every
    attempt excluded): `0.0` would read as "measured, and the model never
    repeated", not as "there was nothing to measure a rate over"."""

    turns_to_green_median: float | None
    """The median `turns` among SCORED attempts that actually passed
    (`a.passed and a.excluded is None`) -- median, not mean, because nine
    attempts landing in one turn and one landing at the turn cap is a
    genuinely different distribution from ten attempts landing at turn
    five, and a mean collapses that shape into one number a median does
    not. `None` when no scored attempt passed -- there is no "turns to
    green" to speak of when the run never went green (invariant 7.2)."""

    repeat_rate: float | None
    """`sum(repeats) / sum(turns)` across every SCORED attempt (`excluded
    is None`), passing or not -- deliberately turn-weighted, not attempt-
    weighted (`mean(a.repeats / a.turns for a in scored)` would let a
    single 2-turn attempt with one repeat (rate 0.5) outweigh a 40-turn
    attempt with three repeats (rate 0.075) by the same margin as if they
    had run the same number of turns, which they did not). `None` only
    when the denominator itself is zero -- no scored attempt ever spent a
    turn at all -- not when the numerator is zero: a run with real turns
    and genuinely no repeats reports `0.0`, a real, measured fact distinct
    from "nothing to measure" (invariant 7.2)."""


def stage5_discipline(attempts: Sequence[Attempt]) -> Stage5:
    """Reduce stage 4's `Attempt`s into `Stage5`. Takes an `Attempt`
    sequence and NOTHING else -- no `client`, no `ModelClient` Protocol
    object, not even accepted as an unused parameter -- which is what
    makes invariant 7.1 ("stage 5 must never cause a model call") true BY
    CONSTRUCTION rather than by an internal promise this function could
    quietly break later: a function with no reference to a client has no
    way to call `.generate` on one, so there is no line of code review, no
    future edit, and no test to write that could reintroduce a model call
    here without ALSO changing this signature -- which is the strongest
    form the guarantee can take, stronger than an assertion or a comment
    that merely states the constraint and trusts the next edit to respect
    it. `test_stage5_never_calls_the_client` (Task 6) is a falsification
    test for the record, not the mechanism the guarantee actually rests on.

    `attempts` is expected to be `Stage4.all_attempts` -- every `Attempt`
    stage 4 produced, scored and excluded alike, exactly as that field's
    own docstring anticipates (it names Task 6's `stage5_discipline` by
    name as the reason it carries the excluded ones at all, even though
    stage 4 itself has no use for them beyond `dropped`'s summary).

    Excluded attempts (`a.excluded is not None`) never gave the model a
    fair chance (spec 4.3.4) -- a broken clone, a suite that would not
    run, a corpus record whose anchor file was missing from this checkout
    -- and are filtered out FIRST, before either metric is computed, so
    they contribute to neither the median's population nor the repeat
    rate's numerator or denominator. This mirrors `Stage4.rate`'s own
    filter exactly (`robigo.profile.repair.stage4_repair`) rather than
    re-deriving a parallel notion of "counts" that could drift from it."""
    scored = [a for a in attempts if a.excluded is None]
    green = [a.turns for a in scored if a.passed]
    turns = sum(a.turns for a in scored)
    return Stage5(
        turns_to_green_median=float(median(green)) if green else None,
        repeat_rate=(sum(a.repeats for a in scored) / turns) if turns else None,
    )
