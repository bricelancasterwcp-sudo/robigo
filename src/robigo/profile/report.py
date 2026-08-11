# src/robigo/profile/report.py
from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from robigo.model.client import ModelClient
from robigo.model.geometry import WindowPlan
from robigo.profile.corpus_io import CorpusRecord
from robigo.profile.discipline import stage5_discipline
from robigo.profile.fixtures import FIXTURES, Fixture
from robigo.profile.repair import stage4_repair
from robigo.profile.schema import (
    ENVELOPE_FIDELITY_MIN,
    CodecResult,
    Profile,
    select_best_codec,
    verdict_for,
)
from robigo.profile.stages import stage0_window, stage1_envelope, stage2_codecs
from robigo.profile.verify import Baseline


def profile_path(family: str) -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "robigo" / "profiles" / f"{family}.json"


def run_profile(
    client: ModelClient,
    plan: WindowPlan,
    *,
    model: str,
    quant: str,
    family: str,
    seeds: int,
    mode: str,
    corpus: str,
    kv_bits: int = 16,
    fixtures: tuple[Fixture, ...] = FIXTURES,
    corpus_dropped: tuple[str, ...] = (),
    repo: Path | None = None,
    records: Sequence[CorpusRecord] = (),
    corpus_baseline: Baseline | None = None,
    turn_cap: int = 8,
    python: str = sys.executable,
) -> Profile:
    """Stages run cheapest-first and gate each other, and each is able to
    STOP the run, not merely skip its own measurement (spec 5's
    architecture: "Three staged probes, cheapest first, each able to stop
    the run").

    `corpus` has no default (task 4, fixing carried debt from plan 03):
    the previous `corpus: str = "fixtures-v1"` kwarg default meant nothing
    tied that string to what was actually run, so the day a real corpus
    replaced the bundled fixtures, every profile produced here would have
    kept saying `fixtures-v1` unless every call site remembered to override
    it by hand. Every caller must now name the corpus explicitly --
    `cli.profile_main` passes `robigo.profile.fixtures.CORPUS_NAME`, the
    identity defined once, beside `FIXTURES` itself, not a second literal
    typed here.

    `fixtures` and `corpus_dropped` DO default (plan 05 task 2, P1.2),
    unlike `corpus` -- to `FIXTURES` and `()`, exactly the values every
    call site that predates `--corpus` already behaves as if it passed,
    so `test_committed_transcripts_replay.py`'s and `test_profile_report.
    py`'s existing calls (none of which name either keyword) keep running
    the bundled fixtures with nothing dropped, unchanged. `fixtures` is
    threaded into `stage2_codecs(client, seeds, fixtures=fixtures)` below
    -- stage 2 measures whichever set was handed to it, real corpus or
    bundled fixtures alike, with no branch here caring which. `dropped`
    is extended with every entry of `corpus_dropped` unconditionally,
    regardless of which stage gate this run lands on: a corpus's own
    losses are a fact about the CORPUS, not about how far stage 0/1
    happened to get, so they belong in every profile `--corpus` ever
    produces, not only the ones that reach stage 2.

    `corpus_dropped` is expected to carry BOTH loss channels
    `cli.profile_main` concatenates before calling here -- what
    `read_corpus`'s third return value reports (the GENERATOR's own
    drops: a target abandoned as barren, a candidate a time budget cut
    short) and what `fixtures_from_corpus` itself dropped as unwrappable
    (`FixturesFromCorpus.dropped`, I4: a mutant whose wrapped body is not
    valid Python at any indent -- measured at ~9.2% of real records, 91
    of 986 from `src/robigo`). Neither is a model failure, and neither may be
    excluded from `dropped` while still being excluded from `fixtures`:
    that would let a harness artifact quietly vanish from BOTH the
    denominator (it is not in `fixtures`, so stage 2 never scores it) and
    the written record (P1.2) -- exactly what this parameter exists to
    prevent, by giving every one of those records an explicit line here
    regardless of which stage gate this run lands on.

    **Stage 0 gates stages 1 and 2.** A family with no verified window has
    nothing for the envelope or codec probes to run against -- both would
    execute at `num_ctx: 0`, where the daemon substitutes its own default,
    which is not the window this profile claims to measure. Before this
    gate existed (whole-branch review I1, ruled 2026-08-10), a profile
    that verified NO window at all still ran stage 1 and stage 2 to
    completion, landing `envelope 100%` and `lands 100%` beside a headline
    `usable_window: 0` -- and the verdict read LIMITED, the same verdict a
    working 4096-token model gets, because `verdict_for` never saw the
    fact that nothing downstream was real. When this gate closes,
    `envelope_fidelity` stays `0.0` and `codecs` stays `{}`, so
    `verdict_for` -- which checks fidelity before it ever looks at
    `codecs` -- independently also returns UNUSABLE, and `dropped` names
    all three stages explicitly.

    **Stage 1 gates stage 2** (unchanged from before this fix): a family
    that cannot reliably drive the action envelope never reaches the
    codec measurement -- there is nothing there to measure. When THIS gate
    closes (stage 0 verified something, but stage 1's fidelity falls short),
    `codecs` stays `{}` and `dropped` gains a line naming why. That empty
    dict must never be read as "stage 2 ran and nothing landed" -- it means
    stage 2 did not run at all, which is a different fact. Two guarantees
    make that distinction hold downstream, not just in this function:

    1. The gate here uses `ENVELOPE_FIDELITY_MIN`, the exact same constant
       `verdict_for` itself checks first (before it ever looks at
       `codecs`). So whenever this function skips stage 2, `verdict_for`
       independently also returns UNUSABLE from fidelity alone, never
       falling through to inspect the (empty) `codecs` it was skipped.
    2. `Profile.best_codec()` returns `None` on an empty `codecs`, not a
       fabricated "0% on every codec" entry -- so nothing downstream can
       quote a codec name for a family that was never measured against
       one.

    A family that clears both gates but still lands nothing under a real
    stage 2 run gets a *non-empty* `codecs` (every requested codec name
    present, each at `lands=0.0`) and no "stage 2" line in `dropped` --
    that is the "we tried and nothing worked" case, and it must stay
    visibly different from "we never tried" in the written profile.

    **`payload_corruption` is always `None`** -- no stage this plan
    builds measures it -- and `dropped` always names it, unconditionally,
    regardless of how far the run got (whole-branch review I4, ruled
    2026-08-10: before this, the field was hardcoded `None` with no
    `dropped` entry naming it, so a reader had no way to learn "not
    measured" from the written profile at all, violating "anything not
    measured is stated as dropped"). Stage 3 (payload-corruption
    generation) is deferred to plan 06, which runs only once THIS plan's
    gate passes -- the `dropped` line names plan 06 explicitly, not just
    "not measured", so a reader can tell "deferred, pending the gate"
    apart from "abandoned".

    **`repeat_rate` (task 7, plan 05) is finally measured**, when stage 5
    runs: `discipline.repeat_rate` (`robigo.profile.discipline.
    stage5_discipline`) if stage 4/5 ran, `None` otherwise. Its `dropped`
    line is the mirror image of `payload_corruption`'s: unlike that field,
    which NO stage in this plan ever measures, `repeat_rate` sometimes IS
    measured, so the line stating "not measured" must disappear exactly
    when it stops being true (spec 8.1, task 7's own honesty rule 1) --
    leaving it unconditional would have this profile declare a field
    unmeasured in the same JSON payload that carries its measurement,
    which is a worse lie than never measuring it at all.

    **Stage 4 (repair) and stage 5 (loop discipline) run only when THREE
    things are all true**, checked in this order, each with its own
    specific `dropped` line naming which one closed the gate (never a
    single generic "stage 4 did not run"):

    1. `select_best_codec(codecs)` (`robigo.profile.schema`) returns a
       codec, not `None`. A `None` here already implies one of two very
       different upstream facts -- stage 0 or stage 1 never verified a
       usable window/envelope at all (`codecs` is `{}`, and the reason is
       already named above as "stage 0" or "stage 2"), or stage 2
       genuinely ran and every codec it tried landed 0% (spec 4.4's
       measured case: a family that clears every gate but never lands a
       single edit) -- and this is the only place in `dropped` that
       SECOND reason is ever recorded, since no earlier gate closed for
       it. `repair_rate` stays `None` here -- NOT MEASURED, not zero --
       because there is no codec to configure a repair loop around.
    2. `repo is not None`. Stage 4 breaks and patches a throwaway git
       clone (`robigo.profile.repair.stage4_repair`/`attempt_repair`); a
       corpus record alone names a defect, it is not a working tree to
       run one against. `run_profile` never clones anything itself --
       that is the caller's job, exactly as `cli.corpus_main` already
       owns cloning for stage 3 generation -- so `repo is None` (this
       function's default) is the ordinary, expected state for every
       caller that has not wired `--repo` through yet, not a failure.
    3. `records` is non-empty. There is nothing to repair without at
       least one verified mutant.
    4. `corpus_baseline is not None`. `attempt_repair`'s own judgement
       compares a repair run's executed-test total against
       `Baseline.executed` (a real single-test regression must not be
       confused with a collection error or an early exit) -- there is no
       safe baseline to assume on the caller's behalf, and guessing one
       (e.g. `Baseline(0, 0, 0.0)`) would silently make every repair
       attempt's suite-state comparison meaningless rather than honestly
       refusing to run at all.

    When all four hold, `stage4_repair(records, repo, client, seeds=seeds,
    codec=best, base=corpus_baseline, turn_cap=turn_cap, python=python)`
    runs the full (record, seed) grid against the shipped tool,
    `stage5_discipline(repair.all_attempts)` derives the two loop-discipline
    numbers from the SAME attempts at zero additional model calls
    (invariant 7.1), and `repair.dropped` (every excluded attempt, spec
    4.3.4) is folded into this function's own `dropped` list -- a
    corpus-and-repo's own losses
    belong in the written profile exactly as `corpus_dropped` already
    does for stage 2's.

    `stage4_repair` can raise `CorruptedCloneError`
    (`robigo.profile.repair`) when `repo` starts this process already
    checked out on a `robigo/*` branch -- a defect in the shared clone
    itself, identical for every attempt in the run, not a per-record
    surprise. This function does NOT catch it: no `try`/`except` wraps the
    `stage4_repair` call, on purpose, so the exception propagates all the
    way to whichever caller can actually fix `repo` (see that exception's
    own docstring for why swallowing it into an `excluded` `Attempt` would
    be silently wrong roughly 940 times over, once per attempt in the
    run, rather than failing loudly once).

    `python` (task 8, fix round 1) is the ONE interpreter `stage4_repair`
    -> `attempt_repair` uses for BOTH halves of every attempt -- the loop
    (`PythonAdapter`) and the judge (`pytest_runner`'s executed-total
    comparison against `corpus_baseline`). Defaults to `sys.executable`,
    not `PythonAdapter`'s own `.venv`/`venv`/`PATH` search, because
    `sys.executable` is what `robigo corpus` was itself running under when
    it measured `corpus_baseline` in the first place -- see
    `repair.attempt_repair`'s own docstring for the live-confirmed failure
    this prevents (every attempt against a fresh third-party clone
    excluded as "loop infrastructure" before the model was ever touched)
    and for why one shared value, not two independently-defaulted ones,
    is the actual property that matters."""
    dropped: list[str] = []
    stage0 = stage0_window(client, plan)
    fidelity = 0.0
    level = 0
    codecs: dict[str, CodecResult] = {}

    if not stage0.verified:
        dropped.append(f"stage 0: {stage0.note}")
        dropped.append(
            "stage 1: not run, stage 0 found no usable window -- there is "
            "no window to run the envelope probe against"
        )
        dropped.append(
            "stage 2: not run, stage 0 found no usable window"
        )
    else:
        stage1 = stage1_envelope(client, seeds)
        fidelity, level = stage1.fidelity, stage1.level
        if fidelity >= ENVELOPE_FIDELITY_MIN:
            codecs = stage2_codecs(client, seeds, fixtures=fixtures).results
        else:
            dropped.append(
                f"stage 2: not run, envelope fidelity {fidelity:.2f} "
                f"below {ENVELOPE_FIDELITY_MIN:.2f}"
            )

    # Stage 4/5 (plan 05 task 7): gated on THREE independent things, each
    # with its own `dropped` line naming which one closed the gate -- see
    # this function's own docstring for why each check exists and why the
    # ORDER matters (a `None` `best` already implies a specific upstream
    # reason worth stating precisely, not folding into one generic line).
    best = select_best_codec(codecs)
    repair = None
    discipline = None
    if best is None:
        # Deliberately does NOT contain the literal substrings "stage 0",
        # "stage 1" or "stage 2" -- other tests in this file (e.g.
        # test_a_stage_two_run_that_lands_nothing_differs_from_one_that_
        # never_ran) assert those substrings are ABSENT from `dropped`
        # for a run where that earlier stage genuinely executed, and this
        # line fires precisely in that case (every codec measured, all at
        # 0%). Naming the earlier stage here would make this line a false
        # positive for those assertions without changing what actually
        # happened.
        dropped.append(
            "stage 4: not run -- no codec ever landed a single edit (spec "
            "4.4). Either an earlier probe already explains why (see the "
            "notes above), or every codec that WAS actually measured "
            "landed on 0% of its attempts -- this is the only place THAT "
            "particular reason is recorded. repair_rate stays None -- NOT "
            "MEASURED, not zero -- because there is no codec here to "
            "configure a repair loop around."
        )
        dropped.append("stage 5: not run, stage 4 did not run")
    elif repo is None:
        dropped.append(
            "stage 4: not run, no --repo given -- a repair attempt needs "
            "a throwaway clone of a real working tree to break and patch; "
            "a corpus record alone names a defect, it is not one"
        )
        dropped.append("stage 5: not run, stage 4 did not run")
    elif not records:
        dropped.append(
            "stage 4: not run, no corpus records given -- there is "
            "nothing to repair without at least one verified mutant"
        )
        dropped.append("stage 5: not run, stage 4 did not run")
    elif corpus_baseline is None:
        dropped.append(
            "stage 4: not run, no corpus baseline given -- attempt_repair's "
            "own judgement compares a repair run's executed-test total "
            "against Baseline.executed, and there is no safe baseline to "
            "assume on the caller's behalf"
        )
        dropped.append("stage 5: not run, stage 4 did not run")
    else:
        repair = stage4_repair(
            records, repo, client, seeds=seeds, codec=best,
            base=corpus_baseline, turn_cap=turn_cap, python=python,
        )
        discipline = stage5_discipline(repair.all_attempts)
        dropped.extend(repair.dropped)

    dropped.append(
        "payload_corruption: not measured -- stage 3 (payload-corruption "
        "generation) is deferred to plan 06, which runs only once this "
        "plan's gate passes; no stage this plan builds measures it"
    )
    # repeat_rate's line, unlike payload_corruption's above, must disappear
    # exactly when it stops being true (spec 8.1, honesty rule 1): stage 5
    # sometimes DOES measure it, so an unconditional line here would have
    # this profile declare the field unmeasured in the same payload that
    # carries its real value.
    if discipline is None:
        dropped.append("repeat_rate: not measured (stage 5 did not run)")
    elif discipline.repeat_rate is None:
        dropped.append(
            "repeat_rate: not measured (stage 5 ran, but no scored "
            "attempt spent a single turn)"
        )
    # Both of `--corpus`'s own loss channels (P1.2): unconditional, same as
    # payload_corruption above -- a corpus's losses are a fact about the
    # CORPUS, not about which stage gate this run happened to land on, so
    # they belong in `dropped` even on a run that never reached stage 2 at
    # all.
    dropped.extend(corpus_dropped)

    # stage0.window is never larger than plan.window (Stage0's own scope
    # boundary), and it is 0 exactly when nothing was verified accepted --
    # falling back to the unverified plan.window here (the brief's own
    # sample did this) would report a hypothesis nothing ever confirmed as
    # though it were the measured usable_window, the same class of
    # overclaim `dropped` exists to prevent (task-6 brief amendment).
    window = stage0.window

    return Profile(
        family=family, model=model, quant=quant,
        # The real training context (`Geometry.training_ctx`, threaded
        # through `WindowPlan` -- whole-branch review C3, ruled
        # 2026-08-10), NOT plan.window: plan.window is min(training_ctx,
        # vram, user_cap), so whenever vram or a user cap actually bound,
        # `plan.window` is that OTHER limit's number, not the model's
        # training context. Reading plan.window here used to make
        # `training_ctx == usable_window` whenever vram bound -- a state
        # no real model can be in, and exactly what the live granite run
        # this review measured against did (`training_ctx: 0`).
        training_ctx=plan.training_ctx, kv_kib_per_token=plan.kv_per_token // 1024,
        kv_bits=kv_bits, usable_window=window, window_limited_by=plan.limited_by,
        envelope_level=level, envelope_fidelity=fidelity,
        codecs=codecs, payload_corruption=None,
        # `repair`/`discipline` are None exactly when stage 4/5 did not
        # run (the gate above) -- `repair_rate`/`turns_to_green_median`
        # stay None in that case too (NOT MEASURED, not zero; spec 4.4),
        # and `repair_attempts`/`repair_records` fall back to 0, which
        # loses no information `repair_rate` does not already carry (see
        # that field's own docstring on Profile).
        repeat_rate=discipline.repeat_rate if discipline else None,
        repair_rate=repair.rate if repair else None,
        repair_attempts=repair.attempts if repair else 0,
        repair_records=repair.records if repair else 0,
        turns_to_green_median=discipline.turns_to_green_median if discipline else None,
        # verdict_for is UNCHANGED (honesty rule 3, task 7 brief): READY/
        # LIMITED/UNUSABLE describe instrument fitness -- can this family's
        # window/envelope/codecs even be measured -- which is a different
        # question from "did it clear the 33.3% repair-rate gate". A
        # future verdict tweak must not be able to silently move that gate.
        verdict=verdict_for(fidelity, codecs, window),
        seeds=seeds, mode=mode, corpus=corpus, python=python,
        dropped=tuple(dropped),
    )


def render_table(profile: Profile) -> str:
    lines = [
        f"{profile.model}",
        f"  window        {profile.usable_window} "
        f"(limited by {profile.window_limited_by}, "
        f"{profile.kv_kib_per_token} KiB/token)",
        f"  envelope      {profile.envelope_fidelity:.0%} "
        f"(level {profile.envelope_level})",
    ]
    for name, result in sorted(profile.codecs.items()):
        ceiling = (
            f"  files <= {result.max_file_tokens} tok"
            if result.max_file_tokens else ""
        )
        lines.append(
            f"  {name:<15} lands {result.lands:.0%} "
            f"of {result.attempts}{ceiling}"
        )
    if profile.repair_rate is None:
        # NOT MEASURED, not zero (spec 4.4) -- "0.0% of 0 attempts" would
        # read as a measured, failing result, not as "the gate was never
        # run against this family at all".
        lines.append("  repair        not measured")
    else:
        lines.append(
            f"  repair        {profile.repair_rate:.1%} of "
            f"{profile.repair_attempts} attempts over "
            f"{profile.repair_records} records"
        )
    # One row for both stage-5 numbers, each independently "not measured"
    # when None -- turns_to_green_median and repeat_rate can each be None
    # (invariant 7.2) while the other is a real value (a run that scored
    # attempts but none of them ever passed still spent real turns, so
    # repeat_rate can be a real number even when turns_to_green_median is
    # not, and the reverse never happens but is not assumed here either).
    turns_to_green = (
        "not measured" if profile.turns_to_green_median is None
        else f"{profile.turns_to_green_median:.1f} turns"
    )
    repeat_rate = (
        "not measured" if profile.repeat_rate is None
        else f"{profile.repeat_rate:.1%}"
    )
    lines.append(
        f"  discipline    turns to green {turns_to_green}, "
        f"repeat rate {repeat_rate}"
    )
    lines.append(f"  verdict       {profile.verdict}")
    for note in profile.dropped:
        lines.append(f"  dropped       {note}")
    lines.append(
        f"  measured      {profile.seeds} seeds, {profile.mode} mode, "
        f"corpus {profile.corpus}, python {profile.python}"
    )
    if profile.mode != "full":
        lines.append("  NOTE          quick mode -- not publishable")
    return "\n".join(lines)
