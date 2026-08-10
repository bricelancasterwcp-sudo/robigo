# src/robigo/profile/report.py
from __future__ import annotations

import os
from pathlib import Path

from robigo.model.client import ModelClient
from robigo.model.geometry import WindowPlan
from robigo.profile.fixtures import FIXTURES, Fixture
from robigo.profile.schema import ENVELOPE_FIDELITY_MIN, CodecResult, Profile, verdict_for
from robigo.profile.stages import stage0_window, stage1_envelope, stage2_codecs


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

    **`payload_corruption` and `repeat_rate` are always `None`** -- no
    stage this plan builds measures either one -- and `dropped` always
    names both, unconditionally, regardless of how far the run got
    (whole-branch review I4, ruled 2026-08-10: before this, both fields
    were hardcoded `None` with no `dropped` entry naming them, so a
    reader had no way to learn "not measured" from the written profile at
    all, violating "anything not measured is stated as dropped").
    """
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

    dropped.append(
        "payload_corruption: not measured (no stage in this plan measures it)"
    )
    dropped.append(
        "repeat_rate: not measured (no stage in this plan measures it)"
    )
    # Both of `--corpus`'s own loss channels (P1.2): unconditional, same as
    # payload_corruption/repeat_rate above -- a corpus's losses are a fact
    # about the CORPUS, not about which stage gate this run happened to
    # land on, so they belong in `dropped` even on a run that never
    # reached stage 2 at all.
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
        codecs=codecs, payload_corruption=None, repeat_rate=None,
        verdict=verdict_for(fidelity, codecs, window),
        seeds=seeds, mode=mode, corpus=corpus, dropped=tuple(dropped),
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
    lines.append(f"  verdict       {profile.verdict}")
    for note in profile.dropped:
        lines.append(f"  dropped       {note}")
    lines.append(
        f"  measured      {profile.seeds} seeds, {profile.mode} mode, "
        f"corpus {profile.corpus}"
    )
    if profile.mode != "full":
        lines.append("  NOTE          quick mode -- not publishable")
    return "\n".join(lines)
