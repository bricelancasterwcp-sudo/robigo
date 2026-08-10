# src/robigo/profile/report.py
from __future__ import annotations

import os
from pathlib import Path

from robigo.model.client import ModelClient
from robigo.model.geometry import WindowPlan
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
    kv_bits: int = 16,
    corpus: str = "fixtures-v1",
) -> Profile:
    """Stages run cheapest-first and gate each other: a family that cannot
    reliably drive the action envelope (stage 1) never reaches the codec
    measurement (stage 2) -- there is nothing there to measure (spec 5).

    When the gate closes, `codecs` stays the empty dict `{}` it was
    initialised to, and `dropped` gains a line naming why. That empty dict
    must never be read as "stage 2 ran and nothing landed" -- it means
    stage 2 did not run at all, which is a different fact. Two guarantees
    make that distinction hold downstream, not just in this function:

    1. The gate here uses `ENVELOPE_FIDELITY_MIN`, the exact same constant
       `verdict_for` itself checks first (before it ever looks at
       `codecs`). So whenever this function skips stage 2, `verdict_for`
       independently also returns UNUSABLE from `stage1.fidelity` alone,
       never falling through to inspect the (empty) `codecs` it was
       skipped. A verdict of UNUSABLE is not just "codecs came back
       empty" -- it is "the envelope gate never opened", corroborated by
       `dropped` naming stage 2 explicitly, not inferred from an empty
       dict that a real, run stage 2 landing 0% would produce identically.
    2. `Profile.best_codec()` returns `None` on an empty `codecs`, not a
       fabricated "0% on every codec" entry -- so nothing downstream can
       quote a codec name for a family that was never measured against
       one.

    A family that clears the gate but still lands nothing under a real
    stage 2 run gets a *non-empty* `codecs` (every requested codec name
    present, each at `lands=0.0`) and no "stage 2" line in `dropped` --
    that is the "we tried and nothing worked" case, and it must stay
    visibly different from "we never tried" in the written profile.
    """
    dropped: list[str] = []
    stage0 = stage0_window(client, plan)
    if not stage0.verified:
        dropped.append(f"stage 0: {stage0.note}")
    # stage0.window is never larger than plan.window (Stage0's own scope
    # boundary), and it is 0 exactly when nothing was verified accepted --
    # falling back to the unverified plan.window here (the brief's own
    # sample did this) would report a hypothesis nothing ever confirmed as
    # though it were the measured usable_window, the same class of
    # overclaim `dropped` exists to prevent (task-6 brief amendment).
    window = stage0.window

    stage1 = stage1_envelope(client, seeds)

    codecs: dict[str, CodecResult] = {}
    if stage1.fidelity >= ENVELOPE_FIDELITY_MIN:
        codecs = stage2_codecs(client, seeds).results
    else:
        dropped.append(
            f"stage 2: not run, envelope fidelity {stage1.fidelity:.2f} "
            f"below {ENVELOPE_FIDELITY_MIN:.2f}"
        )

    return Profile(
        family=family, model=model, quant=quant,
        training_ctx=plan.window, kv_kib_per_token=plan.kv_per_token // 1024,
        kv_bits=kv_bits, usable_window=window, window_limited_by=plan.limited_by,
        envelope_level=stage1.level, envelope_fidelity=stage1.fidelity,
        codecs=codecs, payload_corruption=None, repeat_rate=None,
        verdict=verdict_for(stage1.fidelity, codecs, window),
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
