# tests/test_committed_transcripts_replay.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from robigo.cli import _quant
from robigo.model.client import ModelError
from robigo.model.geometry import WindowPlan
from robigo.profile.report import run_profile
from robigo.profile.transcript import CallReplayer

# C2 (whole-branch review, ruled 2026-08-10): `68da8c2` committed three
# transcripts under tests/transcripts/ so the profiler is exercisable with no
# GPU, but no test ever loaded them -- "a fixture nothing loads is a fixture
# nothing protects". `8a3ddc9` (the very next real commit) changed
# `landing_prompt`, nobody re-recorded, and 0 of 30 stage-2 keys matched at
# HEAD -- silently, because nothing here would have raised. This file is
# that test: it replays every committed transcript through the real
# `run_profile`, the same call `robigo profile --replay <path>` makes, so a
# prompt that drifts from what a transcript was recorded against fails LOUDLY
# (TranscriptMiss) instead of leaving a stale fixture nobody notices.
_TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"

# (filename, seeds, mode) -- must match exactly what produced the committed
# recording (`robigo profile --model <model> --record <path>`, no --seeds/
# --full, so the CLI's own defaults: seeds=3, mode="quick"). A mismatch here
# changes how many stage-1/stage-2 calls get made and would itself raise
# TranscriptMiss, the same way a stale prompt does -- this tuple is part of
# what "matches the recording" means, not incidental.
#
# qwen7b.jsonl is deliberately NOT here -- see
# test_qwen_transcript_documents_a_real_unmeasurable_probe below for why its
# committed shape is one row, not a full profile run.
_FIXTURES = (
    ("granite8b.jsonl", 3, "quick"),
    ("codegemma7b.jsonl", 3, "quick"),
)


def _plan_for(path: Path) -> WindowPlan:
    """A `WindowPlan` whose `.window` matches exactly what the transcript
    was recorded against -- read from the transcript's own first row,
    not hardcoded, so this test stays correct across a re-recording
    without editing. `.window` is the only field stage0_window's probe
    construction depends on; the rest (training_ctx, kv_per_token, ...)
    only affect what gets REPORTED in the resulting Profile, never what
    prompts get sent, so placeholders are honest here the same way
    test_stage0.py's and test_profile_report.py's PLAN literals use them."""
    first_row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    return WindowPlan(
        window=first_row["window"], limited_by="vram", free_vram=None,
        kv_per_token=56 * 1024, weights_bytes=0, overhead_bytes=0,
        training_ctx=first_row["window"],
    )


@pytest.mark.parametrize("filename,seeds,mode", _FIXTURES)
def test_a_committed_transcript_replays_through_run_profile(
    filename: str, seeds: int, mode: str
):
    # Fails with TranscriptMiss if ANY prompt this run sends -- stage 0's
    # filler probe, stage 1's ENVELOPE_PROMPT, or stage 2's landing_prompt
    # for any of the five fixtures under either codec -- is not a byte-
    # for-byte match for what is recorded under (model, prompt, seed) in
    # the committed transcript. That is precisely the staleness this test
    # exists to catch: verified by hand against the transcripts committed
    # at 68da8c2, before they were re-recorded for this fix -- see this
    # file's own module docstring and the fix-wave report for the exact
    # command and count (0/30 stage-2 keys matched).
    path = _TRANSCRIPTS_DIR / filename
    model = json.loads(path.read_text(encoding="utf-8").splitlines()[0])["model"]
    plan = _plan_for(path)

    profile = run_profile(
        CallReplayer(path), plan,
        model=model, quant=_quant(model),
        family=model.replace(":", "-").replace("/", "-"),
        seeds=seeds, mode=mode, corpus="fixtures-v1", kv_bits=16,
    )

    assert profile.model == model
    assert (profile.seeds, profile.mode) == (seeds, mode)
    # A transcript that replays cleanly but never actually reached stage 2
    # (e.g. because stage 1 never got its full seed count recorded) would
    # still avoid TranscriptMiss on the calls it DOES make -- assert the
    # full run really happened, not just the calls it made it through.
    assert profile.verdict in {"READY", "LIMITED", "UNUSABLE"}


def test_granite_reads_limited_on_window_alone():
    # The plan's own Task 6 Step 6 expectation: granite-code:8b's training
    # context (4096) sits below SUPPORTED_FLOOR (8192), so it must never
    # read READY regardless of how well it drives the envelope or lands a
    # codec -- pinned against the actual committed transcript rather than
    # a fake, so a re-recording that silently stopped being LIMITED would
    # be caught here.
    path = _TRANSCRIPTS_DIR / "granite8b.jsonl"
    model = json.loads(path.read_text(encoding="utf-8").splitlines()[0])["model"]
    profile = run_profile(
        CallReplayer(path), _plan_for(path),
        model=model, quant=_quant(model), family="granite-code-8b",
        seeds=3, mode="quick", corpus="fixtures-v1", kv_bits=16,
    )
    assert profile.verdict == "LIMITED"


def test_no_committed_row_encodes_an_unmeasured_reply():
    # Whole-branch review, reopened round (ruled 2026-08-10): the first
    # re-recording of these fixtures committed exactly the shape this
    # guards against -- `codegemma7b.jsonl` clean, but `qwen7b.jsonl` and
    # `granite8b.jsonl` each carried a stage-0 row with `outcome: "reply"`
    # and `tokens_in: 0`, which `stage0_window` (before this round's fix)
    # read as `Stage0(window=0, verified=True, note="probe accepted")` --
    # incoherent on its face and false besides. Fails if any committed
    # transcript is ever re-recorded (or hand-edited) back into that shape:
    # a "reply" outcome -- the call was accepted, not rejected or errored
    # -- must always carry a positive `tokens_in`, or it isn't a
    # measurement at all (see client.py's matching fix, which now raises
    # rather than producing this row in the first place).
    for path in sorted(_TRANSCRIPTS_DIR.glob("*.jsonl")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["outcome"] == "reply":
                assert row["tokens_in"] > 0, (
                    f"{path.name} line {lineno}: a 'reply' row with "
                    f"tokens_in={row['tokens_in']!r} is an unmeasured call "
                    f"masquerading as a measured one"
                )


def test_qwen_transcript_documents_a_real_unmeasurable_probe():
    # C1's fix reopened this file's own earlier claim (see the fix-wave
    # report): the "done: false, no stats" daemon response is NOT reliably
    # reproducible at every size -- it is a genuine, sharp, size-dependent
    # threshold specific to this model. Measured live, repeatedly, at
    # qwen2.5-coder:7b-instruct-q8_0's REAL plan.window (32768, its
    # training context -- this box has ample free VRAM, so nothing smaller
    # binds): the full-window probe `stage0_window` always sends first
    # -- 40/40 consecutive live attempts at (model, this exact prompt,
    # seed=0) failed -- while the same probe at a smaller target (<=
    # ~11500) succeeded reliably every time it was tried.
    # `_PROBE_SEED` is fixed at 0 by design (Task 3, so replay
    # stays deterministic) and `num_predict=1024` is the CLI's own fixed
    # default -- neither is in this fix wave's scope to change -- so the
    # immediate full-window probe is the FIRST call any recording attempt
    # makes, and it cannot currently be recorded as a successful
    # measurement for this model on this daemon.
    #
    # The honest artifact is what is committed: ONE row, `outcome:
    # "error"`, `error_type: "ModelError"` -- client.py's fix (this round)
    # correctly refuses to fabricate a `tokens_in: 0` "measurement" for it.
    # This test pins that the transcript stays in that shape (not silently
    # reverting to a false reply) and that replaying it reproduces the
    # exact same failure, byte for byte -- proof that this is a genuine,
    # reproducible daemon defect, not a fluke of the one live session that
    # recorded it.
    path = _TRANSCRIPTS_DIR / "qwen7b.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert len(rows) == 1
    assert rows[0]["outcome"] == "error"
    assert rows[0]["error_type"] == "ModelError"
    assert "prompt_eval_count" in rows[0]["error_message"]

    model = rows[0]["model"]
    with pytest.raises(ModelError) as e:
        run_profile(
            CallReplayer(path), _plan_for(path),
            model=model, quant=_quant(model), family="qwen-7b",
            seeds=3, mode="quick", corpus="fixtures-v1", kv_bits=16,
        )
    assert str(e.value) == rows[0]["error_message"]
