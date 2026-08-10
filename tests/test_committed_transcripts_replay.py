# tests/test_committed_transcripts_replay.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from robigo.cli import _quant
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
_FIXTURES = (
    ("qwen7b.jsonl", 3, "quick"),
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
        seeds=seeds, mode=mode, kv_bits=16,
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
        seeds=3, mode="quick", kv_bits=16,
    )
    assert profile.verdict == "LIMITED"
