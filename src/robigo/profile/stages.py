# src/robigo/profile/stages.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from robigo.model.client import ContextOverflowError, ModelClient
from robigo.model.geometry import WindowPlan

_PROBE_SEED = 0
"""Fixed, never derived from time, randomness, or call count. Replay
(`robigo.profile.transcript.CallReplayer`) is keyed on
`(model, prompt, seed)` -- a probe seed that drifted between the recording
run and a replay run would change the key on every call, and every
replayed call would raise `TranscriptMiss` instead of reproducing the
profile."""

_CHARS_PER_TOKEN = 3
_FILLER_WORD = "token "


@dataclass(frozen=True)
class Stage0:
    """The result of verifying a `WindowPlan` against a real server.

    `window` is never larger than the `plan.window` that was probed --
    stage 0 only ever confirms or shrinks the planned window, it does not
    search upward for a bigger one (see `stage0_window`'s scope note).
    `verified` is True exactly when some probe was accepted; `note`
    explains what happened in either case.
    """

    window: int
    verified: bool
    note: str


def _default_probe(target: int) -> str:
    """Build a prompt intended to represent roughly `target` tokens, sized
    in characters via a fixed chars-per-token ratio.

    The ratio's accuracy does not matter to correctness. The real
    tokenizer will disagree with it by a few percent (measured: a prompt
    aimed at exactly 512 tokens landed at 530 real tokens), which is
    exactly why `stage0_window` does not trust a single conversion -- it
    bisects on the server's real accept/reject answer instead. A wrong
    ratio here only costs a few extra probe rounds.

    Length is computed directly as `target * _CHARS_PER_TOKEN`, then the
    filler word is repeated and SLICED to that exact character count --
    never rounded down to a whole number of words. That precision is load-
    bearing, not cosmetic (amendment to Task 3, ruled 2026-08-10): an
    earlier version floored to whole words (`target * _CHARS_PER_TOKEN //
    len(_FILLER_WORD)` words), which made length a step function of
    `target` -- every `target` and `target + 1` sharing one floor value
    aliased onto the SAME prompt. Bisection could then land on the larger
    of an aliased pair and report it as "verified", even though its probe
    was byte-identical to the smaller, equally-accepted one -- the exact
    symptom measured against a 6000-character limit: `target=2001` and
    `target=2000` both floored to 1000 words (6000 chars), and the search
    reported 2001 as accepted when nothing distinguished its probe from
    2000's. Slicing to an exact length makes `target -> len(prompt)`
    strictly increasing, so no two distinct targets can ever share a
    probe, and the target bisection lands on is always the unique one that
    was actually, distinguishably tested.
    """
    length = max(target * _CHARS_PER_TOKEN, 1)
    repeats = length // len(_FILLER_WORD) + 1
    return (_FILLER_WORD * repeats)[:length]


def stage0_window(
    client: ModelClient,
    plan: WindowPlan,
    *,
    probe: Callable[[int], str] | None = None,
) -> Stage0:
    """Verify `plan.window` by loading it, rather than trusting the VRAM
    arithmetic that produced it (spec 5, stage 0). `plan.window` is a
    hypothesis computed from KV-cache geometry; only the server's own
    tokenizer, via a real request, can confirm or correct it.

    Built on three facts measured against a live daemon (see the task
    brief's "Verified before execution" section):

    1. Rejection is decided on the prompt alone against `num_ctx` --
       `num_predict` is not counted. So a probe that fills the window
       entirely (this one aims its first attempt at exactly
       `plan.window`) is legitimate and is not rejected merely for
       leaving no room to generate.
    2. A rejection names the real token count from the server's own
       tokenizer -- it is a measurement, not just a "no". This function
       acts on that principle by treating every rejection as information
       to narrow the search with (via bisection on real accept/reject
       answers) rather than as a final verdict on the whole window.
    3. Aiming at exactly the window risks a false negative from the
       char-to-token estimate: a prompt aimed at "exactly N tokens" can
       land a few percent over N on the real tokenizer and be rejected
       even though N genuinely fits. Because this function verifies by
       bisecting on real answers rather than trusting the char/token
       estimate for the final number, that estimate's error only costs
       probe rounds -- it can never cause a correct window to be reported
       wrong, and it can never cause an incorrect window to be reported
       right.

    Scope boundary: `hi` starts at `plan.window` and never rises above
    it -- this function only ever verifies *at or below* the planned
    window, never discovers that the model could do more. That is
    deliberate: `plan.window` is a VRAM-derived ceiling, and a request
    above it risks OOMing the real daemon rather than failing cleanly.

    Every probe uses the fixed `_PROBE_SEED` and a prompt that is a pure
    function of its target token count, so a `CallRecorder` transcript of
    one run replays deterministically with a `CallReplayer` and no live
    client.
    """
    build = probe or _default_probe
    if plan.window <= 0:
        return Stage0(
            window=0,
            verified=False,
            note="planned window is 0; nothing to probe",
        )

    def accepted(target: int) -> bool:
        try:
            client.generate(build(target), seed=_PROBE_SEED)
        except ContextOverflowError:
            return False
        return True

    if accepted(plan.window):
        return Stage0(window=plan.window, verified=True, note="probe accepted")

    # The full window was rejected. Bisect between "definitely accepted"
    # (0, untested but assumed) and "definitely rejected" (plan.window,
    # just tested) to find the largest size the server actually accepts,
    # rather than giving up on the first rejection or falling back to an
    # arbitrary fixed fraction that might overshoot the same way.
    lo, hi = 0, plan.window
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if accepted(mid):
            lo = mid
        else:
            hi = mid

    if lo == 0:
        return Stage0(
            window=0,
            verified=False,
            note=f"every probe from {plan.window} down was rejected",
        )
    return Stage0(
        window=lo,
        verified=True,
        note=f"planned {plan.window} rejected; verified at {lo}",
    )
