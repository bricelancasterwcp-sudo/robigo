# tests/test_stage0.py
from __future__ import annotations

from pathlib import Path

import pytest

from robigo.model.client import Generation, ModelError, ServerContextOverflowError
from robigo.model.geometry import WindowPlan
from robigo.profile.stages import Stage0, _default_probe, stage0_window
from robigo.profile.transcript import CallRecorder, CallReplayer

# weights_bytes/overhead_bytes are required positional fields of the real
# WindowPlan (src/robigo/model/geometry.py) but are not read by stage0_window
# -- it only ever looks at plan.window. Set to 0 rather than omitted: the
# brief's own PLAN literal omits them, which raises TypeError against the
# shipped dataclass (verified by hand before writing this file; see the
# task-3 report). training_ctx (added by the whole-branch review's C3 fix,
# ruled 2026-08-10) is also unused by stage0_window and left at its default.
PLAN = WindowPlan(window=8192, limited_by="vram", free_vram=None,
                  kv_per_token=56 * 1024, weights_bytes=0, overhead_bytes=0)


class _Accepts:
    model = "m"
    # Present so these fakes can also be wrapped in CallRecorder/CallReplayer
    # (see test_a_recorded_stage0_run_replays_without_the_client below) --
    # unused by the four tests ported from the brief.
    window = 8192
    num_predict = 512

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def generate(self, prompt: str, *, seed: int) -> Generation:
        self.sizes.append(len(prompt))
        # Plays the server honestly: `tokens_in` is the prompt's own real
        # length, NOT some fixed placeholder -- whole-branch review C1
        # (ruled 2026-08-10) made stage0_window report this value instead
        # of the char-estimated target the probe was built for, and a
        # fake that always answered `tokens_in=1` regardless of what was
        # actually sent could never catch a regression back to ignoring
        # it (a hardcoded 1 would pass just as well as reading the real
        # field). See test_a_verified_window_is_returned_unchanged, which
        # is the test this specific choice exists to make meaningful.
        return Generation("ok", len(prompt), 1, False)


class _Rejects(_Accepts):
    def __init__(self, until: int) -> None:
        super().__init__()
        self.until = until
        # Every prompt that was NOT rejected -- lets a test check that a
        # returned window corresponds to a probe the fake actually
        # accepted, not merely one smaller than the plan (which
        # boundary-plus-one also satisfies).
        self.accepted: list[str] = []

    def generate(self, prompt: str, *, seed: int) -> Generation:
        self.sizes.append(len(prompt))
        if len(prompt) > self.until:
            raise ServerContextOverflowError("too big")
        self.accepted.append(prompt)
        return Generation("ok", len(prompt), 1, False)


def test_a_verified_window_is_returned_unchanged():
    # Fails if stage0_window ever shrinks or re-notes a window the server
    # actually accepted (e.g. always applying a "safety" fraction even on
    # acceptance), or if the probe it sends is trivially small -- a probe
    # that sends a single token would satisfy "the server accepted it"
    # while verifying nothing about whether the full window fits.
    #
    # The expected window is the LENGTH of the probe actually sent, not
    # PLAN.window (C1, ruled 2026-08-10): `_default_probe(PLAN.window)`'s
    # length is 27033 characters, not 8192 -- and the fake reports that
    # real length back as `tokens_in`, exactly the way a real server
    # reports a real tokenizer's count for what it was actually sent. The
    # pre-fix version of stage0_window reported `plan.window` verbatim
    # here, which is exactly the defect this pins: a probe aimed at 8192
    # tokens can be, and on this project's own committed
    # `codegemma7b.jsonl` transcript was, far smaller in real tokens than
    # what got reported as verified.
    client = _Accepts()
    result = stage0_window(client, PLAN)
    expected = len(_default_probe(PLAN.window))
    assert expected != PLAN.window  # sanity: this test would be vacuous otherwise
    assert result == Stage0(window=expected, verified=True, note="probe accepted")
    # It must actually have sent something near the window, not a token.
    assert max(client.sizes) > 8192


def test_a_rejected_window_falls_back_and_says_so():
    # Fails if a rejection at the full window is treated as a final
    # verdict (returning verified=False immediately) instead of triggering
    # a search for the largest size the server does accept, or if the note
    # doesn't explain that the planned number changed.
    #
    # The reported window must be the SERVER's own count for the tightest
    # probe the search actually found accepted (C1, ruled 2026-08-10) --
    # not the char-estimated TARGET that probe was built for, and not
    # merely "smaller than the plan" (boundary-plus-one satisfies that
    # too). `_Rejects` plays the server honestly here: it reports back
    # exactly `len(prompt)` as `tokens_in`, so the strongest available
    # check is structural -- the reported window must equal the length of
    # the LONGEST prompt the fake actually accepted, whatever the search
    # algorithm's internal target numbers were.
    client = _Rejects(until=6000)
    result = stage0_window(client, PLAN)
    assert result.window < PLAN.window
    assert result.verified is True
    assert "rejected" in result.note
    assert client.accepted, "the search must have found something accepted"
    assert max(len(p) for p in client.accepted) == result.window
    assert result.window <= 6000


def test_a_window_rejected_at_every_size_is_unverified():
    # Fails if stage0_window ever reports verified=True or a positive
    # window when literally nothing was ever accepted (e.g. defaulting to
    # verified=True, or returning the last-tried size instead of 0).
    result = stage0_window(_Rejects(until=0), PLAN)
    assert result.verified is False
    assert result.window == 0


def test_infrastructure_failures_propagate_rather_than_shrinking_the_window():
    # A daemon that is down is not a small window. Conflating them would
    # write a wrong number into the profile (spec 9 law 10).
    # Fails if the except clause is broadened from ContextOverflowError to
    # plain ModelError -- a connection failure would then be silently
    # reported as window=0, verified=False instead of raising.
    class _Down(_Accepts):
        def generate(self, prompt: str, *, seed: int) -> Generation:
            raise ModelError("connection refused")

    with pytest.raises(ModelError):
        stage0_window(_Down(), PLAN)


def test_stage0_is_frozen():
    # Fails if @dataclass(frozen=True) is dropped from Stage0 -- the brief
    # marks it frozen explicitly, and a mutable result could be edited
    # in place by a later stage, silently invalidating the "this number was
    # actually measured" guarantee the rest of the profile depends on.
    result = stage0_window(_Accepts(), PLAN)
    with pytest.raises(Exception):
        result.window = 1  # type: ignore[misc]


def test_the_probe_never_exceeds_the_planned_window():
    # The scope boundary from the brief: plan.window is a VRAM-derived
    # ceiling, and stage 0 must only ever verify AT OR BELOW it, never
    # search upward. Fails if the search range's upper bound is ever
    # widened past plan.window (e.g. probing plan.window * 2 "just in
    # case"), which on a real daemon risks OOM rather than a clean
    # rejection.
    targets: list[int] = []

    def probe(target: int) -> str:
        targets.append(target)
        return "x"  # tiny prompt: an always-reject fake below forces the
        # search to explore the full range down to 0, so the maximum
        # target tried is the strongest evidence of the upper bound used.

    stage0_window(_Rejects(until=-1), PLAN, probe=probe)
    assert max(targets) <= PLAN.window


def test_a_rejected_window_recovers_the_exact_accepted_boundary():
    # Fails if the fallback is a coarse fixed ladder (e.g. 75%/50%/25% of
    # the plan) instead of narrowing toward the true accept/reject
    # boundary -- it would land on some smaller window that happens to be
    # accepted rather than the tightest one the server actually supports,
    # silently discarding real, usable capacity.
    def probe(target: int) -> str:
        return "x" * target  # len(prompt) == target exactly, isolating
        # the search algorithm from the default char/token conversion.

    class _RejectsAbove(_Accepts):
        def __init__(self, boundary: int) -> None:
            super().__init__()
            self.boundary = boundary

        def generate(self, prompt: str, *, seed: int) -> Generation:
            self.sizes.append(len(prompt))
            if len(prompt) > self.boundary:
                raise ServerContextOverflowError("too big")
            return Generation("ok", len(prompt), 1, False)

    result = stage0_window(_RejectsAbove(boundary=5000), PLAN, probe=probe)
    assert result.window == 5000


def test_a_zero_planned_window_is_unverified_without_probing():
    # A vram-exhausted plan (usable_window can legitimately return
    # window=0, see test_geometry.py) has nothing to probe. Fails if
    # stage0_window calls generate() anyway with a degenerate empty/near-
    # empty prompt instead of recognizing there is nothing to verify.
    empty_plan = WindowPlan(window=0, limited_by="vram", free_vram=0,
                            kv_per_token=1, weights_bytes=0, overhead_bytes=0)
    client = _Accepts()
    result = stage0_window(client, empty_plan)
    assert result == Stage0(window=0, verified=False,
                            note="planned window is 0; nothing to probe")
    assert client.sizes == []


def test_a_recorded_stage0_run_replays_without_the_client(tmp_path: Path):
    # Whether a recorded stage-0 run can be replayed depends on every
    # probe's (model, prompt, seed) being a deterministic function of the
    # plan alone -- CallReplayer looks calls up by that exact triple
    # (key_for). Fails if the probe seed or prompt varies between runs of
    # the *same* plan (e.g. a seed derived from id(client) or time.time()),
    # which would make the replayed call raise TranscriptMiss instead of
    # reproducing the recorded profile. Uses _Accepts (verified on the
    # first probe, one call, no rejection) -- the case where the run also
    # includes a rejection is the test directly below.
    path = tmp_path / "stage0.jsonl"
    recorder = CallRecorder(_Accepts(), path)
    recorded = stage0_window(recorder, PLAN)

    replayer = CallReplayer(path)
    replayed = stage0_window(replayer, PLAN)

    expected = len(_default_probe(PLAN.window))
    assert replayed == recorded == Stage0(window=expected, verified=True,
                                          note="probe accepted")


def test_a_run_that_hits_a_rejection_replays_identically(tmp_path: Path):
    # This is the acceptance test for the Task 2 amendment ("record
    # outcomes, not just replies", ruled 2026-08-10). It used to document a
    # real gap (renamed from test_a_run_that_hits_a_rejection_does_not_yet
    # _replay): CallRecorder.generate wrote a transcript row only after a
    # successful return, so a call that raised ServerContextOverflowError
    # left no row, and a recorded run whose bisection ever hit a rejection
    # -- precisely the run worth recording, since it's the one that found
    # a planned window that didn't hold -- replayed as TranscriptMiss on
    # the very first rejected probe instead of reproducing the run.
    #
    # CallRecorder now catches ModelError, records the outcome (including
    # which exact exception type), and re-raises; CallReplayer reconstructs
    # that exact type on replay. Fails end-to-end if either half regresses:
    # if a rejected probe goes unrecorded again, the live run
    # (stage0_window(CallRecorder(...), PLAN)) itself raises TranscriptMiss
    # on the *replay* line below, before the two variables are ever
    # compared -- pytest would report an unhandled exception, not an
    # assertion failure, which is itself diagnostic of which half broke.
    path = tmp_path / "stage0.jsonl"
    live = stage0_window(CallRecorder(_Rejects(until=6000), path), PLAN)

    replayed = stage0_window(CallReplayer(path), PLAN)

    assert replayed == live
    # Sanity: this run actually exercises a rejection (otherwise the test
    # would pass trivially, the same way test_a_recorded_stage0_run_
    # replays_without_the_client already covers the no-rejection case).
    assert live.window < PLAN.window
    assert live.verified is True
