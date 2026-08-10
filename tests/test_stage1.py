# tests/test_stage1.py
from __future__ import annotations

from pathlib import Path

from robigo.model.client import Generation
from robigo.profile.stages import ENVELOPE_PROMPT, stage1_envelope
from robigo.profile.transcript import CallRecorder, CallReplayer


class _Scripted:
    model = "m"
    # Present so this fake can also be wrapped in CallRecorder/CallReplayer
    # (see test_a_recorded_mixed_stage1_run_replays_identically below) --
    # unused by the seven tests ported from the brief.
    window = 8192
    num_predict = 512

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.seeds: list[int] = []

    def generate(self, prompt: str, *, seed: int) -> Generation:
        self.seeds.append(seed)
        return Generation(self.replies[(seed - 1) % len(self.replies)], 5, 2, False)


def test_a_perfect_model_scores_one():
    result = stage1_envelope(_Scripted("read src/target.py"), seeds=4)
    assert result.fidelity == 1.0
    assert result.attempts == 4
    assert result.level == 0


def test_an_unparseable_reply_scores_zero_and_the_text_is_kept():
    result = stage1_envelope(_Scripted("Sure! I'd love to help."), seeds=2)
    assert result.fidelity == 0.0
    # The raw failures are the diagnostic material for the whole project.
    assert "Sure!" in result.failures[0]


def test_a_right_shaped_action_with_the_wrong_verb_does_not_count():
    # Parseable but not what was asked: the family can emit the envelope
    # yet cannot follow the instruction, and those are different findings.
    result = stage1_envelope(_Scripted("run"), seeds=2)
    assert result.fidelity == 0.0
    assert "wrong verb" in result.failures[0]


def test_a_mixed_model_scores_the_fraction():
    result = stage1_envelope(_Scripted("read src/target.py", "nope"), seeds=4)
    assert result.fidelity == 0.5


def test_level_one_is_recommended_when_fidelity_is_middling():
    # Level 1 is the two-step envelope: constrain the header, leave the
    # payload free (spec 2.3).
    assert stage1_envelope(_Scripted("read src/target.py", "no"), seeds=4).level == 1


def test_every_seed_is_used_so_variance_is_visible():
    client = _Scripted("read src/target.py")
    stage1_envelope(client, seeds=5)
    assert client.seeds == [1, 2, 3, 4, 5]


def test_the_prompt_names_exactly_one_expected_action():
    assert "read src/target.py" in ENVELOPE_PROMPT


def test_a_recorded_mixed_stage1_run_replays_identically(tmp_path: Path):
    # Mirrors test_stage0.py's replay tests. Deliberately a MIXED run (some
    # seeds parse, some don't) rather than an all-good one: an all-good
    # replay would still pass even if CallRecorder silently dropped every
    # failure from the transcript, which is exactly the class of bug Task
    # 2's amendment (185d8a7, "record and replay call outcomes, not just
    # successful replies") fixed for stage 0 -- this is the same proof for
    # stage 1.
    path = tmp_path / "stage1.jsonl"
    recorded = stage1_envelope(
        CallRecorder(_Scripted("read src/target.py", "nope"), path), seeds=5
    )
    replayed = stage1_envelope(CallReplayer(path), seeds=5)

    assert replayed == recorded
    # Sanity: this run actually mixes successes and failures -- otherwise
    # this test would prove nothing beyond what an all-good run already
    # would (see the comment above).
    assert 0.0 < recorded.fidelity < 1.0
    assert recorded.fidelity == 0.6
