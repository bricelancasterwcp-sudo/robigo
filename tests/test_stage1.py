# tests/test_stage1.py
from __future__ import annotations

from robigo.model.client import Generation
from robigo.profile.stages import ENVELOPE_PROMPT, stage1_envelope


class _Scripted:
    model = "m"

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
