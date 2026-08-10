# tests/test_transcript.py
from __future__ import annotations

from pathlib import Path

import pytest

from robigo.model.client import Generation
from robigo.profile.transcript import CallRecorder, CallReplayer, TranscriptMiss, key_for


class _Client:
    model = "m"
    window = 4096
    num_predict = 512

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, *, seed: int) -> Generation:
        self.calls += 1
        return Generation(f"reply to {prompt} @{seed}", 10, 5, False)


def test_recording_then_replaying_reproduces_exactly(tmp_path: Path):
    # Fails if replay ever calls the wrapped client, or if it returns
    # anything other than byte-identical Generation objects.
    path = tmp_path / "t.jsonl"
    client = _Client()
    recorder = CallRecorder(client, path)
    first = recorder.generate("hello", seed=1)
    second = recorder.generate("world", seed=2)

    replayer = CallReplayer(path)
    assert replayer.generate("hello", seed=1) == first
    assert replayer.generate("world", seed=2) == second
    # Replay must not touch the model at all.
    assert client.calls == 2


def test_replay_preserves_the_truncated_flag(tmp_path: Path):
    # Fails if the transcript round-trip drops or defaults `truncated`.
    path = tmp_path / "t.jsonl"

    class _Cut(_Client):
        def generate(self, prompt: str, *, seed: int) -> Generation:
            return Generation("cut", 10, 5, True)

    CallRecorder(_Cut(), path).generate("p", seed=1)
    assert CallReplayer(path).generate("p", seed=1).truncated is True


def test_a_missing_key_is_a_loud_failure_not_a_silent_skip(tmp_path: Path):
    # Fails if a prompt that was never recorded returns "" (or anything)
    # instead of raising -- that would let a stage silently measure a
    # broken fixture as a real (bad) result.
    path = tmp_path / "t.jsonl"
    CallRecorder(_Client(), path).generate("recorded", seed=1)
    with pytest.raises(TranscriptMiss) as e:
        CallReplayer(path).generate("never recorded", seed=1)
    # A changed prompt must fail visibly: silently re-running the model
    # would make a "reproduced" profile meaningless.
    assert "re-record" in str(e.value)


def test_a_call_past_the_end_of_the_transcript_is_a_loud_failure(tmp_path: Path):
    # Fails if a second call for the *same* (model, prompt, seed) as a
    # single recorded row returns that row again instead of raising --
    # that would let a loop that calls generate() more times than the
    # fixture has replies loop the last reply forever instead of stopping.
    path = tmp_path / "t.jsonl"
    CallRecorder(_Client(), path).generate("only call", seed=1)
    replayer = CallReplayer(path)
    replayer.generate("only call", seed=1)
    with pytest.raises(TranscriptMiss):
        replayer.generate("only call", seed=1)


def test_repeated_calls_to_the_same_prompt_and_seed_replay_in_recorded_order(
    tmp_path: Path,
):
    # Fails if replay is a plain dict keyed on key_for() -- a second
    # recording under an identical (model, prompt, seed) would then
    # silently overwrite the first, and both replay calls would return
    # the second (real) reply even though the first real call produced a
    # different one.
    path = tmp_path / "t.jsonl"

    class _Sequenced(_Client):
        def generate(self, prompt: str, *, seed: int) -> Generation:
            self.calls += 1
            return Generation(f"reply #{self.calls}", 10, 5, False)

    recorder = CallRecorder(_Sequenced(), path)
    first = recorder.generate("same prompt", seed=1)
    second = recorder.generate("same prompt", seed=1)
    assert first != second  # sanity: the fixture setup actually varies

    replayer = CallReplayer(path)
    assert replayer.generate("same prompt", seed=1) == first
    assert replayer.generate("same prompt", seed=1) == second


def test_the_key_covers_model_prompt_and_seed():
    # Fails if key_for ignores any one of its three arguments.
    assert key_for("m", "p", 1) != key_for("m", "p", 2)
    assert key_for("m", "p", 1) != key_for("n", "p", 1)
    assert key_for("m", "p", 1) == key_for("m", "p", 1)


def test_recording_appends_one_line_per_call(tmp_path: Path):
    # Fails if the recorder batches, truncates, or overwrites the file
    # instead of appending one JSON object per call.
    path = tmp_path / "t.jsonl"
    recorder = CallRecorder(_Client(), path)
    recorder.generate("a", seed=1)
    recorder.generate("b", seed=1)
    assert len(path.read_text().strip().split("\n")) == 2


def test_recorder_carries_model_window_and_num_predict_from_the_wrapped_client(
    tmp_path: Path,
):
    # Fails if the wrapper forwards generate() but drops model/window/
    # num_predict -- it would pass every reply-shaped test here and then
    # break the profiler loop at runtime, since the loop reads
    # `client.num_predict` directly with no fallback.
    client = _Client()
    recorder = CallRecorder(client, tmp_path / "t.jsonl")
    assert recorder.model == client.model
    assert recorder.window == client.window
    assert recorder.num_predict == client.num_predict


def test_replayer_carries_model_window_and_num_predict_from_the_transcript(
    tmp_path: Path,
):
    # Fails if CallReplayer(path) leaves model/window/num_predict at a
    # hardcoded placeholder instead of recovering them from what was
    # recorded -- it would then structurally satisfy ModelClient's type
    # shape while lying about which model/window/num_predict it stands in
    # for, and (worse) key_for() during replay would use the wrong model
    # and every lookup would silently TranscriptMiss.
    path = tmp_path / "t.jsonl"
    client = _Client()
    CallRecorder(client, path).generate("a", seed=1)
    replayer = CallReplayer(path)
    assert replayer.model == client.model
    assert replayer.window == client.window
    assert replayer.num_predict == client.num_predict


def _accepts_model_client(client) -> Generation:
    """Exercises a value the way `robigo.model.client.ModelClient` is used
    by the profiler loop: read `.model`/`.window`/`.num_predict` as plain
    attributes, then call `.generate`. `ModelClient` is a structural
    `Protocol` without `@runtime_checkable` (see client.py), so `isinstance`
    against it raises `TypeError` rather than checking anything -- this is
    the same shape check the rest of the codebase uses instead (see
    `test_loop_budget.py`'s `_MissingNumPredict`)."""
    assert isinstance(client.model, str)
    assert isinstance(client.window, int)
    assert isinstance(client.num_predict, int)
    return client.generate("x", seed=1)


def test_recorder_and_replayer_are_usable_wherever_a_model_client_is(
    tmp_path: Path,
):
    # Fails if either wrapper's model/window/num_predict are missing or
    # non-int/str, or if generate()'s keyword-only `seed` parameter isn't
    # preserved -- any of those breaks structural conformance to
    # ModelClient even though "it has a generate method".
    path = tmp_path / "t.jsonl"
    client = _Client()
    recorder = CallRecorder(client, path)
    _accepts_model_client(recorder)
    replayer = CallReplayer(path)
    _accepts_model_client(replayer)
