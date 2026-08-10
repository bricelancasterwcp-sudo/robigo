# src/robigo/profile/transcript.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from robigo.model.client import Generation, ModelClient


class TranscriptMiss(Exception):
    """Replay was asked for a call the transcript does not contain --
    because the exact (model, prompt, seed) was never recorded, or because
    it was recorded fewer times than replay is now asking for. Loud on
    purpose: silently falling through to a live model, or to an empty or
    repeated reply, would make a "reproduced" profile meaningless
    (spec 5.3)."""


def key_for(model: str, prompt: str, seed: int) -> str:
    """Replay is keyed on (model, prompt, seed), never positional. A NUL
    byte separates the fields because none of the three can otherwise
    contain one, so no combination of shorter/longer field values can
    collide onto the same digest."""
    digest = hashlib.sha256()
    digest.update(f"{model}\x00{seed}\x00{prompt}".encode())
    return digest.hexdigest()


class CallRecorder:
    """Wraps a `ModelClient`, passing every call straight through to it and
    appending one JSON object per call to a JSONL transcript at `path`.

    Carries `model`, `window` and `num_predict` from the wrapped client, so
    a `CallRecorder` is itself a `ModelClient`: the profiler loop can be
    pointed at a live client wrapped in a `CallRecorder` to capture a
    fixture, or at a `CallReplayer` to replay one, without knowing which.
    """

    def __init__(self, client: ModelClient, path: Path) -> None:
        self._client = client
        self._path = path
        self.model = client.model
        self.window = client.window
        self.num_predict = client.num_predict
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch()

    def generate(self, prompt: str, *, seed: int) -> Generation:
        gen = self._client.generate(prompt, seed=seed)
        row = {
            "key": key_for(self.model, prompt, seed),
            "model": self.model,
            "window": self.window,
            "num_predict": self.num_predict,
            "text": gen.text,
            "tokens_in": gen.tokens_in,
            "tokens_out": gen.tokens_out,
            "truncated": gen.truncated,
        }
        with self._path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row) + "\n")
        return gen


class CallReplayer:
    """Replays a `CallRecorder` transcript with no client, no daemon, no
    network -- the whole point of the format (spec 5.3).

    `model`, `window` and `num_predict` are recovered from the transcript
    itself, not supplied by the caller: every recorded row carries them, so
    a `CallReplayer` reports the same identity a `CallRecorder` for the
    same client would, and is itself a `ModelClient` the loop can read
    `num_predict` from directly. An empty transcript (no calls were ever
    recorded) leaves them at the type-correct placeholders `""`/`0`/`0`;
    every `generate()` call on such a file raises `TranscriptMiss` anyway,
    so the placeholder is never mistaken for a real value.

    Replay is keyed on (model, prompt, seed) via `key_for`, not positional
    by call order: a call whose prompt (or seed, or model) does not match
    anything recorded raises `TranscriptMiss` rather than silently
    returning an unrelated reply. That is the deliberate choice -- a
    positional replay would return call N's reply for call N regardless of
    whether the prompt at that position had changed, which would make
    `--replay` prove nothing about whether today's prompts match the
    recording.

    Within a single key, rows ARE replayed positionally: recording the same
    (model, prompt, seed) twice (e.g. a repeat-rate measurement re-running
    one prompt) appends two rows under one key, and replay hands them back
    in the order they were recorded, one row per call. A call for a key
    that was never recorded, and a call for a key that has already
    returned every row recorded under it, both raise `TranscriptMiss` --
    running out is running out, whichever way it happened.
    """

    def __init__(self, path: Path) -> None:
        self.model = ""
        self.window = 0
        self.num_predict = 0
        self._queues: dict[str, list[dict]] = {}
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            self.model = row["model"]
            self.window = row["window"]
            self.num_predict = row["num_predict"]
            self._queues.setdefault(row["key"], []).append(row)

    def generate(self, prompt: str, *, seed: int) -> Generation:
        key = key_for(self.model, prompt, seed)
        queue = self._queues.get(key)
        if not queue:
            raise TranscriptMiss(
                f"no recorded reply left for model {self.model!r}, seed "
                f"{seed}, and this prompt ({len(prompt)} chars). Either "
                "this exact call was never recorded, or replay has already "
                "used every reply recorded for it. The prompt, seed, or "
                "call count changed since the transcript was made -- "
                "re-record it."
            )
        row = queue.pop(0)
        return Generation(
            row["text"], row["tokens_in"], row["tokens_out"], row["truncated"]
        )
