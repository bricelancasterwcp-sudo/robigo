# src/robigo/profile/transcript.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from robigo.model.client import (
    ContextOverflowError,
    Generation,
    ModelClient,
    ModelError,
    ServerContextOverflowError,
)

_KNOWN_ERRORS: dict[str, type[ModelError]] = {
    cls.__name__: cls
    for cls in (ModelError, ContextOverflowError, ServerContextOverflowError)
}
"""Every exception `ModelClient.generate` is allowed to raise (`ModelError`
is infrastructure failure and nothing else -- a rambling or capped reply is
a RESULT carried in `Generation`, not an exception; spec 9 law 10). Keyed
by class name so a recorded row can name exactly which one happened and
replay can reconstruct that same type, not a flattened `ModelError`. This
is what lets stage 0 tell "the server rejected this size"
(`ServerContextOverflowError`) apart from "the model is broken" (bare
`ModelError`) identically under replay (Task 2 amendment, ruled
2026-08-10)."""


class TranscriptMiss(Exception):
    """Replay was asked for a call the transcript does not contain --
    because the exact (model, prompt, seed) was never recorded, or because
    it was recorded fewer times than replay is now asking for. Loud on
    purpose: silently falling through to a live model, or to an empty or
    repeated reply, would make a "reproduced" profile meaningless
    (spec 5.3).

    A call that was recorded as a *failure* is not a `TranscriptMiss` --
    replaying it re-raises the recorded exception type, faithfully, same
    as replaying a recorded reply returns that reply. A `TranscriptMiss`
    only ever means the transcript has nothing at all for this exact call,
    never that what it has is a rejection."""


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

    Records the OUTCOME of every call, not just a returned reply: a call
    that raises a `ModelError` (spec: the only exception a `ModelClient` is
    allowed to raise) is caught, its type and message recorded, and then
    RE-RAISED unchanged -- the live caller sees exactly what it would have
    seen unwrapped, and the failure is on the transcript for replay to
    reproduce. Task 2 amendment (ruled 2026-08-10): a `CallRecorder` that
    only wrote a row on successful return left every rejected probe
    unrecorded, so a stage-0 run whose bisection ever hit a rejection --
    precisely the run worth recording, since it's the one that found a
    planned window that did not hold -- could never be replayed past its
    first rejection. Any exception that is NOT a `ModelError` (a bug in
    the wrapped client, not a call outcome) is left unrecorded and
    propagates immediately, same as before this amendment.
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
        base_row = {
            "key": key_for(self.model, prompt, seed),
            "model": self.model,
            "window": self.window,
            "num_predict": self.num_predict,
        }
        try:
            gen = self._client.generate(prompt, seed=seed)
        except ModelError as exc:
            self._append(
                {
                    **base_row,
                    "outcome": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            raise
        self._append(
            {
                **base_row,
                "outcome": "reply",
                "text": gen.text,
                "tokens_in": gen.tokens_in,
                "tokens_out": gen.tokens_out,
                "truncated": gen.truncated,
            }
        )
        return gen

    def _append(self, row: dict) -> None:
        with self._path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row) + "\n")


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
    in the order they were recorded, one row per call.

    A call for a key that was never recorded, and a call for a key that
    has already returned every row recorded under it, both raise
    `TranscriptMiss` -- but with distinct messages, because they call for
    different fixes. "Never recorded" means the prompt or seed drifted
    from the fixture: the fix is to look at what changed in the code.
    "Already used" (drained) means this call site now asks for this exact
    (model, prompt, seed) more times than were recorded: the fix is to
    re-record, probably with more seeds. A caller (or a human reading the
    exception) must be able to tell which happened from the message alone
    -- `key not in self._queues` (never recorded) is checked separately
    from an empty-but-present queue (drained).

    A row recorded as a failure (`CallRecorder` caught a `ModelError`) is
    NOT a `TranscriptMiss` -- it is a normal replay outcome, and `generate`
    re-raises the exact recorded exception type (looked up in
    `_KNOWN_ERRORS` by the class name `CallRecorder` stored) with the
    recorded message. Re-raising by type, not as a generic `ModelError`,
    matters because stage 0 catches `ServerContextOverflowError`
    specifically to tell "the server rejected this size" apart from "the
    model is broken"; flattening the type on replay would change stage
    0's bisection behaviour under replay while every wrapper-level test
    that only checks "something raised" kept passing (Task 2 amendment,
    ruled 2026-08-10).
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
        if key not in self._queues:
            raise TranscriptMiss(
                f"no reply was ever recorded for model {self.model!r}, "
                f"seed {seed}, and this prompt ({len(prompt)} chars) -- "
                "this exact call was never recorded. The prompt, seed, or "
                "model changed since the transcript was made -- re-record "
                "it."
            )
        queue = self._queues[key]
        if not queue:
            raise TranscriptMiss(
                f"every reply recorded for model {self.model!r}, seed "
                f"{seed}, and this prompt ({len(prompt)} chars) has "
                "already been used by replay. This call site is asking "
                "for this exact (model, prompt, seed) more times than the "
                "transcript has replies for -- re-record it, probably "
                "with more seeds."
            )
        row = queue.pop(0)
        if row["outcome"] == "error":
            raise _KNOWN_ERRORS[row["error_type"]](row["error_message"])
        return Generation(
            row["text"], row["tokens_in"], row["tokens_out"], row["truncated"]
        )
