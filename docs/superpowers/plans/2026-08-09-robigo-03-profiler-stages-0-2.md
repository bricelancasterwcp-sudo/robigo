# robigo 03 — Profiler Stages 0–2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `robigo profile` measures a served model's real window, whether it can emit a parseable action at all, and which codec its patches actually land with — then writes a profile the loop configures itself from, reproducibly enough to re-run with no GPU.

**Architecture:** Three staged probes, cheapest first, each able to stop the run. Every model call passes through a transcript layer that either records or replays, so the whole pipeline is exercisable from a fixture file. A reporter turns stage results into a `Profile` JSON plus a human table, carrying the seeds and mode that produced it so a quick run can never be quoted as a result.

**Tech Stack:** Python 3.12+, stdlib only. Builds on plans 01 and 02.

## Global Constraints

- **Runtime dependencies: none.** Standard library only.
- `requires-python = ">=3.12"`; `from __future__ import annotations` in every module. Type annotations on every **non-test** function signature; pytest test functions are exempt.
- **Stages run cheapest-first and gate each other.** A family that fails stage 1 never reaches stage 2 (spec §5).
- **Stage 0 probes; it does not trust metadata.** The computed window from plan 02 is a *hypothesis* to be verified by an actual load (spec §5, stage 0).
- **Every profile records `seeds` and `mode`.** Quick-mode numbers are never publishable; the reporter marks them (spec §5.5).
- **Per-family only, never pooled.** Floors and ceilings are named as floors and ceilings.
- **Anything dropped for time is stated as dropped** — no silent caps.
- Commit messages: `<type>: <subject>`, single line, no body, no trailers.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/robigo/profile/schema.py` | `Profile`, `CodecResult`, JSON round-trip, verdict rules |
| `src/robigo/profile/transcript.py` | record/replay wrapper around any client |
| `src/robigo/profile/stages.py` | stage 0 window probe, stage 1 envelope, stage 2 codec landing |
| `src/robigo/profile/fixtures/` | five tiny single-defect Python repos for stage 2 |
| `src/robigo/profile/report.py` | orchestration, verdict, JSON + table output |
| `src/robigo/cli.py` | *modified* — `robigo profile` subcommand |

The five bundled fixtures are a stopgap so stage 2 can be built and tested now. Plan 04 replaces them with mutation-generated corpora and this module's interface does not change when it does.

## Verified before execution (2026-08-10)

This plan was written before plans 02 and 02b executed, so every interface it
consumes was checked against the shipped code first. All present and compatible:
`ServerContextOverflowError` and `ContextOverflowError` (`model/client.py`),
`CODECS` and `PatchError` (`action/codec.py`), `parse` (`action/verbs.py`),
`PythonAdapter`, `estimate_tokens` (`context/budget.py`), `plan_window` with the signature this plan calls.

**Correction (2026-08-10): the `WindowPlan` claim above was wrong.** This note
originally said `WindowPlan` had "exactly the four fields this plan constructs".
It has **six** — `weights_bytes` and `overhead_bytes` were added by plan 02b's fix
wave, so the window-0 refusal could print its terms honestly. I verified that the
*names* this plan imports exist and did not check arity, which is the same
not-quite-verified shape this project keeps finding. **Every `WindowPlan(...)`
literal in this plan is therefore stale** — Tasks 3, 4, 5 and 6 all contain one.
Add `weights_bytes=` and `overhead_bytes=` when you transcribe them; Task 3's
implementer hit this and fixed it locally, and Tasks 4-6 will hit it too. `build_client(args)` reads exactly
`backend, model, window, num_predict, host` — the five the plan's `Namespace`
supplies. Nothing in this plan reads run records, so 02b's `rung` → `rungs`
schema change does not touch it.

**The CLI dispatch is sound and non-breaking.** Routing on a leading `profile`
argument leaves the existing flat parser intact, so `robigo "<task>"` keeps
working and no existing CLI test changes. Note the one collision it creates: a
task that is *exactly* the single word `profile` would be routed to the
profiler. `robigo "profile the parser"` is unaffected because the whole task is
one argv element. Acceptable; mention it in the dispatch's comment so the next
reader does not think it was missed.

**Naming ruling: the transcript classes are `CallRecorder` and `CallReplayer`,
not `Recorder`/`Replayer`.** `record.py` already ships `RunRecorder` and
`new_recorder`, which record a *run's* per-turn prompts, replies and test output
to `.robigo/runs/<id>/` for a human to read. This plan's classes wrap a client to
record and replay *model calls* for deterministic re-runs. Two different things
both called "recorder" in one codebase is how this project has repeatedly ended
up with two answers to one question — see `docs/CARRIED-DEBT.md`, where a
five-way disagreement about unreadable files and three names for one output
reservation are both recorded. Name them apart now, while it costs nothing.
Everywhere this plan writes `Recorder`/`Replayer`, read `CallRecorder`/
`CallReplayer`.
---

### Task 1: Profile schema and verdict rules

**Files:**
- Create: `src/robigo/profile/__init__.py`, `src/robigo/profile/schema.py`
- Test: `tests/test_profile_schema.py`

**Interfaces:**
- Produces: `CodecResult(lands: float, attempts: int, max_file_tokens: int | None)` (frozen); `Profile(family, model, quant, training_ctx, kv_kib_per_token, kv_bits, usable_window, window_limited_by, envelope_level, envelope_fidelity, codecs, payload_corruption, repeat_rate, verdict, seeds, mode, corpus, dropped)` (frozen) with `to_json()`/`from_json()`/`best_codec()`; `verdict_for(envelope_fidelity, codecs, usable_window) -> str`; `SUPPORTED_FLOOR = 8192`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_profile_schema.py
from __future__ import annotations

import json

from robigo.profile.schema import (
    SUPPORTED_FLOOR,
    CodecResult,
    Profile,
    verdict_for,
)


def _profile(**kw) -> Profile:
    defaults = dict(
        family="qwen2.5-coder-7b", model="qwen2.5-coder:7b-instruct-q8_0",
        quant="q8_0", training_ctx=32768, kv_kib_per_token=56, kv_bits=16,
        usable_window=32768, window_limited_by="training_ctx",
        envelope_level=0, envelope_fidelity=0.98,
        codecs={"search_replace": CodecResult(0.62, 30, None),
                "whole_file": CodecResult(0.55, 30, 1400)},
        payload_corruption=None, repeat_rate=None, verdict="READY",
        seeds=3, mode="quick", corpus="fixtures-v1", dropped=(),
    )
    return Profile(**{**defaults, **kw})


def test_round_trips_through_json():
    original = _profile()
    assert Profile.from_json(json.loads(original.to_json())) == original


def test_best_codec_is_the_highest_landing_rate():
    assert _profile().best_codec() == "search_replace"


def test_seeds_and_mode_are_always_present_in_the_json():
    # A quick 3-seed profile must never be quotable as a result, so the
    # provenance travels with the numbers (spec 5.5).
    payload = json.loads(_profile().to_json())
    assert payload["measured"]["seeds"] == 3
    assert payload["measured"]["mode"] == "quick"


def test_both_quantization_covariates_are_recorded():
    # Weight quantization AND kv-cache quantization are covariates, not
    # free levers: a 14B-Q4 and a 7B-Q8 are DIFFERENT SUBJECTS, and q8 kv
    # buys window at a cost that must be visible (spec 3.2).
    payload = json.loads(_profile(quant="q4_K_M", kv_bits=8).to_json())
    assert payload["quant"] == "q4_K_M"
    assert payload["kv_bits"] == 8


def test_unusable_when_the_envelope_cannot_be_emitted():
    # Below half, the model cannot reliably say what it wants to do, and
    # nothing downstream is measurable (spec 5, stage 1 gates the rest).
    assert verdict_for(0.4, {"search_replace": CodecResult(0.9, 10, None)}, 32768) == "UNUSABLE"


def test_limited_when_the_window_is_under_the_supported_floor():
    assert SUPPORTED_FLOOR == 8192
    verdict = verdict_for(0.99, {"whole_file": CodecResult(0.7, 10, 1400)}, 4096)
    assert verdict == "LIMITED"


def test_limited_when_no_codec_lands_half_the_time():
    assert verdict_for(0.99, {"search_replace": CodecResult(0.3, 10, None)}, 32768) == "LIMITED"


def test_ready_when_everything_clears():
    assert verdict_for(0.95, {"search_replace": CodecResult(0.6, 30, None)}, 32768) == "READY"


def test_dropped_work_is_recorded_rather_than_silent():
    payload = json.loads(_profile(dropped=("stage2 udiff: time",)).to_json())
    assert payload["dropped"] == ["stage2 udiff: time"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_profile_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.profile'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/profile/schema.py
from __future__ import annotations

import json
from dataclasses import dataclass, replace  # noqa: F401  (replace used by callers)

SUPPORTED_FLOOR = 8192
"""Windows below this are a documented edge case, not a target. A design
that works at 4096 works everywhere, but a 4096 family should never be
recommended for agentic work (spec 3.1)."""

_ENVELOPE_MIN = 0.5
_LANDING_MIN = 0.5


@dataclass(frozen=True)
class CodecResult:
    lands: float
    attempts: int
    max_file_tokens: int | None


@dataclass(frozen=True)
class Profile:
    family: str
    model: str
    quant: str
    training_ctx: int
    kv_kib_per_token: int
    kv_bits: int
    usable_window: int
    window_limited_by: str
    envelope_level: int
    envelope_fidelity: float
    codecs: dict[str, CodecResult]
    payload_corruption: float | None
    repeat_rate: float | None
    verdict: str
    seeds: int
    mode: str
    corpus: str
    dropped: tuple[str, ...]

    def best_codec(self) -> str | None:
        if not self.codecs:
            return None
        return max(self.codecs, key=lambda name: self.codecs[name].lands)

    def to_json(self) -> str:
        return json.dumps(
            {
                "family": self.family, "model": self.model, "quant": self.quant,
                "training_ctx": self.training_ctx,
                "kv_kib_per_token": self.kv_kib_per_token,
                "kv_bits": self.kv_bits,
                "usable_window": self.usable_window,
                "window_limited_by": self.window_limited_by,
                "envelope_level": self.envelope_level,
                "envelope_fidelity": self.envelope_fidelity,
                "codecs": {
                    name: {"lands": r.lands, "attempts": r.attempts,
                           "max_file_tokens": r.max_file_tokens}
                    for name, r in self.codecs.items()
                },
                "payload_corruption": self.payload_corruption,
                "repeat_rate": self.repeat_rate,
                "verdict": self.verdict,
                "measured": {"seeds": self.seeds, "mode": self.mode,
                             "corpus": self.corpus},
                "dropped": list(self.dropped),
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: dict) -> Profile:
        measured = payload["measured"]
        return cls(
            family=payload["family"], model=payload["model"],
            quant=payload["quant"], training_ctx=payload["training_ctx"],
            kv_kib_per_token=payload["kv_kib_per_token"],
            kv_bits=payload["kv_bits"],
            usable_window=payload["usable_window"],
            window_limited_by=payload["window_limited_by"],
            envelope_level=payload["envelope_level"],
            envelope_fidelity=payload["envelope_fidelity"],
            codecs={
                name: CodecResult(r["lands"], r["attempts"], r["max_file_tokens"])
                for name, r in payload["codecs"].items()
            },
            payload_corruption=payload["payload_corruption"],
            repeat_rate=payload["repeat_rate"], verdict=payload["verdict"],
            seeds=measured["seeds"], mode=measured["mode"],
            corpus=measured["corpus"], dropped=tuple(payload["dropped"]),
        )


def verdict_for(
    envelope_fidelity: float, codecs: dict[str, CodecResult], usable_window: int
) -> str:
    if envelope_fidelity < _ENVELOPE_MIN:
        return "UNUSABLE"
    best = max((r.lands for r in codecs.values()), default=0.0)
    if usable_window < SUPPORTED_FLOOR or best < _LANDING_MIN:
        return "LIMITED"
    return "READY"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_profile_schema.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/profile tests/test_profile_schema.py
git commit -m "feat: profile schema with provenance and verdict rules"
```

---

### Task 2: The record/replay transcript

**Files:**
- Create: `src/robigo/profile/transcript.py`
- Test: `tests/test_transcript.py`

**Interfaces:**
- Consumes: `Generation`.
- Produces: `TranscriptMiss(Exception)`; `Recorder(client, path)` and `Replayer(path)`, both exposing `generate(prompt, *, seed) -> Generation`; `key_for(model, prompt, seed) -> str`.

Replay makes the profiler testable in CI with no GPU and reproducible for anyone reading a published number (spec §5.3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transcript.py
from __future__ import annotations

from pathlib import Path

import pytest

from robigo.model.client import Generation
from robigo.profile.transcript import Recorder, Replayer, TranscriptMiss, key_for


class _Client:
    model = "m"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, *, seed: int) -> Generation:
        self.calls += 1
        return Generation(f"reply to {prompt} @{seed}", 10, 5, False)


def test_recording_then_replaying_reproduces_exactly(tmp_path: Path):
    path = tmp_path / "t.jsonl"
    client = _Client()
    recorder = Recorder(client, path)
    first = recorder.generate("hello", seed=1)
    second = recorder.generate("world", seed=2)

    replayer = Replayer(path)
    assert replayer.generate("hello", seed=1) == first
    assert replayer.generate("world", seed=2) == second
    # Replay must not touch the model at all.
    assert client.calls == 2


def test_replay_preserves_the_truncated_flag(tmp_path: Path):
    path = tmp_path / "t.jsonl"

    class _Cut(_Client):
        def generate(self, prompt: str, *, seed: int) -> Generation:
            return Generation("cut", 10, 5, True)

    Recorder(_Cut(), path).generate("p", seed=1)
    assert Replayer(path).generate("p", seed=1).truncated is True


def test_a_missing_key_is_a_loud_failure_not_a_silent_skip(tmp_path: Path):
    path = tmp_path / "t.jsonl"
    Recorder(_Client(), path).generate("recorded", seed=1)
    with pytest.raises(TranscriptMiss) as e:
        Replayer(path).generate("never recorded", seed=1)
    # A changed prompt must fail visibly: silently re-running the model
    # would make a "reproduced" profile meaningless.
    assert "re-record" in str(e.value)


def test_the_key_covers_model_prompt_and_seed():
    assert key_for("m", "p", 1) != key_for("m", "p", 2)
    assert key_for("m", "p", 1) != key_for("n", "p", 1)
    assert key_for("m", "p", 1) == key_for("m", "p", 1)


def test_recording_appends_one_line_per_call(tmp_path: Path):
    path = tmp_path / "t.jsonl"
    recorder = Recorder(_Client(), path)
    recorder.generate("a", seed=1)
    recorder.generate("b", seed=1)
    assert len(path.read_text().strip().split("\n")) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transcript.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.profile.transcript'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/profile/transcript.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from robigo.model.client import Generation


class TranscriptMiss(Exception):
    """Replay was asked for a call the transcript does not contain. Loud
    on purpose: silently falling through to a live model would make a
    "reproduced" profile meaningless (spec 5.3)."""


def key_for(model: str, prompt: str, seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"{model}\x00{seed}\x00{prompt}".encode())
    return digest.hexdigest()


class Recorder:
    """Passes calls through and appends each one to a JSONL transcript."""

    def __init__(self, client, path: Path) -> None:
        self._client = client
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch()

    def generate(self, prompt: str, *, seed: int) -> Generation:
        gen = self._client.generate(prompt, seed=seed)
        row = {
            "key": key_for(self._client.model, prompt, seed),
            "text": gen.text, "tokens_in": gen.tokens_in,
            "tokens_out": gen.tokens_out, "truncated": gen.truncated,
        }
        with self._path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row) + "\n")
        return gen


class Replayer:
    def __init__(self, path: Path) -> None:
        self.model = ""
        self._rows: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                row = json.loads(line)
                self._rows[row["key"]] = row

    def generate(self, prompt: str, *, seed: int) -> Generation:
        for key, row in self._rows.items():
            if key == key_for(self.model, prompt, seed):
                return Generation(row["text"], row["tokens_in"],
                                  row["tokens_out"], row["truncated"])
        raise TranscriptMiss(
            f"no recorded reply for seed {seed} and this prompt "
            f"({len(prompt)} chars). The prompt or the model changed since "
            f"the transcript was made — re-record it."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transcript.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/profile/transcript.py tests/test_transcript.py
git commit -m "feat: record/replay transcript so profiles reproduce without a GPU"
```

---

#### Amendment to Task 2 (ruled 2026-08-10): record outcomes, not just replies

Found by Task 3 and reproduced: `CallRecorder` writes a transcript row only after
`generate` **returns**, so a call that *raises* leaves no row. Stage 0 bisects by
deliberately provoking `ServerContextOverflowError`, so on a model whose planned
window is wrong, 7 of 14 probes go unrecorded and replay dies on the very first
call — the largest probe, which was rejected live.

The consequence is the feature inverted: `--replay` reproduces a profile run only
when stage 0 found nothing wrong. The runs actually worth recording and sharing —
a model whose computed window did not hold — are exactly the ones that cannot be
replayed.

*Invariant:* a transcript records the **outcome** of every call, reply or
exception, and replay reproduces that outcome — re-raising the same exception
type with the recorded message. A recorded run replays identically whether or not
it hit a rejection.

Keep `TranscriptMiss` meaning what it now means: the call was never recorded, or
its recorded outcomes are used up. A recorded *rejection* is not a miss — it is a
faithful replay of a call that failed.

Test it end to end at the stage level, not only at the wrapper: record a stage-0
run against a client that rejects above a threshold, replay it, and assert the
replayed `Stage0` equals the live one. That assertion is the point of the whole
record/replay layer, and it currently fails.

#### Amendment to Task 3 (ruled 2026-08-10): do not report a window one token above what was verified

Measured: against a client rejecting prompts over 6000 characters, stage 0
returns `Stage0(window=2001, verified=True, ...)`. At 3 characters per token the
largest *accepted* probe was 6000 chars — 2000 tokens — so a prompt for 2001
tokens (6003 chars) would have been rejected. The number reported as verified is
one token above anything the server actually accepted.

One token, in the direction that claims more capability than was demonstrated —
the same ±1 class that cost plan 02b a false refusal and a needless
over-degradation. `verified=True` must mean "a probe of this size was accepted",
so report the largest size that actually passed, not the boundary plus one.

### Task 3: Stage 0 — probe the window rather than trust it

**Files:**
- Create: `src/robigo/profile/stages.py`
- Test: `tests/test_stage0.py`

**Interfaces:**
- Consumes: `WindowPlan`, `ServerContextOverflowError`, `ModelError`.
- Produces: `Stage0(window: int, verified: bool, note: str)` (frozen); `stage0_window(client, plan, *, probe=None) -> Stage0`.

The probe sends a prompt sized to fill the planned window and confirms the server accepts it. A planned window the server rejects is a computed hypothesis that was wrong, and the profile must carry the smaller verified number.

#### Verified before execution (2026-08-10) — the rejection boundary, measured

The probe design depends on how the server decides to reject, so it was measured
against the live daemon before this task was dispatched
(`qwen2.5-coder:1.5b-instruct-q8_0`, `num_ctx` 512):

| prompt | num_ctx | num_predict | result |
|---|---|---|---|
| ~430 tok | 512 | 64 | accepted |
| 530 tok | 512 | 64 | rejected — *"request (530 tokens) exceeds the available context size (512 tokens)"* |
| 535 tok | 512 | **256** | rejected, reporting **535** tokens, not 791 |

Three facts follow, and the probe must be built on them rather than on
assumption:

1. **Rejection is on the prompt alone against `num_ctx`; `num_predict` is not
   counted.** The third row proves it — raising `num_predict` fourfold did not
   change the reported figure. So a probe that fills the window entirely is
   legitimate and will not be rejected for leaving no room to generate. Without
   this, the natural worry is that stage 0 would report every model unverified.

2. **The rejection names the real token count**, from the server's own
   tokenizer. Use it. A rejection is therefore not merely a "no" — it is a
   measurement of how far over the probe landed, and it is more trustworthy than
   any local estimate.

3. **Aiming at exactly the window risks a false negative from the char→token
   estimate.** The probe sizes its prompt in characters via a chars-per-token
   ratio, and the real tokenizer will disagree by a few percent. At `num_ctx`
   512 the boundary sat between 430 accepted and 530 rejected — a prompt aimed at
   "exactly 512" can land at 530 on a 3% error and be rejected, making stage 0
   report a *correct* planned window as wrong. Aim deliberately under the window,
   and treat a rejection as information to refine with (using the count from
   fact 2) rather than as a final verdict. A stage that calls a good window bad
   is as much a measurement failure as one that calls a bad window good.

Also confirmed: **`truncate: false` is working.** The oversized prompts were
rejected rather than silently front-truncated, which is the precondition that
makes this probe mean anything — a server that quietly drops the front of the
prompt would accept every size and stage 0 would verify nothing.

**Scope note:** the client sends `num_ctx` from its own `window`, so stage 0 can
only ever verify *at or below* `plan.window`, never discover that the model could
do more. That is correct and deliberate: `plan.window` is a VRAM-derived ceiling
and exceeding it would OOM rather than run. Do not "improve" this into an upward
search.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage0.py
from __future__ import annotations

import pytest

from robigo.model.client import Generation, ModelError, ServerContextOverflowError
from robigo.model.geometry import WindowPlan
from robigo.profile.stages import Stage0, stage0_window

PLAN = WindowPlan(window=8192, limited_by="vram", free_vram=None,
                  kv_per_token=56 * 1024)


class _Accepts:
    model = "m"

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def generate(self, prompt: str, *, seed: int) -> Generation:
        self.sizes.append(len(prompt))
        return Generation("ok", 1, 1, False)


class _Rejects(_Accepts):
    def __init__(self, until: int) -> None:
        super().__init__()
        self.until = until

    def generate(self, prompt: str, *, seed: int) -> Generation:
        self.sizes.append(len(prompt))
        if len(prompt) > self.until:
            raise ServerContextOverflowError("too big")
        return Generation("ok", 1, 1, False)


def test_a_verified_window_is_returned_unchanged():
    client = _Accepts()
    result = stage0_window(client, PLAN)
    assert result == Stage0(window=8192, verified=True, note="probe accepted")
    # It must actually have sent something near the window, not a token.
    assert max(client.sizes) > 8192


def test_a_rejected_window_falls_back_and_says_so():
    result = stage0_window(_Rejects(until=6000), PLAN)
    assert result.window < 8192
    assert result.verified is True
    assert "rejected" in result.note


def test_a_window_rejected_at_every_size_is_unverified():
    result = stage0_window(_Rejects(until=0), PLAN)
    assert result.verified is False
    assert result.window == 0


def test_infrastructure_failures_propagate_rather_than_shrinking_the_window():
    # A daemon that is down is not a small window. Conflating them would
    # write a wrong number into the profile (spec 9 law 10).
    class _Down(_Accepts):
        def generate(self, prompt: str, *, seed: int) -> Generation:
            raise ModelError("connection refused")

    with pytest.raises(ModelError):
        stage0_window(_Down(), PLAN)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage0.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.profile.stages'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/profile/stages.py
from __future__ import annotations

from dataclasses import dataclass

from robigo.model.client import ContextOverflowError
from robigo.model.geometry import WindowPlan

_FILLER = "token "
_STEPS = (1.0, 0.75, 0.5, 0.25)


@dataclass(frozen=True)
class Stage0:
    window: int
    verified: bool
    note: str


def stage0_window(client, plan: WindowPlan, *, probe_chars_per_token: int = 3) -> Stage0:
    """Verify the planned window by actually filling it. The computed
    number from plan 02 is a hypothesis; only a load proves it (spec 5,
    stage 0). ``ContextOverflowError`` shrinks the estimate; every other
    ModelError propagates, because a daemon that is down is not a small
    window."""
    for fraction in _STEPS:
        target = int(plan.window * fraction)
        if target <= 0:
            break
        prompt = _FILLER * (target * probe_chars_per_token // len(_FILLER))
        try:
            client.generate(prompt, seed=1)
        except ContextOverflowError:
            continue
        note = "probe accepted" if fraction == 1.0 else (
            f"planned {plan.window} rejected; verified at {target}"
        )
        return Stage0(window=target, verified=True, note=note)
    return Stage0(
        window=0,
        verified=False,
        note=f"every probe from {plan.window} down was rejected",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage0.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/profile/stages.py tests/test_stage0.py
git commit -m "feat: stage 0 verifies the planned window by filling it"
```

---

### Task 4: Stage 1 — envelope fidelity

**Files:**
- Modify: `src/robigo/profile/stages.py`
- Test: `tests/test_stage1.py`

**Interfaces:**
- Consumes: `parse`, `ActionParseError`.
- Produces: `Stage1(fidelity: float, attempts: int, level: int, failures: tuple[str, ...])` (frozen); `ENVELOPE_PROMPT: str`; `stage1_envelope(client, seeds: int) -> Stage1`.

Stage 1 asks for one specific action and involves no code reasoning at all, so it isolates whether the family can drive the envelope. It gates every later stage.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage1.py -v`
Expected: FAIL — `ImportError: cannot import name 'ENVELOPE_PROMPT'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/robigo/profile/stages.py
from robigo.action.verbs import ActionParseError, parse

ENVELOPE_PROMPT = """Reply with exactly one action and nothing else.

Available actions, one per reply:
  read <path>        show a file
  find <symbol>      locate a symbol
  patch <path>       change a file (needs a fenced payload)
  run                re-run the tests
  done <summary>     finished

Emit exactly this action, on a line of its own:

read src/target.py
"""

_EXPECTED = ("read", "src/target.py")
_LEVEL1_MIN = 0.9
_FAILURE_CHARS = 200


@dataclass(frozen=True)
class Stage1:
    fidelity: float
    attempts: int
    level: int
    failures: tuple[str, ...]


def stage1_envelope(client, seeds: int) -> Stage1:
    """Can this family drive the envelope at all? No code reasoning is
    involved, so a failure here is purely about the action surface -- and
    it gates every later stage (spec 5)."""
    good = 0
    failures: list[str] = []
    for seed in range(1, seeds + 1):
        gen = client.generate(ENVELOPE_PROMPT, seed=seed)
        try:
            action = parse(gen.text)
        except ActionParseError as exc:
            failures.append(f"seed {seed}: {exc} :: {gen.text[:_FAILURE_CHARS]!r}")
            continue
        if (action.verb, action.arg) != _EXPECTED:
            failures.append(
                f"seed {seed}: wrong verb/arg "
                f"{(action.verb, action.arg)} :: {gen.text[:_FAILURE_CHARS]!r}"
            )
            continue
        good += 1
    fidelity = good / seeds if seeds else 0.0
    return Stage1(
        fidelity=fidelity,
        attempts=seeds,
        level=0 if fidelity >= _LEVEL1_MIN else 1,
        failures=tuple(failures),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage1.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/profile/stages.py tests/test_stage1.py
git commit -m "feat: stage 1 envelope fidelity, gating the later stages"
```

---

### Task 5: Stage 2 — codec landing rate and size ceiling

**Files:**
- Modify: `src/robigo/profile/stages.py`
- Create: `src/robigo/profile/fixtures/__init__.py`
- Test: `tests/test_stage2.py`

**Interfaces:**
- Consumes: `CODECS`, `PatchError`, `PythonAdapter`, `parse`, `estimate_tokens`.
- Produces: `Fixture(name: str, filename: str, original: str, expect: str)` (frozen); `FIXTURES: tuple[Fixture, ...]` (five entries); `Stage2(results: dict[str, CodecResult], failures: tuple[str, ...])` (frozen); `stage2_codecs(client, seeds, codecs=("search_replace", "whole_file")) -> Stage2`; `landing_prompt(fixture, codec) -> str`.

"Lands" means the reply parses as a `patch` action, the codec applies it, **and** the result still parses as Python. Whether the edit is semantically right is stage 4's question, not this one.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage2.py
from __future__ import annotations

from robigo.model.client import Generation
from robigo.profile.fixtures import FIXTURES
from robigo.profile.stages import landing_prompt, stage2_codecs


def _sr_reply(fixture) -> str:
    return (
        f"patch {fixture.filename}\n```python\n<<<<<<< SEARCH\n"
        f"{fixture.original}=======\n{fixture.expect}>>>>>>> REPLACE\n```\n"
    )


class _Lands:
    model = "m"

    def generate(self, prompt: str, *, seed: int) -> Generation:
        fixture = next(f for f in FIXTURES if f.filename in prompt)
        return Generation(_sr_reply(fixture), 20, 10, False)


class _Misses(_Lands):
    def generate(self, prompt: str, *, seed: int) -> Generation:
        gen = super().generate(prompt, seed=seed)
        # A one-character transcription slip: the characteristic failure.
        return Generation(gen.text.replace("SEARCH\n", "SEARCH\n "), 20, 10, False)


class _Prose:
    model = "m"

    def generate(self, prompt: str, *, seed: int) -> Generation:
        return Generation("Here is what I would change...", 20, 10, False)


def test_there_are_five_fixtures_and_each_is_self_consistent():
    assert len(FIXTURES) == 5
    for fixture in FIXTURES:
        assert fixture.original != fixture.expect
        assert fixture.original.endswith("\n") and fixture.expect.endswith("\n")


def test_a_landing_model_scores_one():
    result = stage2_codecs(_Lands(), seeds=1, codecs=("search_replace",))
    assert result.results["search_replace"].lands == 1.0
    assert result.results["search_replace"].attempts == 5


def test_a_transcription_slip_scores_zero_and_is_recorded():
    result = stage2_codecs(_Misses(), seeds=1, codecs=("search_replace",))
    assert result.results["search_replace"].lands == 0.0
    assert any("SEARCH block not found" in f for f in result.failures)


def test_prose_scores_zero_without_raising():
    result = stage2_codecs(_Prose(), seeds=1, codecs=("search_replace",))
    assert result.results["search_replace"].lands == 0.0


def test_the_prompt_describes_only_the_codec_under_test():
    fixture = FIXTURES[0]
    assert "SEARCH" in landing_prompt(fixture, "search_replace")
    assert "SEARCH" not in landing_prompt(fixture, "whole_file")


def test_a_size_ceiling_is_recorded_for_whole_file(monkeypatch):
    # whole_file must report the largest file it managed, because at a
    # small window that ceiling is the binding constraint (spec 3.3).
    result = stage2_codecs(_Lands(), seeds=1, codecs=("whole_file",))
    ceiling = result.results["whole_file"].max_file_tokens
    assert ceiling is None or ceiling > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage2.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.profile.fixtures'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/profile/fixtures/__init__.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fixture:
    """One single-defect edit with a known-correct target. A stopgap for
    stage 2 until plan 04's mutation generator replaces it; the interface
    it presents to stages.py does not change when that happens."""

    name: str
    filename: str
    original: str
    expect: str


FIXTURES: tuple[Fixture, ...] = (
    Fixture("off_by_one", "src/counter.py",
            "    return len(items) - 1\n", "    return len(items)\n"),
    Fixture("wrong_operator", "src/scale.py",
            "    return value + factor\n", "    return value * factor\n"),
    Fixture("swapped_args", "src/clamp.py",
            "    return max(high, min(low, value))\n",
            "    return max(low, min(high, value))\n"),
    Fixture("missing_return", "src/total.py",
            "    sum(values)\n", "    return sum(values)\n"),
    Fixture("inverted_test", "src/gate.py",
            "    if not ready:\n", "    if ready:\n"),
)
```

```python
# add to src/robigo/profile/stages.py
from robigo.action.codec import CODECS, PatchError
from robigo.adapters.python_ import PythonAdapter
from robigo.context.budget import estimate_tokens
from robigo.profile.fixtures import FIXTURES, Fixture
from robigo.profile.schema import CodecResult

_CODEC_HELP = {
    "search_replace": (
        "Reply with a patch action whose payload is:\n"
        "<<<<<<< SEARCH\n<the exact existing lines>\n=======\n"
        "<the replacement lines>\n>>>>>>> REPLACE"
    ),
    "whole_file": (
        "Reply with a patch action whose payload is the complete new file."
    ),
}


@dataclass(frozen=True)
class Stage2:
    results: dict[str, CodecResult]
    failures: tuple[str, ...]


def fixture_body(fixture: Fixture) -> str:
    """The file the model is shown. One definition, used by both the
    prompt and the applier -- two copies would drift and the codec would
    be applied to text the model never saw."""
    return (
        f"def f(items, value=0, factor=1, low=0, high=1, ready=True):\n"
        f"{fixture.original}"
    )


def landing_prompt(fixture: Fixture, codec: str) -> str:
    body = fixture_body(fixture)
    return (
        f"--- {fixture.filename} ---\n{body}\n"
        f"Change the line `{fixture.original.strip()}` to "
        f"`{fixture.expect.strip()}` and nothing else.\n\n"
        f"{_CODEC_HELP[codec]}\n\nYour action:\n"
    )


def stage2_codecs(
    client,
    seeds: int,
    codecs: tuple[str, ...] = ("search_replace", "whole_file"),
) -> Stage2:
    """Does a patch PARSE, APPLY, and leave valid Python? Whether the edit
    is semantically right is stage 4's question, not this one."""
    adapter = PythonAdapter()
    results: dict[str, CodecResult] = {}
    failures: list[str] = []
    for codec in codecs:
        landed = 0
        attempts = 0
        ceiling: int | None = None
        for fixture in FIXTURES:
            body = fixture_body(fixture)
            for seed in range(1, seeds + 1):
                attempts += 1
                ok, note = _try_one(client, fixture, codec, body, seed, adapter)
                if ok:
                    landed += 1
                    size = estimate_tokens(body)
                    ceiling = size if ceiling is None else max(ceiling, size)
                else:
                    failures.append(f"{codec}/{fixture.name}/s{seed}: {note}")
        results[codec] = CodecResult(
            lands=landed / attempts if attempts else 0.0,
            attempts=attempts,
            max_file_tokens=ceiling if codec == "whole_file" else None,
        )
    return Stage2(results=results, failures=tuple(failures))


def _try_one(client, fixture, codec, body, seed, adapter) -> tuple[bool, str]:
    gen = client.generate(landing_prompt(fixture, codec), seed=seed)
    try:
        action = parse(gen.text)
    except ActionParseError as exc:
        return False, str(exc)
    if action.verb != "patch":
        return False, f"emitted '{action.verb}', not a patch"
    try:
        new_text = CODECS[codec](body, action.payload or "")
    except PatchError as exc:
        return False, str(exc).split("\n")[0]
    if not adapter.syntax_ok(new_text):
        return False, "result does not parse as Python"
    return True, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage2.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/profile tests/test_stage2.py
git commit -m "feat: stage 2 codec landing rate over bundled fixtures"
```

---

### Task 6: `robigo profile` — orchestration, gating, and output

**Files:**
- Create: `src/robigo/profile/report.py`
- Modify: `src/robigo/cli.py`
- Test: `tests/test_profile_report.py`

**Interfaces:**
- Consumes: all stages, `Profile`, `verdict_for`, `plan_window`, `Recorder`, `Replayer`.
- Produces: `run_profile(client, plan, *, model, quant, family, seeds, mode, corpus="fixtures-v1") -> Profile`; `render_table(profile) -> str`; `profile_path(family) -> Path`; `profile_main(argv) -> int`; CLI dispatch on a leading `profile` argument, with `--seeds`, `--full`, `--record PATH`, `--replay PATH`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_profile_report.py
from __future__ import annotations

from pathlib import Path

from robigo.model.client import Generation
from robigo.model.geometry import WindowPlan
from robigo.profile.fixtures import FIXTURES
from robigo.profile.report import render_table, run_profile
from robigo.profile.transcript import Recorder, Replayer

PLAN = WindowPlan(window=8192, limited_by="vram", free_vram=None,
                  kv_per_token=56 * 1024)


class _Good:
    model = "m"

    def generate(self, prompt: str, *, seed: int) -> Generation:
        if "read src/target.py" in prompt:
            return Generation("read src/target.py", 5, 2, False)
        fixture = next((f for f in FIXTURES if f.filename in prompt), None)
        if fixture:
            return Generation(
                f"patch {fixture.filename}\n```python\n<<<<<<< SEARCH\n"
                f"{fixture.original}=======\n{fixture.expect}>>>>>>> REPLACE\n```\n",
                20, 10, False,
            )
        return Generation("ok", 1, 1, False)


class _CannotEnvelope(_Good):
    def generate(self, prompt: str, *, seed: int) -> Generation:
        if "read src/target.py" in prompt:
            return Generation("I would read the file.", 5, 2, False)
        raise AssertionError("stage 2 must not run after stage 1 fails")


def _run(client, **kw):
    args = dict(model="m", quant="q8_0", family="fam", seeds=1, mode="quick")
    return run_profile(client, PLAN, **{**args, **kw})


def test_a_good_model_profiles_ready_and_records_provenance():
    profile = _run(_Good())
    assert profile.verdict == "READY"
    assert profile.usable_window == 8192
    assert (profile.seeds, profile.mode) == (1, "quick")


def test_stage_one_failure_gates_stage_two():
    # The assertion lives in the fake: if stage 2 runs, it raises.
    profile = _run(_CannotEnvelope())
    assert profile.verdict == "UNUSABLE"
    assert profile.codecs == {}
    assert any("stage 2" in d for d in profile.dropped)


def test_the_table_names_the_window_limit_and_the_mode():
    table = render_table(_run(_Good()))
    assert "vram" in table
    assert "quick" in table
    # A quick profile must be visibly unquotable.
    assert "not publishable" in table


def test_a_profile_replays_identically_from_a_transcript(tmp_path: Path):
    path = tmp_path / "t.jsonl"
    live = _run(Recorder(_Good(), path))
    replayed = _run(Replayer(path))
    assert replayed.to_json() == live.to_json()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_profile_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.profile.report'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/profile/report.py
from __future__ import annotations

import os
from pathlib import Path

from robigo.model.geometry import WindowPlan
from robigo.profile.schema import Profile, verdict_for
from robigo.profile.stages import stage0_window, stage1_envelope, stage2_codecs


def profile_path(family: str) -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "robigo" / "profiles" / f"{family}.json"


def run_profile(
    client,
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
    """Stages run cheapest-first and gate each other: a family that
    cannot emit an action never reaches the codec stage (spec 5)."""
    dropped: list[str] = []
    stage0 = stage0_window(client, plan)
    if not stage0.verified:
        dropped.append(f"stage 0: {stage0.note}")
    stage1 = stage1_envelope(client, seeds)

    codecs = {}
    if stage1.fidelity >= 0.5:
        codecs = stage2_codecs(client, seeds).results
    else:
        dropped.append(
            f"stage 2: not run, envelope fidelity {stage1.fidelity:.2f} "
            f"below 0.50"
        )

    window = stage0.window or plan.window
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
        lines.append("  NOTE          quick mode — not publishable")
    return "\n".join(lines)
```

Then in `src/robigo/cli.py`, add the subcommand dispatch at the top of `main`:

```python
# in src/robigo/cli.py — first lines of main()
def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "profile":
        return profile_main(argv[1:])
```

and add `profile_main` to `cli.py`:

```python
def profile_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="robigo profile")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("ollama", "llamacpp"), default="ollama")
    parser.add_argument("--host", default=None)
    parser.add_argument("--gguf", type=Path, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--full", action="store_true",
                        help="all stages at 10 seeds; the only publishable mode")
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--replay", type=Path, default=None)
    parser.add_argument("--kv-bits", dest="kv_bits", type=int,
                        choices=(16, 8), default=16)
    args = parser.parse_args(argv)

    seeds = 10 if args.full else args.seeds
    mode = "full" if args.full else "quick"
    try:
        plan = plan_window(args.backend, args.model, args.host or "", None,
                           kv_bits=args.kv_bits, gguf_path=args.gguf)
    except (GeometryError, OSError) as exc:
        print(f"cannot determine the usable window: {exc}")
        return OUTCOMES["infrastructure"]

    client = Replayer(args.replay) if args.replay else build_client(
        argparse.Namespace(backend=args.backend, model=args.model,
                           window=plan.window, host=args.host,
                           num_predict=1024)
    )
    if args.record:
        client = Recorder(client, args.record)

    family = args.model.replace(":", "-").replace("/", "-")
    profile = run_profile(client, plan, model=args.model, quant=_quant(args.model),
                          family=family, seeds=seeds, mode=mode,
                          kv_bits=args.kv_bits)
    print(render_table(profile))
    path = profile_path(family)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.to_json(), encoding="utf-8")
    print(f"written to {path}")
    return 0 if profile.verdict != "UNUSABLE" else OUTCOMES["refused"]


def _quant(model: str) -> str:
    tail = model.rsplit("-", 1)[-1]
    return tail if tail.lower().startswith("q") else "unknown"
```

with imports added to `cli.py`:

```python
from robigo.profile.report import profile_path, render_table, run_profile
from robigo.profile.transcript import Recorder, Replayer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_profile_report.py -v` then `pytest -q`
Expected: PASS, 4 tests; full suite green

- [ ] **Step 5: Commit**

```bash
git add src/robigo/profile/report.py src/robigo/cli.py tests/test_profile_report.py
git commit -m "feat: robigo profile with staged gating and replayable transcripts"
```

- [ ] **Step 6: Profile the real roster and record it**

Run:
```bash
robigo profile --model qwen2.5-coder:7b-instruct-q8_0 --record /tmp/qwen7b.jsonl
robigo profile --model granite-code:8b-instruct-q8_0  --record /tmp/granite8b.jsonl
robigo profile --model codegemma:7b-instruct-q8_0     --record /tmp/codegemma7b.jsonl
```
Expected: three profiles written. granite should read `LIMITED` on window
alone (4096 < the 8192 floor). Then verify reproducibility with no GPU:
```bash
robigo profile --model qwen2.5-coder:7b-instruct-q8_0 --replay /tmp/qwen7b.jsonl
```
Expected: byte-identical JSON to the recorded run.

- [ ] **Step 7: Commit the recorded transcripts as replay fixtures**

```bash
mkdir -p tests/transcripts && cp /tmp/*.jsonl tests/transcripts/
git add tests/transcripts
git commit -m "test: recorded profile transcripts for GPU-free replay"
```

---

## Amendment (ruled 2026-08-10): stage 2 must present the loop's envelope

Measured after Task 6, against the live daemon
(`qwen2.5-coder:7b-instruct-q8_0`, seed 1, `search_replace`, all five fixtures):

| prompt | replies that `parse` accepts |
|---|---|
| `landing_prompt` as shipped | **0 / 5** |
| `SYSTEM` + `_CODEC_HELP[codec]` + `landing_prompt` | **5 / 5** |

**Stage 2's 0% landing rate across every roster model is an artifact of its own
prompt, not a property of any family.** `landing_prompt` shows the SEARCH/REPLACE
payload template but never states the *action* syntax — that the reply must be
`patch <path>` on a line of its own with a fenced payload. So the model returns a
unified diff in a code fence, `parse` finds no action, and the codec is never
exercised. Every failure reads `no action found`.

We already knew the model can do this: earlier the same model, at the same codec,
landed a patch and repaired a real bug through the loop — because the loop's
prompt carries `SYSTEM`'s action list. Stage 1 scores 100% for the same reason:
`ENVELOPE_PROMPT` lists the actions. Only stage 2 omits them.

*Invariant:* **stage 2 presents the same envelope the loop presents.** The stage
exists to predict what the loop will get, so it must reuse `render.SYSTEM` and
`render._CODEC_HELP[codec]` rather than paraphrase them — the same principle that
already makes stage 1 use the real `parse` and stage 2 use the real `CODECS`. A
parallel prompt is free to drift, and this is what drift looks like: a profiler
that reports every model unusable.

Falsification test: assert `landing_prompt` contains the action-list text and the
codec help, sourced from those modules rather than duplicated as literals. A test
comparing against a copied string would pass while the loop's own prompt changed
underneath it.

**Second, smaller finding from the same run.** With the envelope added, 2 of the 5
parsed as `patch --- src/clamp.py ---` — the model copied the `--- path ---`
decoration from the file header into the action argument, giving a parse success
with an unusable path. The loop presents files with the same header and does not
usually suffer this, so it is not fatal, but stage 2 should not be measuring the
family against a path the header invited it to mistype. Present the file so the
path to patch is unambiguous, and say in the report what the landing rate becomes
once both are fixed.

**Do not tune the fixtures to raise the number.** The point is to measure what the
loop would get. If the rate is still low after the envelope is correct, that is a
result — and `fixtures-v1` is a stopgap that plan 04 replaces with a
mutation-generated corpus anyway.

## Whole-branch review fix wave (ruled 2026-08-10)

Three Criticals, all confirmed against this branch's own committed data. Two are
mine.

**C1 — stage 0 reports a window twice what any probe demonstrated.** The filler
is `"token "`, which tokenizes at ~6 chars/token, while `_default_probe` sizes at
3. Measured from `tests/transcripts/codegemma7b.jsonl` row 0 — the stage-0 probe
aimed at 8192 tokens:

| aimed at | chars sent | robigo's estimator | **the server counted** | reported |
|---|---|---|---|---|
| 8192 tok | 24576 | 7448 tok | **4119 tok** | `usable_window: 8192, verified=True` |

So `verified=True` was claimed for a window **0.50×** of which was ever accepted.
This is the Task 3 amendment's own defect at four thousand times the scale: I
ruled that `verified=True` must mean "a probe of this size was accepted", fixed
the one-token version, and the ~4000-token version was in the same function.

The fix was already written down and dropped. Measured fact 2 of Task 3's
pre-execution section says the rejection **names the real token count** and to
*use it*. `stages.py` discards the `Generation`, so `tokens_in` — present on every
accepted probe, from the server's own tokenizer — is never read. Read it. Bisect
and report on the server's count, not on a character estimate. And reconcile
`_CHARS_PER_TOKEN = 3` with `budget.CHARS_PER_TOKEN = 3.3`: two ratios 10% apart,
one of them sizing the probe.

Delete the docstring claim that the estimate "can never cause an incorrect window
to be reported right". It is false as written, and a comment asserting the
opposite of the behaviour is worse than the behaviour.

**C2 — all three committed replay fixtures are dead, and the plan's replay
criterion is false at HEAD.** `68da8c2` recorded them; `8a3ddc9` — my envelope fix
— changed `landing_prompt` and nothing was re-recorded. **0 of 30** stage-2 keys
match. No test loads a committed transcript, so nothing caught it. My fix was
right and its blast radius was not checked.

Re-record all three against HEAD. Then add a test that actually replays a
committed transcript, so this cannot go stale silently again — a fixture nothing
loads is a fixture nothing protects.

`granite8b.jsonl` is also **two runs appended into one file**: 67 rows, windows
`{0, 4096}`, 33 duplicate keys, merged silently by `CallReplayer`, which then
reports the last row's window while handing back the first run's replies. Cause:
`CallRecorder` opens `"a"`. Truncate, or refuse an existing path.

**C3 — `Profile.training_ctx` is not the training context.** It is assigned
`plan.window`, which is `min(training_ctx, vram, user_cap)`. Whenever VRAM binds,
the field reports the VRAM-derived window as the model's training context, and
`training_ctx == usable_window` — a state no real model can be in. The live
granite run wrote `training_ctx: 0`. Verbatim plan text, pinned by nothing:
mutating it to `999999` leaves all 90 profile tests green. Thread the real
`Geometry.training_ctx` through, or drop the field and state that it was dropped.

**I1 + I4 — two absences that read as measurements.** A profile that verified no
window at all returns `LIMITED`, the same verdict a working 4096-token model gets,
and stages 1 and 2 still run — at `num_ctx: 0`, where the daemon substitutes its
own default — so the profile ships `usable_window: 0` beside `envelope 100%` and
`lands 100%`. Stage 0 must be able to stop the run, per the architecture's own
"each able to stop the run". And `payload_corruption`/`repeat_rate` are hardcoded
`None` with no `dropped` entry: mutating both to `0.0` — "measured, no corruption"
— leaves all 90 tests green. Anything not measured is stated as dropped.

Everything else in that review is carried debt with rulings. Two items go to
`CARRIED-DEBT.md` addressed to plan 04 specifically, because both sit on its first
code path: `best_codec()` has no landing floor and returns a 0%-landing codec
(granite measured exactly that), and `corpus` is a kwarg default rather than
derived from the fixtures used, so the corpus swap plan 04 exists to perform will
mislabel every profile it produces.

## Done when

- `pytest -q` green; `robigo profile --replay <fixture>` reproduces a
  committed profile byte-for-byte with no model running.
- Three real profiles exist for the local roster, and granite reads `LIMITED`
  on window alone.
- A model that fails stage 1 never runs stage 2, and the profile says so in
  `dropped` rather than reporting an empty codec table with no explanation.
- Quick-mode output is visibly marked not publishable.
- Both quantization covariates (`quant`, `kv_bits`) appear in every profile.

Note what this plan does **not** do: nothing reads the written profile yet.
`robigo run` still takes `--codec` and `--window` from the command line.
Wiring `Profile.best_codec()` and `usable_window` into the loop — spec §5.4's
`READY`/`LIMITED`/`UNUSABLE`/unprofiled behaviour, including the
announce-conservative-defaults rule — is the first task of plan 04, where a
real corpus makes the codec choice worth trusting.

---

## After this plan

Step 6 of the spec's build order is the decision point, not more work:
**read stage 4's number and apply the 40% kill criterion (spec §0.2).** Stage
4 arrives in plan 05; plan 04 is the mutation corpus that makes it meaningful.
Do not start memory (spec §4) or the TypeScript adapter before that gate.
