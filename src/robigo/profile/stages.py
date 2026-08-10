# src/robigo/profile/stages.py
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable

from robigo.action.codec import CODECS, PatchError
from robigo.action.verbs import ActionParseError, parse
from robigo.context.budget import CHARS_PER_TOKEN, estimate_tokens
from robigo.model.client import ContextOverflowError, Generation, ModelClient
from robigo.model.geometry import WindowPlan
from robigo.profile.fixtures import FIXTURES, Fixture
from robigo.profile.schema import CodecResult

_PROBE_SEED = 0
"""Fixed, never derived from time, randomness, or call count. Replay
(`robigo.profile.transcript.CallReplayer`) is keyed on
`(model, prompt, seed)` -- a probe seed that drifted between the recording
run and a replay run would change the key on every call, and every
replayed call would raise `TranscriptMiss` instead of reproducing the
profile."""

_FILLER_WORD = "token "


@dataclass(frozen=True)
class Stage0:
    """The result of verifying a `WindowPlan` against a real server.

    `window` is never larger than the `plan.window` that was probed --
    stage 0 only ever confirms or shrinks the planned window, it does not
    search upward for a bigger one (see `stage0_window`'s scope note). It
    is also never larger than what a real, accepted call actually
    demonstrated: `window` is read from `Generation.tokens_in` -- the
    server's OWN tokenizer's count for the exact prompt it accepted --
    never computed from the char-estimated target that prompt was built
    for (whole-branch review C1, ruled 2026-08-10; see `stage0_window`'s
    docstring). `verified` is True exactly when some probe was accepted;
    `note` explains what happened in either case.
    """

    window: int
    verified: bool
    note: str


def _default_probe(target: int) -> str:
    """Build a prompt intended to represent roughly `target` tokens, sized
    in characters via `budget.CHARS_PER_TOKEN` -- the one chars-per-token
    ratio this project maintains, reused here rather than a second,
    independent guess (whole-branch review C1, ruled 2026-08-10: this
    module used to keep its own `_CHARS_PER_TOKEN = 3`, ten percent off
    `budget.CHARS_PER_TOKEN = 3.3`, sizing every probe on a number nothing
    else in the project used).

    The ratio's accuracy does not affect what `stage0_window` reports --
    see that function's docstring, point 3 -- only how many probe rounds
    the search takes and how close a probe lands to the real boundary.

    Length is computed directly as `int(target * CHARS_PER_TOKEN)`, then
    the filler word is repeated and SLICED to that exact character count
    -- never rounded down to a whole number of words. That precision is
    load-bearing, not cosmetic (amendment to Task 3, ruled 2026-08-10): an
    earlier version floored to whole words (`target * CHARS_PER_TOKEN //
    len(_FILLER_WORD)` words), which made length a step function of
    `target` -- every `target` and `target + 1` sharing one floor value
    aliased onto the SAME prompt. Bisection could then land on the larger
    of an aliased pair and report it as "verified", even though its probe
    was byte-identical to the smaller, equally-accepted one. Slicing to an
    exact length makes `target -> len(prompt)` strictly increasing for any
    `CHARS_PER_TOKEN > 1` (3.3 is), so no two distinct targets can ever
    share a probe.
    """
    length = max(int(target * CHARS_PER_TOKEN), 1)
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
       tokenizer -- it is a measurement, not just a "no". On acceptance,
       the reply itself carries the same kind of measurement:
       `Generation.tokens_in` is the server's own tokenizer's count for
       the exact prompt it just accepted. This function acts on both
       halves of that principle -- treating every rejection as
       information to narrow the search with (via bisection on real
       accept/reject answers), and treating every acceptance's
       `tokens_in` as the number to report, never the char-estimated
       target that built the probe (whole-branch review C1, ruled
       2026-08-10 -- see point 3 below for why the earlier version got
       this half wrong).
    3. Aiming at exactly the window risks a false negative from the
       char-to-token estimate: a prompt aimed at "exactly N tokens" can
       land a few percent over N on the real tokenizer and be rejected
       even though N genuinely fits. The estimate's error therefore only
       ever costs probe rounds and search precision -- it never changes
       what gets REPORTED, because the reported `window` is always the
       winning probe's own `Generation.tokens_in`, read off the server's
       real tokenizer, not derived from the target that built the probe.
       An earlier version of this paragraph claimed the estimate's error
       "can never cause an incorrect window to be reported right" --
       false as written, and the false half is exactly what C1 found:
       that version reported the char-estimated TARGET (or `plan.window`
       verbatim, on immediate acceptance) instead of `tokens_in`, so a
       filler word whose real density the estimate underweighted by
       roughly 2x reported a window twice what the accepted probe
       actually contained. Measured on this project's own committed
       `codegemma7b.jsonl` transcript: a probe aimed at 8192 tokens sent
       24576 characters, the server's own tokenizer counted 4119 tokens
       for it, and `usable_window: 8192, verified=True` shipped anyway --
       half the claimed window was never demonstrated. Reading
       `tokens_in` instead makes that class of error structurally
       impossible: the reported number IS what the server said, not a
       computation that merely hopes to track it.

    Scope boundary: `hi` starts at `plan.window` and never rises above
    it -- this function only ever verifies *at or below* the planned
    window, never discovers that the model could do more. That is
    deliberate: `plan.window` is a VRAM-derived ceiling, and a request
    above it risks OOMing the real daemon rather than failing cleanly.
    (It also follows from the server's own acceptance semantics that
    `tokens_in` can never exceed `plan.window` on an accepted call: every
    probe is sent with `num_ctx = plan.window`, per fact 1 above rejection
    is decided against that exact figure, so a call that does NOT raise
    already proves its real token count fit under it.)

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

    def try_probe(target: int) -> Generation | None:
        try:
            return client.generate(build(target), seed=_PROBE_SEED)
        except ContextOverflowError:
            return None

    gen = try_probe(plan.window)
    if gen is not None:
        return Stage0(window=gen.tokens_in, verified=True, note="probe accepted")

    # The full window was rejected. Bisect between "definitely accepted"
    # (0, untested but assumed) and "definitely rejected" (plan.window,
    # just tested) to find the largest size the server actually accepts,
    # rather than giving up on the first rejection or falling back to an
    # arbitrary fixed fraction that might overshoot the same way. `best`
    # keeps the Generation from the last (== largest target, since lo only
    # ever advances on acceptance) accepted probe, so the final report can
    # read its real `tokens_in` rather than re-deriving a number from `lo`.
    lo, hi = 0, plan.window
    best: Generation | None = None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        gen = try_probe(mid)
        if gen is not None:
            lo = mid
            best = gen
        else:
            hi = mid

    if best is None:
        return Stage0(
            window=0,
            verified=False,
            note=f"every probe from {plan.window} down was rejected",
        )
    return Stage0(
        window=best.tokens_in,
        verified=True,
        note=f"planned {plan.window} rejected; verified at {best.tokens_in}",
    )


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
    """The result of asking the family to drive the action envelope with
    no code reasoning involved (spec 5, stage 1). `fidelity` is the
    fraction of `attempts` (== the `seeds` requested; see `stage1_envelope`)
    whose reply, run through the real `parse`, produced exactly the one
    action that was asked for. `failures` keeps the raw, un-truncated-past-
    `_FAILURE_CHARS` model text for every attempt that did not count -- the
    diagnostic material for why a family failed, not just that it did.

    `level` is NOT "the envelope level this family needs" -- only level 0
    (this stage's own fixed one-action envelope) is ever probed, so `level`
    can only ever report on level 0, never confirm level 1:

    - `0` means level 0 was measured sufficient (`fidelity >= _LEVEL1_MIN`)
      -- a real result, backed by the `seeds` attempts actually run.
    - `1` means level 0 was measured INSUFFICIENT. It is a recommendation
      to try the two-step envelope (spec 2.3: constrain the header, leave
      the payload free) next, not a report that level 1 was tried and
      found to work -- nothing in this stage ever sends a level-1 prompt.
      Measuring level 1 itself is later work (not yet built as of this
      task).

    Do not read a `1` here as "level 1 verified" anywhere downstream
    (including `Profile.envelope_level`, which is `stage1.level` passed
    through unchanged) -- that reading is exactly the class of overclaim
    this project's review has repeatedly caught elsewhere (`meta.json`'s
    `rung`, stage 0's off-by-one `verified=True`): a field naming a
    capability that was never actually run.
    """

    fidelity: float
    attempts: int
    level: int
    """0 if level 0 (the only envelope this stage probes) was measured
    sufficient; 1 if it was measured insufficient. Never confirms level 1
    itself -- see the class docstring."""
    failures: tuple[str, ...]


def stage1_envelope(client: ModelClient, seeds: int) -> Stage1:
    """Can this family drive the envelope at all? No code reasoning is
    involved, so a failure here is purely about the action surface -- and
    it gates every later stage (spec 5): a family that cannot reliably
    emit a parseable, correctly-shaped action never reaches the codec
    measurement, because there is nothing there to measure.

    Every attempt is scored against the real `parse` (`robigo.action.
    verbs`), the same parser the loop uses -- not a reimplementation --
    because the question is what the loop would have accepted, and only
    the loop's own parser can answer that. A reply that raises
    `ActionParseError` counts as a failure; so does a reply that parses
    cleanly but names the wrong verb or argument (spec 2.3's distinction:
    driving the envelope and following the instruction are different
    findings, and only the latter needs no free-form reasoning at all to
    get right).

    Each attempt uses a distinct seed (1..seeds) so replies vary the way a
    real deployment would, and so a `CallRecorder` transcript of this run
    replays deterministically under `CallReplayer` -- (model, prompt,
    seed) is the same fixed `ENVELOPE_PROMPT` paired with a different seed
    each time, never a prompt that itself varies by call count or time.

    The returned `level` is derived from THIS stage's fidelity alone
    (`0` if `fidelity >= _LEVEL1_MIN` else `1`) -- level 1 (the two-step
    envelope) is never itself probed here, so a `1` means "level 0 was
    measured insufficient", not "level 1 was verified sufficient". See
    `Stage1`'s docstring for why that distinction matters.
    """
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


_FUNCTION_HEADER = (
    "def f(items, value=0, factor=1, low=0, high=1, ready=True):\n"
)
_FILLER_BODY = "        pass\n"


@dataclass(frozen=True)
class Stage2:
    """The result of asking the family to land a real edit through a real
    codec (spec 5, stage 2): does the reply parse as a `patch` action, does
    the codec it names apply without raising, and does the result still
    parse as Python? All three, per attempt -- semantic correctness (did
    the edit actually fix anything) is stage 4's question, not this one.

    `results` is keyed by codec name and NEVER pooled across codecs or
    across fixtures into one number -- a family that lands cleanly on
    `search_replace` and never on `whole_file` is two different findings,
    and averaging them together would hide the one the loop actually needs
    to pick a codec with. `failures` keeps one line per non-landing attempt
    (`codec/fixture/seed: reason`), naming which of the three ways it
    failed -- the diagnostic material, same role as `Stage1.failures`.
    """

    results: dict[str, CodecResult]
    failures: tuple[str, ...]


def fixture_body(fixture: Fixture) -> str:
    """The file the model is shown. One definition, used by both the
    prompt and the applier -- two copies would drift and the codec would
    be applied to text the model never saw.

    A fixture whose `original` is a bare compound-statement header (the
    shape `inverted_test` takes: a line ending in `:`, with no suite of
    its own) is not a valid function body by itself -- `ast.parse` demands
    an indented block under it, and the brief's own sample wraps every
    fixture the same way regardless of shape, which made `inverted_test`
    fail `ast.parse` both before AND after its own patch (confirmed by
    hand: `def f(...):\\n    if not ready:\\n` raises "expected an indented
    block after 'if' statement"). A nested `pass` is appended only when the
    line needs one, so the wrapper stays generic across whatever shape a
    fixture's single line takes, rather than special-casing one fixture's
    name. It sits after `fixture.original` and is never inside a matched
    SEARCH span, so a codec's replacement of the header line leaves it in
    place and the result stays syntactically complete either way.
    """
    body = _FUNCTION_HEADER + fixture.original
    if fixture.original.rstrip("\n").endswith(":"):
        body += _FILLER_BODY
    return body


def landing_prompt(fixture: Fixture, codec: str) -> str:
    """Presents the SAME action envelope the loop itself sends -- `render.
    SYSTEM` (the action list: `read`/`find`/`patch`/`run`/`done`, one per
    reply) and `render._CODEC_HELP[codec]` (the codec-specific payload
    format) -- rather than a paraphrase of them (amendment ruled
    2026-08-10). Measured live against `qwen2.5-coder:7b-instruct-q8_0`:
    the prior version, which showed the SEARCH/REPLACE payload template but
    never stated that a reply must be `patch <path>` on a line of its own
    with a fenced payload, scored 0/5 -- every reply was a bare unified
    diff that `parse` correctly rejected as "no action found", because the
    codec was never reached. Adding the envelope text (unchanged) scored
    5/5 on the same daemon and seed. The loop's own prompt already proved
    this shape works (the same model, same codec, repaired a real bug
    through it); a stage that omits the shape it is trying to predict was
    never measuring the model.

    `SYSTEM`, `_CODEC_HELP`, and `_TRAILER` are imported here, inside the
    function, rather than once at module load -- not for the usual
    circular-import reason (`render` does not import anything from
    `profile`), but so this reads `render`'s CURRENT attributes on every
    call. A module-level `from ... import SYSTEM` would instead bind a
    private copy of today's string once, at import time, which is exactly
    the "parallel prompt free to drift" failure mode the amendment names --
    a copy that starts identical to `render.SYSTEM` and silently stops
    tracking it the moment the loop's own prompt changes.
    `test_the_prompt_is_sourced_from_render_not_copied` monkeypatches
    `render.SYSTEM`/`render._CODEC_HELP` and asserts the substitution shows
    up here, which only a live, per-call read can pass.

    Describes ONLY the codec under test (pinned by
    `test_the_prompt_describes_only_the_codec_under_test`): a family being
    scored on `whole_file` must never see SEARCH/REPLACE syntax it could
    borrow from, or a landing measured under one codec's prompt would
    really be measuring a different one.

    The file is introduced as `File to patch: <path>` on its own line, in
    plain prose, rather than the `--- <path> ---` decoration `render.
    _scope_section` uses for real turns (amendment's second finding): on
    the SAME live run, 2 of 5 `whole_file` replies that did state an action
    parsed as `patch --- src/clamp.py ---` -- the model had copied the
    dashes into the argument, a parse success with an unusable path. The
    loop mostly avoids this because by the time a real run reaches `patch`,
    the model has usually already typed that exact path itself in an
    earlier `read` turn; stage 2's first and only turn has no such prior
    turn to anchor on, so the header text is the sole place the path
    appears -- and here, unlike `_CODEC_HELP`, fidelity to the loop's exact
    formatting is not the goal; not inviting a mistyped path is.
    """
    from robigo.context.render import SYSTEM, _CODEC_HELP, _TRAILER

    body = fixture_body(fixture)
    return (
        f"{SYSTEM}\n{_CODEC_HELP[codec]}\n\n"
        f"File to patch: {fixture.filename}\n\n{body}\n"
        f"Change the line `{fixture.original.strip()}` to "
        f"`{fixture.expect.strip()}` and nothing else.\n\n"
        f"{_TRAILER}\n"
    )


def _parses_as_python(text: str) -> bool:
    """"Lands" (spec 5, stage 2) requires the patched result to still be
    valid Python -- exactly what `ast.parse` decides, and nothing more.

    `robigo.adapters.python_.PythonAdapter.syntax_ok` performs this
    identical one-line check, but it lives on a class whose OTHER method
    (`.run`) shells out to pytest and runs the fixture's real test suite --
    exactly what this stage must never do (see `stage2_codecs`'s
    docstring). Importing that class here to reach one pure method would
    sit a test-runner one attribute access away from a landing-rate loop
    that iterates `seeds * fixtures * codecs` times; a bare `ast.parse`
    call gets the identical answer with no such neighbour for a future
    edit to reach for by reflex. Not used: dropped deliberately, not by
    oversight."""
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def stage2_codecs(
    client: ModelClient,
    seeds: int,
    codecs: tuple[str, ...] = ("search_replace", "whole_file"),
) -> Stage2:
    """Does a patch reply PARSE as a patch action, does the named codec
    APPLY it, and does the result still PARSE AS PYTHON? Whether the edit
    is semantically right is stage 4's question, not this one -- so this
    function never runs the fixture's tests, only `ast.parse` on the
    result (spec 5, stage 2).

    Every fixture is synthesized in memory by `fixture_body` and every
    codec (`robigo.action.codec.CODECS`, the same table the loop uses) is
    applied to that in-memory string, never to a file on disk -- there is
    no working tree here for a patch to land in or corrupt, real or
    temporary, because nothing this stage measures is ever written
    anywhere.

    Scored per (fixture, seed) for each codec in `codecs`, `attempts`
    always equals `len(FIXTURES) * seeds` for that codec (five fixtures by
    design -- see `FIXTURES`). `results` is never pooled across codecs:
    each codec gets its own `CodecResult`, because a family's
    `search_replace` landing rate and its `whole_file` landing rate are
    the two different numbers the loop needs to pick between.

    `max_file_tokens` is tracked only for `whole_file`, and only from
    attempts that actually landed -- a size that was ATTEMPTED but did not
    land is not evidence the codec "manages" that size, and reporting it
    would be exactly the ceiling-that-isn't-one Stage 0's `verified=True`
    off-by-one already cost this project once (spec 3.3: at a small
    window, whole_file's inability to re-emit a large file is the binding
    constraint, so this ceiling is the number that constraint is measured
    by). `search_replace`'s payload size does not scale with the file's
    size the way `whole_file`'s does, so its `max_file_tokens` is always
    `None` -- reporting a number there would imply a capability this
    stage never measured.
    """
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
                ok, note = _try_one(client, fixture, codec, body, seed)
                if ok:
                    landed += 1
                    if codec == "whole_file":
                        size = estimate_tokens(body)
                        ceiling = size if ceiling is None else max(ceiling, size)
                else:
                    failures.append(f"{codec}/{fixture.name}/s{seed}: {note}")
        results[codec] = CodecResult(
            lands=landed / attempts if attempts else 0.0,
            attempts=attempts,
            max_file_tokens=ceiling,
        )
    return Stage2(results=results, failures=tuple(failures))


def _try_one(
    client: ModelClient, fixture: Fixture, codec: str, body: str, seed: int
) -> tuple[bool, str]:
    """One (fixture, seed) attempt under one codec. Returns (landed, note)
    rather than raising: a non-landing attempt is a normal, expected
    outcome for this stage (unlike a `ModelError`, which is infrastructure
    failure and does propagate, same as every other stage in this module),
    and the note is kept verbatim in `Stage2.failures` as diagnostic
    material.

    Checks the three ways a patch can fail to land, each with a note that
    names which one happened:
      1. the reply does not parse as a `patch` action at all -- either it
         does not parse as any action (`ActionParseError`), or it parses
         cleanly but names a different verb;
      2. it parses as `patch` but the named codec raises `PatchError`
         applying it (e.g. the SEARCH block does not match);
      3. it applies without raising but the result does not parse as
         Python.
    Any one of the three scores this attempt as not landed; only clearing
    all three counts as landed.
    """
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
    if not _parses_as_python(new_text):
        return False, "result does not parse as Python"
    return True, ""
