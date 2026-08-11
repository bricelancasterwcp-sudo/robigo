# tests/test_profile_report.py
from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

import pytest

from robigo.loop import OUTCOMES, RunResult
from robigo.model.client import ContextOverflowError, Generation
from robigo.model.geometry import WindowPlan
from robigo.profile import report as report_module
from robigo.profile import repair as repair_module
from robigo.profile.corpus_io import CorpusRecord
from robigo.profile.discipline import Stage5
from robigo.profile.fixtures import FIXTURES
from robigo.profile.repair import Stage4
from robigo.profile.report import profile_path, render_table, run_profile
from robigo.profile.schema import SUPPORTED_FLOOR, Profile
from robigo.profile.transcript import CallRecorder, CallReplayer
from robigo.profile.verify import Baseline, SuiteState

# weights_bytes/overhead_bytes are required positional fields of the real
# WindowPlan (src/robigo/model/geometry.py) but are read by nothing this
# module exercises -- only .window, .limited_by and .kv_per_token matter to
# run_profile/render_table. The task-6 brief's own PLAN literal has only
# four fields, which raises TypeError against the shipped six-field
# dataclass (the same stale-sample defect test_stage0.py and test_stage1.py
# already found in earlier tasks; see the task-6 report). Set to 0 here for
# the same reason those files did.
#
# training_ctx=32768 is deliberately far from window=8192 (whole-branch
# review C3, ruled 2026-08-10): Profile.training_ctx used to be assigned
# plan.window, so a PLAN whose training_ctx equalled its window could not
# tell a correct implementation apart from the bug. See
# test_training_ctx_is_the_models_real_training_context_not_the_planned_window.
PLAN = WindowPlan(window=8192, limited_by="vram", free_vram=None,
                  kv_per_token=56 * 1024, weights_bytes=0, overhead_bytes=0,
                  training_ctx=32768)


class _Good:
    model = "m"
    # Present so this fake can also be wrapped in CallRecorder/CallReplayer
    # (see test_a_profile_replays_identically_from_a_transcript and the CLI
    # round-trip test below) -- the brief's own `_Good` literal omits both,
    # which raises AttributeError inside CallRecorder.__init__ the moment
    # it reads client.window (same class of stale-sample defect as the
    # four-field WindowPlan; mirrors test_stage1.py's `_Scripted` fake).
    window = 8192
    num_predict = 512

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
        # The catch-all: stage 0's filler probe, and nothing else (the two
        # branches above account for stage 1's envelope prompt and stage
        # 2's fixture prompts). Reports tokens_in=self.window (8192)
        # rather than a fixed placeholder -- since whole-branch review C1
        # (ruled 2026-08-10) made stage0_window report Generation.tokens_in
        # instead of plan.window, a hardcoded small tokens_in here would
        # make PLAN.window (8192) diverge from the verified usable_window
        # this file's tests assert (e.g. test_a_good_model_profiles_ready_
        # and_records_provenance's usable_window == 8192) for a reason
        # unrelated to what any of those tests actually exercise.
        return Generation("ok", self.window, 1, False)


class _CannotEnvelope(_Good):
    """Fails stage 1 on every seed. The assertion that stage 2 never runs
    lives IN the fake, not in a post-hoc check on the result: a call whose
    prompt names one of the stage-2 fixtures raises, so if `run_profile`
    ever reached stage 2 after this gate should have closed, the call
    itself -- not just its absence from `codecs` -- blows up the test.

    The brief's own version of this fake raises on ANY prompt that isn't
    the envelope one, which also fires on stage 0's window-verification
    probe (plain filler text, sent before stage 1 even runs) -- a false
    positive that made the test fail every time for a reason unrelated to
    what it claims to check. Narrowed here to fire only on a stage-2-shaped
    prompt (one naming a FIXTURES filename, `landing_prompt`'s own marker),
    which stage 0 and stage 1's fixed prompts never contain."""

    def generate(self, prompt: str, *, seed: int) -> Generation:
        if "read src/target.py" in prompt:
            return Generation("I would read the file.", 5, 2, False)
        if any(fixture.filename in prompt for fixture in FIXTURES):
            raise AssertionError("stage 2 must not run after stage 1 fails")
        return super().generate(prompt, seed=seed)


class _EnvelopeOkNothingLands(_Good):
    """Passes stage 1 (drives the envelope perfectly, every seed) but never
    lands a codec edit in stage 2. Distinct from `_CannotEnvelope`: here
    stage 2 genuinely RUNS and genuinely finds nothing, rather than never
    running at all -- the two must be distinguishable in the resulting
    `Profile`, which is the whole reason `Profile.dropped` exists."""

    def generate(self, prompt: str, *, seed: int) -> Generation:
        if "read src/target.py" in prompt:
            return Generation("read src/target.py", 5, 2, False)
        return Generation("I refuse to patch that.", 5, 2, False)


class _HalfEnvelope(_Good):
    """Exactly half of stage 1's seeds drive the envelope correctly --
    fidelity lands EXACTLY on ENVELOPE_FIDELITY_MIN (0.5), not above or
    below it. Pins the gate's `>=` (matching verdict_for's own boundary):
    a `>` mutant would close the gate here even though 0.5 is the
    documented cutoff for "usable"."""

    def generate(self, prompt: str, *, seed: int) -> Generation:
        if "read src/target.py" in prompt:
            if seed == 1:
                return Generation("read src/target.py", 5, 2, False)
            return Generation("nope", 5, 2, False)
        return super().generate(prompt, seed=seed)


class _WindowNeverVerifies:
    """Every stage-0 probe is rejected, at every size the bisection tries
    -- including the smallest one -- so `stage0.window` stays 0 and
    `verified` stays False.

    The assertion that stage 1 and stage 2 never run lives IN the fake
    (same trick as `_CannotEnvelope`): any prompt that is not stage 0's
    plain filler probe raises AssertionError, so if `run_profile` ever
    reached stage 1 or stage 2 after stage 0 found no usable window, the
    call itself -- not just an absence downstream -- blows up the test.
    Before the I1 gate existed (whole-branch review, ruled 2026-08-10),
    this exact scenario's envelope and codec prompts were answered
    exactly like `_Good`'s, landing `envelope 100%` and `lands 100%`
    beside a headline `usable_window: 0`, and the verdict read LIMITED --
    the same verdict a working 4096-token model gets."""

    model = "m"
    window = 8192
    num_predict = 512

    def generate(self, prompt: str, *, seed: int) -> Generation:
        if "read src/target.py" in prompt or any(
            fixture.filename in prompt for fixture in FIXTURES
        ):
            raise AssertionError(
                "stage 1/2 must not run after stage 0 found no usable window"
            )
        raise ContextOverflowError("rejected")


class _ShrinksWindowBelowTheFloor(_Good):
    """Accepts stage-0 probes only up to `_ACCEPTED_CHARS` characters --
    comfortably below both SUPPORTED_FLOOR(8192) and PLAN.window(8192) --
    so stage0_window's bisection VERIFIES a real window smaller than the
    plan, rather than either fully accepting or fully rejecting it. Stage
    1 and stage 2 behave exactly like `_Good`, isolating verdict/window
    interaction from everything else.

    Reports `tokens_in=len(prompt)` on acceptance, the same "play the
    server honestly" convention test_stage0.py's fakes use (C1, ruled
    2026-08-10): stage0_window now reports `Generation.tokens_in`, not a
    char-derived target, so a fake hardcoding some other tokens_in would
    make the verified window it produces arbitrary rather than tied to
    what this fake actually accepted."""

    _ACCEPTED_CHARS = 5000  # comfortably under SUPPORTED_FLOOR once
    # reported back 1:1 as this fake's token count.

    def generate(self, prompt: str, *, seed: int) -> Generation:
        if "read src/target.py" in prompt or any(
            fixture.filename in prompt for fixture in FIXTURES
        ):
            return super().generate(prompt, seed=seed)
        if len(prompt) > self._ACCEPTED_CHARS:
            raise ContextOverflowError("too big")
        return Generation("ok", len(prompt), 1, False)


def _run(client, **kw):
    args = dict(model="m", quant="q8_0", family="fam", seeds=1, mode="quick",
                corpus="fixtures-v1")
    return run_profile(client, PLAN, **{**args, **kw})


# --- Task 7 (plan 05) fixtures: stage 4/5 wiring ---------------------------


def _record(**over) -> CorpusRecord:
    base = dict(
        name="off_by_one", path=Path("src/pkg/mod.py"), line=2,
        broken="    return len(items) - 1\n", fixed="    return len(items)\n",
        test_id="tests/test_mod.py::test_len", diagnostic="exactly one",
        operator="arith", source_repo="/tmp/src", source_sha="deadbeef",
    )
    base.update(over)
    return CorpusRecord(**base)


def _baseline(**over) -> Baseline:
    base = dict(broken=0, executed=1, seconds=0.1)
    base.update(over)
    return Baseline(**base)


def _stage4_repo(tmp_path: Path) -> Path:
    """A real git repo, not just a directory tree -- the identical fixture
    `tests/test_repair.py::_repo` uses, and for the identical reason (its
    own docstring): `attempt_repair`'s `reset_clone` shells out to real
    `git checkout -f`/`git reset --hard`/`git clean -fdq` unconditionally,
    every single attempt, so a tree with no `.git` at all fails every one
    of them before the monkeypatched `run`/`suite_state` below are ever
    reached."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "pkg" / "mod.py").write_text(
        "def n(items):\n    return len(items)\n", encoding="utf-8")
    (tmp_path / "tests" / "test_mod.py").write_text(
        "from pkg.mod import n\ndef test_len():\n    assert n([1]) == 1\n",
        encoding="utf-8")
    run = lambda *argv: subprocess.run(
        ["git", *argv], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "Test")
    run("add", "-A")
    run("commit", "-q", "-m", "initial")
    return tmp_path


def test_a_good_model_profiles_ready_and_records_provenance():
    # Fails if stage 2 is never reached for a family that clears stage 1
    # (verdict would stay LIMITED/UNUSABLE), if usable_window silently
    # diverges from the verified plan, or if seeds/mode are dropped or
    # swapped on their way into the Profile.
    profile = _run(_Good())
    assert profile.verdict == "READY"
    assert profile.usable_window == 8192
    assert (profile.seeds, profile.mode) == (1, "quick")


def test_stage_one_failure_gates_stage_two():
    # Fails (via the fake's own AssertionError) if run_profile ever calls
    # stage2_codecs after stage 1 measured fidelity below the gate -- not
    # merely if codecs ends up non-empty. Also fails if dropped does not
    # name stage 2, or if the gate lets a low-fidelity family read READY.
    profile = _run(_CannotEnvelope())
    assert profile.verdict == "UNUSABLE"
    assert profile.codecs == {}
    assert any("stage 2" in d for d in profile.dropped)


def test_a_stage_two_run_that_lands_nothing_differs_from_one_that_never_ran():
    # This is the assertion the task exists to protect: an empty `codecs`
    # dict must mean ONLY "stage 2 never ran", never "stage 2 ran and
    # landed nothing" -- those are different facts and must stay visibly
    # different in the Profile. Fails if run_profile pools both outcomes
    # into an empty dict, or if it adds a spurious "stage 2" drop note for
    # a stage 2 that genuinely executed.
    ran_and_landed_nothing = _run(_EnvelopeOkNothingLands())
    never_ran = _run(_CannotEnvelope())

    assert ran_and_landed_nothing.codecs != {}
    assert all(r.lands == 0.0 for r in ran_and_landed_nothing.codecs.values())
    assert not any("stage 2" in d for d in ran_and_landed_nothing.dropped)

    assert never_ran.codecs == {}
    assert any("stage 2" in d for d in never_ran.dropped)


def test_run_profile_requires_corpus_with_no_default():
    # Task 4's fix for the carried debt named in CARRIED-DEBT.md (plan 03):
    # `corpus` used to default to the literal "fixtures-v1", which would
    # have silently mislabelled every profile once a real corpus replaced
    # the bundled fixtures. Pinned the same way test_corpus_io.py pins
    # CorpusRecord's own no-default fields -- fails the moment anyone adds
    # `corpus: str = "fixtures-v1"` (or any other default) back.
    sig = inspect.signature(run_profile)
    assert sig.parameters["corpus"].default is inspect.Parameter.empty


def test_identity_and_provenance_arguments_pass_through_unchanged():
    # Fails if run_profile hardcodes, drops, or swaps any of these instead
    # of threading the caller's own values through -- a bug a round-trip
    # comparison against a SECOND run_profile call cannot catch, because a
    # deterministic hardcoding affects both sides of that comparison
    # identically (confirmed by hand: mutating `family=family` to a fixed
    # string left every other test in this file green).
    profile = _run(_Good(), model="real-model", quant="q4_K_M",
                   family="real-family", corpus="fixtures-v2")
    assert profile.model == "real-model"
    assert profile.quant == "q4_K_M"
    assert profile.family == "real-family"
    assert profile.corpus == "fixtures-v2"


def test_fixtures_and_corpus_dropped_thread_through_a_real_run_profile_call():
    """Plan 05 task 2, fix round 1 (reviewer finding, 2026-08-10): the
    CLI-level test (`tests/test_cli_profile.py::
    test_corpus_flag_routes_records_and_carries_dropped`) monkeypatches
    `cli.run_profile` itself before calling `cli.profile_main`, so it only
    proves `cli.py` COMPUTES the right `fixtures=`/`corpus_dropped=`
    kwargs -- it never calls the real `run_profile` body, which means
    neither of the two lines task 2 actually added inside `run_profile`
    (`stage2_codecs(client, seeds, fixtures=fixtures)` and `dropped.
    extend(corpus_dropped)`) was protected by ANY test: the reviewer
    confirmed both can be deleted with the full 630-test suite staying
    green. This test calls `run_profile` directly with a real (non-mocked)
    `_Good` client and the real module `PLAN`, the same pattern every
    other test in this file already uses (see `_run`), so both lines
    actually execute this time.

    `fixtures=(FIXTURES[0],)` -- a ONE-fixture slice of the bundled five,
    never the full default -- proves the threading two independent ways
    at once:

    1. `attempts` for `search_replace` must be exactly `seeds` (1), not
       `len(FIXTURES) * seeds` (5): `stage2_codecs`'s own docstring states
       `attempts == len(fixtures) * seeds` for whichever `fixtures` it was
       actually handed, so a revert back to `stage2_codecs(client, seeds)`
       -- which silently falls back to `stages.FIXTURES`, the bundled
       default -- reports 5 instead of 1.
    2. `lands` must be `1.0`, not merely "did not raise": `_Good.generate`
       recognises a prompt by checking whether ANY bundled `FIXTURES`
       entry's filename appears in it (`next((f for f in FIXTURES if
       f.filename in prompt), None)`, matched against the GLOBAL bundled
       set, not the `fixtures` argument this test passes) and answers
       correctly when it finds one -- so a genuine stage-2 run against
       `FIXTURES[0]` really does land, proving the call actually reached
       the model and got scored, not merely that `attempts` happened to
       come out right by coincidence.

    `corpus_dropped=("mystery-dropped-marker",)` is a string that cannot
    plausibly appear in `profile.dropped` any other way -- unlike
    "stage 2" or "payload_corruption", which `run_profile` itself also
    writes there for unrelated reasons -- so finding it in `profile.
    dropped` is unambiguous proof `dropped.extend(corpus_dropped)` ran,
    not a coincidental match against some other entry.
    """
    profile = _run(_Good(), fixtures=(FIXTURES[0],), seeds=1,
                   corpus_dropped=("mystery-dropped-marker",))
    assert profile.codecs["search_replace"].attempts == 1
    assert profile.codecs["search_replace"].lands == 1.0
    assert "mystery-dropped-marker" in profile.dropped


def test_stage_one_fidelity_exactly_at_the_gate_still_opens_it():
    # Fails if the gate uses `>` instead of `>=` -- a family measured at
    # exactly the documented cutoff must still reach stage 2, the same
    # boundary verdict_for itself treats as usable (spec: both sides of
    # this comparison must agree on 0.5, or a family sits in a state
    # where run_profile skipped stage 2 but verdict_for would not have
    # called it UNUSABLE from fidelity alone).
    profile = _run(_HalfEnvelope(), seeds=2)
    assert profile.envelope_fidelity == 0.5
    assert profile.codecs != {}
    assert not any("stage 2" in d for d in profile.dropped)


def test_a_totally_unverified_window_stops_the_run_before_stage_one():
    # I1 (whole-branch review, ruled 2026-08-10): a totally unverified
    # window must STOP the run, not merely report usable_window=0 while
    # stage 1 and stage 2 still execute at num_ctx: 0 (where the daemon
    # substitutes its own default) -- the pre-fix behaviour landed
    # `envelope 100%` and `lands 100%` beside `usable_window: 0`, and the
    # verdict read LIMITED, the same verdict a working 4096-token model
    # gets. Fails (via the fake's own AssertionError) if run_profile ever
    # calls stage1_envelope or stage2_codecs after stage 0 found nothing
    # usable -- not merely if the resulting fields happen to end up empty,
    # which a coincidentally-matching fake could satisfy without any gate
    # existing at all. Also fails if run_profile still falls back to the
    # unverified plan.window (the brief's own sample line, `stage0.window
    # or plan.window`, does this; not used here).
    profile = _run(_WindowNeverVerifies())
    assert profile.usable_window == 0
    assert any("stage 0" in d for d in profile.dropped)
    assert any("stage 1" in d for d in profile.dropped)
    assert any("stage 2" in d for d in profile.dropped)
    assert profile.envelope_fidelity == 0.0
    assert profile.codecs == {}
    assert profile.verdict == "UNUSABLE"


def test_training_ctx_is_the_models_real_training_context_not_the_planned_window():
    # C3 (whole-branch review, ruled 2026-08-10): Profile.training_ctx
    # used to be assigned plan.window -- min(training_ctx, vram,
    # user_cap) -- so whenever vram or a user cap bound, the profile
    # reported THAT limit's number as though it were the model's training
    # context, and training_ctx == usable_window, a state no real model
    # can be in (the live granite run this review measured against wrote
    # training_ctx: 0). PLAN.training_ctx (32768) is deliberately far from
    # PLAN.window (8192) so a regression back to `training_ctx=plan.window`
    # is caught by an exact-value assertion, not just an inequality.
    profile = _run(_Good())
    assert profile.training_ctx == 32768
    assert profile.training_ctx != profile.usable_window


def test_unmeasured_fields_are_none_and_named_in_dropped():
    # I4 (whole-branch review, ruled 2026-08-10): payload_corruption was
    # hardcoded None -- no stage this plan builds measures it at all --
    # and mutating it to 0.0 (which reads as "measured, no corruption")
    # left every profile test green, because nothing checked the value OR
    # that "not measured" was ever stated anywhere a reader could see it.
    # `_run(_Good())` never wires a --repo, so stage 4/5 do not run here
    # either (task 7), and repeat_rate is None for that separate reason --
    # see test_repeat_rate_still_states_dropped_when_stage_5_does_not_run
    # and test_repeat_rates_dropped_line_disappears_once_it_is_measured,
    # below, for the case where repeat_rate IS measured. Both properties
    # are pinned here: the value assertions catch a 0.0 mutation directly;
    # the dropped assertions catch a fix that nulls the value but never
    # says so, which would still violate "anything not measured is stated
    # as dropped".
    profile = _run(_Good())
    assert profile.payload_corruption is None
    assert profile.repeat_rate is None
    assert any("payload_corruption" in d for d in profile.dropped)
    assert any("repeat_rate" in d for d in profile.dropped)


def test_verdict_uses_the_verified_window_not_the_unverified_plan():
    # Fails if verdict_for is ever called with plan.window instead of the
    # window stage 0 actually verified: this family's REAL window (4096)
    # sits below SUPPORTED_FLOOR even though the plan it started from
    # (8192, at the floor) does not, so a verdict computed from the
    # unverified plan would wrongly read READY instead of LIMITED.
    profile = _run(_ShrinksWindowBelowTheFloor())
    assert profile.usable_window < SUPPORTED_FLOOR
    assert profile.verdict == "LIMITED"


def test_the_table_names_the_window_limit_and_the_mode():
    # Fails if render_table drops the limiting factor, the mode string, or
    # the not-publishable warning a quick run must carry.
    table = render_table(_run(_Good()))
    assert "vram" in table
    assert "quick" in table
    # A quick profile must be visibly unquotable.
    assert "not publishable" in table


def test_a_full_run_is_not_marked_unpublishable():
    # The other direction of the assertion above: fails if "not
    # publishable" is unconditional (present regardless of mode) rather
    # than gated on mode != "full".
    table = render_table(_run(_Good(), seeds=10, mode="full"))
    assert "not publishable" not in table


def test_a_full_run_records_ten_seeds_and_full_mode():
    # Fails if run_profile ignores its seeds/mode arguments (e.g. a
    # provenance field hardcoded to "quick"/1, which the quick-mode tests
    # above alone could never catch since they also pass seeds=1/quick).
    profile = _run(_Good(), seeds=10, mode="full")
    assert (profile.seeds, profile.mode) == (10, "full")


def test_a_profile_replays_identically_from_a_transcript(tmp_path: Path):
    # Fails if any stage sends a prompt that is not a pure function of
    # (model, prompt, seed) -- e.g. one derived from wall-clock time or
    # call count -- which would make the replayed run diverge from the
    # recorded one despite reading the same transcript.
    path = tmp_path / "t.jsonl"
    live = _run(CallRecorder(_Good(), path))
    replayed = _run(CallReplayer(path))
    assert replayed.to_json() == live.to_json()


def test_the_written_profile_round_trips_through_the_filesystem(tmp_path: Path):
    # The JSON is the artifact this task exists to produce -- an
    # in-memory-only comparison (already covered by test_profile_schema.py)
    # cannot prove a file written to disk and read back is usable. Fails if
    # to_json/from_json disagree on any field (a type coercion, a missing
    # key, an ordering-sensitive comparison) that in-memory equality alone
    # would not exercise.
    profile = _run(_Good())
    target = tmp_path / "profile.json"
    target.write_text(profile.to_json(), encoding="utf-8")

    reloaded = Profile.from_json(json.loads(target.read_text(encoding="utf-8")))
    assert reloaded == profile


# --- Task 7 (plan 05): stage 4/5 wired into run_profile --------------------


def test_payload_corruption_stays_dropped_and_names_plan_06():
    # Honesty rule 2 (task 7 brief): payload_corruption KEEPS its dropped
    # line, unconditionally -- stage 3 is deferred to plan 06, not
    # forgotten, and the line must say so explicitly rather than reading
    # like every other "not measured" note.
    profile = _run(_Good())
    line = next(d for d in profile.dropped if "payload_corruption" in d)
    assert "plan 06" in line


def test_stage_4_does_not_run_without_a_best_codec():
    # Spec 4.4: repair_rate None is "never measured", which is NOT a pass
    # at the gate. _EnvelopeOkNothingLands clears stage 0 and stage 1 and
    # genuinely RUNS stage 2 (codecs is non-empty, every entry landing at
    # 0.0) -- select_best_codec(codecs) is None here for a reason no
    # earlier "stage 0"/"stage 1"/"stage 2" dropped line already covers,
    # which is exactly the case this function's own "stage 4" line exists
    # to name.
    profile = _run(_EnvelopeOkNothingLands())
    assert profile.codecs != {}
    assert profile.repair_rate is None
    assert any("stage 4" in d for d in profile.dropped)


def test_stage_4_does_not_run_without_a_repo():
    # _Good() clears every upstream gate (best_codec() names a real
    # codec), but no --repo is given -- the SECOND of the three things
    # stage 4 needs, checked independently of the first.
    profile = _run(_Good())
    assert profile.repair_rate is None
    assert any("stage 4" in d and "repo" in d for d in profile.dropped)


def test_stage_4_does_not_run_without_corpus_records(tmp_path: Path):
    # Third gate: repo and a baseline are both given, but records defaults
    # to () -- there is nothing to repair without at least one verified
    # mutant, and this must be named as its own reason, not silently
    # folded into "no --repo" (which is not what happened here).
    profile = _run(_Good(), repo=tmp_path, corpus_baseline=_baseline())
    assert profile.repair_rate is None
    assert any("stage 4" in d and "record" in d for d in profile.dropped)


def test_stage_4_does_not_run_without_a_corpus_baseline(tmp_path: Path):
    # Fourth gate: repo and records are both given, but no baseline --
    # attempt_repair's own judgement needs Baseline.executed to compare
    # a repair run's suite state against, and there is no safe one to
    # assume on the caller's behalf.
    profile = _run(_Good(), repo=tmp_path, records=(_record(),))
    assert profile.repair_rate is None
    assert any("stage 4" in d and "baseline" in d for d in profile.dropped)


def test_repeat_rate_still_states_dropped_when_stage_5_does_not_run():
    profile = _run(_Good())
    assert profile.repeat_rate is None
    assert any("repeat_rate" in d for d in profile.dropped)


def test_repeat_rates_dropped_line_disappears_once_it_is_measured(monkeypatch):
    # Honesty rule 1 (task 7 brief, spec 8.1): leaving repeat_rate's
    # dropped line unconditional would have the profile declare a field
    # unmeasured in the same payload that carries its real value.
    # stage4_repair/stage5_discipline are monkeypatched at the report
    # module boundary (report.py's own wiring is what this test protects,
    # not repair.py's or discipline.py's internals -- those already have
    # their own dedicated test files) so this is deterministic and does
    # not need a real git repo or pytest subprocess.
    fake_stage4 = Stage4(rate=0.5, attempts=10, records=5, per_record={},
                         dropped=(), all_attempts=())
    fake_stage5 = Stage5(turns_to_green_median=3.0, repeat_rate=0.2)
    monkeypatch.setattr(report_module, "stage4_repair",
                        lambda *a, **k: fake_stage4)
    monkeypatch.setattr(report_module, "stage5_discipline",
                        lambda *a, **k: fake_stage5)

    profile = _run(_Good(), repo=Path("/fake-repo"), records=(_record(),),
                   corpus_baseline=_baseline())

    assert profile.repeat_rate == 0.2
    assert not any("repeat_rate" in d for d in profile.dropped)


def test_stage_4_and_5_results_are_wired_into_the_profile(monkeypatch):
    # Beyond repeat_rate alone: every one of the four new Profile fields
    # this task adds (repair_rate, repair_attempts, repair_records,
    # turns_to_green_median) must come from the real Stage4/Stage5 the
    # gate produced, not be dropped, swapped, or hardcoded on the way in.
    fake_stage4 = Stage4(rate=0.31, attempts=940, records=94, per_record={},
                         dropped=("r seed 3: excluded",), all_attempts=())
    fake_stage5 = Stage5(turns_to_green_median=4.5, repeat_rate=0.12)
    monkeypatch.setattr(report_module, "stage4_repair",
                        lambda *a, **k: fake_stage4)
    monkeypatch.setattr(report_module, "stage5_discipline",
                        lambda *a, **k: fake_stage5)

    profile = _run(_Good(), repo=Path("/fake-repo"), records=(_record(),),
                   corpus_baseline=_baseline())

    assert profile.repair_rate == 0.31
    assert profile.repair_attempts == 940
    assert profile.repair_records == 94
    assert profile.turns_to_green_median == 4.5
    assert profile.repeat_rate == 0.12
    assert "r seed 3: excluded" in profile.dropped
    assert not any("stage 4" in d for d in profile.dropped)


def test_stage_4_receives_the_real_arguments_run_profile_was_given(monkeypatch):
    # Fails if run_profile hardcodes, drops, or swaps any of records/repo/
    # client/seeds/codec/base/turn_cap/python on their way into
    # stage4_repair -- a round-trip comparison against the returned
    # Profile alone (the test above) cannot catch a wrong argument that
    # stage4_repair's OWN fake ignores, so this inspects the call itself.
    captured = {}
    record = _record()
    repo = Path("/fake-repo")
    base = _baseline()

    def fake_stage4_repair(records, repo_arg, client, *, seeds, codec, base,
                           turn_cap, python):
        captured.update(records=records, repo=repo_arg, seeds=seeds,
                        codec=codec, base=base, turn_cap=turn_cap,
                        python=python)
        return Stage4(rate=1.0, attempts=1, records=1, per_record={},
                     dropped=(), all_attempts=())

    monkeypatch.setattr(report_module, "stage4_repair", fake_stage4_repair)
    monkeypatch.setattr(report_module, "stage5_discipline",
                        lambda *a, **k: Stage5(None, None))

    _run(_Good(), repo=repo, records=(record,), corpus_baseline=base,
        seeds=1, turn_cap=3, python="/custom/python")

    assert captured["records"] == (record,)
    assert captured["repo"] == repo
    assert captured["seeds"] == 1
    assert captured["codec"] == "search_replace"
    assert captured["base"] == base
    assert captured["turn_cap"] == 3
    assert captured["python"] == "/custom/python"


def test_run_profiles_own_python_default_is_sys_executable(monkeypatch):
    # The default matters as much as the plumbing (task 8, fix round 1,
    # bullet 4): `sys.executable` is what `robigo corpus` was itself
    # running under when it measured the frozen corpus's Baseline, so a
    # caller of run_profile who never thinks about --python at all must
    # still get an interpreter that AGREES with that baseline, not
    # PythonAdapter's own .venv/venv/PATH search (which would resolve
    # relative to --repo, a completely different tree, and need not have
    # anything installed at all).
    import sys as sys_module

    captured = {}

    def fake_stage4_repair(records, repo_arg, client, *, seeds, codec, base,
                           turn_cap, python):
        captured["python"] = python
        return Stage4(rate=1.0, attempts=1, records=1, per_record={},
                     dropped=(), all_attempts=())

    monkeypatch.setattr(report_module, "stage4_repair", fake_stage4_repair)
    monkeypatch.setattr(report_module, "stage5_discipline",
                        lambda *a, **k: Stage5(None, None))

    _run(_Good(), repo=Path("/fake-repo"), records=(_record(),),
        corpus_baseline=_baseline(), seeds=1)

    assert captured["python"] == sys_module.executable


def test_a_full_run_profile_call_drives_a_real_stage_4_and_stage_5(
    tmp_path: Path, monkeypatch
):
    # The genuine end-to-end path, with no report-level monkeypatching at
    # all: a real git repo (the identical fixture test_repair.py uses),
    # run through the real stage4_repair/stage5_discipline, monkeypatching
    # only the INNERMOST seam (robigo.profile.repair.run/suite_state, the
    # same seam tests/test_repair.py itself patches) so this test needs no
    # daemon and no real pytest subprocess, while still proving run_profile
    # wires `records`/`repo`/`client`/`seeds`/`codec`/`corpus_baseline`/
    # `turn_cap` all the way through to a real Stage4/Stage5 computation,
    # not merely to a function named the right thing.
    repo = _stage4_repo(tmp_path)
    record = _record(source_repo=str(repo))
    base = _baseline()

    monkeypatch.setattr(repair_module, "run", lambda *a, **k: RunResult(
        outcome="pass", turns=2, exit_code=0, branch=None,
        detail="tests pass", repeats=1))
    monkeypatch.setattr(repair_module, "suite_state", lambda *a, **k: SuiteState(
        broken=0, executed=1, broken_ids=(), incomplete=None))

    profile = _run(_Good(), repo=repo, records=(record,),
                   corpus_baseline=base, seeds=1)

    assert profile.repair_rate == 1.0
    assert profile.repair_attempts == 1
    assert profile.repair_records == 1
    assert profile.turns_to_green_median == 2.0
    assert profile.repeat_rate == pytest.approx(0.5)  # 1 repeat / 2 turns
    assert not any("stage 4" in d for d in profile.dropped)
    assert not any("repeat_rate" in d for d in profile.dropped)


def test_the_table_shows_not_measured_for_repair_and_discipline_by_default():
    # render_table (step 5): a profile whose stage 4/5 never ran must
    # print "not measured", never a fabricated "0.0% of 0 attempts" that
    # would read as a real, failing measurement.
    table = render_table(_run(_Good()))
    assert "repair        not measured" in table
    assert "discipline    turns to green not measured, repeat rate not measured" in table


def test_the_table_shows_repair_and_discipline_numbers_when_measured(monkeypatch):
    fake_stage4 = Stage4(rate=0.5, attempts=10, records=5, per_record={},
                         dropped=(), all_attempts=())
    fake_stage5 = Stage5(turns_to_green_median=3.0, repeat_rate=0.2)
    monkeypatch.setattr(report_module, "stage4_repair",
                        lambda *a, **k: fake_stage4)
    monkeypatch.setattr(report_module, "stage5_discipline",
                        lambda *a, **k: fake_stage5)

    table = render_table(_run(_Good(), repo=Path("/fake-repo"),
                              records=(_record(),), corpus_baseline=_baseline()))

    assert "repair        50.0% of 10 attempts over 5 records" in table
    assert "discipline    turns to green 3.0 turns, repeat rate 20.0%" in table


# --- CLI wiring: `robigo profile`, its --full/--seeds handling, and its --
# write to profile_path -----------------------------------------------------


def _stub_plan_window(*args, **kwargs) -> WindowPlan:
    # Same reasoning as test_cli.py's own _stub_plan_window: these tests
    # are about argument wiring and file output, not window detection, and
    # model "m" does not exist on any real daemon.
    return PLAN


def test_leading_profile_argument_dispatches_to_profile_main(monkeypatch):
    # Fails if `main` still routes a leading "profile" argv element into
    # the ordinary task parser (which requires --model and would behave
    # completely differently), instead of into profile_main.
    import robigo.cli as cli_module

    captured = {}

    def fake_profile_main(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(cli_module, "profile_main", fake_profile_main)
    code = cli_module.main(["profile", "--model", "m", "--full"])
    assert code == 0
    assert captured["argv"] == ["--model", "m", "--full"]


def test_full_flag_requests_ten_seeds_and_full_mode(monkeypatch, tmp_path: Path):
    # This is the test the task calls out by name: a run_profile call that
    # always receives seeds=3/mode="quick" regardless of --full would still
    # pass every OTHER test in this file (they mostly exercise seeds=1 or
    # pass seeds/mode straight through in-process), so this has to inspect
    # what profile_main actually hands run_profile when --full is given,
    # not just what a hand-built run_profile call produces.
    import robigo.cli as cli_module

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(cli_module, "plan_window", _stub_plan_window)
    monkeypatch.setattr(cli_module, "build_client", lambda args: _Good())

    captured = {}
    real_run_profile = cli_module.run_profile

    def spy_run_profile(client, plan, **kw):
        captured.setdefault("calls", []).append(kw)
        return real_run_profile(client, plan, **kw)

    monkeypatch.setattr(cli_module, "run_profile", spy_run_profile)

    code = cli_module.profile_main(["--model", "m", "--full"])
    assert code == 0
    assert (captured["calls"][0]["seeds"], captured["calls"][0]["mode"]) == (10, "full")


def test_without_full_flag_the_default_seed_count_is_used_and_marked_quick(
    monkeypatch, tmp_path: Path
):
    # The other direction of the test above: fails if --full's absence is
    # not correctly wired to the (default) quick mode and the default seed
    # count, or if seeds/mode get swapped between the two branches.
    import robigo.cli as cli_module

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(cli_module, "plan_window", _stub_plan_window)
    monkeypatch.setattr(cli_module, "build_client", lambda args: _Good())

    captured = {}
    real_run_profile = cli_module.run_profile

    def spy_run_profile(client, plan, **kw):
        captured.setdefault("calls", []).append(kw)
        return real_run_profile(client, plan, **kw)

    monkeypatch.setattr(cli_module, "run_profile", spy_run_profile)

    code = cli_module.profile_main(["--model", "m"])
    assert code == 0
    assert (captured["calls"][0]["seeds"], captured["calls"][0]["mode"]) == (3, "quick")


def test_profile_main_derives_corpus_from_fixtures_own_identity_not_a_hardcoded_literal(
    monkeypatch, tmp_path: Path
):
    # Task 4's fix, pinned at the CLI level where it actually matters: a
    # profile_main that hardcodes `corpus="fixtures-v1"` at the call site
    # (instead of importing `robigo.profile.fixtures.CORPUS_NAME`) would
    # pass every OTHER test in this file, since none of them ever change
    # what that name is -- monkeypatching the name cli.py actually reads
    # and asserting the WRITTEN profile follows it is the only thing that
    # can tell "derived" apart from "typed twice and happened to agree".
    import robigo.cli as cli_module

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(cli_module, "plan_window", _stub_plan_window)
    monkeypatch.setattr(cli_module, "build_client", lambda args: _Good())
    monkeypatch.setattr(cli_module, "CORPUS_NAME", "totally-different-corpus-name")

    code = cli_module.profile_main(["--model", "m", "--seeds", "1"])
    assert code == 0

    written = Profile.from_json(
        json.loads(profile_path("m").read_text(encoding="utf-8"))
    )
    assert written.corpus == "totally-different-corpus-name"


def test_profile_main_exits_refused_when_the_verdict_is_unusable(
    monkeypatch, tmp_path: Path
):
    # None of the other CLI-level tests ever produce an UNUSABLE verdict
    # (they all use _Good), so a profile_main that always `return 0`
    # regardless of profile.verdict would otherwise pass this whole file.
    # Fails if the exit code stops tracking the verdict, in either
    # direction: 0 for UNUSABLE, or non-zero for READY/LIMITED.
    import robigo.cli as cli_module

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(cli_module, "plan_window", _stub_plan_window)
    monkeypatch.setattr(cli_module, "build_client", lambda args: _CannotEnvelope())

    code = cli_module.profile_main(["--model", "m", "--seeds", "1"])
    assert code == OUTCOMES["refused"]


def test_the_record_flag_writes_a_transcript_that_replays_the_same_profile(
    monkeypatch, tmp_path: Path
):
    # Fails if --record is accepted but silently ignored (client never
    # wrapped in CallRecorder), which would leave no transcript on disk, or
    # if the transcript it writes does not actually reproduce the profile
    # profile_main itself printed.
    import robigo.cli as cli_module

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(cli_module, "plan_window", _stub_plan_window)
    monkeypatch.setattr(cli_module, "build_client", lambda args: _Good())
    transcript = tmp_path / "recorded.jsonl"

    code = cli_module.profile_main([
        "--model", "m", "--seeds", "1", "--record", str(transcript),
    ])
    assert code == 0
    assert transcript.exists() and transcript.stat().st_size > 0

    written = Profile.from_json(
        json.loads(profile_path("m").read_text(encoding="utf-8"))
    )
    replayed = _run(CallReplayer(transcript), seeds=1, mode="quick",
                    quant="unknown", family="m", model="m")
    assert written == replayed


def test_profile_main_writes_a_profile_that_round_trips_through_the_filesystem(
    monkeypatch, tmp_path: Path
):
    # Exercises the actual write path in cli.profile_main (path.write_text),
    # not just Profile.to_json/from_json in memory. Fails if profile_main
    # writes somewhere other than profile_path(family), writes a payload
    # from_json cannot parse back, or writes a profile that disagrees with
    # what an independent replay of the same transcript produces.
    import robigo.cli as cli_module

    transcript = tmp_path / "t.jsonl"
    # Record with the exact client, plan, and seed count profile_main will
    # use below (seeds=1, quick, no --full) so replay lines up call-for-call.
    _run(CallRecorder(_Good(), transcript), seeds=1, mode="quick")

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(cli_module, "plan_window", _stub_plan_window)

    code = cli_module.profile_main([
        "--model", "m", "--seeds", "1", "--replay", str(transcript),
    ])
    assert code == 0

    written_path = profile_path("m")
    assert written_path == tmp_path / "robigo" / "profiles" / "m.json"
    on_disk = Profile.from_json(json.loads(written_path.read_text(encoding="utf-8")))

    expected = _run(CallReplayer(transcript), seeds=1, mode="quick", quant="unknown",
                    family="m", model="m")
    assert on_disk == expected
