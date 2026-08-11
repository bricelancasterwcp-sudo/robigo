from __future__ import annotations

import dataclasses
import json

from robigo.profile.schema import (
    SUPPORTED_FLOOR,
    CodecResult,
    Profile,
    select_best_codec,
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
        payload_corruption=None, repeat_rate=None,
        repair_rate=None, repair_attempts=0, repair_records=0,
        turns_to_green_median=None,
        verdict="READY",
        seeds=3, mode="quick", corpus="fixtures-v1",
        python="/usr/bin/python3", dropped=(),
    )
    return Profile(**{**defaults, **kw})


def test_round_trips_through_json():
    original = _profile()
    assert Profile.from_json(json.loads(original.to_json())) == original


def test_best_codec_is_the_highest_landing_rate():
    assert _profile().best_codec() == "search_replace"


def test_best_codec_returns_none_when_every_codec_landed_zero_percent():
    # THE measured case, pinned directly (CARRIED-DEBT.md, plan 03):
    # granite-code:8b landed 0% on every codec it was profiled against, and
    # the pre-fix `max()` still named one of them "best". A floor tested
    # only against a PASSING codec (the test above, 0.62/0.55) proves
    # nothing about this one -- both codecs here are genuinely 0.0, not
    # merely low.
    profile = _profile(codecs={
        "search_replace": CodecResult(0.0, 10, None),
        "whole_file": CodecResult(0.0, 10, None),
    })
    assert profile.best_codec() is None


def test_best_codec_still_names_a_codec_that_lands_below_the_ready_floor():
    # The floor is "landed at all" (> 0.0), not `_LANDING_MIN` (0.5) --
    # fails for an implementation that reuses verdict_for's READY threshold
    # here instead of its own, narrower one. A codec landing 20% of the
    # time is real signal plan 05 can still configure the loop around.
    profile = _profile(codecs={
        "search_replace": CodecResult(0.0, 10, None),
        "whole_file": CodecResult(0.2, 10, 1000),
    })
    assert profile.best_codec() == "whole_file"


def test_profile_best_codec_delegates_to_the_module_level_function():
    # Task 7 brief: "Do not compute the best codec by constructing a
    # throwaway Profile... refactor it to a module-level function both
    # call. One definition, not two." Pinned directly: calling
    # `select_best_codec` on the SAME `codecs` dict a `Profile` carries
    # must produce the identical answer `Profile.best_codec()` does, for
    # every one of the three shapes the tests above already exercise (a
    # real winner, an all-zero dict, and a below-floor-but-nonzero
    # winner). This fails if `Profile.best_codec` is ever reverted to its
    # own private copy of the `max(...)`-with-floor logic that happens to
    # agree with `select_best_codec` today but is free to drift from it
    # tomorrow -- exactly the "three copies of a codec list" defect class
    # CARRIED-DEBT.md already names.
    real_winner = {"search_replace": CodecResult(0.62, 30, None),
                   "whole_file": CodecResult(0.55, 30, 1400)}
    all_zero = {"search_replace": CodecResult(0.0, 10, None),
                "whole_file": CodecResult(0.0, 10, None)}
    below_floor = {"search_replace": CodecResult(0.0, 10, None),
                   "whole_file": CodecResult(0.2, 10, 1000)}
    for codecs in (real_winner, all_zero, below_floor, {}):
        assert select_best_codec(codecs) == _profile(codecs=codecs).best_codec()


def test_seeds_and_mode_are_always_present_in_the_json():
    # A quick 3-seed profile must never be quotable as a result, so the
    # provenance travels with the numbers (spec 5.5).
    payload = json.loads(_profile().to_json())
    assert payload["measured"]["seeds"] == 3
    assert payload["measured"]["mode"] == "quick"


def test_python_is_recorded_alongside_seeds_mode_and_corpus():
    # Fix round 2, IMPORTANT 2: `--python` is a knob that can change or
    # void `repair_rate`, and it was invisible in the published artifact
    # (grep -c python schema.py -> 0 before this fix). Recorded in the
    # SAME "measured" provenance group as seeds/mode/corpus, for the
    # identical reason: a Profile is what the kill criterion is read
    # from, and a reader must be able to see which interpreter produced
    # it.
    payload = json.loads(_profile(python="/custom/python3.11").to_json())
    assert payload["measured"]["python"] == "/custom/python3.11"


def test_python_round_trips_through_json():
    # The dedicated round-trip the review asked for by name, isolated
    # from the whole-object equality check above (test_round_trips_
    # through_json) so a regression specifically in `python`'s own
    # to_json/from_json wiring fails legibly rather than as an opaque
    # "the whole Profile differs" assertion.
    original = _profile(python="/opt/pyenv/shims/python3.12")
    reloaded = Profile.from_json(json.loads(original.to_json()))
    assert reloaded.python == "/opt/pyenv/shims/python3.12"
    assert reloaded == original


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


# --- Task 7 (plan 05): repair_rate / repeat_rate None-vs-zero honesty ------
#
# `repair_rate: None` means NOT MEASURED; `repair_rate: 0.0` means measured,
# nothing repaired. These are different facts and must stay distinguishable
# after a to_json/from_json round trip -- a family whose repair_rate is None
# has not passed the gate, because it has not been measured at all.


def test_none_and_zero_repair_rates_are_distinguishable_in_json():
    never = _profile(repair_rate=None).to_json()
    zero = _profile(repair_rate=0.0).to_json()
    assert never != zero
    assert Profile.from_json(json.loads(never)).repair_rate is None
    assert Profile.from_json(json.loads(zero)).repair_rate == 0.0


def test_none_and_zero_turns_to_green_median_are_distinguishable_in_json():
    # The identical honesty property, pinned for turns_to_green_median too:
    # this project has already shipped the None-vs-zero collapse once, for
    # CodecResult.max_file_tokens (see that field's own docstring) -- this
    # is the same class of field, and the same falsification must hold.
    never = _profile(turns_to_green_median=None).to_json()
    zero = _profile(turns_to_green_median=0.0).to_json()
    assert never != zero
    assert Profile.from_json(json.loads(never)).turns_to_green_median is None
    assert Profile.from_json(json.loads(zero)).turns_to_green_median == 0.0


def test_every_new_repair_and_discipline_field_round_trips():
    p = _profile(repair_rate=0.31, repair_attempts=1000, repair_records=100,
                 turns_to_green_median=2.0, repeat_rate=0.18)
    assert Profile.from_json(json.loads(p.to_json())) == p


# --- Boundary coverage for verdict_for's thresholds ------------------------
#
# The tests above only exercise a clearly-passing case and a clearly-failing
# case for each threshold, which cannot distinguish `<` from `<=`. These
# pin the exact comparison at SUPPORTED_FLOOR, the landing minimum, and the
# envelope minimum, one value on each side of the line.

_READY_CODECS = {"search_replace": CodecResult(0.6, 30, None)}


def test_window_exactly_at_the_floor_is_not_limited_by_window():
    assert verdict_for(0.99, _READY_CODECS, SUPPORTED_FLOOR) == "READY"


def test_window_one_below_the_floor_is_limited():
    assert verdict_for(0.99, _READY_CODECS, SUPPORTED_FLOOR - 1) == "LIMITED"


def test_window_one_above_the_floor_is_not_limited_by_window():
    assert verdict_for(0.99, _READY_CODECS, SUPPORTED_FLOOR + 1) == "READY"


def test_landing_rate_exactly_at_the_minimum_is_not_limited():
    codecs = {"search_replace": CodecResult(0.5, 30, None)}
    assert verdict_for(0.99, codecs, 32768) == "READY"


def test_landing_rate_just_below_the_minimum_is_limited():
    codecs = {"search_replace": CodecResult(0.49, 30, None)}
    assert verdict_for(0.99, codecs, 32768) == "LIMITED"


def test_landing_rate_just_above_the_minimum_is_not_limited():
    codecs = {"search_replace": CodecResult(0.51, 30, None)}
    assert verdict_for(0.99, codecs, 32768) == "READY"


def test_envelope_fidelity_exactly_at_the_minimum_is_not_unusable():
    assert verdict_for(0.5, _READY_CODECS, 32768) == "READY"


def test_envelope_fidelity_just_below_the_minimum_is_unusable():
    assert verdict_for(0.49, _READY_CODECS, 32768) == "UNUSABLE"


def test_envelope_fidelity_just_above_the_minimum_is_not_unusable():
    assert verdict_for(0.51, _READY_CODECS, 32768) == "READY"


# --- Structural coverage: every field must be wired, not just equal --------
#
# Whole-object equality (test_round_trips_through_json) catches VALUE drift,
# but it cannot catch a field that was never wired into to_json/from_json at
# all: if the field carries a default, from_json fills the gap silently,
# the original carries the same default, and equality holds even though the
# JSON never mentioned it. These walk the actual serialised structure and
# demand every dataclass field appear as a key somewhere in it, independent
# of what value it holds.


def _collect_keys(obj: object) -> set[str]:
    """Every dict key present anywhere in a JSON-like structure, at any
    nesting depth (``codecs`` maps a name to a nested object, and
    ``seeds``/``mode``/``corpus`` live under ``measured``)."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.add(key)
            keys |= _collect_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _collect_keys(item)
    return keys


def test_every_profile_field_is_wired_into_the_json_payload():
    payload = json.loads(_profile().to_json())
    keys = _collect_keys(payload)
    missing = [f.name for f in dataclasses.fields(Profile) if f.name not in keys]
    assert not missing, f"Profile field(s) missing from the JSON payload: {missing}"


def test_every_codec_result_field_is_wired_into_a_serialised_codec_entry():
    payload = json.loads(_profile().to_json())
    entry = payload["codecs"]["search_replace"]
    missing = [f.name for f in dataclasses.fields(CodecResult) if f.name not in entry]
    assert not missing, f"CodecResult field(s) missing from a codec entry: {missing}"
