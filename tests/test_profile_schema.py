from __future__ import annotations

import dataclasses
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
