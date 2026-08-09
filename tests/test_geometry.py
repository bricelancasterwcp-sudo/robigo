# tests/test_geometry.py
from __future__ import annotations

import pytest

from robigo.model.geometry import (
    OVERHEAD_BYTES,
    Geometry,
    GeometryError,
    WindowPlan,
    free_vram_bytes,
    from_model_info,
    usable_window,
)

GIB = 1024**3

# Verbatim shapes of Ollama /api/show model_info, 2026-08-09.
QWEN7B = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 28,
    "qwen2.attention.head_count": 28,
    "qwen2.attention.head_count_kv": 4,
    "qwen2.attention.key_length": 128,
    "qwen2.context_length": 32768,
    "qwen2.embedding_length": 3584,
}
CODEGEMMA = {
    "general.architecture": "gemma",
    "gemma.block_count": 28,
    "gemma.attention.head_count": 16,
    "gemma.attention.head_count_kv": 16,
    "gemma.attention.key_length": 256,
    "gemma.context_length": 8192,
    "gemma.embedding_length": 3072,
}


def test_kib_per_token_matches_the_measured_values():
    assert from_model_info(QWEN7B).kv_bytes_per_token / 1024 == 56
    # 8x the 7B for a model of the same size: this is the whole reason the
    # advertised context length cannot be trusted (spec 3.1).
    assert from_model_info(CODEGEMMA).kv_bytes_per_token / 1024 == 448


def test_kv_bytes_scales_with_tokens_and_halves_under_q8():
    g = from_model_info(QWEN7B)
    assert g.kv_bytes(32768) == 56 * 1024 * 32768
    assert g.kv_bytes(32768, kv_bits=8) == g.kv_bytes(32768) // 2


def test_head_dim_falls_back_to_embedding_over_heads():
    info = dict(QWEN7B)
    del info["qwen2.attention.key_length"]
    assert from_model_info(info).key_dim == 3584 // 28


def test_a_per_layer_kv_head_array_takes_the_maximum():
    # Newer architectures publish head_count_kv per layer. Taking the max
    # over-reserves slightly, which is the safe direction.
    info = dict(QWEN7B, **{"qwen2.attention.head_count_kv": [4, 4, 8]})
    assert from_model_info(info).kv_heads == 8


def test_missing_geometry_raises_rather_than_guessing():
    info = dict(QWEN7B)
    del info["qwen2.attention.head_count_kv"]
    with pytest.raises(GeometryError) as e:
        from_model_info(info)
    assert "head_count_kv" in str(e.value)


def test_geometry_is_frozen():
    with pytest.raises(Exception):
        from_model_info(QWEN7B).layers = 99  # type: ignore[misc]


def test_key_and_value_dims_are_summed_separately():
    # No fixture supplies both, so nothing else pins this: a regression to
    # `2 * key_dim` would pass every other test in this file.
    info = dict(QWEN7B, **{"qwen2.attention.value_length": 64})
    g = from_model_info(info)
    assert (g.key_dim, g.value_dim) == (128, 64)
    assert g.kv_bytes_per_token == 28 * 4 * (128 + 64) * 2


def test_the_training_context_is_carried_through():
    # The field the never-exceed-training-context law reads downstream.
    assert from_model_info(QWEN7B).training_ctx == 32768


def test_a_missing_architecture_says_what_to_do():
    with pytest.raises(GeometryError) as e:
        from_model_info({"qwen2.block_count": 28})
    assert "--window" in str(e.value)


def test_a_malformed_field_raises_geometry_error_not_a_raw_type_error():
    # Task 5 catches GeometryError to fall back to --window; a raw TypeError
    # would escape that handler entirely.
    info = dict(QWEN7B, **{"qwen2.attention.head_count_kv": None})
    with pytest.raises(GeometryError) as e:
        from_model_info(info)
    assert "malformed" in str(e.value)
    assert "attention.head_count_kv" in str(e.value)


def test_a_malformed_head_count_raises_geometry_error():
    info = dict(QWEN7B, **{"qwen2.attention.head_count": "not-a-number"})
    del info["qwen2.attention.key_length"]
    with pytest.raises(GeometryError) as e:
        from_model_info(info)
    assert "attention.head_count" in str(e.value)


def test_a_malformed_embedding_length_on_the_fallback_path_is_named():
    info = dict(QWEN7B, **{"qwen2.embedding_length": None})
    del info["qwen2.attention.key_length"]
    with pytest.raises(GeometryError) as e:
        from_model_info(info)
    assert "embedding_length" in str(e.value)


def test_a_zero_head_count_does_not_divide_by_zero():
    info = dict(QWEN7B, **{"qwen2.attention.head_count": 0})
    del info["qwen2.attention.key_length"]
    with pytest.raises(GeometryError) as e:
        from_model_info(info)
    assert "--window" in str(e.value)


def test_an_empty_kv_head_list_is_geometry_error_not_a_max_failure():
    info = dict(QWEN7B, **{"qwen2.attention.head_count_kv": []})
    with pytest.raises(GeometryError):
        from_model_info(info)


def test_a_malformed_element_of_a_per_layer_kv_list_is_named_not_raw():
    """A JSON null inside the list is the same defect as a malformed scalar
    field, and Task 5's --window fallback catches only GeometryError."""
    info = dict(QWEN7B, **{"qwen2.attention.head_count_kv": [4, None]})
    with pytest.raises(GeometryError) as e:
        from_model_info(info)
    assert "attention.head_count_kv" in str(e.value)


def test_an_empty_per_layer_kv_list_says_it_is_empty():
    info = dict(QWEN7B, **{"qwen2.attention.head_count_kv": []})
    with pytest.raises(GeometryError) as e:
        from_model_info(info)
    assert "empty list" in str(e.value)


def test_a_non_positive_block_count_raises_geometry_error():
    # A zero here is not a harmless edge case: it makes kv_bytes_per_token
    # zero, and usable_window then divides by it and raises a bare
    # ZeroDivisionError that Task 5's `except GeometryError` fallback cannot
    # catch. _as_int only validates type, not value, so this needs its own
    # guard (amendment 2).
    info = dict(QWEN7B, **{"qwen2.block_count": 0})
    with pytest.raises(GeometryError) as e:
        from_model_info(info)
    assert "block_count" in str(e.value)


def test_a_non_positive_head_count_kv_raises_geometry_error():
    info = dict(QWEN7B, **{"qwen2.attention.head_count_kv": -4})
    with pytest.raises(GeometryError) as e:
        from_model_info(info)
    assert "head_count_kv" in str(e.value)


def test_an_infinite_metadata_field_raises_geometry_error_not_overflow_error():
    # int(float('inf')) raises OverflowError, and stdlib json.loads accepts
    # the bare `Infinity` token by default, so this is reachable from a
    # hostile /api/show response, not just a crafted dict.
    info = dict(QWEN7B, **{"qwen2.block_count": float("inf")})
    with pytest.raises(GeometryError) as e:
        from_model_info(info)
    assert "block_count" in str(e.value)


def test_the_training_context_is_never_exceeded_even_with_vram_to_spare():
    # llama-server refuses a slot larger than the model was trained on,
    # and Ollama accepts it silently and rope-degrades (law 1).
    plan = usable_window(from_model_info(QWEN7B), free_vram=15 * GIB,
                         weights_bytes=8 * GIB)
    assert plan.window == 32768
    assert plan.limited_by == "training_ctx"


def test_vram_binds_when_the_cache_is_expensive():
    """codegemma costs 448 KiB/token, so 896 MiB of spare VRAM buys far less
    than its 8192-token training context. The report must name vram as the
    binding constraint -- a user asking why their window is small needs that
    answer without reading the code."""
    geometry = from_model_info(CODEGEMMA)
    free, weights = 10 * GIB + GIB // 8, 9 * GIB
    # 448 KiB/token is the measured figure from the spec's table, restated
    # here on purpose so this does not depend on the property under test.
    expected = (free - weights - OVERHEAD_BYTES) // (448 * 1024)
    plan = usable_window(geometry, free_vram=free, weights_bytes=weights)
    assert plan.limited_by == "vram"
    assert plan.window == expected == 2048
    assert plan.window < geometry.training_ctx


def test_kv_quantization_buys_window():
    """8-bit KV halves the cache, so it exactly doubles the window while vram
    is still binding. This is the cheapest rung on Task 4's ladder."""
    args = dict(free_vram=10 * GIB + GIB // 8, weights_bytes=9 * GIB)
    at_16 = usable_window(from_model_info(CODEGEMMA), **args, kv_bits=16)
    at_8 = usable_window(from_model_info(CODEGEMMA), **args, kv_bits=8)
    assert at_16.window == 2048 and at_8.window == 4096
    assert at_8.window == at_16.window * 2
    assert at_8.limited_by == "vram"


def test_kv_bits_zero_raises_value_error():
    with pytest.raises(ValueError) as e:
        usable_window(from_model_info(QWEN7B), free_vram=15 * GIB,
                      weights_bytes=8 * GIB, kv_bits=0)
    assert "kv_bits" in str(e.value)


def test_kv_bits_negative_raises_value_error():
    # The reviewer's exact reproduction: kv_bits=-16 used to silently
    # produce window=-126391, limited_by='vram' with no error at all --
    # silent wrong data beats a crash for damage, so this must not be a
    # ZeroDivisionError-adjacent near-miss either; it must raise loudly.
    with pytest.raises(ValueError) as e:
        usable_window(from_model_info(QWEN7B), free_vram=15 * GIB,
                      weights_bytes=8 * GIB, kv_bits=-16)
    assert "kv_bits" in str(e.value)


def test_a_user_cap_wins_and_is_reported():
    plan = usable_window(from_model_info(QWEN7B), free_vram=15 * GIB,
                         weights_bytes=8 * GIB, user_cap=4096)
    assert (plan.window, plan.limited_by) == (4096, "user_cap")


def test_an_unmeasurable_vram_returns_a_training_ctx_plan():
    plan = usable_window(from_model_info(QWEN7B), free_vram=None,
                         weights_bytes=8 * GIB)
    assert (plan.window, plan.limited_by) == (32768, "training_ctx")
    assert plan.free_vram is None


def test_free_vram_parses_nvidia_smi():
    assert free_vram_bytes(lambda: "5759\n") == 5759 * 1024 * 1024


def test_free_vram_returns_none_when_the_tool_is_absent():
    def boom() -> str:
        raise FileNotFoundError("nvidia-smi")

    assert free_vram_bytes(boom) is None


def test_window_plan_is_frozen():
    plan = usable_window(from_model_info(QWEN7B), free_vram=15 * GIB,
                         weights_bytes=8 * GIB)
    assert isinstance(plan, WindowPlan)
    with pytest.raises(Exception):
        plan.window = 99  # type: ignore[misc]


def test_exhausted_vram_yields_a_zero_window_attributed_to_vram():
    # weights_bytes alone already consumes all of free_vram, so
    # max(spare, 0) // per_token is 0 -- a legitimate min() winner that
    # must not be confused with an unmeasured (free_vram=None) plan, and
    # must not go negative if the max(spare, 0) floor were ever dropped.
    plan = usable_window(from_model_info(QWEN7B), free_vram=8 * GIB,
                         weights_bytes=8 * GIB)
    assert (plan.window, plan.limited_by) == (0, "vram")
    assert plan.free_vram == 8 * GIB
