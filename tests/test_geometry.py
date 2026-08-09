# tests/test_geometry.py
from __future__ import annotations

import pytest

from robigo.model.geometry import Geometry, GeometryError, from_model_info

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
