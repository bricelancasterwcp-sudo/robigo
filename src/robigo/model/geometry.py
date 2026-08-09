# src/robigo/model/geometry.py
from __future__ import annotations

from dataclasses import dataclass


class GeometryError(Exception):
    """Geometry could not be determined. Raised rather than guessed: a
    wrong bytes-per-token silently sizes the window wrong, and the
    failure surfaces later as unexplained truncation or OOM."""


@dataclass(frozen=True)
class Geometry:
    arch: str
    layers: int
    kv_heads: int
    key_dim: int
    value_dim: int
    training_ctx: int

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V, every layer, at f16. Uses key and value dims
        separately because they differ on some architectures."""
        return self.layers * self.kv_heads * (self.key_dim + self.value_dim) * 2

    def kv_bytes(self, tokens: int, kv_bits: int = 16) -> int:
        return self.kv_bytes_per_token * tokens * kv_bits // 16


def _as_int(key: str, value: object) -> int:
    """Convert one metadata field, naming it if it is malformed. Task 5
    catches GeometryError to fall back to an explicit --window, so a raw
    TypeError or ValueError escaping here would defeat that fallback."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise GeometryError(
            f"{key} is malformed ({value!r}); the KV cache size cannot be "
            f"computed, so the usable window is unknown. Pass --window "
            f"explicitly."
        ) from exc


def _kv_head_count(key: str, value: object) -> int:
    """Reduce `head_count_kv` — which some architectures report per layer —
    to its largest value. `max()` is a conversion like any other and must not
    escape as a raw TypeError, so every element goes through `_as_int` first
    and the comparison only ever sees ints."""
    if not isinstance(value, list):
        return _as_int(key, value)
    if not value:
        raise GeometryError(
            f"{key} is an empty list, so the number of key/value heads is "
            f"unknown and the KV cache size cannot be computed. Pass "
            f"--window explicitly."
        )
    return max(_as_int(key, element) for element in value)


def from_model_info(info: dict) -> Geometry:
    arch = info.get("general.architecture")
    if not isinstance(arch, str):
        raise GeometryError(
            "general.architecture missing from model metadata; the KV cache "
            "size cannot be computed, so the usable window is unknown. Pass "
            "--window explicitly."
        )

    def need(key: str) -> object:
        full = f"{arch}.{key}"
        if full not in info:
            raise GeometryError(
                f"{full} missing from model metadata; cannot compute the "
                f"KV cache size, so the usable window is unknown. Pass "
                f"--window explicitly."
            )
        return info[full]

    kv_heads = _kv_head_count(
        f"{arch}.attention.head_count_kv", need("attention.head_count_kv")
    )
    key_dim = info.get(f"{arch}.attention.key_length")
    if key_dim is None:
        heads = _as_int(f"{arch}.attention.head_count", need("attention.head_count"))
        if heads <= 0:
            raise GeometryError(
                f"{arch}.attention.head_count is {heads}, so the head "
                f"dimension cannot be derived from embedding_length. Pass "
                f"--window explicitly."
            )
        key_dim = _as_int(f"{arch}.embedding_length", need("embedding_length")) // heads
    value_dim = info.get(f"{arch}.attention.value_length", key_dim)
    return Geometry(
        arch=arch,
        layers=_as_int(f"{arch}.block_count", need("block_count")),
        kv_heads=kv_heads,
        key_dim=_as_int(f"{arch}.attention.key_length", key_dim),
        value_dim=_as_int(f"{arch}.attention.value_length", value_dim),
        training_ctx=_as_int(f"{arch}.context_length", need("context_length")),
    )
