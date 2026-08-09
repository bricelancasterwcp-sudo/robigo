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

    kv_heads = need("attention.head_count_kv")
    heads = int(need("attention.head_count"))
    key_dim = info.get(f"{arch}.attention.key_length")
    if key_dim is None:
        key_dim = int(need("embedding_length")) // heads
    value_dim = info.get(f"{arch}.attention.value_length", key_dim)
    try:
        kv_heads = max(kv_heads) if isinstance(kv_heads, list) else kv_heads
        return Geometry(
            arch=arch,
            layers=int(need("block_count")),
            kv_heads=int(kv_heads),
            key_dim=int(key_dim),
            value_dim=int(value_dim),
            training_ctx=int(need("context_length")),
        )
    except (TypeError, ValueError) as exc:
        raise GeometryError(
            f"{arch} metadata has a malformed geometry field ({exc}); the KV "
            f"cache size cannot be computed, so the usable window is unknown. "
            f"Pass --window explicitly."
        ) from exc
