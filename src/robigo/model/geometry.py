# src/robigo/model/geometry.py
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable


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
    except (TypeError, ValueError, OverflowError) as exc:
        # OverflowError: int(float('inf')) raises it, and stdlib json.loads
        # accepts the bare `Infinity` token by default, so this is reachable
        # from a hostile /api/show response, not just a crafted dict.
        raise GeometryError(
            f"{key} is malformed ({value!r}); the KV cache size cannot be "
            f"computed, so the usable window is unknown. Pass --window "
            f"explicitly."
        ) from exc


def _positive(key: str, value: int) -> int:
    """Reject a non-positive dimension. `_as_int` checks that a field is an
    integer; this checks it is a usable one. A zero here is not a harmless
    edge case: it makes `kv_bytes_per_token` zero, and `usable_window` then
    divides by it and raises a bare ZeroDivisionError that Task 5's
    `except GeometryError` fallback cannot catch."""
    if value <= 0:
        raise GeometryError(
            f"{key} is {value}; a model cannot have a non-positive value "
            f"there, so the KV cache size cannot be computed and the usable "
            f"window is unknown. Pass --window explicitly."
        )
    return value


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
        layers=_positive(
            f"{arch}.block_count", _as_int(f"{arch}.block_count", need("block_count"))
        ),
        kv_heads=_positive(f"{arch}.attention.head_count_kv", kv_heads),
        key_dim=_positive(
            f"{arch}.attention.key_length",
            _as_int(f"{arch}.attention.key_length", key_dim),
        ),
        value_dim=_positive(
            f"{arch}.attention.value_length",
            _as_int(f"{arch}.attention.value_length", value_dim),
        ),
        training_ctx=_positive(
            f"{arch}.context_length",
            _as_int(f"{arch}.context_length", need("context_length")),
        ),
    )


OVERHEAD_BYTES = 256 * 1024 * 1024
"""A margin ON TOP OF the slack `weights_bytes` already carries, not an
estimate of total overhead.

Measured 2026-08-09 by loading two models at three context sizes each and
fitting `/api/ps`'s `size_vram`: the fitted intercept sits 430 MiB (7B) to
531 MiB (8B) BELOW the on-disk `size` that `weights_bytes` reports, because
on-disk size overstates VRAM residency. Subtracting a full overhead estimate
on top of that double-counts and refuses windows that genuinely fit — and
context is the scarcest resource this project has.

This 256 MiB covers the observed per-context-class step (qwen's 4096→8192
interval cost 92 KiB/token against a 58 KiB/token steady state) plus
allocator fragmentation. Under-reserving trades a clear refusal for an OOM
mid-run, so the direction of the margin matters more than its precision."""


@dataclass(frozen=True)
class WindowPlan:
    window: int
    limited_by: str
    free_vram: int | None
    kv_per_token: int
    weights_bytes: int
    overhead_bytes: int
    training_ctx: int = 0
    """The model's real training context (`Geometry.training_ctx`),
    carried through even when it is NOT the limit that decided `.window`
    (whole-branch review C1/C3, ruled 2026-08-10). Before this field
    existed, a caller building a `Profile` had only `.window` -- `min(
    training_ctx, vram, user_cap)` -- to report as "the training context",
    so whenever vram or a user cap actually bound, that field reported the
    BINDING limit's number, not the model's training context; a run where
    vram bound at 0 reported `training_ctx: 0`, a state no real model is
    in. Defaults to 0 only for callers (mostly tests) that never read it;
    `usable_window`, the only production constructor, always sets it from
    the real `Geometry` it was handed."""


def free_vram_bytes(runner: Callable[[], str] | None = None) -> int | None:
    """Free VRAM in bytes, or None when it cannot be measured — in which
    case the caller must fall back to the training context or an explicit
    --window rather than assuming a number."""
    try:
        text = (runner or _nvidia_smi)()
    except Exception:
        return None
    try:
        # nvidia-smi prints one line per GPU; robigo loads one model onto
        # one device, so GPU 0's line is the deliberate choice here, not an
        # oversight that silently ignores the other cards on a multi-GPU box.
        return int(text.strip().split("\n")[0]) * 1024 * 1024
    except ValueError:
        return None


def _nvidia_smi() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    ).stdout


def usable_window(
    geometry: Geometry,
    *,
    free_vram: int | None,
    weights_bytes: int,
    kv_bits: int = 16,
    overhead_bytes: int = OVERHEAD_BYTES,
    user_cap: int | None = None,
) -> WindowPlan:
    """The largest context that fits, as the minimum of three independent
    limits: the model's training context, what free VRAM allows once the
    weights and a margin are subtracted, and an optional user cap. The
    advertised context length is never the answer by itself — it is one of
    these three inputs, and not always the binding one.

    `free_vram` must be measured BEFORE the model is loaded. This function
    subtracts `weights_bytes` from it, which is only correct when those
    bytes are not already resident. If the model is already loaded (Ollama
    keeps one hot for five minutes by default), `memory.free` already
    excludes the weights, and subtracting them again understates the window
    by roughly the size of the model. Callers own that ordering.
    """
    if kv_bits <= 0:
        # ValueError, not GeometryError: a bad kv_bits is a caller's
        # programming error, not a property of the model file, and must not
        # be swallowed by a fallback meant for unreadable metadata. Without
        # this guard, kv_bits=-16 silently produced a negative window with
        # no error at all -- silent wrong data beats a crash for damage.
        raise ValueError(f"kv_bits must be positive, got {kv_bits}")
    per_token = geometry.kv_bytes_per_token * kv_bits // 16
    limits: list[tuple[int, str]] = [(geometry.training_ctx, "training_ctx")]
    if free_vram is not None:
        spare = free_vram - weights_bytes - overhead_bytes
        # max(spare, 0) keeps this a well-defined (if useless) 0-token
        # window instead of a negative one when VRAM is already exhausted.
        limits.append((max(spare, 0) // per_token, "vram"))
    if user_cap is not None:
        limits.append((user_cap, "user_cap"))
    window, limited_by = min(limits, key=lambda pair: pair[0])
    return WindowPlan(
        window=window,
        limited_by=limited_by,
        free_vram=free_vram,
        kv_per_token=per_token,
        # Carried through so a caller printing a refusal (a window of 0
        # cannot serve any request, and no degradation rung can help --
        # the ladder shrinks the scope, not the KV cache) can show the
        # exact arithmetic without a second, redundant measurement.
        weights_bytes=weights_bytes,
        overhead_bytes=overhead_bytes,
        # The real training context, independent of whether it was the
        # limit that decided `window` above -- see WindowPlan.training_ctx.
        training_ctx=geometry.training_ctx,
    )
