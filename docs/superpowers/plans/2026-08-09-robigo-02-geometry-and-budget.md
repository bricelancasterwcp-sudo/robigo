# robigo 02 — Geometry and Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `--window 8192` with a computed usable window derived from the model's real KV-cache geometry and the card's free VRAM, plus a deterministic scope-degradation ladder that refuses loudly rather than overflowing.

**Architecture:** `model/geometry.py` reads KV geometry from GGUF metadata (or Ollama's `/api/show`) and computes bytes-per-token; `usable_window()` combines that with free VRAM and the training context to produce the real ceiling. `context/budget.py` seats the fixed costs, hands the remainder to scope, and degrades scope through five fixed steps before refusing. All of it is pure arithmetic over fixtures except the two probes.

**Tech Stack:** Python 3.12+, stdlib only. `struct` for GGUF header parsing. `nvidia-smi` shelled out for free VRAM, with an explicit-override fallback.

## Global Constraints

- **Runtime dependencies: none.** Standard library only.
- `requires-python = ">=3.12"`; `from __future__ import annotations` in every module. Type annotations on every **non-test** function signature; pytest test functions are exempt.
- **Never return a window above the model's training context.** `llama-server` refuses a slot larger than the model was trained on, and Ollama accepts it silently and rope-degrades (spec §9 law 1).
- **The advertised context length is never the usable window** (spec §3.1). It is one of three inputs, and not the binding one.
- Window and quantization are recorded, per-family, as covariates — never pooled (spec §3.2, §9 law 9).
- A prompt that cannot fit after all five degradation steps is a **refusal before turn 1**, printing the arithmetic — never a truncated attempt (spec §3, step 5).
- Commit messages: `<type>: <subject>`, single line, no body, no trailers.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/robigo/model/gguf.py` | GGUF header/metadata reader — `struct`-level, no deps |
| `src/robigo/model/geometry.py` | `Geometry`, bytes-per-token, `usable_window`, free-VRAM probe |
| `src/robigo/context/budget.py` | `Budget`, fixed-cost seating, the five-step degradation ladder |
| `src/robigo/context/scope.py` | *modified* — `Scope.degrade(step)` |
| `src/robigo/cli.py` | *modified* — `--window auto` becomes the default |

## Verified before execution (2026-08-09)

Plan 01's dominant defect was asserting specifics about external formats that
were never checked. This plan had two such assertions; both were tested against
real files and a real daemon before any task was dispatched. One held, one did
not.

**The GGUF reader (Task 2) is correct as written.** Transcribed verbatim and run
against the blob store: 14 of 14 GGUF blobs parsed with zero failures, and it
independently reproduced 5 of the 6 geometries `/api/show` reports — qwen2.5-coder:7b
(56 KiB/tok), granite-code:8b (144), codegemma:7b (448), qwen3:14b (160),
phi4:14b (200). The sixth simply is not among the largest blobs. It also
confirmed the 192 KiB/token figure the design spec §3.1 estimated for the
14B class.

**`attention.key_length` is absent for `qwen2`**, so `from_model_info`'s
`embedding_length / head_count` fallback is **load-bearing, not a nicety** —
without it the recommended model class yields no head dimension at all. Task 1's
`test_head_dim_falls_back_to_embedding_over_heads` is therefore pinning real
behaviour, not a hypothetical.

**`kv_bytes_per_token` from geometry is right; `OVERHEAD_BYTES = 700MB` is
not.** Measured by loading each model at three context sizes and reading
`/api/ps`'s `size_vram`:

| model | ctx points | marginal KiB/token | vs geometric |
|---|---|---|---|
| qwen2.5-coder:7b-q8 | 4096 / 8192 / 16384 | 92.0 then **58.0** | 1.64× then **1.04×** |
| granite-code:8b-q8 | 1024 / 2048 / 4096 | **145.0** then **145.0** | **1.01×** both |

Two corrections follow.

1. **The geometric figure is accurate at the margin** (1.01–1.04×). An earlier
   two-point measurement suggested 1.24× and was wrong — an artifact of the
   step described below. Keep the arithmetic exactly as Task 1 specifies.
2. **On-disk size overstates VRAM residency**, so subtracting a further 700MB
   double-counts. Fitted intercepts against `/api/tags` `size`:

   | model | fitted intercept | on-disk `size` | difference |
   |---|---|---|---|
   | qwen2.5-coder:7b | 7,293 MiB | 7,723 MiB | −430 MiB |
   | granite-code:8b | 8,275 MiB | 8,806 MiB | −531 MiB |

   `weights_bytes` from `/api/tags` already carries ~480 MiB of slack.
   **`OVERHEAD_BYTES` becomes `256 * 1024 * 1024`**, and its docstring says
   why: it is a margin *on top of* that measured slack, sized to cover the
   per-context-class step below and allocator fragmentation — not an estimate
   of total overhead, which would double-count. Task 3's test that asserts on
   `OVERHEAD_BYTES` uses the constant rather than a literal, so it follows.

**There is a step between context classes.** qwen's 4096→8192 interval cost
92 KiB/token marginal against a 58 KiB/token steady state, so something is
allocated once when crossing into a larger class. A linear model therefore
under-predicts near the low end. This is the reason the arithmetic here is a
**hypothesis** and plan 03's stage 0 probes the window by actually loading it:
where the two disagree, the probe wins.

Reference measurements to test against, taken from local models on 2026-08-09:

| model | layers | kv heads | head dim | training ctx | KiB/token |
|---|---|---|---|---|---|
| qwen2.5-coder:7b | 28 | 4 | 128 | 32768 | 56 |
| granite-code:8b | 36 | 8 | 128 | 4096 | 144 |
| qwen3:14b | 40 | 8 | 128 | 40960 | 160 |
| phi4:14b | 40 | 10 | 128 | 16384 | 200 |
| gemma2:9b | 42 | 8 | 256 | 8192 | 336 |
| codegemma:7b | 28 | 16 | 256 | 8192 | 448 |

---

### Task 1: Geometry and bytes-per-token

**Files:**
- Create: `src/robigo/model/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Produces: `Geometry(arch: str, layers: int, kv_heads: int, key_dim: int, value_dim: int, training_ctx: int)` (frozen), with property `kv_bytes_per_token`, method `kv_bytes(tokens: int, kv_bits: int = 16) -> int`; `GeometryError(Exception)`; `from_model_info(info: dict) -> Geometry`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.model.geometry'`

- [ ] **Step 3: Write minimal implementation**

```python
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
        raise GeometryError("general.architecture missing from model metadata")

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
    if isinstance(kv_heads, list):
        kv_heads = max(kv_heads)
    layers = int(need("block_count"))
    heads = int(need("attention.head_count"))
    key_dim = info.get(f"{arch}.attention.key_length")
    if key_dim is None:
        key_dim = int(need("embedding_length")) // heads
    value_dim = info.get(f"{arch}.attention.value_length", key_dim)
    return Geometry(
        arch=arch,
        layers=layers,
        kv_heads=int(kv_heads),
        key_dim=int(key_dim),
        value_dim=int(value_dim),
        training_ctx=int(need("context_length")),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_geometry.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/model/geometry.py tests/test_geometry.py
git commit -m "feat: KV-cache geometry from model metadata"
```

#### Amendment (ruled 2026-08-09): pin what the tests only imply

Three gaps, all in the reference code above.

**1. Malformed-but-present fields raise the wrong exception type.** `need()`
guards *absence*, but `int()` and `max()` on a present-and-broken value —
`head_count_kv: null`, a non-numeric string — raise raw `TypeError`/`ValueError`.
Task 5 catches `GeometryError` specifically to fall back to an explicit
`--window`, so a malformed field produces an uncaught crash instead of the
documented fallback. Wrap the conversions:

```python
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
```

**2. The `general.architecture` guard names the missing key but no remedy**,
unlike every `need()` message. Make it consistent, and test it:

```python
        raise GeometryError(
            "general.architecture missing from model metadata; the KV cache "
            "size cannot be computed, so the usable window is unknown. Pass "
            "--window explicitly."
        )
```

**3. "Key and value dims used separately" is unpinned.** Both fixtures omit
`attention.value_length`, so `value_dim` always falls back to `key_dim` and a
regression conflating them would pass every existing test. Same shape as the
six vacuous guarantees mutation testing found in plan 01.

Four tests. The first is the one that matters — verify it fails if
`(key_dim + value_dim)` is replaced by `2 * key_dim`:

```python
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
```

---

### Task 2: GGUF metadata reader

**Files:**
- Create: `src/robigo/model/gguf.py`
- Test: `tests/test_gguf.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `read_metadata(path: Path) -> dict[str, object]`; `GGUFError(Exception)`.

Needed because `llama-server` does not expose KV head counts over HTTP — the file is the only source of truth on that path.

GGUF layout: magic `b"GGUF"`, `uint32` version, `uint64` tensor count, `uint64` KV count, then KV pairs of `(string key, uint32 type, value)`. Type ids: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 string, 9 array, 10 u64, 11 i64, 12 f64. Strings are `uint64` length then raw bytes. Arrays are `uint32` element type, `uint64` count, then elements.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gguf.py
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from robigo.model.gguf import GGUFError, read_metadata


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv_u32(key: str, value: int) -> bytes:
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_str(key: str, value: str) -> bytes:
    return _s(key) + struct.pack("<I", 8) + _s(value)


def _kv_arr_u32(key: str, values: list[int]) -> bytes:
    head = _s(key) + struct.pack("<I", 9) + struct.pack("<I", 4) + struct.pack("<Q", len(values))
    return head + b"".join(struct.pack("<I", v) for v in values)


def _gguf(pairs: list[bytes]) -> bytes:
    return (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", len(pairs))
        + b"".join(pairs)
    )


def test_reads_strings_ints_and_arrays(tmp_path: Path):
    path = tmp_path / "m.gguf"
    path.write_bytes(_gguf([
        _kv_str("general.architecture", "qwen2"),
        _kv_u32("qwen2.block_count", 28),
        _kv_arr_u32("qwen2.attention.head_count_kv", [4, 4]),
    ]))
    info = read_metadata(path)
    assert info["general.architecture"] == "qwen2"
    assert info["qwen2.block_count"] == 28
    assert info["qwen2.attention.head_count_kv"] == [4, 4]


def test_stops_reading_after_the_metadata_block(tmp_path: Path):
    # Real GGUFs are gigabytes; the reader must not walk the tensor data.
    path = tmp_path / "m.gguf"
    path.write_bytes(_gguf([_kv_str("general.architecture", "qwen2")]) + b"\x00" * 4096)
    assert read_metadata(path) == {"general.architecture": "qwen2"}


def test_rejects_a_file_that_is_not_gguf(tmp_path: Path):
    path = tmp_path / "not.gguf"
    path.write_bytes(b"ORDINARY FILE")
    with pytest.raises(GGUFError) as e:
        read_metadata(path)
    assert "not a GGUF" in str(e.value)


def test_rejects_an_unknown_value_type(tmp_path: Path):
    path = tmp_path / "m.gguf"
    path.write_bytes(_gguf([_s("k") + struct.pack("<I", 99)]))
    with pytest.raises(GGUFError) as e:
        read_metadata(path)
    assert "99" in str(e.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gguf.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.model.gguf'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/model/gguf.py
from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

_SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_STRING = 8
_ARRAY = 9


class GGUFError(Exception):
    """The file is not a readable GGUF."""


def read_metadata(path: Path) -> dict[str, object]:
    """Key-value metadata only. Returns before the tensor data, so cost
    is independent of the model's size."""
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise GGUFError(f"{path} is not a GGUF file")
        struct.unpack("<I", handle.read(4))          # version, unused
        struct.unpack("<Q", handle.read(8))          # tensor count, unused
        count = struct.unpack("<Q", handle.read(8))[0]
        return {_string(handle): _value(handle, _u32(handle)) for _ in range(count)}


def _u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", handle.read(4))[0]


def _u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", handle.read(8))[0]


def _string(handle: BinaryIO) -> str:
    return handle.read(_u64(handle)).decode("utf-8", errors="replace")


def _value(handle: BinaryIO, kind: int) -> object:
    if kind == _STRING:
        return _string(handle)
    if kind == _ARRAY:
        element = _u32(handle)
        return [_value(handle, element) for _ in range(_u64(handle))]
    if kind in _SCALARS:
        fmt, size = _SCALARS[kind]
        return struct.unpack(fmt, handle.read(size))[0]
    raise GGUFError(f"unknown GGUF value type {kind}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gguf.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/model/gguf.py tests/test_gguf.py
git commit -m "feat: dependency-free GGUF metadata reader"
```

---

### Task 3: Free VRAM and the usable window

**Files:**
- Modify: `src/robigo/model/geometry.py`
- Modify: `tests/test_geometry.py`

**Interfaces:**
- Produces: `free_vram_bytes(runner: Callable[..., str] | None = None) -> int | None`; `WindowPlan(window: int, limited_by: str, free_vram: int | None, kv_per_token: int)` (frozen); `usable_window(geometry, *, free_vram, weights_bytes, kv_bits=16, overhead_bytes=OVERHEAD_BYTES, user_cap=None) -> WindowPlan`; `OVERHEAD_BYTES: int`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_geometry.py
from robigo.model.geometry import (
    OVERHEAD_BYTES,
    WindowPlan,
    free_vram_bytes,
    usable_window,
)

GIB = 1024**3


def test_the_training_context_is_never_exceeded_even_with_vram_to_spare():
    # llama-server refuses a slot larger than the model was trained on,
    # and Ollama accepts it silently and rope-degrades (law 1).
    plan = usable_window(from_model_info(QWEN7B), free_vram=15 * GIB,
                         weights_bytes=8 * GIB)
    assert plan.window == 32768
    assert plan.limited_by == "training_ctx"


def test_vram_binds_when_the_cache_is_expensive():
    plan = usable_window(from_model_info(CODEGEMMA), free_vram=13 * GIB,
                         weights_bytes=9 * GIB)
    # 13 - 9 - overhead, divided by 448 KiB/token
    budget = 13 * GIB - 9 * GIB - OVERHEAD_BYTES
    assert plan.window == budget // (448 * 1024)
    assert plan.limited_by == "vram"
    assert plan.window < 8192      # cannot even reach its advertised max


def test_kv_quantization_buys_window():
    args = dict(free_vram=13 * GIB, weights_bytes=9 * GIB)
    f16 = usable_window(from_model_info(CODEGEMMA), **args)
    q8 = usable_window(from_model_info(CODEGEMMA), kv_bits=8, **args)
    assert q8.window == f16.window * 2


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_geometry.py -v`
Expected: FAIL — `ImportError: cannot import name 'usable_window'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/robigo/model/geometry.py
import subprocess
from typing import Callable

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


def free_vram_bytes(runner: Callable[[], str] | None = None) -> int | None:
    """Free VRAM in bytes, or None when it cannot be measured — in which
    case the caller must fall back to the training context or an explicit
    --window rather than assuming a number."""
    try:
        text = (runner or _nvidia_smi)()
    except Exception:
        return None
    try:
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
    per_token = geometry.kv_bytes_per_token * kv_bits // 16
    limits: list[tuple[int, str]] = [(geometry.training_ctx, "training_ctx")]
    if free_vram is not None:
        spare = free_vram - weights_bytes - overhead_bytes
        limits.append((max(spare, 0) // per_token, "vram"))
    if user_cap is not None:
        limits.append((user_cap, "user_cap"))
    window, limited_by = min(limits, key=lambda pair: pair[0])
    return WindowPlan(
        window=window,
        limited_by=limited_by,
        free_vram=free_vram,
        kv_per_token=per_token,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_geometry.py -v`
Expected: PASS, 13 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/model/geometry.py tests/test_geometry.py
git commit -m "feat: usable window from geometry, free VRAM, and training ctx"
```

---

### Task 4: The budget and the degradation ladder

**Files:**
- Create: `src/robigo/context/budget.py`
- Modify: `src/robigo/context/scope.py`
- Test: `tests/test_budget.py`

**Interfaces:**
- Consumes: `Scope`, `render`, `Geometry`.
- Produces: `SYSTEM_TOKENS: int`; `DIAGNOSTIC_TOKENS: int`; `Budget(window, reserve_out, system, diagnostic, history)` (frozen) with `scope_budget` property; `BudgetExhausted(Exception)`; `estimate_tokens(text: str) -> int`; `reserve_for(codec, file_tokens) -> int`; `fit(scope, budget, root) -> tuple[Scope, int]`; and on `Scope`, `degrade(step: int) -> Scope`.

Degradation order is fixed, not heuristic (spec §3): 1 hop-2 signatures only (the default), 2 hop-2 dropped, 3 hop-1 to signatures, 4 anchor windowed, 5 refuse.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_budget.py
from __future__ import annotations

from pathlib import Path

import pytest

from robigo.context.budget import (
    Budget,
    BudgetExhausted,
    estimate_tokens,
    fit,
    reserve_for,
)
from robigo.context.scope import Scope


@pytest.fixture
def scope(tmp_path: Path) -> Scope:
    (tmp_path / "anchor.py").write_text("def test_x():\n" + "    assert 0\n" * 80)
    (tmp_path / "hop1.py").write_text("def f():\n" + "    x = 1\n" * 80)
    (tmp_path / "hop2.py").write_text("def g():\n" + "    y = 1\n" * 80)
    return Scope(tmp_path / "anchor.py",
                 (tmp_path / "anchor.py", tmp_path / "hop1.py"),
                 (tmp_path / "hop2.py",))


def test_scope_budget_is_the_window_minus_the_fixed_costs():
    budget = Budget(window=4096, reserve_out=512, system=350, diagnostic=600,
                    history=200)
    assert budget.scope_budget == 4096 - 512 - 350 - 600 - 200


def test_reserve_for_whole_file_covers_the_file_plus_margin():
    # whole_file must reserve the entire file: this is why weak families
    # are least able to afford the codec easiest for them (spec 3.3).
    assert reserve_for("whole_file", file_tokens=1000) == 1150
    assert reserve_for("search_replace", file_tokens=1000) == 512
    assert reserve_for("udiff", file_tokens=1000) == 384


def test_estimate_tokens_is_conservative_for_code():
    # Deliberately crude and deliberately NOT authoritative: the server's
    # tokenizer always outranks it (spec 3.3).
    assert estimate_tokens("x" * 36) == 11


def test_a_generous_window_keeps_the_full_scope(scope: Scope, tmp_path: Path):
    fitted, step = fit(scope, Budget(32768, 512, 350, 600, 200), tmp_path)
    assert step == 1
    assert fitted.full == scope.full and fitted.signatures == scope.signatures


def test_a_tight_window_drops_hop_two_then_reduces_hop_one(scope: Scope, tmp_path: Path):
    fitted, step = fit(scope, Budget(700, 128, 60, 60, 0), tmp_path)
    assert step >= 2
    assert fitted.signatures == ()


def test_an_impossible_window_refuses_and_prints_the_arithmetic(scope: Scope, tmp_path: Path):
    with pytest.raises(BudgetExhausted) as e:
        fit(scope, Budget(200, 128, 60, 60, 0), tmp_path)
    message = str(e.value)
    for token in ("window 200", "--scope", "reserve"):
        assert token in message


def test_degrade_step_three_reduces_hop_one_to_signatures(scope: Scope):
    reduced = scope.degrade(3)
    assert reduced.full == (scope.anchor,)
    assert scope.full[1] in reduced.signatures


def test_degrade_step_four_windows_the_anchor(scope: Scope):
    assert scope.degrade(4).anchor_window is not None


def test_render_honours_the_anchor_window(scope: Scope, tmp_path: Path):
    # If render ignored anchor_window, budget would compute a cost the
    # prompt does not have -- the arithmetic would say "fits" and the
    # server would reject it. The two must agree.
    from robigo.adapters.base import Diagnostic
    from robigo.context.render import render

    diag = Diagnostic(False, "anchor.py", 40, "AssertionError", "raw")
    windowed = scope.degrade(4)
    out = render(windowed, diag, (), "search_replace", tmp_path)
    full = render(scope, diag, (), "search_replace", tmp_path)
    assert len(out) < len(full)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_budget.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.context.budget'`

- [ ] **Step 3: Write minimal implementation**

First extend `Scope` in `src/robigo/context/scope.py`:

```python
# replace the Scope dataclass in src/robigo/context/scope.py
@dataclass(frozen=True)
class Scope:
    anchor: Path
    full: tuple[Path, ...]
    signatures: tuple[Path, ...]
    anchor_window: tuple[int, int] | None = None

    def degrade(self, step: int) -> Scope:
        """One step down a FIXED ladder (spec section 3). Fixed rather
        than heuristic so the result is reproducible and testable without
        a model."""
        if step <= 1:
            return self
        if step == 2:
            return Scope(self.anchor, self.full, (), self.anchor_window)
        if step == 3:
            return Scope(self.anchor, (self.anchor,), self.full[1:], None)
        return Scope(self.anchor, (self.anchor,), self.full[1:], (-60, 60))
```

Then teach `render` to honour the window, or the budget's arithmetic and the
actual prompt disagree. In `src/robigo/context/render.py`, replace the
full-text loop with:

```python
    for path in scope.full:
        text = path.read_text(encoding="utf-8")
        label = _rel(path, root)
        if path == scope.anchor and scope.anchor_window:
            text = _window_text(text, scope.anchor_window)
            label += " (windowed around the failure)"
        parts.append(f"--- {label} ---")
        parts.append(text)
```

and add, at the bottom of `render.py`:

```python
def _window_text(text: str, span: tuple[int, int]) -> str:
    """Must stay identical to budget._window, or the estimate and the
    prompt diverge and 'it fits' becomes a lie."""
    lines = text.split("\n")
    middle = len(lines) // 2
    return "\n".join(lines[max(0, middle + span[0]) : middle + span[1]])
```

Then:

```python
# src/robigo/context/budget.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robigo.context.scope import Scope, signatures_of

CHARS_PER_TOKEN = 3.3
"""Calibrated for code. Deliberately crude: it exists to catch an overrun
early, and the server's real tokenizer always outranks it (spec 3.3)."""

SYSTEM_TOKENS = 350
DIAGNOSTIC_TOKENS = 600
MAX_STEP = 5


class BudgetExhausted(Exception):
    """Scope cannot fit after every degradation step. Raised BEFORE any
    generation: with no evidence yet, a refusal is honest and a truncated
    attempt is not (spec section 3, step 5)."""


@dataclass(frozen=True)
class Budget:
    window: int
    reserve_out: int
    system: int = SYSTEM_TOKENS
    diagnostic: int = DIAGNOSTIC_TOKENS
    history: int = 200

    @property
    def scope_budget(self) -> int:
        return (
            self.window - self.reserve_out - self.system
            - self.diagnostic - self.history
        )


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def reserve_for(codec: str, file_tokens: int) -> int:
    if codec == "whole_file":
        return int(file_tokens * 1.15)
    return 512 if codec == "search_replace" else 384


def fit(scope: Scope, budget: Budget, root: Path) -> tuple[Scope, int]:
    """The first degradation step whose rendered scope fits, or refuse."""
    for step in range(1, MAX_STEP):
        candidate = scope.degrade(step)
        if _cost(candidate) <= budget.scope_budget:
            return candidate, step
    raise BudgetExhausted(
        f"scope cannot fit the window after all {MAX_STEP - 1} degradation "
        f"steps.\n"
        f"  window {budget.window}   system {budget.system}   "
        f"reserve {budget.reserve_out}   diagnostic {budget.diagnostic}\n"
        f"  available for scope {budget.scope_budget}   "
        f"smallest scope {_cost(scope.degrade(MAX_STEP - 1))}\n"
        f"Narrow it with --scope, or use a model with a larger window."
    )


def _cost(scope: Scope) -> int:
    total = 0
    for path in scope.full:
        text = path.read_text(encoding="utf-8")
        if path == scope.anchor and scope.anchor_window:
            text = _window(text, scope.anchor_window)
        total += estimate_tokens(text)
    for path in scope.signatures:
        total += estimate_tokens(signatures_of(path.read_text(encoding="utf-8")))
    return total


def _window(text: str, span: tuple[int, int]) -> str:
    """Imported from render rather than reimplemented: two copies of this
    would drift, and the drift shows up as an estimate that says a prompt
    fits when the server disagrees."""
    from robigo.context.render import _window_text

    return _window_text(text, span)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_budget.py tests/test_scope.py tests/test_render.py -v`
Expected: PASS — 9 budget tests, plus the 5 scope and 4 render tests from
plan 01 still green

- [ ] **Step 5: Commit**

```bash
git add src/robigo/context tests/test_budget.py
git commit -m "feat: budget arithmetic and a fixed scope degradation ladder"
```

---

### Task 5: Wire `--window auto` into the CLI and the loop

**Files:**
- Modify: `src/robigo/cli.py`
- Modify: `src/robigo/loop.py`
- Create: `src/robigo/model/detect.py`
- Test: `tests/test_window_auto.py`

**Interfaces:**
- Produces: `detect_geometry(backend, model, host, gguf_path=None) -> Geometry`; `plan_window(backend, model, host, user_cap) -> WindowPlan`; CLI accepts `--window auto|<int>` with `auto` as the default, and `--kv-bits {16,8}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_window_auto.py
from __future__ import annotations

from pathlib import Path

import pytest

from robigo.model.detect import detect_geometry, plan_window

# Repeated rather than imported from tests/test_geometry.py: cross-test
# imports need tests/ to be an importable package, which it is not.
QWEN7B = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 28,
    "qwen2.attention.head_count": 28,
    "qwen2.attention.head_count_kv": 4,
    "qwen2.attention.key_length": 128,
    "qwen2.context_length": 32768,
    "qwen2.embedding_length": 3584,
}


def test_ollama_geometry_comes_from_api_show(monkeypatch):
    monkeypatch.setattr(
        "robigo.model.detect._show",
        lambda model, host: {"model_info": QWEN7B, "size": 8 * 1024**3},
    )
    geometry = detect_geometry("ollama", "qwen2.5-coder:7b", "")
    assert (geometry.layers, geometry.kv_heads) == (28, 4)


def test_plan_window_reports_what_bound_it(monkeypatch):
    monkeypatch.setattr(
        "robigo.model.detect._show",
        lambda model, host: {"model_info": QWEN7B, "size": 8 * 1024**3},
    )
    monkeypatch.setattr("robigo.model.detect.free_vram_bytes", lambda: 15 * 1024**3)
    plan = plan_window("ollama", "m", "", user_cap=None)
    assert (plan.window, plan.limited_by) == (32768, "training_ctx")


def test_a_user_cap_above_the_training_context_is_clamped_not_honoured(monkeypatch):
    monkeypatch.setattr(
        "robigo.model.detect._show",
        lambda model, host: {"model_info": QWEN7B, "size": 8 * 1024**3},
    )
    monkeypatch.setattr("robigo.model.detect.free_vram_bytes", lambda: 15 * 1024**3)
    # Asking for 65536 on a 32768-trained model must NOT be granted:
    # Ollama would accept it silently and rope-degrade (law 1).
    plan = plan_window("ollama", "m", "", user_cap=65536)
    assert plan.window == 32768


def test_cli_accepts_the_word_auto(monkeypatch, tmp_path: Path):
    from robigo.cli import main
    import subprocess

    (tmp_path / "test_x.py").write_text("def test_x():\n    assert 0\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.setattr("robigo.cli.plan_window",
                        lambda *a, **k: pytest.importorskip("robigo.model.geometry")
                        .WindowPlan(4096, "vram", None, 56 * 1024))
    # The model does not exist, so this must end as infrastructure (4),
    # proving the window was resolved and the loop was entered.
    assert main(["--root", str(tmp_path), "--model", "nope",
                 "--window", "auto", "fix"]) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_window_auto.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.model.detect'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/model/detect.py
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from robigo.model.geometry import (
    Geometry,
    GeometryError,
    WindowPlan,
    free_vram_bytes,
    from_model_info,
    usable_window,
)
from robigo.model.gguf import read_metadata

OLLAMA_HOST = "http://127.0.0.1:11434"


def _show(model: str, host: str) -> dict:
    req = urllib.request.Request(
        f"{(host or OLLAMA_HOST).rstrip('/')}/api/show",
        data=json.dumps({"model": model}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def detect_geometry(
    backend: str, model: str, host: str, gguf_path: Path | None = None
) -> Geometry:
    """Ollama publishes the geometry over HTTP; llama-server does not, so
    on that path the GGUF file is the only source of truth."""
    if backend == "ollama":
        return from_model_info(_show(model, host).get("model_info", {}))
    if gguf_path is None:
        raise GeometryError(
            "llama.cpp does not expose KV geometry over HTTP. Pass "
            "--gguf <path> so the window can be computed, or --window <int>."
        )
    return from_model_info(read_metadata(gguf_path))


def weights_bytes(backend: str, model: str, host: str, gguf_path: Path | None) -> int:
    if backend == "ollama":
        return int(_show(model, host).get("size", 0))
    return gguf_path.stat().st_size if gguf_path else 0


def plan_window(
    backend: str,
    model: str,
    host: str,
    user_cap: int | None,
    *,
    kv_bits: int = 16,
    gguf_path: Path | None = None,
) -> WindowPlan:
    geometry = detect_geometry(backend, model, host, gguf_path)
    return usable_window(
        geometry,
        free_vram=free_vram_bytes(),
        weights_bytes=weights_bytes(backend, model, host, gguf_path),
        kv_bits=kv_bits,
        user_cap=user_cap,
    )
```

Then in `src/robigo/cli.py`, replace the `--window` argument and add resolution:

```python
# in src/robigo/cli.py — replace the --window line with:
    parser.add_argument("--window", default="auto",
                        help="'auto' (default) computes it from model "
                             "geometry and free VRAM; an integer caps it")
    parser.add_argument("--kv-bits", dest="kv_bits", type=int,
                        choices=(16, 8), default=16)
    parser.add_argument("--gguf", type=Path, default=None,
                        help="GGUF path, required with --backend llamacpp "
                             "when --window is auto")
```

and immediately after `args = parser.parse_args(argv)`:

```python
    cap = None if args.window == "auto" else int(args.window)
    try:
        plan = plan_window(args.backend, args.model, args.host or "", cap,
                           kv_bits=args.kv_bits, gguf_path=args.gguf)
    except (GeometryError, OSError) as exc:
        print(f"cannot determine the usable window: {exc}")
        return OUTCOMES["infrastructure"]
    args.window = plan.window
    print(f"window {plan.window} (limited by {plan.limited_by}, "
          f"{plan.kv_per_token // 1024} KiB/token)")
```

with imports added at the top of `cli.py`:

```python
from robigo.loop import OUTCOMES, run
from robigo.model.detect import plan_window
from robigo.model.geometry import GeometryError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_window_auto.py -v` then `pytest -q`
Expected: PASS, 4 tests; full suite green

- [ ] **Step 5: Commit**

```bash
git add src/robigo/model/detect.py src/robigo/cli.py tests/test_window_auto.py
git commit -m "feat: --window auto computed from geometry and free VRAM"
```

- [ ] **Step 6: Verify against real models**

Run:
```bash
robigo --model granite-code:8b-instruct-q8_0 --window auto --root /tmp/x "fix" ; echo "exit $?"
robigo --model codegemma:7b-instruct-q8_0 --window auto --root /tmp/x "fix" ; echo "exit $?"
```
Expected: granite prints a window of 4096 limited by `training_ctx`; codegemma
prints **below** its advertised 8192, limited by `vram`, at 448 KiB/token.
That second line is the whole point of this plan — an advertised window the
card cannot actually hold.

---

## Done when

- `pytest -q` green.
- `--window auto` is the default and prints what bound it.
- codegemma resolves to a window below its advertised 8192, attributed to
  `vram`; granite resolves to exactly 4096, attributed to `training_ctx`.
- A window request above the training context is clamped, never granted.
- An impossible scope refuses with the arithmetic printed, before any
  generation.
