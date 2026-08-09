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

#### Amendment 2 (ruled 2026-08-09): guard every conversion, and name the field

Amendment 1's `try` block was placed wrong — it starts *after* `heads` and
`key_dim` are computed, so two conversions sit outside it. Three escapes
reproduced, all still bypassing Task 5's `GeometryError` fallback:

| input | escapes as |
|---|---|
| `attention.head_count: "not-a-number"` | raw `ValueError` |
| `embedding_length` malformed, `key_length` absent | raw `ValueError` |
| `attention.head_count: 0`, `key_length` absent | **`ZeroDivisionError`** |

The third is the instructive one: `ZeroDivisionError` is an `ArithmeticError`,
so the chosen `except (TypeError, ValueError)` would not have caught it even
with the lines moved inside. And the message embeds the raw exception text
rather than naming the offending field, which is weaker than every `need()`
message beside it.

Replace the blanket `try` with a per-field converter, so every conversion is
guarded and every message names its own key:

```python
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
```

and use it for every field, guarding the division explicitly rather than
catching its exception:

```python
    kv_heads = need("attention.head_count_kv")
    if isinstance(kv_heads, list):
        kv_heads = max(kv_heads) if kv_heads else None
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
        kv_heads=_as_int(f"{arch}.attention.head_count_kv", kv_heads),
        key_dim=_as_int(f"{arch}.attention.key_length", key_dim),
        value_dim=_as_int(f"{arch}.attention.value_length", value_dim),
        training_ctx=_as_int(f"{arch}.context_length", need("context_length")),
    )
```

`max(kv_heads) if kv_heads else None` handles an empty list, which would
otherwise raise `ValueError` from `max()` outside any guard.

#### Amendment 3 (ruled 2026-08-09): the `max()` is a conversion too

Amendment 2 reasoned about the *empty* list and stopped there. A non-empty
list with a malformed element still escapes, because `max()` compares before
`_as_int` ever sees the values:

| input | escapes as |
|---|---|
| `head_count_kv: [4, None]` | raw `TypeError: '>' not supported between …` |
| `head_count_kv: [4, "x"]` | raw `TypeError: '>' not supported between …` |

This is a **regression against fix round 1**, whose blanket `try` covered the
`max()` incidentally; replacing it with per-field guards dropped that cover.
It is the original finding relocated from a scalar field to a list element,
and it is reachable: this metadata arrives JSON-decoded from Ollama's
`/api/show` as well as from GGUF, and a JSON array can hold `null` or a string
where a GGUF typed array cannot.

Reduce the list through a converter of its own, so no comparison ever sees a
non-int, and replace both the `isinstance` branch and the later `_as_int` on
`kv_heads`:

```python
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
```

```python
    kv_heads = _kv_head_count(
        f"{arch}.attention.head_count_kv", need("attention.head_count_kv")
    )
```

with `kv_heads=kv_heads,` in the `Geometry(...)` call — the field is already
an `int` by then, and a second `_as_int` on it would be dead.

This also sharpens the empty-list message, which previously normalised `[]` to
`None` and then reported `head_count_kv is malformed (None)` — telling a user
debugging an empty list that they had passed nothing at all.

Two tests, and the existing valid-per-layer-list test must keep passing:

```python
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
```

**The four tests below belong to Amendment 2**, not to Amendment 3 above —
Amendment 3 was inserted at the point it corrects, which split this block from
its heading. They were already added in fix round 2; do not add them twice.
The last of them, `test_an_empty_kv_head_list_is_geometry_error_not_a_max_failure`,
is **superseded** by Amendment 3's `test_an_empty_per_layer_kv_list_says_it_is_empty`,
which asserts the same behaviour and additionally pins the message.

Four tests. The existing malformed-field test also gains a field-name
assertion, since naming the field is the point:

```python
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

#### Amendment (ruled 2026-08-09): every read must be an exact read

The four tests above cover bad magic and an unknown type id. They do not cover
a **short** read, and the parser as written handles it badly. Measured against
the shipped module, not reasoned about — eight probes, five raw escapes:

| input | result |
|---|---|
| well-formed header | parses |
| empty file | `GGUFError` ✓ |
| just the magic | raw `struct.error` |
| truncated mid-header | raw `struct.error` |
| kv count says 1, file ends | raw `struct.error` |
| truncated mid-key-string | raw `struct.error` |
| key length claims 8 EiB | raw `OverflowError` |
| kv count claims 2**40 | raw `struct.error` |

Two of these matter beyond the exception type:

- **"kv count says 1, file ends" is the interrupted-`ollama pull` shape.** A
  partial blob is the realistic malformed input for this reader, not a hostile
  one. "Real GGUF files won't truncate mid-header" is true only of *complete*
  files, and the reader's job is to tell you when it doesn't have one.
- **A short read inside `_string` does not raise at all.** `handle.read(300)`
  returning 3 bytes decodes cleanly to a 3-character key, so the metadata dict
  gets a *wrong key name* rather than an error. In the probe the mistake
  surfaced one field later, by luck, on the next `unpack`. That is a silent
  corruption path and it is the reason this amendment exists.

`read` returns a short buffer at EOF instead of raising, so every read must go
through one helper that treats short as fatal:

```python
_MAX_READ = 64 * 1024 * 1024
"""No single GGUF field approaches this. The largest metadata values are the
tokenizer arrays, and those are read element by element, so no individual
read is big. A claimed length above this means a corrupt length field, and
honouring it would allocate that much memory."""


def _read_exactly(handle: BinaryIO, n: int, what: str) -> bytes:
    """Read exactly `n` bytes or raise. `read` returns a short buffer at EOF
    rather than raising, which reaches `struct.unpack` as `struct.error` and
    reaches `bytes.decode` as a silently truncated string."""
    if not 0 <= n <= _MAX_READ:
        raise GGUFError(
            f"{what} claims {n} bytes, beyond anything a real GGUF field "
            f"contains; the file is corrupt."
        )
    data = handle.read(n)
    if len(data) != n:
        raise GGUFError(
            f"file ends mid-{what}: wanted {n} bytes, got {len(data)}; the "
            f"file is truncated."
        )
    return data
```

Route **every** `handle.read(...)` in the module through it — the version,
the tensor count, the kv count, `_u32`, `_u64`, `_string`, and the scalar read
in `_value` — passing a `what` that names the field being read. The magic
check may stay as it is: a short read there already fails the `!= b"GGUF"`
comparison and produces the right message, which the empty-file probe
confirms.

The absurd-count cases need no separate guard once this is in place: a kv or
array count of 2**40 makes the first element read hit EOF, and that read now
raises `GGUFError`.

Six tests, one per probe row that currently escapes, plus the silent-corruption
case stated as an assertion about the *value* rather than the exception:

```python
def _header(kv_count: int = 1) -> bytes:
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack(
        "<Q", kv_count
    )


def test_a_file_holding_only_the_magic_is_a_gguf_error(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF")
    with pytest.raises(GGUFError):
        read_metadata(path)


def test_a_header_truncated_mid_field_is_a_gguf_error(tmp_path):
    path = tmp_path / "t.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + b"\x01\x02")
    with pytest.raises(GGUFError):
        read_metadata(path)


def test_a_count_promising_more_pairs_than_the_file_holds_is_a_gguf_error(tmp_path):
    """The interrupted-download shape: the header is intact and claims a
    pair, and the file simply stops."""
    path = tmp_path / "p.gguf"
    path.write_bytes(_header(1))
    with pytest.raises(GGUFError) as e:
        read_metadata(path)
    assert "truncated" in str(e.value)


def test_a_key_string_running_past_the_end_does_not_silently_shorten(tmp_path):
    """The value-level assertion. A short read used to decode cleanly, so the
    key came back as 'abc' instead of failing."""
    path = tmp_path / "s.gguf"
    path.write_bytes(_header(1) + struct.pack("<Q", 300) + b"abc")
    with pytest.raises(GGUFError) as e:
        read_metadata(path)
    assert "truncated" in str(e.value)


def test_an_absurd_string_length_is_refused_before_allocating(tmp_path):
    path = tmp_path / "h.gguf"
    path.write_bytes(_header(1) + struct.pack("<Q", 2**63) + b"abc")
    with pytest.raises(GGUFError) as e:
        read_metadata(path)
    assert "corrupt" in str(e.value)


def test_an_absurd_kv_count_ends_at_the_first_missing_byte(tmp_path):
    path = tmp_path / "c.gguf"
    path.write_bytes(_header(2**40))
    with pytest.raises(GGUFError):
        read_metadata(path)
```

**Re-verified against real files after the rewrite** (controller, 2026-08-09).
Exact reads touch every read path, so the parser's original evidence — "it
parses real blobs" — had to be re-earned rather than assumed. Result: **20 of
20 real GGUF blobs parse, and `from_model_info` derives geometry for all 20**,
reproducing every one of the six independently-known geometries. Nothing in
`_read_exactly` or the 64 MiB `_MAX_READ` rejects a real file. Widest real
spread observed, which is the case for `--window auto` later: 56 KiB/token
(qwen2, 28×4×128) to **800 KiB/token** (a llama with `head_count_kv` 40 —
no GQA at all), a 14× range, and one model advertising `context_length`
1024000.

#### Amendment 2 (ruled 2026-08-09): the two remaining escapes, and a cost test that tests cost

The task review reproduced three defects. All three are the same shape as the
ones already fixed — malformed input escaping as something other than
`GGUFError`, and a test that cannot fail — so they close here rather than
being deferred.

**Both fixes were measured against all 32 real GGUF blobs before being
specified**, because the obvious form of each would have rejected working
models:

| question | measured answer |
|---|---|
| keys failing strict UTF-8 | **0** of 32 blobs |
| string values failing strict UTF-8 | **0**, including ~150k-token tokenizer arrays |
| blobs using nested arrays | **0**; deepest value nesting is 2 (an array of scalars) |

So `errors="strict"` rejects no real file, and refusing nested arrays outright
costs nothing real.

**1. Nested arrays raise `RecursionError`.** `_value` recurses on the array
branch with no depth limit. Each level costs 12 bytes on the wire (`u32`
element type + `u64` count), so a ~24 KB file with 2000 levels exhausts the
stack. Since no real blob nests arrays at all, refuse the construct instead of
counting depth — a depth limit invites picking a number, and a clear refusal is
the honest bound:

```python
    if kind == _ARRAY:
        element = _u32(handle)
        if element == _ARRAY:
            raise GGUFError(
                "nested arrays are not supported; no real GGUF model uses "
                "them, and honouring them would let a crafted file recurse "
                "until the stack is exhausted."
            )
        return [_value(handle, element) for _ in range(_u64(handle))]
```

**2. `errors="replace"` absorbs corrupt bytes silently.** A string of the
correct declared length holding invalid UTF-8 decodes to U+FFFD with no error,
so a bit-flip inside a correctly-sized string yields a mangled value. That is
the same silent corruption Amendment 1 exists to prevent — Amendment 1 only
guarded the *length*. A mangled `general.architecture` then surfaces
downstream as `GeometryError: ... missing from model metadata`, which names
absence when the truth is corruption:

```python
def _string(handle: BinaryIO, what: str = "string") -> str:
    raw = _read_exactly(handle, _u64(handle), what)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GGUFError(
            f"{what} is not valid UTF-8 ({raw[:24]!r}); the file is corrupt."
        ) from exc
```

**3. `test_stops_reading_after_the_metadata_block` does not test cost.** It
asserts only on the parsed dict, and the review demonstrated that a
`path.open("rb").read()` implementation — the exact O(filesize) behaviour the
brief forbids — passes it unchanged. Assert on **bytes actually read**, which
is the property the requirement is about. No large or sparse file is needed:

```python
class _CountingHandle:
    """Wraps a binary handle and totals the bytes read through it. The cost
    requirement is about reads, so the test has to observe reads; asserting
    on the returned dict cannot distinguish a bounded reader from one that
    slurps the file first."""

    def __init__(self, handle):
        self._handle = handle
        self.read_bytes = 0

    def read(self, size=-1):
        data = self._handle.read(size)
        self.read_bytes += len(data)
        return data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._handle.close()
        return False


def test_the_reader_stops_at_the_end_of_the_metadata(tmp_path, monkeypatch):
    path = tmp_path / "tail.gguf"
    metadata = _header(1) + _kv_u32("qwen2.block_count", 28)
    path.write_bytes(metadata + b"\x00" * (1024 * 1024))

    opened = []
    real_open = Path.open

    def counting_open(self, *args, **kwargs):
        handle = _CountingHandle(real_open(self, *args, **kwargs))
        opened.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", counting_open)
    info = read_metadata(path)

    assert info == {"qwen2.block_count": 28}
    assert opened, "read_metadata did not open the path via Path.open"
    assert opened[0].read_bytes <= len(metadata), (
        f"read {opened[0].read_bytes} bytes for {len(metadata)} bytes of "
        f"metadata; the reader is not stopping at the metadata block"
    )
```

Keep the original equality test as well — it pins the parse, and this one pins
the cost.

Tests for the first two, both of which must fail before the fix:

```python
def test_a_nested_array_is_refused_not_a_recursion_error(tmp_path):
    """12 bytes per level on the wire, so a small file can exhaust the
    stack. RecursionError is not a GGUFError."""
    path = tmp_path / "n.gguf"
    nest = _s("deep") + struct.pack("<I", 9)
    nest += b"".join(struct.pack("<I", 9) + struct.pack("<Q", 1) for _ in range(2000))
    path.write_bytes(_header(1) + nest)
    with pytest.raises(GGUFError) as e:
        read_metadata(path)
    assert "nested arrays" in str(e.value)


def test_a_string_of_the_right_length_holding_invalid_utf8_is_refused(tmp_path):
    """Amendment 1 guarded the length; this guards the bytes. Silently
    replacing them yields a mangled value and no error."""
    path = tmp_path / "u.gguf"
    path.write_bytes(_header(1) + struct.pack("<Q", 3) + b"\xff\xfe\xfd")
    with pytest.raises(GGUFError) as e:
        read_metadata(path)
    assert "UTF-8" in str(e.value)
```

After this lands, **re-run the real-blob check again** — `errors="strict"` and
the nested-array refusal both sit on the path every real file takes, and the
20-of-20 result above was measured before them.

**A real-blob test must resolve the blob directory from `OLLAMA_MODELS`.** On
this machine Ollama is a systemd service with
`OLLAMA_MODELS=/mnt/extra/ollama-models`; `~/.ollama/models` holds only
`.modelfile` text and **no `blobs/` directory at all**. A test whose `skipif`
hardcodes `~/.ollama/models/blobs` therefore never runs here and reports
`skipped`, which reads as coverage while guaranteeing nothing. Resolve
`os.environ.get("OLLAMA_MODELS")` first, fall back to
`~/.ollama/models`, and skip only when neither yields a readable `blobs/`.

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

#### Verified before execution (2026-08-09)

`nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits` was run on
the target box before this task was dispatched. It exits 0 and prints a bare
integer in MiB on one line — `14571` — so the parse in `free_vram_bytes` is
correct as written, and `nounits` does mean MiB.

Two facts from that run that the code must reflect rather than discover later:

- **The machine is the target hardware, and "16 GB" is not 16 GiB of headroom.**
  Total is 16303 MiB, used 1269 MiB, **free 14571 MiB**, with no model loaded —
  the desktop and compositor hold ~1.7 GiB before robigo asks for anything.
  This is precisely why the window comes from *measured free* VRAM and never
  from `memory.total`, and it is the number the degradation ladder in Task 4
  will actually be working against.
- **`split("\n")[0]` takes GPU 0 deliberately, not accidentally.** This box has
  one GPU, so the behaviour is unobservable here, but on a multi-GPU machine
  the command prints one line per GPU. robigo loads one model on one device,
  so GPU 0 is the choice; say so in a comment at that line, because an
  unexplained `[0]` reads as a bug that silently ignores the other cards.

**`free_vram` must be read before the model is resident.** `usable_window`
subtracts `weights_bytes` from `free_vram`, which is right only when the
weights are not yet loaded. If the model is already resident — Ollama keeps
one hot for five minutes by default — `memory.free` already excludes those
bytes and subtracting them again understates the window by roughly the size
of the model. Task 5 owns that ordering; state the precondition in
`usable_window`'s docstring here so the caller cannot get it wrong silently.

#### Amendment (ruled 2026-08-09): two of these tests contradict their own fixture

The implementer reported BLOCKED rather than guessing, and was right.
`test_vram_binds_when_the_cache_is_expensive` and
`test_kv_quantization_buys_window` cannot pass as written, because the numbers
in them do not produce the outcome they assert. Verified by calculation, not
by hand:

    CODEGEMMA is 448 KiB/token with training_ctx 8192
    spare  = 13 GiB - 9 GiB - 256 MiB = 3.75 GiB
    vram   = 3.75 GiB // 448 KiB      = 8777 tokens
    min(8192 training_ctx, 8777 vram)  -> 8192, limited_by "training_ctx"

So the vram limit lands *above* the training context, `training_ctx` binds, and
both `limited_by == "vram"` and `window < 8192` fail. The fixture was fine; my
chosen `free_vram` was.

Use `free_vram = 10 * GIB + GIB // 8` (10.125 GiB), which leaves 896 MiB of
spare and divides **exactly**: 2048 tokens at 16-bit KV and 4096 at 8-bit, with
no floor truncation, so the doubling is exact rather than approximate.

Derive the expected window in the test from the **measured** 448 KiB/token
figure rather than from `geometry.kv_bytes_per_token`. Restating it
independently is the point: a test that recomputes the expectation with the
same property it is testing would pass even if that property were wrong, and a
wrong bytes-per-token is the failure this whole plan exists to prevent.

```python
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
```

The `== 2048` and `== 4096` literals are deliberate alongside the derived
expression: they pin `OVERHEAD_BYTES` at its measured 256 MiB, so
re-measuring that constant fails here loudly instead of quietly shifting every
window the tool reports.

**Keep the two supplementary tests the implementer added** — a frozen-`WindowPlan`
check and the exhausted-VRAM zero-window case. Both cover constraints this plan
states in prose and never tested, which is the gap that has cost this plan the
most.

#### Amendment 2 (ruled 2026-08-09): harden the model layer before Task 4 builds on it

Task 3's review approved the task and raised two Important robustness gaps,
judging them unreachable through the call sites this plan currently wires up.
One of them is reachable from a **file**, which changes the calculus:

`from_model_info({..., "qwen2.block_count": 0, ...})` returns a `Geometry`
whose `kv_bytes_per_token` is `0`, because `_as_int` validates *type* and not
*value*. `usable_window` then divides by it and raises an uncaught
`ZeroDivisionError` — escaping the `except GeometryError` fallback that Task 5
depends on, from ordinary malformed metadata rather than programmer error.
Task 1 already guards `heads <= 0` for exactly this reason; it simply never
generalised to the other fields.

Fix that at the source, in `from_model_info`, where every other malformed-field
refusal already lives. A `Geometry` should not be constructible in a state that
makes the window planner crash:

```python
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
```

Apply it to every dimension `Geometry` carries — `block_count`,
`head_count_kv`, `key_length`, `value_length`, `context_length` — by wrapping
each `_as_int(...)` result at the constructor call. The existing `heads <= 0`
guard stays where it is: it must fire *before* the division that derives
`key_dim`, which is earlier than this check.

Second, guard the division in `usable_window`. `kv_bits <= 0` is the worse of
the two failures the review found, because a negative `kv_bits` yields a
**negative window with no error at all** — observed `window=-126391,
limited_by='vram'`. Silent wrong data beats a crash for damage:

```python
    if kv_bits <= 0:
        raise ValueError(f"kv_bits must be positive, got {kv_bits}")
```

`ValueError` rather than `GeometryError` is deliberate: a bad `kv_bits` is a
caller's programming error, not a property of the model file, and it should not
be swallowed by a fallback meant for unreadable metadata.

Two carried Minors close in the same pass, since both touch these files:

- **`_as_int` does not catch `OverflowError`.** `int(float('inf'))` raises it,
  and stdlib `json.loads` accepts the `Infinity` token by default, so it is
  reachable from a hostile `/api/show` response. Add `OverflowError` to the
  except tuple — one word, same message.
- **Both `_string` call sites in `gguf.py` pass the default `what`**, so a
  corrupt-file message cannot say whether the key or the value was bad. Pass
  `"key"` at the key site and `"string value"` at the value site.

Tests: a non-positive `block_count` and a non-positive `head_count_kv` each
raise `GeometryError` naming the field; `kv_bits=0` and `kv_bits=-16` each raise
`ValueError`; `float('inf')` in a metadata field raises `GeometryError`; and a
corrupt key versus a corrupt value produce distinguishable messages.

**Re-run the real-blob check afterwards.** Positivity validation sits on the
path every real file takes. All 20 blobs measured earlier derive geometry with
positive dimensions throughout, so nothing real should be rejected — if
something is, that is a finding, not a reason to loosen the guard.

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

#### Amendment (ruled 2026-08-09): the ladder's rung 4 is dead, and its label lies

The implementer reported BLOCKED and was right on every count. I then measured
each rung's cost against the fixture rather than deriving it, and the
measurement found two defects worse than the one reported. **This amendment
states invariants and the tests that falsify them, not code to transcribe** —
plan 01's first process lesson, which the last seven amendment errors ignored.

Measured, fixture as specified above (`reserve_out=128, system=60,
diagnostic=60, history=0`, so 248 tokens of fixed cost):

| rung | cost | saves vs previous | window band that lands `fit` here |
|---|---|---|---|
| 1 | 569 | — | ≥ 817 |
| 2 | 566 | **3** | 814 – 816 (**width 3**) |
| 3 | 323 | 243 | 571 – 813 |
| 4 | 323 | **0** | **empty — unreachable** |

**Defect 1 — rung 4 is unreachable and rung 2 is untestable.** `cost(4)`
equals `cost(3)`, so no window exists for which `fit` returns step 4: a scope
falls from rung 3 straight to refusal, and a fixed five-step ladder is really
four. Rung 2's band is 3 tokens wide because dropping hop-2 removes only its
*signatures* — `def g():`, about 3 tokens. Both are fixture artifacts, and both
hid a real bug behind an untestable gap.

**Defect 2 — the window is centred on the wrong thing, and the label says so
falsely.** `_window_text` centres on `len(lines) // 2`, the middle of the
*file*, and `render` labels the result `" (windowed around the failure)"`. For a
failure at line 350 of a 400-line file, rung 4 hands the model lines 140–260 —
the failing code excluded — while asserting the opposite. That is the same
category as a truncated prompt: the model answers about code it cannot see,
and here it is additionally told it can. It also explains defect 1: a window
of `(-60, 60)` around the middle of an 81-line file is the whole file, so
rung 4 saves nothing for any file of 120 lines or fewer, which is most files.

**Invariants to satisfy.** Do not transcribe a fix; make these true and prove
each with a test that fails when it is violated:

1. **The window contains the failing line.** After `degrade(4)`, the rendered
   anchor includes the diagnostic's line whenever that line is in the file.
   The diagnostic's line is available where the `Scope` is built — `resolve`
   already receives `diag` — and is not available inside `degrade`, so carry it
   on the `Scope` rather than recomputing it. Give it a default so every
   existing construction stays valid.
2. **The estimate and the prompt agree exactly.** `budget`'s cost for a
   windowed scope and `render`'s output for the same scope must window
   identically. Both must derive the centre from the `Scope` alone — if one
   uses the diagnostic and the other uses the file's middle, "it fits" becomes
   a lie in the direction that OOMs or truncates. The single-implementation
   plus thin-delegating-wrapper arrangement stays; that is what makes this
   invariant structural.
3. **The label is true.** If the text is windowed, say so and say around what.
   If a window would not shrink the file, do not claim a window was applied.
4. **Every rung is individually reachable through `fit`.** For each of steps
   1–4 there must exist a window that makes `fit` return exactly that step,
   and a test that pins it. Rung 4 must save real tokens against rung 3.

**Fixture, so the bands are wide enough to test.** Make the anchor **400
lines** — windowing to 120 lines then removes ~70% of it — and give `hop2.py`
**40 separate `def`s** so its signatures cost something real. Measured with
that fixture and middle-centred windowing:

| rung | cost | saves | band | width |
|---|---|---|---|---|
| 1 | 1958 | — | ≥ 2206 | — |
| 2 | 1827 | 131 | 2075 – 2205 | 131 |
| 3 | 1584 | 243 | 1832 – 2074 | 243 |
| 4 | 476 | **1108** | 724 – 1831 | 1108 |

**Re-measure after implementing invariant 1** — centring on the failure changes
what rung 4 contains — and choose each test's window from the **middle** of its
measured band, not its edge. Report the measured table. A budget picked at a
band edge is one estimator tweak away from testing the neighbouring rung, which
is how the original numbers came to assert rung 2's shape while landing on
rung 3.

**Corrected assertions for the test that failed.** Its name is right and its
body described the wrong rung: at step 3, hop-1 has become a signature, so
`signatures` is `(hop1,)` and `full` is `(anchor,)` — `== ()` is rung 2's
shape. Pin the exact rung, since the ladder is fixed rather than heuristic:

```python
def test_a_tight_window_reduces_hop_one_to_signatures(scope: Scope, tmp_path: Path):
    fitted, step = fit(scope, Budget(<mid of rung 3 band>, 128, 60, 60, 0), tmp_path)
    assert step == 3
    assert fitted.full == (scope.anchor,)
    assert fitted.signatures == (scope.full[1],)
```

**And the anchor-window test must isolate windowing.** As written it compares
`degrade(4)` against the undegraded scope, so it passes on the length drop from
rung 3's hop-1 collapse — the implementer proved this by mutation: a
`_window_text` that returns its input unchanged leaves the test green. Compare
against `degrade(3)` instead. Those two scopes differ *only* in
`anchor_window`, so the difference can come from nothing else:

```python
def test_windowing_the_anchor_is_what_shrinks_step_four(scope, tmp_path):
    diag = Diagnostic(False, "anchor.py", 40, "AssertionError", "raw")
    at_3 = render(scope.degrade(3), diag, (), "search_replace", tmp_path)
    at_4 = render(scope.degrade(4), diag, (), "search_replace", tmp_path)
    assert len(at_4) < len(at_3)
```

#### Amendment 2 (ruled 2026-08-09): make invariant 2 structural, not partial

Task 4's review confirmed all four invariants and then found that invariant 2
holds only for the *windowed file text*, which is narrower than the invariant
was meant to be. Three of its findings share one root cause: `_cost` and
`render` each build their own idea of the scope's text, so they can disagree.
Measured against this fixture — `_cost` reports **445** where the rendered
prompt costs **721**, a 276-token gap.

Three concrete divergences, all verified live by the reviewer:

1. **The estimate is systematically low.** `_cost` counts file contents only,
   never the `--- {label} ---` header render emits per file, nor the windowing
   suffix. Low is the dangerous direction: it is the one that says "fits" and
   then gets truncated or OOMs. It is currently masked by the fixed
   `system`+`diagnostic` reserve, but Task 5 sets those to their real values,
   and then nothing covers the headers.
2. **`_cost` crashes where `render` degrades.** `_cost` calls bare
   `path.read_text(encoding="utf-8")`; `render` uses the guarded `_read` and
   substitutes `_UNREADABLE`. On a non-UTF-8 or deleted file, `render` produces
   a clean prompt while `_cost` raises `UnicodeDecodeError` — so `fit` crashes
   instead of fitting or refusing with arithmetic, which is the one guarantee
   this task exists to provide.
3. **The honest fallback label is untested.** The branch used when
   `anchor_window` is set but `anchor_line` is `None` has no coverage:
   collapsing it to print the literal `windowed around line None` leaves the
   whole suite green. Reachable whenever an adapter cannot determine a line.

**The invariant, stated properly.** For any `Scope`, the cost `fit` reasons
about must equal the cost of the text `render` actually emits for that scope's
files:

    _cost(scope) == estimate_tokens(<exactly what render emits for scope's files>)

Satisfy it **structurally** — extract the scope's file section into one
function that both call, rather than keeping two implementations in step by
discipline. Two copies is what produced all three findings above, and the
`_window_text`/`_window` delegation already in this task is the pattern to
follow: one implementation, one thin caller.

Do not thread `diag`, `history` or `codec` into `_cost`. Those are the caller's
fixed costs and are already seated in `Budget`; the scope's own text is what
must agree.

Falsification tests — each must fail if the two diverge:

```python
def test_the_estimate_equals_the_cost_of_what_render_emits(scope, tmp_path):
    """Not 'close to'. Two implementations of the same text drift, and the
    drift is invisible until a prompt the arithmetic called safe comes back
    truncated."""
    for step in (1, 2, 3, 4):
        candidate = scope.degrade(step)
        assert _cost(candidate) == estimate_tokens(
            _scope_section(candidate, tmp_path)
        )


def test_an_unreadable_file_is_costed_the_way_it_is_rendered(scope, tmp_path):
    """render substitutes a placeholder; the estimate must cost that same
    placeholder rather than raising. A crash here is worse than a bad
    number: fit's contract is fit-or-refuse-with-arithmetic."""
    (tmp_path / "hop1.py").write_bytes(b"\xff\xfe not utf-8")
    cost = _cost(scope)                      # must not raise
    assert cost == estimate_tokens(_scope_section(scope, tmp_path))


def test_a_windowed_scope_with_no_known_line_is_labelled_honestly(tmp_path):
    """anchor_window set, anchor_line None: the label must not claim a line
    it does not have, and must not print the string 'None'."""
    ...  # build the scope directly, assert "None" not in the rendered label
```

Also give `degrade` an explicit upper bound. `step >= 5` currently returns
rung 4's scope silently; `fit` never asks for it today, but a caller that
miscomputes a step should get a `ValueError`, not a plausible-looking scope.

**Re-measure the rung table afterwards** — adding header tokens raises every
rung's cost — and re-pick each test's budget from the middle of its new band.
Report the new table. Every existing test must stay green, including plan 01's
render tests, which are the proof that extracting the shared section did not
change what `render` emits.

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

#### Amendment (ruled 2026-08-09): `/api/show` does not return `size`

Verified against the live daemon before this task was dispatched, and the
plan's `weights_bytes` is wrong on its Ollama path.

`POST /api/show` returns exactly these top-level keys: `capabilities`,
`details`, `license`, `model_info`, `modelfile`, `modified_at`, `system`,
`template`, `tensors`. **There is no `size`.** So

```python
return int(_show(model, host).get("size", 0))
```

silently returns **0** for every Ollama model, and the `.get(..., 0)` default is
what hides it — the same "plausible-looking default" this plan forbids
elsewhere, in the same dangerous direction.

Zero weights means `usable_window` believes the whole card is free. Measured on
this box: free VRAM 14571 MiB, and `qwen2vl` weighs 14540 MiB. Correctly, spare
is `14571 − 14540 − 256 < 0`, so the window is 0 and the run must refuse. With
`weights_bytes` returning 0, spare becomes 14315 MiB and the window comes back
as its full advertised **128000** — a request that cannot possibly be served.
The failure converts "this model does not fit" into the largest window in the
table.

`GET /api/tags` **does** carry `size`, per model, and needs no model load. Each
entry has `capabilities, details, digest, model, modified_at, name, size` —
confirmed `qwen2.5-coder:7b-instruct-q8_0` → 8,098,539,207 bytes (7.54 GiB).

**Invariant: `weights_bytes` never guesses.** It returns a real measured size
or raises. There is no default, because a wrong weights figure is invisible
until the server refuses the slot or the allocation OOMs.

Take the size from `/api/tags`, matching the model by name. 12 of the 30 names
on this box end in `:latest`, so a bare `model` argument may need that suffix
appended to match — try the exact name first, then `f"{model}:latest"`. If
neither matches, raise `GeometryError` naming the model and listing what the
daemon does know, rather than returning a number.

Falsification tests, both offline with an injected responder — no test may
contact a daemon:

```python
def test_weights_come_from_tags_not_show(...):
    """/api/show has no `size` field at all. A .get(..., 0) default here
    reports a 0-byte model and hands back the largest window in the table."""
    ...  # a responder whose /api/show lacks `size` and whose /api/tags has it;
         # assert the real byte count is returned


def test_an_unknown_model_raises_rather_than_reporting_zero_weights(...):
    ...  # /api/tags without the requested model; assert GeometryError naming it
```

Add one `@pytest.mark.live` test — the marker already means "requires a running
model daemon" (`pyproject.toml:19-20`) and is deselected by default — asserting
that the real `/api/show` response for a locally-present model still lacks
`size`, so this regression is caught if Ollama ever adds the field back and the
workaround becomes unnecessary.

#### Amendment 2 (ruled 2026-08-09): a zero window must refuse, and two messages are wrong

Task 5 was exercised end to end against the live daemon on the target box. The
window arithmetic is right — `--window auto` gives
`32768 (limited by training_ctx, 28 KiB/token)`, `--window 999999` clamps to
`32768` by `training_ctx`, `--window 2048` reports `user_cap`, and the 14.2 GB
model correctly reports `window 0 (limited by vram, 56 KiB/token)` where the
pre-amendment `.get("size", 0)` bug would have handed back 128000. Three
defects remain, all in what happens around that arithmetic.

**1. A zero window prints and then proceeds.** `window 0` means the weights plus
the margin already exceed free VRAM: not one token of context fits, and no
degradation rung can help, because the ladder shrinks the *scope*, not the KV
cache. The run must stop there. Today it prints the line and continues into
adapter setup, so on a repo where the adapter succeeds it would go on to build
a prompt against a 0-token budget. This is the plan's own law — a refusal
before turn 1, printing the arithmetic, never an attempt — and `window 0` is
the most clear-cut case of it.

Refuse with `OUTCOMES["refused"]` (exit 3), not `infrastructure`: nothing is
broken in the environment, the model simply does not fit this card, and exit 4
is reserved for a harness that could not run at all. Print the arithmetic that
produced it — free VRAM, the weights, the margin, and the per-token cost — so
the user can see which term to change. A bare `window 0` tells them nothing
actionable; `free 14571 MiB − weights 14540 MiB − margin 256 MiB` tells them to
pick a smaller quantisation.

**2. The llama.cpp geometry message advises a flag the user already passed.**
With `--backend llamacpp --window 4096` and no `--gguf`, the run aborts with
*"Pass `--gguf <path>` so the window can be computed, or `--window <int>`."* —
recommending exactly what was just supplied. Same family as plan 01's messages
naming flags that did not exist: the advice does not match the situation. Make
the text depend on whether a cap was given. With an explicit `--window`, the
honest statement is that the training context cannot be verified without
`--gguf`, so the cap cannot be safely clamped.

**3. The `codegemma` criterion below was never true on this hardware.** Measured:
codegemma weighs 8.454 GiB, free VRAM is 14571 MiB, so spare is 5.525 GiB and
buys 12931 tokens at 448 KiB each — comfortably above its 8192 training
context, which therefore binds. The example was miscalibrated: 8192 × 448 KiB
is only 3.5 GiB, which fits beside 8.45 GiB of weights on a 16 GB card. It is
replaced below with the case that does exercise a `vram`-bound result, taken
from a real run rather than constructed.

---

## Done when

- `pytest -q` green.
- `--window auto` is the default and prints what bound it.
- granite resolves to exactly 4096, attributed to `training_ctx`; codegemma to
  exactly 8192, also `training_ctx` (its 448 KiB/token still fits beside 8.45
  GiB of weights on a 16 GB card — see amendment 2).
- A model whose weights leave no room — the 14.2 GB one on this box — resolves
  to `window 0` attributed to `vram`, **and refuses before turn 1 with the
  arithmetic printed**.
- A window request above the training context is clamped, never granted.
- An impossible scope refuses with the arithmetic printed, before any
  generation.

---

## Whole-branch review fix wave (ruled 2026-08-09)

The whole-branch review found six defects that five passing per-task reviews
could not see, exactly as `CARRIED-DEBT.md` lesson 2 predicts. Each is stated
as the invariant to restore.

**1. `GGUFError` escapes the CLI contract (Critical).** `gguf.py` defines it
independent of `GeometryError`, `detect.py` does not wrap it, and `cli.py`
catches only `(GeometryError, OSError)`. Grep confirms no production caller
catches it. A truncated `--gguf` — the interrupted-download shape
`tests/test_gguf.py` itself names — escapes as a traceback and exit 1, outside
the five contract codes.

*Invariant:* every failure to determine geometry reaches the caller as
`GeometryError`, whatever its source. Wrap at `detect.py`, the boundary that
promises geometry, rather than coupling `gguf.py` to `geometry.py` — the GGUF
reader is deliberately standalone. Add a CLI test on a corrupt `--gguf`
asserting a contract exit code.

**2. Three raw exceptions escape `detect.py` (Critical).** All reproduced:
`{"model_info": null}` → `AttributeError`, because `.get(k, {})` defaults only
when the key is *absent*, never when its value is `null` — the same shape as
the `.get("size", 0)` bug one level up. `{"models": null}` → `TypeError`.
`size: Infinity` → `OverflowError`, which `geometry.py:40` catches *with a
comment explaining why* while its sibling does not.

*Invariant:* a malformed daemon response raises `GeometryError`. Validate the
envelope's shape — `/api/show` a `dict`, `/api/tags`'s `models` a `list` — and
catch `OverflowError` alongside `TypeError`/`ValueError`. Prefer one shared
converter over three call sites drifting again.

**3. `Budget.history = 200` under-reserves by 12x (Important).** Measured
against `loop.py`'s `_READ_CAP = 4000` chars and two turns kept: two capped
`read` results render to 2449 tokens. `fit()` therefore *accepts prompts that
overflow* — at a comfortable window 4096 it returns rung 1 and the real prompt
exceeds the window by 1095 tokens. That is the says-fits-then-truncates
direction.

*Invariant:* no fixed cost is a literal that a second module can contradict.
Derive the default from the same constant `loop.py` caps reads with, times the
turns it keeps, and round toward over-reserving — over-reserving costs scope,
under-reserving costs the run. The exact figure belongs to the caller that
knows its real turns; the *default* must be safe, not optimistic.

**4. The refusal's arithmetic does not reconcile (Important).** `history` is
subtracted from `scope_budget` but never printed, so the printed terms do not
produce the printed total: `window 800 / system 60 / reserve 128 /
diagnostic 60` shown against `available for scope 352`, where those terms give
552. The law is that a refusal prints *the arithmetic*; a number the user
cannot reproduce is not that. The existing test cannot see it — it passes
`history=0` and only asserts three substrings exist.

*Invariant:* every term subtracted appears in the message, and the printed
terms reconcile to the printed total. Test with a non-zero `history`.

**5. Invariant 2's own test cannot fail (Important).** `tests/test_budget.py`
asserts `_cost(scope, root) == estimate_tokens(_scope_section(scope, root))`,
but `_cost` *is* that expression — both sides are one call and `render` is
never invoked. Proven by giving `render` a private copy of `_scope_section`
differing only in the header delimiter: the whole suite stayed green. The
delegation is real in shipped code, so this is a test defect, but it is the
only thing holding the invariant the amendment was written for.

*Invariant:* the test must relate the costed text to **`render`'s actual
output** — assert the section appears verbatim inside `render(...)`. A test
whose two sides are the same expression pins nothing.

**6. `--kv-bits 8` doubles the window and is never sent to any server
(Important).** Measured: codegemma with 896 MiB spare returns 2048 at 16 bits
and 4096 at 8, but the server still allocates f16, so that window needs
1792 MiB against 896 MiB — a 2.0x overcommit, OOM at load. Neither payload
carries a cache-type field, and neither backend accepts one per request:
Ollama takes `OLLAMA_KV_CACHE_TYPE` at server start, llama.cpp
`--cache-type-k` at launch.

*Ruling:* keep the flag, because robigo cannot set the server's KV type and
should not pretend to — but it **describes** the server's existing
configuration rather than requesting one. Say so in the flag's help, and say
that a value the server does not match overcommits VRAM by that ratio. A flag
whose help implies it changes the server is the defect; the arithmetic is
correct given a truthfully-declared server.

**7. Small ones in the same wave.** `degrade` raises for `step >= 5` but
silently returns `self` for `0` and negatives — the reason given for the upper
bound applies below the ladder too. And four added tests are not honest: a
`pytest.raises(Exception)` broad enough to pass if the constructor itself
raises; an `importorskip` on a module already imported at the top of the file,
so it can never fire; a test that stubs `plan_window` but not `build_client`
and therefore **POSTs to a real daemon**, violating the offline-tests
constraint outright; and `estimate_tokens("x" * 36) == 11` under a name
claiming it tests conservatism for code.
