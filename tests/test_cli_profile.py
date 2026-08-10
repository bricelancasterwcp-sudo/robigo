# tests/test_cli_profile.py
"""P2 (2026-08-10 design, docs/superpowers/specs/2026-08-10-robigo-05-repair-gate-design.md
§3 P2): `robigo profile` had no way to cap the window it asks the daemon for.
`plan_window` already accepted a `user_cap` fourth positional argument and
already folded it into `min(training_ctx, vram, user_cap)`
(`src/robigo/model/geometry.py::usable_window`) -- `cli.profile_main` simply
never passed anything but `None` through. On this box that is not a cosmetic
gap: `qwen2.5-coder:7b`, the best-measured family, resolves to its full 32768
training context because VRAM never binds here, stage 0 then probes past this
box's Ollama daemon's measured ~11.5k prompt-token ceiling, and the run dies
before stage 0 finishes -- the best family cannot be profiled at all without
an explicit ceiling. These tests cover the plumbing (the flag reaches
`plan_window` as `user_cap`, and its absence still passes `None` rather than
some other default) and the invariant the flag exists to preserve (a cap
above the training context changes nothing -- it is a ceiling, never a
floor)."""
from __future__ import annotations

import pytest

from robigo import cli
from robigo.model.geometry import Geometry, WindowPlan, usable_window


def test_window_flag_is_passed_to_plan_window_as_the_user_cap(monkeypatch):
    """`--window 4096` must arrive at `plan_window` as its 4th positional
    argument, `user_cap` -- that positional slot, not a keyword, is what
    `plan_window(backend, model, host, user_cap, *, kv_bits, gguf_path)`
    exposes (confirmed against `src/robigo/model/detect.py` before writing
    this test; the brief's assertion about the 4th positional was correct).
    `run_profile` is stubbed to raise rather than return, so this test never
    depends on stage 0-2 actually running -- it only proves the cap reaches
    `plan_window`, nothing downstream of it."""
    seen = {}

    def fake_plan_window(backend, model, host, user_cap, *, kv_bits=16, gguf_path=None):
        seen["user_cap"] = user_cap
        return WindowPlan(window=4096, limited_by="user_cap", free_vram=None,
                          kv_per_token=57344, weights_bytes=0, overhead_bytes=0,
                          training_ctx=32768)

    monkeypatch.setattr(cli, "plan_window", fake_plan_window)
    monkeypatch.setattr(cli, "run_profile", lambda *a, **k: pytest.skip("not reached"))
    with pytest.raises(BaseException):
        cli.profile_main(["--model", "m", "--window", "4096"])
    assert seen["user_cap"] == 4096


def test_no_window_flag_still_passes_none(monkeypatch):
    """Absent `--window`, the default must stay `None` -- not `0`, not some
    other sentinel that `usable_window` would treat as a real cap. `0` in
    particular would look like a legitimate (if useless) user cap rather
    than "no cap given", so this is not a redundant restatement of the test
    above; it pins the default's identity, not just its truthiness."""
    seen = {}

    def fake_plan_window(backend, model, host, user_cap, *, kv_bits=16, gguf_path=None):
        seen["user_cap"] = user_cap
        raise SystemExit(99)

    monkeypatch.setattr(cli, "plan_window", fake_plan_window)
    with pytest.raises(SystemExit):
        cli.profile_main(["--model", "m"])
    assert seen["user_cap"] is None


def test_window_above_training_ctx_does_not_raise_the_window():
    """P2.1, proved against the real `usable_window`, not a stub: a cap
    above the training context must change nothing, because `usable_window`
    takes `min(training_ctx, vram, user_cap)` and `--window` only ever adds
    a fourth term to that `min`, never replaces it.

    The brief transcribed this call from memory as
    `Geometry(layers=28, kv_heads=4, head_dim=128, training_ctx=4096)` and
    `usable_window(g, free_vram=None, user_cap=999_999, kv_bits=16)`. Both
    are wrong against the real source (`src/robigo/model/geometry.py`):
    `Geometry` is `(arch, layers, kv_heads, key_dim, value_dim,
    training_ctx)` -- there is no `head_dim` field, and `key_dim`/`value_dim`
    are tracked separately because they differ on some architectures
    (`Geometry.kv_bytes_per_token`'s own docstring). `arch` has no default,
    so it must be supplied. And `usable_window`'s `weights_bytes` is a
    required keyword-only parameter with no default at all -- omitting it,
    as the brief's call does, raises `TypeError` before the invariant is
    even exercised. `weights_bytes=0` here is deliberate: with `free_vram`
    also `None`, the vram limit never enters the `min()` (see
    `usable_window`'s `if free_vram is not None` guard), so the zero is
    inert and only the training_ctx/user_cap comparison this test cares
    about is live.
    """
    g = Geometry(arch="test", layers=28, kv_heads=4, key_dim=128, value_dim=128,
                 training_ctx=4096)
    capped = usable_window(g, free_vram=None, weights_bytes=0, user_cap=999_999,
                           kv_bits=16)
    uncapped = usable_window(g, free_vram=None, weights_bytes=0, user_cap=None,
                             kv_bits=16)
    assert capped.window == uncapped.window
    assert capped.limited_by == "training_ctx"
