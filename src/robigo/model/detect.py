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


def _tags(host: str) -> dict:
    """`GET /api/tags`: the only Ollama endpoint that carries `size`, and the
    only one this module calls that is guaranteed not to need `model` at all
    -- confirmed 2026-08-09 that `/api/show` itself does not load the model
    either (free VRAM measured unchanged before/after), but `weights_bytes`
    still uses this endpoint rather than that fact, per the amendment."""
    req = urllib.request.Request(f"{(host or OLLAMA_HOST).rstrip('/')}/api/tags")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _no_gguf_message(what: str, user_cap: int | None) -> str:
    """Wording depends on whether a cap was already given. With no cap,
    `--window <int>` is a real alternative and worth naming. With one
    already supplied, naming it back is the exact defect this function
    exists to avoid: it once told a user who ran `--window 4096` with no
    `--gguf` to pass `--window <int>`, recommending what they had just
    supplied -- the same family as plan 01 shipping messages that named
    flags which did not exist. Kept generic across both callers
    (`detect_geometry`'s "KV geometry" and `weights_bytes`'s "weights
    size") rather than naming "training context" specifically, since that
    phrase is only true for the geometry caller."""
    if user_cap is None:
        return (
            f"llama.cpp does not expose {what} over HTTP. Pass --gguf "
            f"<path> so it can be measured, or --window <int>."
        )
    return (
        f"llama.cpp does not expose {what} over HTTP, so it cannot be "
        f"verified and --window {user_cap} cannot be safely honoured "
        f"without it. Pass --gguf <path>."
    )


def detect_geometry(
    backend: str,
    model: str,
    host: str,
    gguf_path: Path | None = None,
    *,
    user_cap: int | None = None,
) -> Geometry:
    """Ollama publishes the geometry over HTTP; llama-server does not, so
    on that path the GGUF file is the only source of truth.

    `user_cap` affects only the wording of a raised GeometryError, never the
    logic: it lets the message say the honest thing when a --window value
    was already supplied (see `_no_gguf_message`).
    """
    if backend == "ollama":
        return from_model_info(_show(model, host).get("model_info", {}))
    if gguf_path is None:
        raise GeometryError(_no_gguf_message("KV geometry", user_cap))
    return from_model_info(read_metadata(gguf_path))


def weights_bytes(
    backend: str,
    model: str,
    host: str,
    gguf_path: Path | None,
    *,
    user_cap: int | None = None,
) -> int:
    """A real measured size, or a raise -- never a default.

    `POST /api/show` does not return a `size` field at all (verified
    against the live daemon before this was written: its top-level keys are
    capabilities, details, license, model_info, modelfile, modified_at,
    system, template, tensors). A `.get("size", 0)` default there silently
    reports a 0-byte model, and `usable_window` then believes the whole card
    is free and hands back the largest window in the table for a model that
    may not fit at all -- the exact bug this function exists to not have.

    `GET /api/tags` does carry `size`, per model, and needs no model load.
    12 of 30 names on the reference box end in `:latest`, so a bare model
    argument may need that suffix appended to match: the exact name is
    tried first, then `f"{model}:latest"`. If neither matches, this raises
    GeometryError naming the model and listing what the daemon does know,
    rather than returning a number.

    `user_cap` only changes the wording of the no-`--gguf` message, same as
    `detect_geometry` (`_no_gguf_message`) -- in the CLI's actual call
    order `plan_window` calls `detect_geometry` first, so that message is
    the one a user sees; this one is reached by any direct caller instead.
    """
    if backend != "ollama":
        if gguf_path is None:
            raise GeometryError(_no_gguf_message("weights size", user_cap))
        return gguf_path.stat().st_size
    models = _tags(host).get("models", [])
    by_name = {
        entry.get("name"): entry for entry in models if isinstance(entry, dict)
    }
    entry = by_name.get(model) or by_name.get(f"{model}:latest")
    if entry is None or "size" not in entry:
        known = ", ".join(sorted(n for n in by_name if isinstance(n, str)))
        raise GeometryError(
            f"{model!r} is not a model /api/tags knows about (tried "
            f"{model!r} and {model + ':latest'!r}); the daemon knows: "
            f"{known or '(none)'}. Weights size cannot be measured, so the "
            f"usable window is unknown. Pass --window explicitly."
        )
    try:
        return int(entry["size"])
    except (TypeError, ValueError) as exc:
        raise GeometryError(
            f"{model!r}'s /api/tags size is malformed ({entry['size']!r}); "
            f"the weights size cannot be measured, so the usable window is "
            f"unknown. Pass --window explicitly."
        ) from exc


def plan_window(
    backend: str,
    model: str,
    host: str,
    user_cap: int | None,
    *,
    kv_bits: int = 16,
    gguf_path: Path | None = None,
) -> WindowPlan:
    """Free VRAM is read FIRST, before `detect_geometry`'s `/api/show` or
    `weights_bytes`'s `/api/tags`. `usable_window`'s own precondition is
    that `free_vram` is measured before the model is loaded; reading it
    before any network call at all makes that hold regardless of what those
    two read-only endpoints do internally (confirmed neither one loads the
    model, by measuring `nvidia-smi` unchanged across a real `/api/show`
    call), rather than relying on that fact staying true.
    """
    free = free_vram_bytes()
    geometry = detect_geometry(backend, model, host, gguf_path, user_cap=user_cap)
    weights = weights_bytes(backend, model, host, gguf_path, user_cap=user_cap)
    return usable_window(
        geometry,
        free_vram=free,
        weights_bytes=weights,
        kv_bits=kv_bits,
        user_cap=user_cap,
    )
