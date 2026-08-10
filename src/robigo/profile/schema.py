from __future__ import annotations

import json
from dataclasses import dataclass, replace  # noqa: F401  (replace used by callers)

SUPPORTED_FLOOR = 8192
"""Windows below this are a documented edge case, not a target. A design
that works at 4096 works everywhere, but a 4096 family should never be
recommended for agentic work (spec 3.1)."""

_ENVELOPE_MIN = 0.5
_LANDING_MIN = 0.5


@dataclass(frozen=True)
class CodecResult:
    lands: float
    attempts: int
    max_file_tokens: int | None


@dataclass(frozen=True)
class Profile:
    family: str
    model: str
    quant: str
    training_ctx: int
    kv_kib_per_token: int
    kv_bits: int
    usable_window: int
    window_limited_by: str
    envelope_level: int
    envelope_fidelity: float
    codecs: dict[str, CodecResult]
    payload_corruption: float | None
    repeat_rate: float | None
    verdict: str
    seeds: int
    mode: str
    corpus: str
    dropped: tuple[str, ...]

    def best_codec(self) -> str | None:
        if not self.codecs:
            return None
        return max(self.codecs, key=lambda name: self.codecs[name].lands)

    def to_json(self) -> str:
        return json.dumps(
            {
                "family": self.family, "model": self.model, "quant": self.quant,
                "training_ctx": self.training_ctx,
                "kv_kib_per_token": self.kv_kib_per_token,
                "kv_bits": self.kv_bits,
                "usable_window": self.usable_window,
                "window_limited_by": self.window_limited_by,
                "envelope_level": self.envelope_level,
                "envelope_fidelity": self.envelope_fidelity,
                "codecs": {
                    name: {"lands": r.lands, "attempts": r.attempts,
                           "max_file_tokens": r.max_file_tokens}
                    for name, r in self.codecs.items()
                },
                "payload_corruption": self.payload_corruption,
                "repeat_rate": self.repeat_rate,
                "verdict": self.verdict,
                "measured": {"seeds": self.seeds, "mode": self.mode,
                             "corpus": self.corpus},
                "dropped": list(self.dropped),
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: dict) -> Profile:
        measured = payload["measured"]
        return cls(
            family=payload["family"], model=payload["model"],
            quant=payload["quant"], training_ctx=payload["training_ctx"],
            kv_kib_per_token=payload["kv_kib_per_token"],
            kv_bits=payload["kv_bits"],
            usable_window=payload["usable_window"],
            window_limited_by=payload["window_limited_by"],
            envelope_level=payload["envelope_level"],
            envelope_fidelity=payload["envelope_fidelity"],
            codecs={
                name: CodecResult(r["lands"], r["attempts"], r["max_file_tokens"])
                for name, r in payload["codecs"].items()
            },
            payload_corruption=payload["payload_corruption"],
            repeat_rate=payload["repeat_rate"], verdict=payload["verdict"],
            seeds=measured["seeds"], mode=measured["mode"],
            corpus=measured["corpus"], dropped=tuple(payload["dropped"]),
        )


def verdict_for(
    envelope_fidelity: float, codecs: dict[str, CodecResult], usable_window: int
) -> str:
    if envelope_fidelity < _ENVELOPE_MIN:
        return "UNUSABLE"
    best = max((r.lands for r in codecs.values()), default=0.0)
    if usable_window < SUPPORTED_FLOOR or best < _LANDING_MIN:
        return "LIMITED"
    return "READY"
