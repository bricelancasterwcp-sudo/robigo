from __future__ import annotations

import json
from dataclasses import dataclass, replace  # noqa: F401  (replace used by callers)

SUPPORTED_FLOOR = 8192
"""Windows below this are a documented edge case, not a target. A design
that works at 4096 works everywhere, but a 4096 family should never be
recommended for agentic work (spec 3.1)."""

ENVELOPE_FIDELITY_MIN = 0.5
"""Below this, `verdict_for` reports UNUSABLE regardless of window or
codecs (spec 5, stage 1 gates the rest). Public -- not just an internal
threshold of this module's own function -- because `robigo.profile.report.
run_profile` gates stage 2 on this exact same question ("can this family
drive the envelope at all") and must use the identical number, not a
second 0.5 written by hand at the call site. Two independent literals that
happen to agree today is exactly how a threshold drifts silently: a future
change to the number here, without a matching edit at the other site,
would make `run_profile` run stage 2 for a family `verdict_for` still
calls UNUSABLE (or the reverse), and either way `dropped` would then
describe a gate that no longer matches what the profile's own verdict is
built from."""
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
        """The codec plan 05 should configure the loop around, or `None`
        if not one of them ever landed anything -- `max()` alone (the
        pre-fix implementation, CARRIED-DEBT.md's carried item from plan
        03) names a codec even when EVERY codec landed 0%, which is not
        "best", it is "none of these ever landed a single edit". Measured
        live: granite-code:8b returned exactly that, a 0%-landing codec
        quoted as the family's best.

        The floor is `> 0.0`, not `_LANDING_MIN` (0.5): that constant
        answers a different question (`verdict_for`'s "is this family
        READY", which already gates on it independently, before this
        method is ever called) -- a codec that lands 20% of the time is
        real, useful signal for plan 05 to configure around, and this
        method's only job is to refuse a codec that never landed at all,
        the exact case that was measured wrong."""
        if not self.codecs:
            return None
        name, result = max(self.codecs.items(), key=lambda item: item[1].lands)
        return name if result.lands > 0.0 else None

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
    if envelope_fidelity < ENVELOPE_FIDELITY_MIN:
        return "UNUSABLE"
    best = max((r.lands for r in codecs.values()), default=0.0)
    if usable_window < SUPPORTED_FLOOR or best < _LANDING_MIN:
        return "LIMITED"
    return "READY"
