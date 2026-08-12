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


def select_best_codec(codecs: dict[str, CodecResult]) -> str | None:
    """The codec stage 4's repair loop should be configured around, or
    `None` if none of them ever landed a single edit -- `max()` alone (the
    pre-fix implementation, CARRIED-DEBT.md's carried item from plan 03)
    names a codec even when EVERY codec landed 0%, which is not "best", it
    is "none of these ever landed a single edit". Measured live:
    granite-code:8b returned exactly that, a 0%-landing codec quoted as
    the family's best.

    The floor is `> 0.0`, not `_LANDING_MIN` (0.5): that constant answers
    a different question (`verdict_for`'s "is this family READY", which
    already gates on it independently, before this function is ever
    called) -- a codec that lands 20% of the time is real, useful signal
    to configure a repair loop around, and this function's only job is to
    refuse a codec that never landed at all, the exact case that was
    measured wrong.

    Module-level, not a method reimplemented wherever a caller needs "the
    best codec": `Profile.best_codec` (below) and `robigo.profile.report.
    run_profile`'s stage 4 gate (plan 05 task 7) both need this exact
    answer, and `run_profile` has `codecs` in hand well before it has
    everything else a `Profile` requires (family, model, verdict, and
    every other field are not yet known at that point in the function) --
    constructing a throwaway `Profile` there just to call `.best_codec()`
    on it would still be a SECOND definition of "what counts as best"
    wearing a different shape, not a shared one. This project's own
    `docs/CARRIED-DEBT.md` already names exactly this pattern -- multiple
    copies of one list or one rule, free to drift apart -- as a recurring
    defect class (see e.g. `robigo.profile.verify.suite_state`'s
    docstring, which cites the identical shape for a different pair of
    modules: "five path resolvers with one guarded ValueError, three
    copies of a codec list"). `Profile.best_codec` is now a one-line
    delegation to this function, so there is exactly one place this logic
    lives, not two that happen to agree today."""
    if not codecs:
        return None
    name, result = max(codecs.items(), key=lambda item: item[1].lands)
    return name if result.lands > 0.0 else None


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
    repair_rate: float | None
    """Stage 4's `Stage4.rate` (`robigo.profile.repair.stage4_repair`),
    carried through unchanged: passes / scored attempts across every
    (record, seed) pair stage 4 judged (spec 4.5). `None` means NOT
    MEASURED -- stage 4 never ran, because `select_best_codec` found no
    codec worth configuring a repair loop around, or because no `--repo`,
    no corpus records, or no corpus baseline were given to run one
    against (`robigo.profile.report.run_profile`'s stage 4 gate names
    which one, in `dropped`). `0.0` means measured, and every single
    attempt failed. These are different facts, and the project's kill
    criterion (spec 0.2/1.4: below 33.3%, robigo ships as a benchmark repo
    rather than a tool) reads this field, unqualified -- conflating
    "never measured" with "measured and it was zero" would let an
    UNMEASURED family read as though it had already failed the gate, a
    claim this profile is not entitled to make. A family whose
    `repair_rate` is `None` has NOT passed the gate; it has also not
    failed it. It has not been measured."""
    repair_attempts: int
    """Stage 4's `Stage4.attempts` -- the number of (record, seed) pairs
    that actually got scored (spec 4.3.4: an excluded attempt, which never
    gave the model a fair chance, counts toward neither this nor
    `repair_records`). Deliberately a plain `int`, never `int | None`:
    unlike `repair_rate`, no case needs this field alone to carry the
    never-measured/measured-zero distinction -- `repair_rate` is `None`
    exactly when `repair_attempts == 0` (`Stage4.rate` is `(passes /
    scored) if scored else None`), so a reader who needs "was this
    measured at all" already has an unambiguous field to check; a second
    Optional here would only be a second way to ask the same question,
    free to drift from the first."""
    repair_records: int
    """Stage 4's `Stage4.records` -- the number of DISTINCT corpus records
    with at least one scored attempt, as opposed to `repair_attempts`
    (every scored (record, seed) pair, summed). A family measured at 10
    seeds across 94 records with nothing excluded reports
    `repair_attempts=940` and `repair_records=94`; `render_table` prints
    both together ("31% of 940 attempts over 94 records") because a bare
    rate alone cannot tell a reader whether it was measured against a
    handful of records or the whole corpus."""
    repair_per_record: dict[str, tuple[int, int]] | None
    """Stage 4's `Stage4.per_record` -- `record name -> (passes, scored)`
    for every record with at least one scored attempt -- carried through
    unchanged, because this profile is the ONLY artifact that survives
    the ~12h run and the record-level 95% confidence interval (spec 4.5,
    spec 6.1; plan 05 Task 11 step 2) is computed FROM this breakdown:
    seeds within one record share the same defect, file, and starting
    tree, so an attempt-level interval over `repair_attempts` would claim
    roughly sqrt(seeds) more precision than the design has. Discovered
    missing by the 2026-08-12 Task 10 dry run: `Stage4.per_record`
    existed in memory with exactly this rationale in its docstring, but
    was dropped at serialization -- after a real Task 11 run there would
    have been no way to compute the interval the gate's write-up
    requires. `None` means stage 4 never ran (same NOT-MEASURED rule as
    `repair_rate`); `{}` means it ran and every attempt was excluded
    before scoring. JSON stores the pairs as 2-element lists;
    `from_json` restores tuples so a round-tripped `Profile` compares
    equal to the one that was written."""
    turns_to_green_median: float | None
    """Stage 5's `Stage5.turns_to_green_median` (`robigo.profile.
    discipline.stage5_discipline`), carried through unchanged -- the
    median turn count among SCORED attempts that actually passed. `None`
    both when stage 5 did not run at all and when it ran but no scored
    attempt ever passed: there is no "turns to green" to speak of when
    nothing went green, and `0.0` there would read as "reaches green in
    zero turns", the opposite of the truth (`Stage5`'s own docstring,
    invariant 7.2). This project has already shipped the identical
    None-vs-zero collapse once, for a different field
    (`CodecResult.max_file_tokens`) -- this field is typed `float | None`
    specifically so it cannot repeat that mistake."""
    verdict: str
    seeds: int
    mode: str
    corpus: str
    python: str
    """The interpreter stage 4 ran the loop and judge under
    (`robigo.profile.repair.stage4_repair`'s own `python` parameter,
    default `sys.executable`), recorded here for the identical reason
    `seeds`/`mode`/`corpus` already are (fix round 2, 2026-08-10 review):
    a `Profile` is the artifact the project's kill criterion is read from,
    and `--python` is a knob that can change OR VOID `repair_rate` --
    `InterpreterMismatchError` refuses outright on a real mismatch, but a
    `--python` that merely resolves DIFFERENT test collection than
    whatever measured the corpus (skips, version-dependent behaviour)
    while still agreeing on `executed` would not trip that guard, and a
    reader comparing this profile's `repair_rate` against another
    family's has no way to know it unless the interpreter that produced
    it travels with the number. Always a real, concrete string -- never
    `None` -- because `report.run_profile`'s own `python` parameter
    always resolves to one (`sys.executable` by default) whether or not
    stage 4 actually ran; recorded unconditionally, exactly as `mode`
    is recorded even for a family that never reached stage 2."""
    dropped: tuple[str, ...]

    def best_codec(self) -> str | None:
        """Delegates to the module-level `select_best_codec` -- see that
        function's own docstring both for the "why `> 0.0`, not
        `_LANDING_MIN`" reasoning and for why this is now a SHARED
        definition rather than one `Profile` re-derives on its own. Kept
        as a method, rather than removed in favour of every caller going
        straight to the module function, because every existing caller of
        `Profile.best_codec()` -- this module's own tests included --
        already reads it off a constructed `Profile`, and there is no
        reason to force a second migration onto call sites that were
        never the problem."""
        return select_best_codec(self.codecs)

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
                "repair_rate": self.repair_rate,
                "repair_attempts": self.repair_attempts,
                "repair_records": self.repair_records,
                "repair_per_record": (
                    None if self.repair_per_record is None
                    else {name: list(pair)
                          for name, pair in self.repair_per_record.items()}
                ),
                "turns_to_green_median": self.turns_to_green_median,
                "verdict": self.verdict,
                "measured": {"seeds": self.seeds, "mode": self.mode,
                             "corpus": self.corpus, "python": self.python},
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
            repeat_rate=payload["repeat_rate"],
            repair_rate=payload["repair_rate"],
            repair_attempts=payload["repair_attempts"],
            repair_records=payload["repair_records"],
            repair_per_record=(
                None if payload["repair_per_record"] is None
                else {name: (pair[0], pair[1])
                      for name, pair in payload["repair_per_record"].items()}
            ),
            turns_to_green_median=payload["turns_to_green_median"],
            verdict=payload["verdict"],
            seeds=measured["seeds"], mode=measured["mode"],
            corpus=measured["corpus"], python=measured["python"],
            dropped=tuple(payload["dropped"]),
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
