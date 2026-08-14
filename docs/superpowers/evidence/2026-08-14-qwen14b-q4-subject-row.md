# Second subject: qwen2.5-coder-14B-Q4 — 0/940, and the reason is geometry, not cognition

2026-08-14. A **new subject row** on the frozen `boltons-gate-v1`
corpus under the identical apparatus that measured qwen2.5-coder-7B-Q8
at 1.06% ([the gate write-up](2026-08-12-stage4-gate.md)). The gate
itself was consumed by that first reading; no decision rides on this
row, and the 33.3% line appears here only as the tool-viability
reference.

## The number

**Strict repair: 0.0% — 0 of 940 attempts** (94 records × 10 seeds,
940/940 scored, zero exclusions). Record-level 95% upper bound: 4.0%.
`repeat_rate` 39.1%; `turns_to_green_median` undefined (nothing went
green). Same `--window 8192` request, same seeds, same corpus, same
daemon as the 7B row. Transcript: `docs/transcripts/qwen14b-q4-stage4.jsonl`
(7,253 calls, replayable).

## Why this is a serving-geometry result first

The two rows did not run at the same effective context, and the
transcript pins the mechanism:

- Stage 0 verified a **1,783-token window for the 14B** against
  **4,535 for the 7B** — 39% of the context, same requested cap.
- The first stage-0 probes show why: at the fixed `num_ctx=8192`
  request shape this daemon serves 14B-Q4 prompts of ~1.2k and ~1.7k
  tokens and **errors** (empty, count-free replies) on everything
  larger. Not truncation — refusal.
- The same daemon serves the same 14B blob to **16,384 tokens clean**
  when `num_ctx` is right-sized per request — measured independently
  the same day by assay's ceiling probe, which widens `num_ctx` per
  call. The failure is the *request shape*, not the model or the
  hardware ceiling.
- Stage 0 therefore did exactly its job: it refused to ship the
  requested 8192 and verified the 1,783 that actually worked — the
  window law ("computed, never read") converting a daemon quirk into a
  stated instrument condition instead of a silent one.

At 1,783 tokens the degradation ladder sits at its lowest rungs almost
everywhere: signatures-only context, anchor slices, no room for
hop-context. The model's loop discipline was fine — envelope fidelity
1.0, `whole_file` codec landing 97.3% on stage-2 probes — **it simply
could not see the code it was asked to repair.** The higher repeat
rate (39.1% vs 29.8%) is what looping on an unreadable task looks
like.

## What this answers, and what it does not

This is the **fixed-VRAM comparison the spec pre-registered** ("7B-Q8
vs 14B-Q4 as different subjects, never '14B vs 7B'"), and at fixed
VRAM the answer is sharp: **the 14B's parameter count is paid for in
KV geometry (192 vs 56 KiB/token) and serving fragility, and on this
stack that tax zeroes the context an agentic repair loop needs.**
Doubling parameters at half precision bought nothing here — not
because the model reasons worse, but because the loop never got to ask
it a fair question.

What this row does **not** answer is the capability-window question
(whether 14B-class cognition clears what 7B cognition fell below):
that requires the 14B at the 7B's effective window. The daemon can
serve it (assay proved 16k works right-sized), so a third run — 14B
with a window request this daemon honors, e.g. `--window 4096` or a
right-sized `num_ctx` path — would isolate cognition from geometry.
~5h GPU; queued for the owner's call, not run unilaterally.

## Provenance

Fresh boltons clone at `580a9c2d` (branch `gate`); corpus
`boltons-gate-v1` (the known name-collision means `repair_records`
reads 93-for-94, both fused records passless, rate unaffected — see
CARRIED-DEBT plan 05); interpreter = robigo's venv for loop and judge;
run detached 17:55–01:25 (7h30m), wrapper
`~/workspace/robigo-14b-run.sh`, exit 0. Profile preserved beside this
document as
[`2026-08-14-qwen14b-q4-profile.json`](2026-08-14-qwen14b-q4-profile.json).
