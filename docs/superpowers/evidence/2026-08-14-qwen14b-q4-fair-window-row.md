# Third run: 14B-Q4 at a fair window — 0/940, and this time it means it

2026-08-14. The capability-window experiment the second row could not
be: same frozen corpus, same seeds, same `--window 8192` request as the
7B row, under stage-0 filler v2 (see the row-2 erratum). Stage 0
verified **3,777 tokens** against the 7B row's 4,535 — 83%, the same
order of context — and the run executed 940/940 attempts with **zero
daemon errors**.

## The number

**Strict repair: 0.0% — 0 of 940** (record-level 95% upper bound 4.0%).
Even `pathutils-dropped_return-183`, the record 7B-Q8 passed 10/10,
failed all ten seeds. `repeat_rate` 38.9% — statistically identical to
the context-starved run's 39.1%, at double the context.

## What failure looks like with a fair window

The transcript (7,7xx calls, replayable) shows a model that ENGAGES the
task and cannot close it:

- **A third of all turns are bare `run`** (1,711 calls): execute the
  suite, read the failure, execute again. The repeat rate is not
  context starvation — it reproduces exactly at 2× the window.
- **18% of replies are truncated** (934 hit the `num_predict=1024`
  cap). The 14B writes long — patches wrapped in commentary and
  multi-block edits — and robigo's truncation-is-a-veto rule (a capped
  reply is never applied) discards them. Verbosity is self-sabotage
  under an honest cap that the terse 7B rarely hit.
- It patches **real source files across the corpus** (typeutils,
  gcutils, funcutils, iterutils, …) where the 7B mostly ping-ponged
  read/find and patched the test — more parameters bought broader
  engagement — but the patches that survive the cap do not land
  strictly, and anchor violations (patching test files) still occur.

## Reading the three rows together

| subject | window (verified) | strict repair | signature |
|---|---|---|---|
| 7B-Q8 | 4,535 | **1.06%** (one record, 10/10) | terse; narrow engagement; one deterministic success |
| 14B-Q4, starved | 1,783 | 0.0% | probe artifact — see erratum |
| 14B-Q4, fair | 3,777 | **0.0%** | broad engagement; run-loops; verbosity vetoed |

At fixed consumer VRAM, doubling parameters at half precision made this
loop **worse, not better** — while the same model beats the 7B on
assay's single-shot probes (whole_file applies 5/5 vs 2/5). The
capability window for THIS loop did not open at 14B-Q4; what opened is
a new named failure mode: **verbosity × truncation-veto**. Whether it
is Q4 quantization, qwen-14B chat tuning, or the model class is not
separable from these two subjects (the spec's own rule: different
subjects, never "14B vs 7B").

## Honest limits

- Windows are comparable (83%), not identical; `num_predict=1024` and
  every other loop parameter identical across rows.
- One corpus, one repo, one daemon; the three-row table is this box's
  story, not a general law.
- A `num_predict`-raised variant (e.g. 2048) would test the
  verbosity-veto mechanism directly — recorded as the natural fourth
  run, not run unilaterally.

Profile: [`2026-08-14-qwen14b-q4-r3-profile.json`](2026-08-14-qwen14b-q4-r3-profile.json).
Transcript: `docs/transcripts/qwen14b-q4-stage4-r2.jsonl`.
