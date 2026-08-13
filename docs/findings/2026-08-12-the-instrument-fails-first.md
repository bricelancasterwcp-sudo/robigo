# The instrument fails before the model does — and then the model fails too

*robigo findings — 2026-08-12. Companion pieces:
[black-oxide's findings series](https://github.com/bricelancasterwcp-sudo/black-oxide)
(language-side) and [assay](https://github.com/bricelancasterwcp-sudo/assay)
(the probe instrument these findings produced). All numbers are
reproducible from committed artifacts in this repository and assay's;
sources are linked inline.*

robigo's thesis was that generic harnesses fail small local models for
**instrument** reasons — window, envelope, codec — before capability is
ever reached, and that a harness co-designed with the VRAM budget would
make 7–14B models useful for single-defect repair. The project
pre-registered a kill criterion before building the measurement: below
33.3% strict repair on a frozen corpus, robigo ships as a
benchmark-and-findings repository, not a tool.

Both halves of the thesis got answered in the same week. The instrument
claims replicated, transferred, and produced a reusable prober. The
capability claim died at **1.06%**. This document is the systems-side
findings, in the order they earn their keep.

## 1. A context window is geometry, not configuration

The KV cache costs `2 × layers × kv_heads × head_dim × bytes` per token,
and across real GGUF blobs on one machine that spans **14×**:

| family (example) | layers × kv_heads × head_dim | KV per token |
|---|---|---|
| qwen2.5-coder-7b | 28 × 4 × 128 | **56 KiB** |
| granite-code-8b | 36 × 8 × 128 | **144 KiB** |
| codegemma-7b | 28 × 16 × 192 | **336 KiB** |
| a llama with no GQA | 40 × 40 × 128 | **800 KiB** |

The practical consequence is routinely backwards from intuition:
codegemma's 8k training window costs more VRAM to serve than qwen's full
32k. So robigo's law is **the window is computed, never read**:
`usable = min(training_ctx, (free_vram − weights − overhead)/kv_per_token,
user_cap)`, always reporting *which term bound it*. The first two rows
above were measured independently twice — robigo's GGUF reader
([README](../../README.md)) and assay's `/api/show` arithmetic
([live validation](https://github.com/bricelancasterwcp-sudo/assay/blob/master/docs/superpowers/evidence/2026-08-12-live-validation.md))
— and agree exactly.

Margins at the edge are not academic. The repository's standing
demonstration is a real bug repaired by a 7B **in an 1100-token window
whose worst turn used 1097 tokens** — three to spare. A harness that
estimates instead of measures does not get to have that margin: token
estimates are non-additive and under-count, which is fatal precisely at
the edge where small-VRAM operation lives.

## 2. The serving path lies, and it lies *transiently*

On 2026-08-10, this machine's Ollama daemon failed **40/40** attempts
above ~11.8k prompt tokens in a specific, nasty way: HTTP 200, plausible
content, and no `prompt_eval_count`/`eval_count`/`done_reason` — a
protocol-breaking reply indistinguishable from success unless you check
the stats. Prompts at 11.5k succeeded reliably. The workaround (cap the
window at 8192) shaped every measurement after it, including the gate
run.

On 2026-08-12, assay's ceiling probe could not reproduce it — and a
manual reproduction under the original conditions (15,792-token prompt,
`num_ctx: 32768`) came back with **intact stats** from the **same daemon
process**, unrestarted since 2026-07-27, same version. The bug is
state-dependent: not the binary, not the config, but something in the
serving state's history. Two conclusions, both uncomfortable:

- **An environmental bug is not a stable target.** assay's success
  criterion pinned "detect the ~11.5k ceiling"; the honest evaluation
  had to record that the criterion named an address, not a behavior
  class, and the address had moved.
- **Capability profiles are point-in-time measurements of a serving
  state.** This is the strongest argument for probing at configuration
  time (what assay is for) rather than assuming a model's serving
  envelope from its card. A related observation from the same probe:
  at 15.8k the protocol held but the *output* was degenerate repetition
  that ignored its instruction — quality dies before protocol does,
  and only a canary check sees it.

## 3. "Does an edit land" is a property of the instrument

Two probes measured qwen2.5-coder-7b's search/replace landing on the
same daemon: robigo's stage 2 measured **100%**; assay's v1 measured
**0/15** — with every failing reply a *semantically correct fix*. The
difference decomposed, after falsifying two hypotheses live
(temperature; fencing), into two instrument axes:

- **Presentation.** robigo's probe presents the loop's full action
  envelope — and robigo's own recorded history already contained the
  phenomenon: an earlier probe that showed the payload template without
  the envelope scored **0/5**; adding the envelope text scored **5/5**
  on the same daemon and seed ("a stage that omits the shape it is
  trying to predict was never measuring the model"). Under the full
  envelope qwen copies indentation faithfully; under a minimal
  instruction it strips it, 30/30, at temperature 0.8 *and* 0.2.
- **Landing definition.** robigo scores "parses as an action + codec
  applies + result parses as Python"; assay v1 scores byte-equality
  against an expected file. On whole-file replacement, byte-equality
  measures compliance-with-incidentals: the model fixed the defect and
  rewrote a comment, and scored zero.

Neither number is wrong. Both are true **under a named lens** — and the
transferable rule is that a landing rate quoted without its lens
(presentation, applier semantics, sampler, success predicate) is not a
model property. The v1.1 consequence for assay is that verdicts must
name their lens and probes should accept the consumer's own prompt
shape, because that is the only landing rate that predicts the
consumer's reality.

## 4. The number: 1.06%, and what it was made of

The pre-registered gate ran once, in full: 94 frozen boltons mutants ×
10 seeds = 940 attempts, every one scored, zero infrastructure
exclusions ([the gate write-up](../superpowers/evidence/2026-08-12-stage4-gate.md)).
**Strict repair — target test green, whole suite green, anchor test file
byte-identical, within an 8-turn cap — landed at 1.06%**, record-level
95% CI [0%, 3.2%], cluster bootstrap [0%, 3.5%], against the 33.3%
floor. robigo is therefore a benchmark-and-findings repository, per the
protocol, with no extension and no re-run.

The structure of the failure says more than the rate:

- **Success was bimodal, not thin.** All ten passes came from one
  record, which passed 10/10 seeds. Not "3% everywhere" — determinism
  on one easy defect and nothing anywhere else.
- **Even the success was expensive**: median 7 turns of the 8-turn cap.
- **The failure modes were behavioral, not infrastructural**: a 29.8%
  verbatim-repeat rate across turns; attempts spent patching the
  read-only failing test instead of the source; literal `patch <path>`
  placeholder imitation; `done` claims with nothing green (correctly
  scored as failures by the strict judge).

The instrument thesis survived its own null: the apparatus scored
940/940 fairly, separated infrastructure from model failure throughout,
and the one mid-plan defect (the per-record breakdown never serialized)
was caught by the small-N dry run *before* the 12-hour measurement —
which is exactly what dry runs are for.

## 5. What a pre-registered kill criterion is worth

The number came in at **1/31st of the threshold**. At that moment every
temptation the protocol was written against arrived on schedule: try
another corpus, raise the turn cap, widen the window, re-run with a
better prompt. The pre-registration — threshold derived from an
attempts-per-success argument, corrected once *before* any instrument
existed, point-estimate-decides written down in advance — meant the
result shipped the same day instead of becoming a month of motivated
iteration. The null is the artifact: the corpus, the transcripts (8,009
replayable calls), the profiles, and the findings above.

## Honest limits

- One repo (boltons), one mutation-derived corpus, one family measured
  at the gate (the best available by profile); the window was capped at
  8k by the (then-present) daemon ceiling and stage-0's known ~0.55
  probe-density pessimism narrowed it further — both biases run
  *against* the capability thesis and are stated in the gate write-up.
- The landing comparison in §3 is one model on one day; its value is
  the demonstrated instrument-dependence, not the specific rates.
- The KV table's spread depends on which architectures you own; 14× is
  this machine's blobs, not a universal constant.
