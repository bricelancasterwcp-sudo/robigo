# robigo — design

A coding agent for local models, designed against a VRAM budget rather than
adapted to one.

Status: design agreed 2026-08-09. Not yet implemented.

---

## 0. What this is

`robigo` runs an edit→test loop against a locally-served model on a single
consumer GPU, and it profiles that model first so the harness can configure
itself to what the model measurably does rather than what a spec sheet
claims.

The pitch is deliberately narrow: **your local model is more capable than a
generic harness lets it be — here is how much, per family, and here is where
the wall actually is.**

Latin *robigo*: the rust on iron. Robigalia was the Roman festival invoked to
avert rust and blight. Sibling to the `black-oxide` language project, from
which most of the discipline below is inherited.

### 0.1 The gap this fills

A survey of the field (2026-08-08) found no coding agent that treats a VRAM
budget as a design constraint or ships measurement of its own effect:

| project | ★ | relevance |
|---|---|---|
| `SWE-agent/mini-swe-agent` | 6.3k | closest shape — one bash tool, radically simple. Small-model-friendly by accident |
| `Aider-AI/aider` | 48k | the only project measuring **edit formats**; independently corroborates the ergonomics finding below |
| `Nano-Collective/nanocoder` | 2.3k | "local-first, bring your own model" — zero measurement, `llama3.1` as its example |
| `cline`, `Roo-Code`, `OpenHands`, `goose` | 24k–83k | system prompt + tool schemas alone exceed a small model's usable window |

Searches for VRAM-budgeted agents returned a 4GB hobby repo and two 0★ field
reports. The niche is empty, and the credible differentiator is an apparatus,
not a feature count.

### 0.2 The thesis, and the kill criterion

The thesis is that a harness co-designed with a small-window model budget
makes 9–14B models useful for single-defect repair, and that generic
harnesses fail them for **instrument** reasons before capability is ever
reached.

The counter-evidence is strong and comes from this project's own sibling: a
7B writing Oxide from a card scores **2/20 against 20/20 for Rust**, and that
gap was concluded to be pretraining exposure, not language design. Multi-turn
tool use is out-of-distribution in the same way.

**Decision, taken while neutral:** if quick-profile stage 4 (§5) lands below
**40% strict** on the best available family, `robigo` becomes a
benchmark-and-findings repo and the agent is not shipped as a usable tool.
Rationale: below 40%, a fix needs more than three attempts on average, which
is not a tool anyone should be handed. *This number is the one figure in this
document chosen by the author rather than derived; it wants a sanity check
before implementation starts.*

A null result here is publishable and would be the most valuable artifact in
the niche. It is not, however, the product — and the point of fixing the
threshold now is that after three weeks of work nobody wants to reach it.

---

## 1. Architecture

Python, stdlib only. Files kept in the 200–400 line band.

```
src/
├── cli.py                argv, config, dispatch
├── loop.py               the turn loop — the only stateful module
├── action/
│   ├── verbs.py          read · find · patch · run · done  (+ parse)
│   ├── codec.py          search_replace | whole_file | udiff
│   └── envelope.py       header GBNF, stop sequences, two-step split
├── context/
│   ├── scope.py          test-anchored resolution: anchor + N hops
│   ├── budget.py         token accounting, output reserve, degradation
│   └── render.py         prompt assembly
├── adapters/
│   ├── base.py           run(scope) · diagnose(out) · imports(file)
│   ├── python_.py        pytest + ast
│   ├── typescript.py     vitest + regex imports
│   └── generic.py        user test command, explicit scope
├── model/
│   ├── client.py         llama.cpp + Ollama, OpenAI-compatible
│   ├── families.py       detection → geometry, codec default, quirks
│   ├── geometry.py       GGUF metadata → KiB/token, real window ceiling
│   └── tokens.py         per-family counting
├── apply/
│   ├── patch.py          apply · atomic write · verify
│   └── safety.py         git branch, snapshot, refusal rules
├── memory/
│   ├── store.py          markdown vault read/write
│   ├── keys.py           structural retrieval keys
│   └── decay.py          hit/miss accounting, eviction
└── profile/
    ├── probe.py          corpus runner
    ├── corpus.py         mutation-based corpus generation
    └── report.py         → ~/.config/robigo/profiles/<family>.json
```

Everything except `loop.py` is a pure function over fixtures, testable
without a model in the picture: codecs, scope resolution, budget arithmetic,
geometry, memory keys, adapters.

### 1.1 One invocation

1. `cli` resolves provider + model → `families.detect` → load profile, or run
   with conservative defaults **and say so**
2. `adapters` runs the tests → `diagnose` → structured diagnostic anchored at
   `file:line`
3. `scope` takes that file as anchor, walks imports 1–2 hops; hop 2 arrives as
   signatures only
4. `budget` computes the window from measured geometry (§3), then fills:
   system → scope → diagnostic → last-K turns
5. `loop` renders, calls the model (header-constrained, or two-step per
   profile), parses exactly one action
6. `apply` writes atomically inside a git branch; a parse failure returns a
   loud diagnostic and costs a turn
7. re-run tests → new diagnostic, or PASS
8. terminate on PASS · turn cap · budget exhaustion · refusal

### 1.2 Invariants

- **The window is never overflowed.** Running out is a *session result*, not a
  crash.
- **Exactly one action per turn.** No batching, no parallel tool calls.
- **The model never sees the repo, only the scope.** Repo size cannot affect
  prompt size.
- **Every write is atomic and inside a branch**, so any run is one
  `git checkout` from undone.

---

## 2. The action surface

Five verbs. Only `patch` carries a payload.

```
read <path> [start:end]        pull code into the window
find <symbol>                  file:line locations only, never bodies
patch <path>                   + fenced payload
run [-k filter]                the adapter's test command, not a shell
done <one-line summary>
```

`run` is **not** arbitrary shell. A model that can write `rm -rf` eventually
does, and an agent that cannot be trusted unattended is not worth shipping.
`find` lets the model reach outside scope without the repo entering the
window — locations are ~20 tokens, bodies are not.

**Escape hatch:** an `--allow-run` allowlist (regex, empty by default) for the
legitimate "let me run my linter" request, which splits it cleanly from "let
the model run arbitrary bash."

### 2.1 The envelope

````
patch src/fog.ts
```ts
<<<<<<< SEARCH
  const r = computeRadius(t)
=======
  const r = computeRadius(t) * scale
>>>>>>> REPLACE
```
````

The format is **borrowed, not invented** — it is aider's, and aider's 48k
stars mean the format is throughout the training data. Same law as the
method-syntax finding in black-oxide: do not design an elegant surface, adopt
the one the model has already read a million times. Inventing an XML dialect
here would be the agent-shaped version of expecting a 7B to write Oxide from
a card.

### 2.2 Codecs

| codec | output cost | fails when |
|---|---|---|
| `search_replace` | changed region only | copying fidelity is weak — one wrong semicolon and SEARCH misses |
| `whole_file` | = file size | long emissions drift; capped by remaining budget |
| `udiff` | smallest | line numbers and hunk headers must be right |

Landing rate = parses **and** applies cleanly **and** the file still parses.
Python gets a real `ast` check; TypeScript gets brace balance plus the test
run, and that limitation is stated in the README rather than papered over.

Selection is **`codec(family, file_tokens, remaining_budget)`**, not
`codec(family)` — see §3.3 for why.

`udiff` ships mainly so the profiler can demonstrate cheaply that it is bad,
and because a family heavily trained on diffs may surprise us.

### 2.3 Constraint policy

Inherited law: **constrain the envelope, never the payload.**

- **Level 0, default: no grammar at all.** Stop sequences and a strict parse.
  Nothing can be steered because nothing is constrained.
- **Level 1: two-step.** Request A constrained to a header grammar
  (`verb path`), stopping at newline. Request B unconstrained, with that
  header already placed in the assistant turn, producing only the payload.
  GBNF applies to a whole completion, so "constrain the envelope" *requires*
  two calls.
- **Level 2: refuse.** A family that cannot land a header even constrained is
  marked `LIMITED` and declines patch work.

Rationale is the `mut acc` → `mutacc` finding (black-oxide SPEC §54): a
grammar cannot reject a token, it steers to the nearest legal string. Escaping
a code payload into JSON under a grammar is that failure geometry exactly —
the nearest legal string to a raw newline inside a JSON string is something
else, silently. Every project in §0.1 ships this unexamined.

**`payload_corruption` is therefore a measured quantity** (§5), obtained by
running identical patch tasks with and without payload constraint and
diffing byte-for-byte.

### 2.4 Parse failures are prompt surface

They name **both ends**, per the `OX0403` lesson — a diagnostic that points at
one location and shrugs is how confident wrong repairs happen.

```
ACTION PARSE FAILED
  header   ok    patch src/fog.ts
  payload  bad   SEARCH block not found in src/fog.ts

  closest line in file   L42  const r = computeRadius(t);
  your SEARCH line            const r = computeRadius(t)
  difference             trailing semicolon

  re-emit the SEARCH block copied exactly from the file above.
```

Costs a fuzzy-locate implementation. Capped at a token budget so one bad turn
cannot eat the window.

---

## 3. Context construction under budget

```
window       = usable_window(model)          # §3.1 — NOT the spec sheet
reserve_out  = f(codec)   search_replace 512 · udiff 384 · whole_file ≈ file+15%
system       ≈ 350        # five verbs + rules, measured at build time
diagnostic   ≤ 600        # capped
history      = last-K full, older turns elided to one line each
scope_budget = window − reserve_out − system − diagnostic − history
```

When scope does not fit, it degrades in a **fixed, deterministic order** — a
pure function over fixtures, testable with no model:

1. hop-2 → signatures only *(the default)*
2. hop-2 → dropped
3. hop-1 → signatures plus whole bodies of functions touching the span
4. anchor → windowed ±N lines around the failing line
5. still does not fit → **refuse before turn 1**, printing the arithmetic

Step 5 is the zero-evidence branch: refuse loudly rather than start a session
that can only fabricate a result.

### 3.1 The usable window is computed, not read

**The advertised context length is not the usable window.** Measured from
local GGUF metadata (2026-08-09), KV cache cost per token varies ~8× across
models of similar size:

| model | max ctx | KiB/token | KV at 8k | KV at 32k |
|---|---|---|---|---|
| qwen2.5-coder:7b-q8 | 32768 | **56** | 0.44 GiB | 1.75 GiB |
| granite-code:8b-q8 | 4096 | 144 | — | — |
| qwen3:14b | 40960 | 160 | 1.25 GiB | 5.00 GiB |
| phi4:14b | 16384 | 200 | 1.56 GiB | — |
| gemma2:9b | 8192 | 336 | 2.62 GiB | — |
| **codegemma:7b-q8** | 8192 | **448** | 3.50 GiB | 14.00 GiB |

`KiB/token = 2 · layers · kv_heads · head_dim · 2 bytes / 1024`

codegemma's 8k window costs more VRAM than qwen-7B's full 32k. So
`model/geometry.py` reads block count, KV head count and head dim from GGUF
metadata and computes:

```
usable_window = min(
    training_ctx,                                   # never rope-scale past it
    (free_vram − weights − compute_overhead) / kv_per_token,
    user_cap,
)
```

**Recommended model tiers** follow from this arithmetic at ~15 GiB usable:

- **Supported floor: 8192.** `granite-code:8b` (4096) is a *test case*, not a
  target — a design that works at 4096 works everywhere, but its profile
  reads `LIMITED` permanently and it should never be recommended for agentic
  work.
- **Recommended band: 9–14B in the 8–9 GB weight band.** A 14B at Q4_K_M
  leaves ~5.3 GB for KV, giving ~28k usable tokens. A 7B at Q8_0 reaches its
  full 32k with ~4.5 GB spare.
- **Not recommended: `gemma2:9b`** — strictly dominated. Hard 8192 ceiling,
  and 336 KiB/token means 8k costs 6× what qwen-7B pays for the same window.
- The README's recommended-models table is **generated from profile output**,
  never asserted.

### 3.2 Quantization is a covariate, and an experiment

At a fixed ~9 GB weight budget one may buy 7B-Q8 **or** 14B-Q4 — total bits
are near-identical (7×8.5 ≈ 60 Gbit vs 14×4.5 ≈ 63 Gbit). The VRAM ceiling
therefore makes quantization a first-class experimental axis: **at a fixed
memory budget, more parameters or more precision?** Unanswered for agentic
repair, and the profiler is the instrument for it.

**Methodological requirement.** black-oxide pins Q8_0 uniformly because
"uniform quantization is the control that keeps the capability curve from
being confounded with precision." Accordingly:

- quantization is recorded in every manifest as a covariate
- `qwen2.5-coder:14b-q4_K_M` and `:7b-q8_0` are **different subjects**, never
  reported as "14B vs 7B"
- KV-cache quantization (`-ctk q8_0`) is a second such lever: it halves the
  cache and buys real window, and it is **not free** — it goes in the manifest

### 3.3 The whole_file / budget tension

`whole_file` forces `reserve_out ≥ the entire file`, so at a 4096 window a
200-line file (~2,600 tokens) is unpatchable once scope and system are
seated. **Weak families are precisely those least able to afford the codec
easiest for them.** This is a capability boundary, not a problem to be
cleverly solved, and the profiler publishes it as one:

```
granite-code-8b   whole_file      lands 71%   but only for files ≤ ~55 lines
                  search_replace  lands 34%   any file size
```

Token counting uses the backend's real tokenizer where exposed, and otherwise
a calibrated estimate of ≈3.3–3.6 chars per token for typical code. The
server always outranks the estimate: any estimate has a pathological case —
law 5 in §9 records punctuation-dense input where a 4-chars-per-token
estimate under-counted by **3.6×** — so the estimate is a cheap early guard,
never an authority. (The two figures share a digit string by coincidence: one
is a chars-per-token ratio for ordinary code, the other an under-count factor
in a worst case.)

---

## 4. Memory

Hindsight (19.3k★) and the Hermes memory sidecar are the reference designs.
Their goals are right; their mechanism is impossible here:

```
Hermes on a frontier model    4k recall / 200k window  =   2% of the window
a 4096-token family           4k recall /  4k window   = 100% of the window
```

Hindsight also requires PostgreSQL, Docker, and its own LLM provider for
`retain`/`reflect` — a second model in VRAM or an API bill, breaking the
premise twice more. Same goals, inverted mechanism:

**Memory acts on the harness, not on the prompt.** Tiers are ordered by
*where a fact acts*, not by recency.

| tier | what it is | window cost | acts on |
|---|---|---|---|
| **0 · config** | family geometry, codec + size ceiling, test command, tracer overrides | **0** | how the run is configured |
| **1 · selection** | path-alias maps, "the fog logic actually lives in vision.ts", files that always change together | **0** | *which* code enters the window |
| **2 · in-window** | repo conventions; already-tried-and-failed notes | **≤256 tok** | the prompt itself |
| **3 · curated vault** | hand-written notes, markdown | **0 unless pinned** | on request only |

Tier 1 is where a small-window agent gets its leverage: it *replaces* content
rather than adding it. Knowing the tracer misses your path aliases does not
cost 200 tokens, it saves 400 by pulling the right file.

**Tier 2 ships disabled.** Only two things could earn it — repo conventions
(semicolons, indent, quote style, because they determine whether a SEARCH
block matches) and anti-loop notes. Both are hypotheses. It is enabled
per-family only when the profiler's stage-5 repeat rate shows that family
actually re-emits identical failing patches. The profiler earns the feature;
the design does not assert it.

### 4.1 Storage

One fact per file, frontmatter + `[[wikilinks]]`, no database. Openable in
Obsidian, diffable, hand-editable. One fact per file specifically so
concurrent runs do not produce merge conflicts.

```
.robigo/memory/                          # in-repo, git-tracked, reviewable in PRs
  conventions/typescript-semicolons.md
  selection/alias-at-slash.md
~/.config/robigo/memory/              # global, cross-repo
  family/granite8b-whole-file-ceiling.md
```

### 4.2 Retrieval is structural, not semantic

Every other memory system guesses relevance because it is handed free text. A
coding agent **already knows the key**: family slug, repo id, file path, test
id, error code, codec. Retrieval is a dict lookup plus a stdlib inverted index
for the free-text case. No vector store, no embedder occupying VRAM beside the
model. Semantic search solves a problem this domain does not have.

### 4.3 Writes are mechanical only

Every fact cites the observation that produced it — run id, turn, `file:line`.
The model does not write memory: a small model authoring memory notes poisons
the vault, and poisoned memory is strictly worse than none because it costs
window *and* misleads.

```markdown
---
tier: conventions
key: repo=a3f9c1 · lang=ts
hits: 7   misses: 0   confidence: 0.88
origin: run g0u-s3 turn 2 — SEARCH miss ×3, all trailing `;`
---
This repo's TypeScript terminates statements with semicolons.
Copy target lines byte-exactly. See [[alias-at-slash]].
```

Model-proposed notes land in a queue for `robigo memory review`, never
auto-accepted. Facts carry hits/misses and are demoted then deleted when they
stop predicting, or the vault becomes a landfill charging rent in the one
currency that is scarce.

**Memory is an eval arm**: paired on/off, per family, same corpus. As far as
the §0.1 survey showed, nobody has run agent memory against a control on the
actual coding task. If it does not move landing rate, that is published too.

`robigo memory doctor` exists because a recall path that silently stops firing
is the characteristic failure of systems like this.

---

## 5. The profiler

The component everything else trusts, so it must be cheap enough that people
run it and honest enough to publish.

**Staged, cheapest-first.** Models die at the shallowest stage, so measure per
stage and stop early rather than spending an hour discovering a family cannot
emit a parseable action.

| stage | measures | cost |
|---|---|---|
| 0 · geometry | KiB/token, and the largest window that **actually loads** | seconds |
| 1 · envelope | can it emit `verb path` at all? no code involved | seconds |
| 2 · codec landing | parses **and** applies **and** file still parses — per codec, per size bucket | minutes |
| 3 · payload corruption | same task, constrained vs unconstrained, byte-diff | minutes |
| 4 · repair | full loop on a single-defect task — does the test go green? | slow |
| 5 · loop discipline | turns-to-green, identical-failing-patch repeat rate | slow |

Stage 0 **probes** rather than trusts: load at successively larger windows
until it fails, and record the real ceiling. Stage 1 gates the rest.

**Two budgets, one instrument.** `robigo profile` runs stages 0–2, 3 seeds,
~20 tasks — minutes, enough to configure the harness. `robigo profile --full`
runs all six at 10 seeds — overnight, and is the only mode whose numbers may
be published.

**Grid ordering is by model residency**, not by stage: only one model fits at a
time and a swap costs seconds paid hundreds of times. All stages for one
family, then swap.

### 5.1 The corpus is generated, not authored

Twenty hand-written mechanically-verified fixtures across two languages is the
largest single cost in this design. Instead, **mutation**:

1. mutate one line of real code
2. keep the mutant only if **exactly one** test fails
3. record the diagnostic and the reverse-patch as ground truth

Real code, real failures, mechanically verified by construction, and it scales
to any repo the user points it at. black-oxide's own 1327-test suite is a
corpus mine. `.robigo/runs/` records (§6) double as corpus candidates.

The verification property is inherited unchanged: a record may not enter the
corpus unless the broken form fails with the intended diagnostic **and nothing
else**, and the reference patch makes the test pass.

### 5.2 Output

```json
{
  "family": "granite-code-8b", "quant": "q8_0",
  "training_ctx": 4096, "kv_kib_per_token": 144,
  "usable_window": 4096, "envelope_level": 1,
  "codecs": {
    "whole_file":     {"lands": 0.71, "max_file_tokens": 1400},
    "search_replace": {"lands": 0.34, "max_file_tokens": null},
    "udiff":          {"lands": 0.05, "max_file_tokens": null}
  },
  "payload_corruption": 0.12, "repeat_rate": 0.31,
  "verdict": "LIMITED",
  "measured": {"seeds": 3, "mode": "quick", "corpus": "v1"}
}
```

### 5.3 Self-validation

`robigo profile --replay` runs the whole pipeline against recorded model
outputs and must reproduce a known profile exactly. The profiler is
load-bearing, so it gets an instrument of its own — and this makes it testable
in CI with no GPU.

### 5.4 Profile → behaviour

- `READY` → normal operation
- `LIMITED` → refuse `patch` above the size ceiling, warn at startup, suggest
  `--scope`
- `UNUSABLE` → refuse the run, printing the measurement that says why
- **unprofiled** → conservative defaults, announced on every run. Never a
  silent assumption.

### 5.5 Honesty rules in the reporter

Per-family only, never pooled. Floors and ceilings named as such. `seeds` and
`mode` printed in the table so a quick profile cannot be quoted as a result.
Anything dropped for time stated as dropped.

---

## 6. Safety and failure modes

The threat model is not malice. It is a small model with weak
instruction-following holding a write handle to a working tree.

- **A patch whose generation stopped at the token cap is rejected, never
  applied.** A truncated `whole_file` emission is the likeliest real data
  loss: the codec would faithfully write a gutted file. `done_reason ==
  "length"` is promoted from telemetry to a veto.
- **The anchor test file is read-only.** A small model will edit the failing
  test until it passes, and every layer above would report success.
  `--allow-test-edits` opts out for when the test really is wrong.
- **Scope escape is refused structurally** — paths resolved and compared
  against the scope root. No `..`, no absolutes, no symlink hop.
- **`run` is the adapter's test command**, with a wall-clock cap, output
  truncated to the diagnostic budget before entering the window.
- **Git is the undo.** Every run branches (`robigo/<slug>-<n>`), commits a
  snapshot *before* the first patch — including a dirty tree, so nothing is
  lost — then commits each applied patch. Not in a repo → refuse, unless
  `--no-git`, which prints what is being given up.
- **Stalling is detected, not waited out.** Turn cap, wall-clock cap, and a
  no-progress check: unchanged test signature plus rejected patches for K
  turns ends the run as `stalled`.

**A failing test is a hard input requirement.** `robigo` refuses to start
without one and says why: *write the failing test first — that is the
interface.* v1 cannot do "add feature X", and turning that into a stated
stance is more honest than letting every first attempt fail.

### 6.1 Terminal states

Infrastructure must never be reported as a model result, or the reverse.

```
0  pass               3  refused before starting (scope won't fit, UNUSABLE)
1  stalled            4  infrastructure (daemon down, adapter missing)
2  budget-exhausted
```

### 6.2 Run records

`.robigo/runs/<id>/` holds prompts, **raw model outputs verbatim**, patches,
adapter output, and terminal state — the same format as `eval/results/`, so a
user's failing run is already a replayable corpus candidate.

---

## 7. Measurement, and what the README may claim

The harness is a **deliberate fork** of black-oxide's `eval/`, marked
"forked at `<sha>`, intentionally not synced". Two research repos with
different task shapes will drift; pretending to share produces two sets of
numbers that disagree with no way to adjudicate.

**Three isolated tables — the credible numbers**, each moving one thing:

- codec landing rate per family, per file-size bucket
- payload corruption, constrained vs unconstrained
- memory on vs off, paired
- (and, once §3.2 is run: fixed-VRAM-budget parameters vs precision)

**One bundled table, labelled as bundled.** "This harness vs a generic
harness" is the number people want, and it is legitimate as external validity
— but it is six differences at once and must say so in the same breath. A
bundle is not a mechanism.

**`CORRECTIONS.md` exists from the first commit.** The most
credibility-building thing in black-oxide's README is the section where a
headline is withdrawn and the arithmetic that killed it is shown. Shipping the
file empty with a stated commitment to use it is worth more than any number in
the initial table.

**Never claimed:** that small models are good coding agents; that this beats a
frontier model; a single magnitude for anything.

---

## 8. Build order

Deliberately sequenced so the first honest number arrives before the
expensive features.

1. `loop` + `action` + `adapters/python_` + `apply` — the edit→test loop
2. `model/geometry` + `context/budget` — the arithmetic, with fixtures
3. profiler stages 0–2 + `--replay`
4. mutation corpus generator
5. profiler stages 3–5
6. **read the number, apply §0.2's kill criterion**
7. memory tiers 0 and 1
8. `adapters/typescript`
9. speculative decoding (draft model config), then publish

Memory and TypeScript come after step 6 because if stage 4 returns 18% they
were both wasted.

**The first implementation plan covers steps 1–3 only.** That is the smallest
increment that produces a running loop plus the arithmetic and the profiler
that configures it — enough to be exercised on a real repo. Steps 4–5 are a
second plan, and step 6 is a decision point, not work. Planning past a gate
whose outcome could cancel the remaining steps is how the wasted-effort
scenario above actually happens.

**Speculative decoding** is a free win specific to this hardware: a 0.5B draft
model against a 7–14B target fits alongside it in 16 GB, and code is the most
predictable text there is. Turns × tokens is the agent's real cost, and no
project in §0.1 ships this configured.

---

## 9. Laws inherited from black-oxide

Adopted rather than re-derived. Each was paid for.

1. **Never rope-scale past the training context.** `llama-server` refuses
   outright; a window larger than the model was trained on is physically
   unsatisfiable, not a policy choice.
2. **The window is a per-family covariate, arm-fair within a family.** Do not
   equalize across families; hold it identical across arms inside one.
3. **Overflow is evidence-gated, not type-gated.** ≥1 attempt submitted → a
   session result, attempts preserved. Zero attempts → a loud abort with the
   cause recorded, because at a small window a zero-attempt "result" repeats
   identically across every seed and fabricates a whole grid.
4. **Two overflow detectors, both kept** — a cheap client-side estimate and
   the server's real tokenizer. The decision does not depend on which fired;
   the record still says which did.
5. **Ollama silently front-truncates by default.** `truncate: false` is
   mandatory and **top-level** — nested in `options` it is ignored. Measured:
   a 3160-token prompt into a 256-token window returned 200 with
   `prompt_eval_count: 130`, head canary gone. Also measured: the crude
   4-chars-per-token estimate under-counted a real prompt by **3.6×**
   (1845 estimated, 6648 actual).
6. **Constrain the envelope, never the payload** (§54: `mut acc` →
   `mutacc`). A grammar cannot reject a token; it steers to the nearest legal
   string.
7. **Diagnostics name both ends** (`OX0403`). One end plus a shrug produces
   confident wrong repairs.
8. **Corpus records are mechanically verified in both directions** before
   admission.
9. **Per-family, never pooled.** codegemma's wall was 100% instrument;
   granite's was real incompetence. Same number, opposite meaning.
10. **Infrastructure and model failures are never conflated in either
    direction.** The first biases toward the null; the second drops the
    worst-performing cells.
11. **Ergonomics can dominate reasoning.** Method syntax was worth +42pp
    against ownership's ≈+10pp. Meet the pretraining prior before designing
    anything elegant.
12. **Do not invent a DSL.** 2/20 vs 20/20. Pretraining exposure beats design.

---

## 10. Open questions

- The 40% kill threshold in §0.2 — the one number chosen rather than derived.
- Whether stage-4 tasks should be drawn from the user's own repo by default
  (more relevant, less comparable) or from a fixed corpus (comparable, less
  relevant). Current lean: fixed corpus for published numbers, user repo for
  `--replay`-able local sanity checks.
- Whether `search_replace` fuzzy-matching should be allowed to normalize
  whitespace. It would raise landing rates and it would also hide the
  convention-memory effect the Tier-2 experiment is trying to measure.
- TypeScript syntax verification without a dependency. Brace balance is weak;
  the alternative is shelling out to a project-local `tsc`, which is a
  dependency by another name.
