# robigo

A coding agent for local models, designed against a VRAM budget.

robigo repairs one failing test at a time using a model running on your own
hardware. It is built for the case where context is genuinely scarce — a single
consumer GPU — so it computes how much context actually fits before it writes a
prompt, and shrinks what it sends until it does fit. If it cannot fit, it refuses
and prints the arithmetic rather than sending a prompt the server will truncate.

**Status: early.** The edit→test loop, the context budget and the degradation
ladder work and are covered by 302 tests. The measurement harness that will
decide whether the whole approach is worth pursuing is not built yet — see
[Honest status](#honest-status).

## Why a VRAM budget

The usual assumption is that a model's advertised context length is the context
you have. On a 16 GB card that is not close to true, for two reasons this project
measures rather than assumes.

**The KV cache dominates, and its cost per token varies enormously between
models of similar size.** Measured across the models on one machine:

| model | layers × kv-heads × head-dim | KV cost |
|---|---|---|
| qwen2.5-coder 7B | 28 × 4 × 128 | **56 KiB/token** |
| granite-code 8B | 36 × 8 × 128 | 144 KiB/token |
| gemma2 9B | 42 × 8 × 256 | 336 KiB/token |
| codegemma 7B | 28 × 16 × 256 | 448 KiB/token |
| a llama with no GQA | 40 × 40 × 128 | **800 KiB/token** |

That is a **14× spread** at comparable parameter counts. A 32k context costs
1.75 GB on the first row and 25 GB on the last. Two 7B models, same quantisation,
and one of them cannot hold a tenth of the other's window.

**A 16 GB card does not have 16 GB free.** Measured on an idle desktop: 16,303
MiB total, **14,558 MiB free** — the compositor and desktop hold the rest before
anything is asked for.

So robigo takes the smallest of three numbers: what free VRAM allows once the
weights and a margin are subtracted, the model's training context, and any cap
you set. It reports which one bound the result:

```
$ robigo --model qwen2.5-coder:7b-instruct-q8_0 "fix the failing test"
window 32768 (limited by training_ctx, 56 KiB/token)
```

Ask for more than the model was trained on and it clamps rather than obliging —
`llama-server` refuses an oversized slot outright, and Ollama accepts it silently
and degrades the rope scaling, which is worse.

## The degradation ladder

When the prompt does not fit, the scope shrinks in a fixed order — fixed so a run
is reproducible, not heuristic:

1. two-hop dependencies as signatures only
2. two-hop dependencies dropped
3. one-hop dependencies reduced to signatures
4. the failing file windowed around the failing line
5. refuse

The rung is chosen by **measurement, not arithmetic**: each candidate is rendered
and checked against the real window, and the first that fits is sent. This is not
fussiness. The token estimate is not additive, so a decomposed budget can
under-count the assembled prompt by a token, and a token is enough — on a real
run in a 1100-token window, the worst turn used **1097**.

Windowing centres on the failing line, not the middle of the file, and the prompt
says so. An earlier version centred on the file's midpoint while labelling it
"windowed around the failure", which for a failure at line 350 of a 400-line file
handed the model an excerpt that did not contain the bug.

Every rung the run used is recorded, so a run that quietly degraded is
distinguishable from one that never had to.

## What it does in a run

Reads the failing test, resolves scope by following imports from it, builds a
prompt that fits, and asks the model for one action at a time: `read`, `find`,
`patch`, `run`, or `done`. Patches are applied atomically and verified before the
write. Every turn is recorded to `.robigo/runs/<id>/` with the exact prompt, the
reply, and the test output.

Work happens on a `robigo/*` branch with a snapshot commit taken before the first
patch, so the undo instruction printed at the end is one the tool can actually
honour — including when your tree was dirty to begin with.

Exit codes are the outcome: `0` pass, `1` stalled, `2` budget exhausted, `3`
refused, `4` infrastructure.

## Install

Python 3.12+, and no runtime dependencies — the standard library only.

```bash
git clone https://github.com/<you>/robigo && cd robigo
pip install -e .
```

You need a local model server. Either:

- **Ollama** — geometry is read over HTTP, nothing else needed.
- **llama.cpp** — `llama-server` does not expose KV head counts over HTTP, so
  pass `--gguf /path/to/model.gguf` and robigo reads the file directly.

## Usage

```bash
# from a repo with exactly one failing test
robigo "make the failing test pass"

# a specific model and a hard cap on the window
robigo --model qwen2.5-coder:7b-instruct-q8_0 --window 8192 "fix the parser"

# llama.cpp, where the GGUF is the only source of geometry
robigo --backend llamacpp --gguf ~/models/qwen.gguf "fix the parser"

# narrow the scope by hand when the ladder refuses
robigo --scope src/thing.py tests/test_thing.py "fix the rounding"
```

`--window auto` is the default. `--kv-bits {16,8}` **describes** the precision
your server is already running (Ollama's `OLLAMA_KV_CACHE_TYPE`, llama.cpp's
`--cache-type-k`, both set at server launch) — robigo cannot change it over the
API, and telling it 8 when the server runs 16 will overcommit VRAM by 2×.

## Design rules

- **Never send a prompt that does not fit.** A truncated prompt makes the model
  answer about code it cannot see. Refusing is honest; truncating is not.
- **Measure, never guess.** Where a number can be read from the system it is
  read. A plausible default that stands in for a measurement is treated as a
  defect — one such default reported every model as weighing zero bytes and
  handed back the largest window in the table.
- **Refuse before turn 1, with the arithmetic printed.** A user whose window came
  back small gets the terms, not a summary.
- **Constrain the envelope, never the payload.** Structure is enforced around the
  model's output, never inside it.
- **Say what happened.** Records name the rung, the window, the outcome and the
  branch. A run that degraded and one that did not leave different records.

## Honest status

What works: the edit→test loop, the action surface, patch application and safety,
scope resolution, geometry from Ollama or a GGUF file, the window budget, and the
degradation ladder — 302 tests, no runtime dependencies.

What is verified against real models: geometry parsed from all 20 real GGUF blobs
on one machine; a 7B model repairing a real bug inside an 1100-token window at a
degraded rung, changing one line in the source and leaving the failing test
alone.

What is not built: the measurement harness. The open question this project exists
to answer is whether a local model under these constraints resolves enough real
tasks to be worth using, and that has a number attached — if it cannot reach 40%
on a fixed corpus, the honest outcome is to say so publicly rather than keep
polishing. Nothing here should be read as a claim that it does yet.

Only a Python adapter exists. Only `search_replace` and `whole_file` codecs are
implemented.

Known gaps and deferred decisions are recorded in
[`docs/CARRIED-DEBT.md`](docs/CARRIED-DEBT.md) rather than left implicit, and
withdrawn claims go to [`CORRECTIONS.md`](CORRECTIONS.md) with the arithmetic
that killed them.

## Licence

MIT. See [LICENSE](LICENSE).
