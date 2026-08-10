# Carried debt from plan 01

Written at plan 01's merge so it is not re-derived as findings later. Every
item below was found by review, consciously not fixed, and ruled on. The
execution ledger that produced this list was scratch and has been deleted;
this file is the record.

Plan 01 shipped as 51 commits, 152 offline tests, one live test. Twelve task
reviews, a whole-branch review, one fix wave and a scoped re-review all came
back clean.

## Parked, with rulings

Three honesty gaps on rarely-taken paths. Each is real; none blocks.

- **`loop.py` — the escape record claims `branch=None` even when a branch
  exists.** `run`'s `except BaseException` handler hardcodes `branch`, so an
  escape after `start_branch` writes a `meta.json` naming no branch while a
  `robigo/*` branch is on disk — the same dishonesty `_execute`'s own comment
  forbids. `branch` is not in scope there without threading it out of
  `_execute`.
- **`loop.py` — a user's Ctrl-C is recorded as `outcome: "infrastructure",
  exit_code: 4`.** The process behaves correctly: `KeyboardInterrupt` is
  re-raised past `cli.main`'s `except Exception`. Only the transcript
  misattributes the interrupt.
- **`cli.py` — on a detached or unborn HEAD, no undo instruction prints at
  all**, even though the snapshot SHA was captured and the user's dirty work
  is inside that commit. Strictly better than what preceded it, which also
  failed there.

## Test hardening, as a named slice

Coverage gaps on real behaviour, several mutation-proven. Worth one focused
pass rather than opportunistic fixes:

`_preflight`'s `OSError` branch and the `-k` filter; a test-file `SyntaxError`
producing a refused run (unpinned in both directions); `signatures_of`
truncating a multi-line `def` to its first physical line — the model reads that
outline; regression tests for the sweep-in, gitignored and zero-commit git
cases; `write_atomic`'s cleanup path under failure injection;
`parse_http_error`'s `None` arm and the generic retry arm; the `done` verb arm
and both `ModelError` arms; `_find`'s `OSError` containment; the CLI root
guard; `slug` and `new_recorder`.

Two notes that make this slice worth doing properly:

- **`OllamaClient`'s `options["stop"]` path is now exercised by no test.** The
  CLI stopped populating `stop=` (it could cut a patch payload mid-stream), so
  only llamacpp's top-level `stop` is covered.
- **Mutation testing is the tool that found most of these.** Two traps: this
  project is installed editable, so naive in-place mutation silently tests the
  original source — force `PYTHONPATH=src` or mutate a `git archive` copy and
  assert `robigo.__file__` — and the mutations worth injecting are the
  thresholds and predicates the fix rounds were spent on (`stall_cap - 1`,
  `lines[start:]`, `add -A`), not arbitrary operators.

## Opportunistic

Fix alongside adjacent work: the CRLF-normalisation comment overclaims;
capitalised near-miss verbs fall through to the generic parse error;
`_miss_message` inspects only a multi-line SEARCH block's first line; a stray
`f` prefix on a placeholder-free string; `_strip_nested_fence`'s trailing-blank
walk-back is untested; two unit tests pass an unresolved `tmp_path` as `root`
and would fail where `/tmp` is a symlink; `hops` collapses to boolean
semantics; the `SYSTEM` size bound is slack at 3× the actual length; `codec:
str` could be a `Literal`; `if diag.line` treats line 0 as falsy; `--scope`'s
`nargs="+"` is argparse-greedy and swallows a trailing positional task.

## Inherited law gaps — for plan 02's preamble

Two of the spec's §9 laws are not fully enforced, and both land in plan 02's
territory (`model/geometry.py`, `context/budget.py`). They are carried debt,
not new findings:

- **Law 4 wants two overflow detectors; there is one.** The client-side
  pre-request estimate does not exist — `ContextOverflowError` is never raised,
  only its `ServerContextOverflowError` subclass — so nothing catches an
  oversized prompt before the request, and the record does not say which check
  fired.
- **Law 1 is unenforced.** `--window` defaults to 8192 with nothing comparing
  it to the model's training context, so a window above what the model was
  trained on can still be requested.

## Process lessons worth keeping

From the whole-branch review, and borne out by the execution record:

1. **The plan's reference code was the dominant defect source**, not any
   implementation. Roughly twenty amendments, and almost every one fixed the
   plan rather than the code. Two error messages shipped naming flags that did
   not exist (`--scope`, `--python`), and the whole-branch Critical was
   verbatim plan text. Prefer plans that state **invariants and their
   falsification tests** over plans that state code to transcribe: a plan
   saying *"`find` must return the same answer regardless of where the repo
   lives on disk"* survives, where `if _SKIP.intersection(path.parts)` ships
   the bug into twelve reviews that each read it as "matches the plan."
2. **Per-task review cannot see cross-module inconsistency.** Five path
   resolvers where one guarded `ValueError`; three modules giving three
   different answers about one unreadable file; two truncation conventions;
   three copies of the codec list. Every one passed twelve reviews because each
   review saw one diff. A mechanical whole-branch sweep for repeated shapes —
   `.resolve()`, `read_text(`, every `except` — is twenty minutes and catches
   this class.
3. **Verify external contracts; never declare the last one verified.** The
   `--tb=line` output format was assumed twice and wrong twice. The llama.cpp
   response shape was assumed once and right. Keep an explicit
   "unverified external contract" list, and let items leave it only with a
   captured response attached.
4. **Anything a user is told to type is production surface.** `git checkout -`
   shipped through twelve reviews because nobody ran it.
5. **A guarantee that holds in the development environment can fail in the
   shipped one.** Three findings had this shape, and the sharpest was invisible
   from inside this repo by construction: run records were being committed into
   the user's history, and only a scratch repo whose `.gitignore` did *not*
   cover `.robigo/` could show it.

---

# Carried debt from plan 02 (geometry and budget)

Written at plan 02's merge, for the same reason as the section above: the
execution ledger is git-ignored scratch and gets deleted. Plan 02 shipped as 39
commits and 259 tests, through five task reviews, a whole-branch review and a
three-round fix wave.

## Deferred with rulings, all judged non-blocking by the whole-branch review

- **`Geometry.kv_bytes` truncates for `kv_bits` outside {8, 16}.** Effectively a
  dead method — no production caller; `usable_window` does its own arithmetic
  and `--kv-bits` is `choices=(16, 8)`.
- **`free_vram_bytes` wraps its runner in a bare `except Exception`**, so a
  caller-injected runner's own bug reads as "no GPU"; and a runner *returning* a
  non-`str` escapes as `AttributeError`. `runner` is a test-only seam — the
  production path calls it with no argument.
- **`min()` attributes an exact `vram`/`training_ctx` tie to `training_ctx`** by
  list order. The window value is identical either way; only the label differs.
  Undocumented.
- **The real-blob GGUF test samples the largest 10 blobs.** Four of the 32 real
  blobs here are legitimately not causal LMs — three CLIP projectors and a
  nomic-bert embedder — for which `GeometryError` is the *correct* answer. They
  pass only because they sort below the sample window, so a vision- or
  embedding-heavy store would fail this test spuriously. The fix is skipping
  non-causal architectures, not loosening the assertion.
- **A `Scope` with both `full=()` and `signatures=()` renders one extra blank
  line** versus pre-refactor. Unreachable via `resolve`/`explicit`/`degrade`,
  which all guarantee `full` is non-empty.
- **`--host` without a URL scheme fails with `unknown url type`.** Inherited from
  plan 01's `client.py`; deliberately not fixed, because a `detect.py`-only fix
  would let window resolution succeed where the generation call then fails
  identically moments later. Plan 02 gives that future fix two duplicated
  `OLLAMA_HOST` constants to unify.
- **Rung 4 collapses toward rung 3's cost on a file shorter than the window.**
  Inherent to a fixed-width window; the label correctly suppresses itself.

## Routed to plan 01's named test-hardening slice

Found by plan 02's whole-branch review, but living in plan 01's code:

- **`python_.imports` crashes on a non-UTF-8 `.py`** — `ast.parse(read_text(...))`
  guarded for `(OSError, SyntaxError)`, and `UnicodeDecodeError` is a
  `ValueError`. This is the **fifth** answer this codebase gives about an
  unreadable file, and the only one that crashes. Reachable from
  `scope.resolve`, including the mid-loop re-resolve, so it can kill a run
  *after* patches have landed.
- **`record.py` catches `OSError` but not `UnicodeDecodeError`**, so a
  `.robigo/.gitignore` holding invalid UTF-8 surfaces as exit 4 *after* a
  successful repair — the outcome `_write`'s docstring exists to prevent,
  through a different door.
- **Three unlinked names for one output reservation**: `--num-predict`,
  `Budget.reserve_out`, `reserve_for`. Nothing makes them agree, `reserve_for`
  has no production caller, and its tested `"udiff"` arm is a codec the CLI
  rejects. **This is a wiring hazard for the ladder slice.**
- Bytes-per-token has two names (`Geometry.kv_bytes_per_token` vs
  `WindowPlan.kv_per_token`); `WindowPlan` mixes `free_vram` with
  `weights_bytes`/`overhead_bytes`; `MAX_STEP = 5` names the *refusal* step so
  `MAX_STEP - 1` appears three times; `OLLAMA_HOST` is defined twice.

## For the ladder-wiring slice specifically

- **`Budget`/`fit()` are not called by any production path.** `loop.py` reserves
  the `budget_exhausted` outcome and exit code 2 but never triggers them, and
  calls `render()` directly with an unbounded history tuple. The brief's own
  `Files:` list named `loop.py` while no step touched it, so this was intended
  and then assigned to no task.
- **`history` must be seated from the turns actually about to be rendered.** The
  default is now derived (`DEFAULT_HISTORY_TOKENS`, 2500, from `loop.py`'s read
  cap × turns kept) and errs toward over-reserving, but a caller that computes
  it from real content will do better. Before that derivation, `fit()` accepted
  a prompt that overflowed a 4096 window by 1095 tokens.
- **`Budget.history`'s default is lazy on purpose** — `field(default_factory=…)`.
  An eager module-level call made `loop.py` fail to import with a circular
  `ImportError` the moment it imported `context.budget`, which is that slice's
  natural first line. Do not "simplify" it back to a constant.

## Process lessons, added to plan 01's

1. **State invariants and their falsification tests, not code to transcribe.**
   Plan 01's lesson 1, ignored for the first eight of plan 02's amendments and
   then applied: every one of those eight was a defect in *my* text, and the two
   amendments written as invariants produced correct implementations first try.
2. **Measure before specifying, and never hand-derive a number that a test will
   assert.** Two amendments shipped arithmetic that could not hold — a rung-2
   assertion against a rung-3 budget, and a `free_vram` figure that made
   `training_ctx` bind where the test claimed `vram`. Both were caught by
   implementers reporting BLOCKED. Three BLOCKED reports on this plan were all
   correct.
3. **Mutation-test every new test; a passing suite is not evidence.** Seven
   vacuous tests were found — including one that stayed green with a completely
   broken helper, one whose `skipif` could never fire, one whose assertions sat
   inside `except Exception: pass`, one asserting `"m" in str(e.value)` where
   `"m"` is a substring of the word "model", and — worst placed — the test
   guarding the estimate/render agreement, whose two sides were the same
   expression so `render` was never called.
4. **Prove offline-ness at the socket, not by grep.** Running the suite with
   `socket.socket.connect` patched to raise found a test that POSTed to a real
   daemon. Grep finds the stubs you thought of.
5. **`.get(key, default)` substitutes only when the key is absent — never when
   its value is `null`.** This shape produced the plan's worst bug
   (`.get("size", 0)` reporting every model as weightless, returning the largest
   window in the table) and then produced it twice more, one level up, in the
   fix for the first instance.
6. **An enumeration of a function's outputs is not an enumeration of its
   behaviours.** The `detect.py` audit traced everything downstream of `_show`
   returning and never asked what happens when it raises — leaving
   `json.JSONDecodeError` escaping the CLI contract through two rounds of fixes.
7. **Run the thing.** Three defects in the last task — a zero window that
   printed and then proceeded, a message advising a flag the user had just
   passed, and a completion criterion that was never true on this hardware —
   were invisible to 239 passing tests and obvious within one CLI invocation.

---

# Carried debt from plan 02b (wiring the ladder into the loop)

Written at merge, for the same reason as the sections above: the execution ledger
is git-ignored scratch. Plan 02b shipped as 8 commits and 302 tests, through one
task review, a whole-branch review and a fix wave.

**What this slice actually settled.** The five-rung ladder is now reachable at
runtime, and the rung is chosen by **measurement, not arithmetic**: the loop
walks the fixed order from rung 1, renders each candidate, and takes the first
whose rendered prompt satisfies `estimate_tokens(prompt) + num_predict <=
window`. `fit()` remains the module's arithmetic API and supplies the refusal
message, but it no longer decides where the ladder stops. Two defects forced
that design — a seated decomposition that under-counts by 1 token at ~5% of
length residues, and an arithmetic-only refusal that rejected a run whose
smallest rung fit exactly.

Verified live once VRAM was free: a 7B local model repaired a real bug inside an
**1100-token window** at a degraded rung, one correct line changed in the source
with the failing test untouched, worst turn using **1097 of 1100 tokens**. Three
tokens of headroom is the margin that makes measuring rather than estimating a
correctness property rather than a preference.

## Deferred with rulings

- **`usable_window`'s docstring still states the precondition this slice
  removed** — "`free_vram` must be measured BEFORE the model is loaded … Callers
  own that ordering." `plan_window` now handles residency itself, so two
  docstrings answer one question two ways.
- **Three duplications.** The exact-name-then-`:latest` matching rule and its
  `by_name` comprehension are byte-identical in `weights_bytes` and
  `resident_bytes`; the advice sentence "Narrow it with `--scope`, or use a model
  with a larger window." exists in both `budget.py` and `loop.py`;
  `(host or OLLAMA_HOST).rstrip('/')` is in three places. Each is one drift from
  two modules disagreeing.
- **Neither refusal message names `--num-predict`, which is often the binding
  term.** With the default 1024 against a small auto-computed window, the reserve
  alone can exceed what the fixed costs leave, and no amount of `--scope`
  narrowing helps — yet `--scope` is the only lever named. Same family as plan
  01's messages naming the wrong flag, inverted.
- **`SYSTEM_TOKENS`, `DIAGNOSTIC_TOKENS` and `_default_history_tokens` now have
  no production caller.** `measure` always passes all three, so the dataclass
  defaults are unreachable from `src/`, and their docstrings describe a caller
  that exists only in tests. `_default_history_tokens`'s 25-line circular-import
  warning now guards a test-only path — **do not delete it without reading that
  warning**, since the import-order hazard it documents is real.
- **`/api/ps` has no captured contract for its *resident* shape.** The idle shape
  (`{"models": []}`, never `null`) was verified live; the `name`/`size_vram`
  field names on a loaded entry are asserted only in comments. `/api/show`'s
  workaround got a `live`-marked sentinel; this did not. Also: `plan_window` runs
  even for an explicit `--window`, so on an Ollama predating `/api/ps` the 404
  escapes as `OSError` printing "HTTP Error 404: Not Found" with no endpoint
  named and no `--window` escape.
- **Free VRAM and residency are read at different times and combined as if
  simultaneous.** A model loading in the gap over-credits by up to its full size
  — the direction that overcommits. Needs a sub-second race, so rare; the
  opposite race merely reproduces the old under-count.
- **None of the three HTTP endpoints guards HTTP status or `URLError` by
  endpoint**; all surface as `OSError` to a CLI message that names no endpoint.
  Pre-existing, now with a third instance.
- **`loop.py`'s `try: fit(...) except BudgetExhausted: raise` is a functional
  no-op.** A bare call behaves identically. Left deliberately for readability
  beside its commentary; harmless, but a future reader may take it for something.

## Process lessons, added to the two lists above

1. **A correction applied in one direction is half a fix.** "Arithmetic proposes,
   measurement decides" was wired into the accept path and not the refuse path,
   so the branch shipped a *new* one-token defect of the same family it was
   fixing. When a fix compensates for an imprecision, check both directions of
   that imprecision before calling it done.
2. **A criterion that correct code cannot satisfy is a bad criterion.** "A real
   run completes a repair" depended on whether a 7B model lands a fix, which no
   amount of correct wiring changes. State what the system owns — that
   degradation happens, is recorded, and yields a prompt the server accepts.
3. **Record the sequence, not the summary.** `meta.json` kept the last turn's
   rung while its own comment claimed the field distinguished a degraded run from
   one that never degraded. A run measured `[2, 2, 3]` recorded as `3`. When a
   consumer is a future analysis tool, store what happened, not a reduction of it.
4. **Run it twice.** The single worst user-facing defect in two plans —
   `window 0` and a refusal on every second consecutive run — was invisible to
   300 passing tests, to five reviews, and to running the tool once.

---

# Carried debt from plan 03 (profiler stages 0-2)

Written at the whole-branch fix wave's close (ruled 2026-08-10). Three
Criticals (a window reported twice what any probe demonstrated; three dead
replay fixtures; `training_ctx` assigned `plan.window` instead of the real
training context) and two Importants (an unverified window reading `LIMITED`
beside fabricated 100% numbers; two unmeasured fields shipping as `null` with
no `dropped` entry) were fixed. Everything below was found by the same review,
consciously not fixed, and ruled on.

## For plan 04 specifically — both sit on its first code path

**Both items below are RESOLVED, in plan 04 itself (task 4) and confirmed
still fixed by its whole-branch fix wave (I7, ruled 2026-08-10) — this
section previously listed them as open, which would have stated two
falsehoods about the code plan 05 builds on. Kept here, marked, rather than
deleted, so the record of what was found and when is not lost.**

- ~~**`best_codec()` has no landing floor.**~~ **Fixed.**
  `Profile.best_codec()` (`src/robigo/profile/schema.py`) now requires
  `lands > _LANDING_MIN` before naming a codec "best", returning `None`
  when every codec landed at or below the floor — a 0%-landing codec is
  never quoted as best.
- ~~**`corpus` is a kwarg default (`"fixtures-v1"`).**~~ **Fixed.**
  `run_profile`'s `corpus: str` parameter (`src/robigo/profile/report.py`)
  carries no default; every caller must name the corpus explicitly.
  `cli.profile_main` passes `robigo.profile.fixtures.CORPUS_NAME` — one
  definition, not a second literal — and `robigo corpus`'s own output
  (`cli.corpus_main`) derives its `name` from `--repo`, never a fixed
  string.

## Deferred with rulings, non-blocking

- **`verdict_for` cannot express "not measured".** It only ever returns
  READY/LIMITED/UNUSABLE — there is no verdict for "stage 0 never ran" versus
  "stage 0 ran and found nothing", so a totally unverified window's UNUSABLE
  (via `envelope_fidelity=0.0`, which this fix wave's I1 gate now forces) reads
  identically to a family that genuinely cannot drive the envelope. `dropped`
  is the only place the two are told apart today.
- **`max_file_tokens=None` carries two facts.** It means both "not tracked for
  this codec" (`search_replace`, always) and "tracked, but nothing landed"
  (`whole_file`, when `lands == 0`) — the same collapse `dropped` exists to
  prevent for `codecs` as a whole, one level down.
- **Stage 2 re-implements `render._preamble`'s composition** (assembling
  `SYSTEM` + codec help + trailer by hand in `landing_prompt`) rather than
  calling `_preamble` itself, even though this fix wave's own amendment
  established the principle ("stage 2 presents the same envelope the loop
  presents") that argues for calling the real function, not a parallel
  assembly of its three pieces.
- **Stage 1's `ENVELOPE_PROMPT` paraphrases `render.SYSTEM`** rather than
  building on it — the same drift class the stage-2 amendment fixed for
  `landing_prompt`, not yet applied to stage 1.
- **`run_profile` does not check `client.window` against `plan.window`.**
  Nothing asserts the client it was handed was actually built with
  `window=plan.window`; a caller wiring a mismatched pair would get a profile
  whose stage-0 probes target one number while the client enforces another,
  silently.
- **The seeds↔mode invariant is only enforced at the CLI.** `run_profile`
  itself accepts any `(seeds, mode)` pair, including `mode="full"` with
  `seeds=1` — only `profile_main`'s `--full` flag keeps the two in lockstep.
  A caller of `run_profile` directly (as this fix wave's own
  `test_committed_transcripts_replay.py` and the pre-existing
  `test_profile_report.py` both are) can produce a "full" profile that never
  ran ten seeds.
- **The codec tuple `("search_replace", "whole_file")` is hand-written** at
  `stage2_codecs`'s default and in `_CODEC_HELP`'s keys, rather than derived
  from `action.codec.CODECS`'s own keys — a third codec added to `CODECS`
  would silently never get profiled.
- **`_KNOWN_ERRORS` lookup (`transcript.py`) is unguarded.** `CallReplayer`
  looks up `row["error_type"]` with `_KNOWN_ERRORS[...]`, a bare `KeyError` if
  a transcript ever named an exception class outside the three currently
  recordable — unreachable from `CallRecorder` today, but a hand-edited or
  future-format transcript would raise a `KeyError` rather than a message
  naming what happened.

- **Stage 0's verified window is honest but systematically pessimistic — roughly
  half the model's real capability.** C1's fix made stage 0 report the server's own
  `tokens_in` for the largest accepted probe, which removed a 2× overclaim. But the
  probe is still *generated* from a fixed 3-chars-per-token estimate while its
  `"token "` filler tokenizes at about 6, so a probe aimed at N tokens carries
  roughly 0.55N. Measured on the committed fixtures: codegemma's `plan.window` is
  8192 and stage 0 verifies **4528**; granite's is 4096 and it verifies **2262**.

  Nothing false is claimed — those token counts really were accepted — but the
  reported `usable_window` is bounded by *how densely the probe packs tokens*, not
  by what the model can hold. A reader, and the loop that configures itself from
  this field, will under-use the window by ~45%.

  The remedy is in the plan's own measured fact 2, one step further than it was
  taken: the server's count is feedback, so the probe should grow until its
  *reported* `tokens_in` reaches the target rather than being sized once from a
  character estimate. Note the daemon threshold above interacts with this — on this
  box a probe cannot exceed ~11.5k tokens at all, so a 32768-token window is not
  verifiable here regardless of filler density, and any fix must report that as
  "could not verify to the plan's window" rather than silently settling lower.

## Found during the fix wave, fixed on reopen (ruled 2026-08-10, second round)

- **`OllamaClient.generate` trusted `prompt_eval_count`/`eval_count` without
  checking they were present.** First surfaced while re-recording this fix
  wave's transcripts (`usable_window: 0` on `qwen7b.jsonl`/`granite8b.jsonl`,
  originally written up here as an unfixed concern). Reopened when the
  coordinator could not reproduce the claim at `num_ctx: 8192` and asked for
  it to be re-verified rather than left asserted. Re-verification found the
  earlier characterization was WRONG in one respect and right in another:
  the daemon response (a 200 with valid `content` but no `prompt_eval_count`/
  `eval_count`/`done_reason` at all, `"done": false`) is real and
  reproducible, but it is **not** simply "these two models, this daemon" --
  it is a sharp, size-dependent threshold. Measured against
  `qwen2.5-coder:7b-instruct-q8_0` at seed 0: targets up to ~11500 succeeded
  reliably (multiple repeats, 4/4 or better each), targets at ~11800 and
  above failed 100% (40/40 across two separate batches) -- and this model's
  REAL `plan.window` (32768, its training context; VRAM is not the binding
  limit on this box) sits far past that threshold, so its immediate
  full-window probe cannot currently be recorded as a successful measurement
  at all. `granite-code:8b`'s real window (4096) sits well under its own
  threshold and was merely flaky (one failure in several dozen calls,
  resolved by retrying) -- a different, milder manifestation of the same
  underlying defect.

  **Fixed**, per the coordinator's explicit direction (not carried): (1)
  `OllamaClient.generate` now raises `ModelError` naming the model and which
  field(s) are missing instead of defaulting either count to 0 --
  `.get(key, 0)` standing in for a measurement is the identical shape as
  plan 02's `.get("size", 0)` bug. (2) `stage0_window` independently treats
  an accepted call whose `Generation.tokens_in <= 0` the same as a rejected
  one, so `verified=True` at `window=0` -- incoherent on its face -- can
  never be returned regardless of what any client does. Both mutation-tested
  (`tests/test_client.py`, `tests/test_stage0.py`).

  **Consequence for the committed transcripts**: `codegemma7b.jsonl` and
  `granite8b.jsonl` are both re-recorded and clean (every "reply" row has
  `tokens_in > 0`, pinned by `test_no_committed_row_encodes_an_unmeasured_
  reply`). `qwen7b.jsonl` could NOT be re-recorded as a full run — every
  attempt at its real training-context window fails before stage 0 even
  finishes, now as a loud, immediate `ModelError` rather than a silent
  false zero. Its committed shape is deliberately one row (`outcome:
  "error"`), the honest artifact of what happens right now, pinned by
  `test_qwen_transcript_documents_a_real_unmeasurable_probe` (replay
  reproduces the exact same error). The underlying daemon defect itself
  remains unfixed — fixing Ollama, or working around it by changing the
  fixed `_PROBE_SEED` or the CLI's fixed `num_predict=1024`, is out of this
  wave's scope either way.
