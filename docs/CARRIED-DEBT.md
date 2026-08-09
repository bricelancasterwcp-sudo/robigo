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
