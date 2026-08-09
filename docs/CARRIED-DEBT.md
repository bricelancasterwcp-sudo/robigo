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
