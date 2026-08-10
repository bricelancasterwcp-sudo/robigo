# Wire the Degradation Ladder Into the Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make plan 02's five-rung degradation ladder reachable at runtime, so a prompt that does not fit is shrunk until it does — or refused before turn 1 with the arithmetic printed — instead of being sent and truncated.

**Architecture:** `fit()` runs once per turn inside `_execute`, immediately before `render`, because `history` grows each turn and the scope is re-resolved mid-loop. The `Budget` is assembled from values the loop already has: `client.window`, `client.num_predict`, and the actual `Turn` tuple and `Diagnostic` about to be rendered. Nothing new is threaded through `run()`'s signature.

**Tech Stack:** Python 3.12+, standard library only.

## Global Constraints

- **Runtime dependencies: none.** Standard library only.
- `requires-python = ">=3.12"`; `from __future__ import annotations` in every module. Type annotations on every **non-test** function signature; pytest test functions are exempt.
- **Degradation order is fixed, not heuristic** (spec §3): 1 hop-2 signatures only, 2 hop-2 dropped, 3 hop-1 to signatures, 4 anchor windowed, 5 refuse. **History is not a rung** — ruled 2026-08-09. If the prompt does not fit after all five scope rungs, the run refuses. Do not add a history-shedding step.
- **A prompt that cannot fit is a refusal before turn 1 with the arithmetic printed — never a truncated attempt.**
- **Every test must pass with no daemon, no GPU and no network.** Prove it by running the suite with `socket.socket.connect` patched to raise, not by grep.
- Commit messages: `<type>: <subject>`, single line, no body, no trailers.

## Context you need before starting

Read `docs/CARRIED-DEBT.md`'s plan-02 section, in particular "For the ladder-wiring slice specifically". Three traps are recorded there. Also note two corrections to earlier belief:

- **`budget_exhausted` is already triggered**, at `loop.py:186`, for a server-reported `ContextOverflowError` when `turn > 1`. It is `fit()`'s `BudgetExhausted` that is unwired, not the outcome. That existing line is the convention this slice mirrors.
- **`client.window` and `client.num_predict` are already reachable from the loop** — `loop.py:97` reads them for the recorder. `reserve_out` comes from `client.num_predict`, which resolves the three-unlinked-names problem without new plumbing.

## File Structure

| File | Responsibility |
|---|---|
| `src/robigo/context/render.py` | *modified* — expose the non-scope prompt parts as named helpers, the way `_scope_section` already is |
| `src/robigo/context/budget.py` | *modified* — seat the fixed costs from what `render` emits, rather than from constants |
| `src/robigo/loop.py` | *modified* — build a `Budget` per turn, `fit()` before `render`, map `BudgetExhausted` to an outcome |
| `tests/test_budget.py` | *modified* — the accounting invariant |
| `tests/test_loop_budget.py` | new — the wiring, the outcome mapping, and the end-to-end fit |

---

### Task 1: Seat the fixed costs from what `render` actually emits

**Files:**
- Modify: `src/robigo/context/render.py`
- Modify: `src/robigo/context/budget.py`
- Test: `tests/test_budget.py`

**Interfaces:**
- Consumes: `render`'s existing `SYSTEM`, `_CODEC_HELP`, `_scope_section`, `Turn`, `Diagnostic`.
- Produces: helpers on `render` for the non-scope sections, and a way to build a `Budget` whose seated costs are measured rather than assumed. Name them as the module's existing `_scope_section` is named; the exact names are yours.

`render` assembles exactly these parts, joined with `"\n"`: `SYSTEM`, `_CODEC_HELP[codec]`, `""`, the scope section, `"--- failing test ---"`, the `where + message` line, `""`, one block per history turn, and the trailer `"Your action:"`.

`Budget` seats only `system`, `diagnostic` and `history`. **The trailer and the joining newlines are seated by nobody.** That is the same defect class the whole-branch review found when `_cost` ignored the per-file headers — small, and in the direction that says "fits" when it does not.

**Invariant 1 — the parts account for the whole.** For any scope, diagnostic, history and codec:

    estimate_tokens(render(scope, diag, history, codec, root))
      == budget.system + budget.diagnostic + budget.history + _cost(scope, root)

when the `Budget` is built by the new derivation. Not "approximately" — exactly. Two independent ways of counting the same string is the shape that produced three separate defects in plan 02, and an equality is the only assertion that catches drift.

**Invariant 2 — the derivation reads one source.** The seated costs must be measured from the same helpers `render` calls. A second copy of the preamble, the diagnostic format or the history format is the defect restated; `render`'s `_scope_section`/`budget._section` delegation is the pattern to follow.

**Invariant 3 — measured beats assumed, and unmeasured is refused.** `SYSTEM_TOKENS = 350` and `DIAGNOSTIC_TOKENS = 600` were conservative guesses standing in for a measurement (real values: ~233 and 16–80). Once the real cost is measurable, the constants stop being the default. Keep them only as an explicitly-named fallback for a caller with no `diag`/`history` to measure, and say so where they are defined.

Falsification tests — each must fail when the invariant is broken:

- Invariant 1 across a matrix: undegraded and windowed scopes × zero, one and two history turns × both codecs. Assert equality, not a bound.
- A history turn holding a `_READ_CAP`-sized result is seated at its real cost, not at `DEFAULT_HISTORY_TOKENS`.
- Mutation: drop the trailer from the seated cost and confirm invariant 1's test goes RED. If it stays green the test is measuring the wrong thing.

**Steps:** write the invariant-1 test first and watch it fail against today's constants; then extract the helpers; then derive; then re-run. Commit.

---

### Task 2: Fit before rendering, every turn

**Files:**
- Modify: `src/robigo/loop.py`
- Test: `tests/test_loop_budget.py`

**Interfaces:**
- Consumes: `fit`, `Budget`, `BudgetExhausted` from `robigo.context.budget`; Task 1's derivation; `client.window`, `client.num_predict`.
- Produces: no signature changes to `run` or `_execute`.

**Invariant 4 — the prompt sent always fits, and measurement is the authority.** For every turn, the prompt actually passed to `client.generate` satisfies

    estimate_tokens(prompt) + client.num_predict <= client.window

**Amended before execution (2026-08-09), from measurement.** The original wording made this a tripwire — verify, and treat a violation as the accounting being broken. Measuring Task 1's decomposition shows a violation is *expected* at a small rate, so a tripwire would refuse runs that had room.

Task 1 seats `system`/`diagnostic`/`history` as telescoping deltas between successive prefixes of the rendered string, which is exact for the scope it measured. `fit()` then degrades the scope, and `estimate_tokens` is `int(len/3.3) + 1` — not additive — so each delta shifts with the new length's residue. Swept across 60 length variants × 4 rungs:

| delta between seated sum and actual rendered cost | samples |
|---|---|
| −1 (**under-counts** — says fits when it may not) | 12 |
| 0 (exact) | 164 |
| +1 (over-counts, safe) | 64 |

One token, 5% of the time, in the direction that matters. Re-measuring inside `fit` would need `diag`/`history`/`codec` threaded into its signature, which this plan forbids.

So: **arithmetic proposes, measurement decides.** `fit()` chooses a rung — it encodes the fixed ladder and is already tested. The loop then renders that rung and measures it against the real window. If it does not fit, step to the next rung down and measure again; if rung 4 does not fit, refuse. Rendering a candidate is cheap (its files are already read) and there are at most four.

This makes the ladder's decision procedure honest end to end without weakening the fixed order: the order still comes from the spec, only the *stopping point* is decided by measurement rather than by an estimate of an estimate. **Never send a prompt that fails the check.** A silent overrun is the exact failure the ladder was built to prevent, and this is the one place it can be caught unconditionally.

A test must pin the step-down specifically: construct a case where the seated arithmetic accepts a rung whose rendered prompt exceeds the window, and assert the loop sends the next rung down instead of either sending the too-large prompt or refusing.

**Invariant 5 — the outcome mirrors the existing evidence gate.** `loop.py:186` already distinguishes: a context overflow with at least one turn behind it is `budget_exhausted`, with none it is `refused`. `BudgetExhausted` from `fit()` takes the same mapping for the same reason — with evidence the work so far stands, without it there is nothing to preserve. Do not invent a third outcome.

**Invariant 6 — the refusal prints the arithmetic.** `BudgetExhausted`'s message already carries the window, the seated terms and the smallest scope's cost. It must reach the user, not be replaced by a summary.

**Invariant 7 — fitting is per turn, and the degradation is recorded.** History grows and the scope is re-resolved mid-loop, so the rung can differ between turns. The rung actually used must reach the run record; a run that silently degraded to rung 4 and one that ran at rung 1 are not the same run, and the profiler in plan 03 needs to tell them apart.

Falsification tests:

- A window generous enough for rung 1 renders the full scope; assert the rung recorded is 1.
- A window that only fits a degraded scope produces a *shorter* prompt and records the higher rung — assert on the rung, not merely on length.
- A window too small for rung 5 refuses at turn 1 with outcome `refused`, exit 3, zero calls to `client.generate`. Assert the client was never called; "it refused" and "it refused before generating" are different claims.
- The same refusal after a successful turn yields `budget_exhausted`, exit 2, and preserves the branch and undo info.
- Invariant 4 holds for every prompt in a multi-turn run, asserted inside a fake client that checks each prompt it receives.
- Mutation: remove the `fit()` call and confirm the tight-window tests go RED rather than merely rendering more.

**Steps:** tests first, confirm each fails for the stated reason, then wire, then re-run. Run the CLI end to end against a real model afterwards — three defects in plan 02's last task were invisible to 239 passing tests and obvious in one invocation. Commit.

---

---

### Task 3: The window must not depend on whether a previous run left the model hot

**Files:**
- Modify: `src/robigo/model/detect.py`
- Test: `tests/test_window_auto.py`

**Interfaces:**
- Consumes: `/api/ps`, alongside the `/api/show` and `/api/tags` calls already there.
- Produces: no signature changes. `plan_window`'s result becomes independent of residency.

**Found by running the tool twice**, which is how the previous plan's last three defects were found too. `usable_window` subtracts `weights_bytes` from free VRAM, which is correct only when the weights are not already loaded — a precondition plan 02 wrote into its docstring and assigned to the CLI. The CLI does not honour it, and Ollama keeps a model resident for five minutes by default, so:

| | first run | second run, within 5 minutes |
|---|---|---|
| `nvidia-smi` free | 14,485 MiB | 4,849 MiB |
| this model resident (`/api/ps` `size_vram`) | 0 | 9,285 MiB |
| weights (`/api/tags`) | 7,723 MiB | 7,723 MiB |
| free − weights − 256 | 6,506 MiB | **−3,130 → clamped to 0** |
| reported window | 32768 (`training_ctx`) | **0 (`vram`) → refuses** |

So `robigo` works, and then refuses when run again on the same model — the second invocation being the more likely one, since the first turn rarely lands the repair. `size_vram` exceeds `weights` because it includes the previous run's KV cache, which is also legitimately available for reuse.

**Invariant 8 — window resolution is idempotent across consecutive runs.** Two runs of the same model on an otherwise-idle machine resolve to the same window and the same `limited_by`, whether or not the first left the model resident. What is available to a model includes whatever that model is already holding.

**Invariant 9 — still measured, never guessed.** Take the resident figure from `/api/ps`, matching by name the way `weights_bytes` matches against `/api/tags` (exact name, then `:latest`). A model that is not resident contributes nothing. If `/api/ps` cannot be read or its shape is wrong, that is the same malformed-response case as its two sibling endpoints: raise `GeometryError`, never assume a number. **Do not** substitute a default — a wrong residency figure moves the window in the direction that overcommits.

Falsification tests, all offline with injected responders:

- Two `plan_window` calls, the second with `/api/ps` reporting the model resident at a plausible `size_vram` and free VRAM reduced by that much, return the identical `WindowPlan`. This is invariant 8 stated directly; it must fail if the residency term is dropped.
- A *different* model being resident does not inflate this model's window — only the requested model's own residency is added back.
- `/api/ps` returning `{"models": null}`, a list, a string, or a non-JSON body each raise `GeometryError`, matching the guards its sibling endpoints already carry.
- Mutation: drop the residency term and confirm the idempotence test goes RED with the `window 0` shape from the table above.

Then run the CLI **twice in a row** against a real model and confirm both runs report the same window.

---

## Done when

- `pytest -q` green, and green with `socket.socket.connect` patched to raise.
- A tight window degrades the scope and records which rung it used.
- An impossible window refuses before turn 1, exit 3, with the arithmetic printed and no model call made.
- The same failure mid-run is `budget_exhausted`, exit 2, with the branch preserved.
- Every prompt sent satisfies `estimate_tokens(prompt) + num_predict <= window`.
- Two consecutive CLI runs against the same model report the same window.
- A real CLI run with `--window` low enough to force degradation sends a prompt at a degraded rung, records that rung, and reaches the model — **corrected 2026-08-09**: this bullet previously required the run to *complete a repair*, which is a property of the model, not of robigo. A 1.5b–7b local model frequently fails a repair for reasons visible in its own output (malformed action, wrong file, mismatched SEARCH block), and no amount of correct wiring changes that. What this slice must demonstrate is that degradation happens, is recorded, and produces a prompt the server accepts.

---

## Whole-branch review fix wave (ruled 2026-08-09)

Five Importants, all reproduced end to end by the review. Finding 1 is a defect
in my own amendment and it changes the design; the rest are honesty fixes in
records and messages.

**1. Make measurement the authority in BOTH directions.** The amendment above
said "arithmetic proposes, measurement decides" and then wired measurement into
the *accept* path only. `fit()`'s own arithmetic-only `BudgetExhausted`
propagates unguarded, so a run whose smallest rung really fits is refused —
reproduced at window 608 / reserve 64, where rung 4's real rendered prompt costs
544 and `544 + 64 == 608` fits exactly, while the seated arithmetic said
`291 > 290`. One token. The same asymmetry has a milder twin (review Minor 6):
because the seated terms were measured against the *undegraded* scope, `fit` can
also propose a rung **lower** than necessary, dropping a file's body for a
1-token artifact, and the step-down never steps back up.

Both directions have one root cause: the stopping point is chosen by an estimate
of an estimate, then corrected in one direction only. So stop correcting and
choose it by measurement:

*Invariant 4, restated.* Select the rung by walking the ladder in its fixed
order — rung 1 upward — rendering each candidate and taking **the first whose
rendered prompt measures as fitting**: `estimate_tokens(prompt) + reserve_out <=
window`. If none of rungs 1 through `MAX_STEP - 1` fits, refuse. This yields the
largest rung that actually fits, makes the invariant true by construction rather
than by correction, and removes both the false refusal and the needless
over-degradation. At most four renders per turn, of files already read.

`fit` keeps its tests and stays the module's arithmetic API — it is what
produces the refusal's arithmetic for the message. It is simply no longer the
authority on where the ladder stops. Say that in `_select_rung`'s docstring, and
correct the docstring's current claim that it refuses "only once rung
`MAX_STEP - 1` … still does not fit", which is what this finding falsified.

**2. A refusal in which nothing was generated must record zero turns.** The
`BudgetExhausted` arm passes `turn` (=1) where every other pre-generation
refusal passes `0`. Reproduced: `outcome=refused turns=1 rung=None`, zero
`generate` calls, and a run directory holding no turn transcript at all. The
plan's own criterion says "refuses **before turn 1**". Plan 03 will read `turns`
beside `rung`, and `turns=1, rung=null, 0 transcripts` is three claims that
contradict each other. The `ContextOverflowError` arm is different and correct —
a request really was sent there.

**3. Record the rung sequence, not the last rung.** `record.py`'s comment says
the field lets a run that silently degraded be told apart from one that never
left rung 1. It cannot: only the last turn's rung survives. Reproduced over a
real two-hop repo with per-turn rungs `[1, 2, 3, 1]` — recorded as `1`,
identical to a run that never degraded. Replace the scalar with the **per-turn
sequence**; it is strictly more information, it is unambiguous, and plan 03 —
the field's only consumer, and the next slice — can derive the last, the worst,
or the shape of the degradation from it. Update the comment to describe what is
actually stored.

**4. Do not print a computed figure under a measurement's name.**
`WindowPlan.free_vram` now carries free-plus-this-model's-residency, and the
window-0 refusal prints it as `free {N} MiB` — a number `nvidia-smi` never
reported, followed by advice to "free VRAM" when part of the shortfall is the
model's own residency already credited back. Name it for what it is in the
message. Keep the field itself differing between a cold and a hot call: that
difference is what kills the double-credit mutation.

**5. Declare `num_predict` on the `ModelClient` Protocol.** The loop reads
`getattr(client, "num_predict", 0)`, which the Protocol does not declare, so a
conforming client silently gets `reserve_out = 0`. The review built one and sent
a prompt occupying 1730 of a 1730-token window with **nothing left for the
reply**, while invariant 4 read as satisfied. One line on the Protocol closes
it. Note `tests/test_window_auto.py`'s `_AbsentModel` is such a client today, so
the silent default is already load-bearing for a test — fix the test rather than
keeping the default.

Not in this wave, to be carried in `CARRIED-DEBT.md` instead: the duplicated
`:latest`-matching rule and refusal-advice sentence; `usable_window`'s docstring
still stating the precondition this branch removed; the now-unreachable
`SYSTEM_TOKENS`/`DIAGNOSTIC_TOKENS`/`_default_history_tokens` defaults; `/api/ps`
having no captured contract for its *resident* shape; and the two-reads race
between free VRAM and residency.
