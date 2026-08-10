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

**Invariant 4 — the prompt sent always fits.** For every turn, the prompt actually passed to `client.generate` satisfies

    estimate_tokens(prompt) + client.num_predict <= client.window

This is the guarantee the whole layer exists for, and it is checkable at the point of use. `fit()` picks the rung by arithmetic; the loop then verifies the assembled result. If a rendered prompt violates this, the accounting is wrong — surface it as the refusal path with a message naming both numbers. **Never send it anyway.** A silent overrun here is the exact failure the ladder was built to prevent, and the only place it can be caught unconditionally.

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

## Done when

- `pytest -q` green, and green with `socket.socket.connect` patched to raise.
- A tight window degrades the scope and records which rung it used.
- An impossible window refuses before turn 1, exit 3, with the arithmetic printed and no model call made.
- The same failure mid-run is `budget_exhausted`, exit 2, with the branch preserved.
- Every prompt sent satisfies `estimate_tokens(prompt) + num_predict <= window`.
- A real CLI run against a small local model completes a repair with `--window` set low enough to force degradation.
