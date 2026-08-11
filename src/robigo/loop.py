# src/robigo/loop.py
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from robigo.action.codec import PatchError
from robigo.action.verbs import ActionParseError, parse
from robigo.adapters.base import Adapter, AdapterError, Diagnostic
from robigo.apply.patch import apply_patch
from robigo.apply.safety import (
    RefusedError,
    commit_all,
    current_branch,
    ensure_repo,
    head_sha,
    is_dirty,
    refuse_ignored,
    snapshot,
    start_branch,
)
from robigo.context.budget import MAX_STEP, BudgetExhausted, estimate_tokens, fit, measure
from robigo.context.render import Turn, render
from robigo.context.scope import Scope, ScopeError, explicit, resolve
from robigo.model.client import ContextOverflowError, Generation, ModelClient, ModelError
from robigo.paths import OutsideRepo, contain
from robigo.record import RunRecorder, new_recorder, slug

_READ_CAP = 4000
_HISTORY_TURNS = 2
"""How many recent turns `history` keeps below -- see the `[-_HISTORY_TURNS:]`
slice. Named rather than left as the literal `2` in that slice: `context.
budget.Budget`'s default `history` reserve is derived from this constant and
`_READ_CAP` together (ruled 2026-08-09), so a change to either number here
must not be able to silently under-reserve there."""
_SKIP = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__", ".robigo"})

OUTCOMES: dict[str, int] = {
    "pass": 0,
    "stalled": 1,
    "budget_exhausted": 2,
    "refused": 3,
    "infrastructure": 4,
}


@dataclass(frozen=True)
class UndoInfo:
    """What a user needs to get back to where they started. `git checkout -`
    is not enough: it relies on a reflog another shell can clobber, and it
    silently drops a dirty tree into the snapshot commit."""

    original_branch: str | None
    snapshot_sha: str | None
    started_dirty: bool


@dataclass(frozen=True)
class RunResult:
    outcome: str
    turns: int
    exit_code: int
    branch: str | None
    detail: str
    undo: UndoInfo | None = None
    rungs: tuple[int, ...] = ()
    """The ladder rung (1-4) EVERY turn's prompt actually used, one entry
    per turn that actually got a prompt rendered and sent, in order --
    NOT a scalar (whole-branch review finding 3, ruled 2026-08-09). A
    single "last rung" cannot tell a run that silently degraded
    (`[1, 2, 3, 1]`) apart from one that never left rung 1 (`[1]`
    repeated): only the sequence can. `len(rungs) == turns` holds for
    every outcome, including a turn-1 `BudgetExhausted` refusal, where
    both are 0 -- nothing was ever rendered. Plan 03's profiler is this
    field's only consumer, and can derive the last, the worst, or the
    shape of the degradation from the sequence itself."""
    repeats: int = 0
    """How many turns re-emitted an (action, reply) pair this run had
    already tried -- a TOTAL, not the consecutive streak `stall_cap`
    watches. `stalls` resets to 0 on any non-repeat, so a run cycling
    A, B, A, C, A never trips the stall cap (its longest consecutive
    repeat streak is a single turn: turn 3 following turn 1, and turn 5
    following either) even though it re-emits an already-tried pair
    TWICE across the whole run -- turn 3 repeats turn 1's pair, and turn
    5 repeats it again. A `stalls`-only view reports neither of those;
    it only ever reports "0 consecutive", identical to a run that never
    repeated at all. `repeats` is the total Stage 5's identical-
    failing-patch metric (`repeat_rate` in `robigo.profile.discipline`)
    needs -- Task 6 verified this exact sequence directly
    (`test_repeats_counts_every_repeat_not_just_consecutive_ones`) rather
    than trusting the count by inspection: it is easy to conflate "the
    same reply appeared 3 times" with "3 repeats" when only 2 of those 3
    appearances followed a prior one. The stall cap is a different
    question -- "has progress visibly stopped right now" -- and keeps its
    own, separate counter; this field never feeds back into that
    decision."""


def _result(
    outcome: str,
    turns: int,
    branch: str | None,
    detail: str,
    undo: UndoInfo | None = None,
    rungs: tuple[int, ...] = (),
    repeats: int = 0,
) -> RunResult:
    return RunResult(
        outcome, turns, OUTCOMES[outcome], branch, detail, undo, rungs, repeats
    )


def run(
    task: str,
    root: Path,
    client: ModelClient,
    adapter: Adapter,
    *,
    codec: str,
    turn_cap: int = 8,
    allow_test_edits: bool = False,
    use_git: bool = True,
    stall_cap: int = 3,
    scope_paths: Sequence[Path] | None = None,
    recorder: RunRecorder | None = None,
) -> RunResult:
    """Wrapper only. `_execute` is the loop; this exists so `finish` runs on
    every return path without touching any of them."""
    if recorder is None:
        recorder = new_recorder(root, task)
    model, window = getattr(client, "model", "?"), getattr(client, "window", 0)
    try:
        result = _execute(
            task, root, client, adapter, codec=codec, turn_cap=turn_cap,
            allow_test_edits=allow_test_edits, use_git=use_git,
            stall_cap=stall_cap, scope_paths=scope_paths, recorder=recorder,
        )
    except BaseException as exc:
        # An escaping exception is infrastructure, never a model result, and
        # the record must exist either way. Re-raised so the traceback
        # survives for debugging; `cli.main` is what turns it into exit 4.
        result = _result("infrastructure", 0, None, f"internal error: {exc!r}")
        recorder.finish(result, model=model, window=window, codec=codec)
        raise
    recorder.finish(result, model=model, window=window, codec=codec)
    return result


def _execute(
    task: str,
    root: Path,
    client: ModelClient,
    adapter: Adapter,
    *,
    codec: str,
    turn_cap: int = 8,
    allow_test_edits: bool = False,
    use_git: bool = True,
    stall_cap: int = 3,
    scope_paths: Sequence[Path] | None = None,
    recorder: RunRecorder,
) -> RunResult:
    branch: str | None = None
    undo: UndoInfo | None = None
    try:
        if use_git:
            ensure_repo(root)
        diag = adapter.run(root, None)
        if diag.passed:
            raise RefusedError(
                "the suite already passes, so there is no failing test to "
                "anchor on. Write the failing test first: that is the "
                "interface."
            )
        scope = (
            explicit(diag, root, scope_paths) if scope_paths
            else resolve(diag, adapter, root)
        )
        if use_git:
            # Checked before a branch exists: an ignored scope file is a
            # refusal, not an infrastructure failure, and it must not leave
            # a `robigo/*` branch behind for a run that never really started.
            refuse_ignored(root, scope.full)
            # Captured BEFORE the branch exists, because that is the state
            # the undo recipe has to name. `git checkout -` cannot: it reads
            # a reflog any other shell can clobber, and it says nothing about
            # a dirty tree that snapshot is about to fold into a commit.
            original, dirty = current_branch(root), is_dirty(root)
            branch = start_branch(root, slug(task))
            snapshot(root, "robigo: snapshot before first patch")
            undo = UndoInfo(original, head_sha(root), dirty)
    except (RefusedError, ScopeError) as exc:
        return _result("refused", 0, branch, str(exc), undo)
    except (ModelError, AdapterError) as exc:
        # AdapterError means the project's tests cannot be run at all --
        # infrastructure, never a model result (Task 4's amendment).
        return _result("infrastructure", 0, branch, str(exc), undo)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        # git can be missing from PATH entirely, or any of the git helpers
        # above (ensure_repo, start_branch, snapshot) can fail -- an
        # infrastructure problem, never a model result. `branch` may already
        # be set if the failure came from `snapshot`, and that must be
        # reported honestly rather than claimed as None.
        return _result("infrastructure", 0, branch, f"git failed: {exc}", undo)

    window = getattr(client, "window", 0)
    # `num_predict` IS on the `ModelClient` Protocol (whole-branch review
    # finding 5, ruled 2026-08-09) -- read directly, not defensively. The
    # earlier `getattr(client, "num_predict", 0)` let a client conforming
    # to the Protocol's declared surface but missing this attribute
    # silently reserve 0 output tokens, formally satisfying invariant 4
    # while leaving no room for a reply at all.
    reserve_out = client.num_predict
    history: tuple[Turn, ...] = ()
    seen: set[str] = set()
    stalls = 0
    repeats = 0
    rungs: tuple[int, ...] = ()
    for turn in range(1, turn_cap + 1):
        try:
            prompt, rung = _select_rung(
                scope, diag, history, codec, root, window, reserve_out
            )
        except BudgetExhausted as exc:
            # Same evidence gate as the ContextOverflowError branch below,
            # for the same reason (invariant 5): with at least one attempt
            # already made this is a session RESULT, with none there is
            # nothing to preserve. Unlike that branch, NOTHING was
            # generated for the current turn -- `_select_rung` raised
            # before `client.generate` was ever called -- so this turn
            # does not count: `turn - 1`, not `turn` (whole-branch review
            # finding 2, ruled 2026-08-09). `rungs` is passed as
            # accumulated so far, unextended, for the same reason.
            outcome = "budget_exhausted" if turn > 1 else "refused"
            return _result(
                outcome, turn - 1, branch, str(exc), undo, rungs=rungs, repeats=repeats
            )
        rungs = rungs + (rung,)
        try:
            gen = client.generate(prompt, seed=turn)
        except ContextOverflowError as exc:
            # Law 3, the evidence gate: with at least one attempt already
            # made this is a session RESULT and the work so far stands;
            # with none, there is nothing to preserve and it is a refusal.
            # Which check caught it does not matter -- only whether
            # evidence exists. Unlike `BudgetExhausted` above, a request
            # really was sent this turn, so it counts: `turn`, not
            # `turn - 1`.
            recorder.turn(prompt, f"<no reply: {exc}>", diag.raw)
            outcome = "budget_exhausted" if turn > 1 else "refused"
            return _result(
                outcome, turn, branch, str(exc), undo, rungs=rungs, repeats=repeats
            )
        except ModelError as exc:
            recorder.turn(prompt, f"<no reply: {exc}>", diag.raw)
            return _result(
                "infrastructure", turn, branch, str(exc), undo, rungs=rungs,
                repeats=repeats,
            )

        action_text, result_text, applied, target = _take_turn(
            gen, root, scope, adapter, codec, allow_test_edits
        )
        recorder.turn(prompt, gen.text, diag.raw)
        history = (history + (Turn(action_text, result_text),))[-_HISTORY_TURNS:]

        if applied:
            try:
                # Only the path actually written is staged. Staging the
                # whole tree would fold a concurrent hand-edit into a
                # commit titled as the model's patch; a bare `run` action
                # writes nothing, so it commits nothing.
                if target is not None:
                    commit_all(root, f"robigo: {action_text}", [target])
                diag = adapter.run(root, None)
            except AdapterError as exc:
                # Not git's fault. Blaming git for a vanished pytest sent
                # every reader of the record looking in the wrong place.
                return _result(
                    "infrastructure", turn, branch, str(exc), undo, rungs=rungs,
                    repeats=repeats,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                return _result(
                    "infrastructure", turn, branch, f"git failed: {exc}", undo,
                    rungs=rungs, repeats=repeats,
                )
            if diag.passed:
                return _result(
                    "pass", turn, branch, "tests pass", undo, rungs=rungs,
                    repeats=repeats,
                )
            # Mid-loop re-resolution can fail where the first one could not:
            # a timed-out or unanchorable run returns file=None, and resolve
            # refuses that. Keep the scope we already have and let the model
            # see the new diagnostic — aborting here would throw away a
            # recoverable turn (and, unguarded, crash out of the loop).
            #
            # The ignored-file check is re-run, not just done at setup: a new
            # import trace can pull in a gitignored `.py`, which `check_target`
            # would admit, `apply_patch` would write, and `commit_all` would
            # then fail to stage -- an unrecoverable edit to a user file with
            # no pre-image. The new scope is adopted only once it is accepted.
            try:
                new_scope = resolve(diag, adapter, root)
                if use_git:
                    refuse_ignored(root, new_scope.full)
                scope = new_scope
            except ScopeError:
                pass
            except RefusedError as exc:
                return _result(
                    "refused", turn, branch, str(exc), undo, rungs=rungs,
                    repeats=repeats,
                )

        key = f"{action_text}\n{gen.text}"
        # `repeats` is a TOTAL across the whole run, deliberately tracked
        # separately from `stalls`: `stalls` resets to 0 on any non-repeat
        # turn (see `RunResult.repeats`'s docstring for why that makes it
        # blind to a run that cycles through several distinct dead ends,
        # re-trying each one). `seen` already accumulates every key this
        # run has ever produced, so `key in seen` here answers "has THIS
        # exact (action, reply) pair been tried before, at any point, not
        # just last turn" -- exactly what `repeats` needs and `stalls`
        # cannot answer.
        if key in seen:
            repeats += 1
        stalls = stalls + 1 if key in seen else 0
        seen.add(key)
        if stalls >= stall_cap - 1:
            return _result(
                "stalled", turn, branch, "no progress; repeating", undo, rungs=rungs,
                repeats=repeats,
            )

    return _result(
        "stalled", turn_cap, branch, f"turn cap {turn_cap} reached", undo, rungs=rungs,
        repeats=repeats,
    )


def _select_rung(
    scope: Scope,
    diag: Diagnostic,
    history: tuple[Turn, ...],
    codec: str,
    root: Path,
    window: int,
    reserve_out: int,
) -> tuple[str, int]:
    """The prompt to send this turn, and the rung it was rendered at.
    Measurement is the authority in BOTH directions, not just the accept
    path (whole-branch review finding 1, ruled 2026-08-09, correcting the
    task-2 amendment this docstring used to describe). The ladder is
    walked in its fixed order, rung 1 upward, RENDERING each candidate for
    real and taking the first whose rendered prompt measures as fitting:
    `estimate_tokens(prompt) + reserve_out <= window`. This yields the
    largest rung that actually fits, by construction, rather than by
    correcting an estimate that can be wrong in either direction.

    `fit`'s own seated `Budget` (via `measure`, against the ORIGINAL,
    undegraded scope) is an estimate of an estimate: `estimate_tokens` is
    `int(len/CHARS_PER_TOKEN) + 1`, not additive, so seating
    system/diagnostic/history once against the undegraded scope and then
    comparing a DIFFERENT candidate's own length against that one seating
    can disagree with the candidate's real rendered length by +/-1 token
    -- in EITHER direction. That is what the task-2 amendment's
    step-down-only correction missed: it is not only that arithmetic can
    ACCEPT a rung that does not really fit (needing a step down), it can
    also REFUSE a rung that really does fit (reproduced at window 608,
    reserve 64: rung 4's real prompt is 544, `544 + 64 == 608` fits
    exactly, while the seated arithmetic said `291 > 290`) -- or propose a
    rung lower than necessary, dropping a file's body for a 1-token
    artifact a step-down-only search can never step back up from. Walking
    every rung by real measurement, ascending, has neither failure mode:
    it never trusts `fit`'s accept/reject verdict for either direction.

    `fit` keeps its own tests and stays the module's arithmetic API -- it
    is asked, below, ONLY for the refusal's arithmetic once real
    measurement has already found no rung fits. Its own verdict is not
    trusted even there: if it still returns instead of raising (its
    narrower, single-seating view disagreeing with what real measurement,
    rendering the actual candidate, already established), this refuses
    anyway rather than send what measurement rejected. Rendering a
    candidate is cheap (its files are already read) and there are at most
    four, so this costs nothing to do honestly."""
    budget = measure(scope, diag, history, codec, root, window, reserve_out)
    smallest_prompt = ""
    for step in range(1, MAX_STEP):
        candidate = scope.degrade(step)
        prompt = render(candidate, diag, history, codec, root)
        if estimate_tokens(prompt) + reserve_out <= window:
            return prompt, step
        smallest_prompt = prompt
    try:
        fit(scope, budget, root)
    except BudgetExhausted:
        raise
    # `fit` disagreed with real measurement and returned instead of
    # raising -- measurement is the authority; refuse anyway, with the
    # arithmetic built from what was actually just measured rather than
    # `fit`'s message (which this path proves cannot be trusted here).
    over = estimate_tokens(smallest_prompt) + reserve_out - window
    raise BudgetExhausted(
        f"scope cannot fit the window: even rung {MAX_STEP - 1} (of "
        f"{MAX_STEP - 1}), the smallest, renders a prompt costing "
        f"{estimate_tokens(smallest_prompt)} tokens, which plus the "
        f"{reserve_out}-token output reserve exceeds the "
        f"{window}-token window by {over}.\n"
        f"  window {window}   reserve {reserve_out}   "
        f"system {budget.system}   diagnostic {budget.diagnostic}"
        f"   history {budget.history}\n"
        f"Narrow it with --scope, or use a model with a larger "
        f"window."
    )


def _take_turn(
    gen: Generation,
    root: Path,
    scope: Scope,
    adapter: Adapter,
    codec: str,
    allow_test_edits: bool,
) -> tuple[str, str, bool, Path | None]:
    """→ (action label, result text fed back, whether a file changed, the
    path written or None). Only the `patch` success path has a target;
    every other return -- including a bare `run` -- passes None."""
    try:
        action = parse(gen.text)
    except ActionParseError as exc:
        return "<unparseable>", f"ACTION PARSE FAILED\n{exc}", False, None

    label = f"{action.verb} {action.arg}".strip()
    if action.verb == "patch":
        if gen.truncated:
            # The likeliest real data loss in the design: a whole-file
            # emission cut at the cap would faithfully write a gutted
            # file (spec section 6).
            return label, (
                "REJECTED: your reply was cut off at the token limit, so "
                "the patch was not applied. Send a smaller edit."
            ), False, None
        try:
            target = apply_patch(action, root, scope, adapter, codec, allow_test_edits)
        except (PatchError, RefusedError) as exc:
            return label, f"PATCH REJECTED\n{exc}", False, None
        return label, "applied", True, target
    if action.verb == "read":
        return label, _read(root, action.arg), False, None
    if action.verb == "find":
        return label, _find(root, action.arg), False, None
    if action.verb == "done":
        return label, "the tests do not pass yet; keep going", False, None
    return label, "tests re-run", True, None


def _read(root: Path, arg: str) -> str:
    """Model-facing text, never an exception. A bare `read` and a `read` of a
    binary file are both ordinary model output, not crashes."""
    parts = arg.split()
    if not parts:
        return "read needs a path, e.g. `read src/fog.py`"
    missing = f"cannot read '{parts[0]}': no such file in this repository"
    try:
        path = contain(root, parts[0])
    except OutsideRepo:
        return missing
    if not path.is_file():
        return missing
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Strict, like every other surface. `errors="replace"` handed back
        # text with U+FFFD in it that can never match as a SEARCH block, so a
        # model that trusted it looped to the stall cap -- while `render`
        # called the same file unreadable and `patch` called it possibly
        # deleted, all in one turn.
        return (
            f"cannot read '{parts[0]}': not valid UTF-8, so robigo cannot "
            f"patch it"
        )
    except OSError as exc:
        return f"cannot read '{parts[0]}': {exc.strerror}"
    return text[:_READ_CAP] + "\n<truncated>\n" if len(text) > _READ_CAP else text


def _find(root: Path, symbol: str) -> str:
    if not symbol:
        return "find needs a symbol, e.g. `find computeRadius`"
    hits: list[str] = []
    for path in sorted(root.rglob("*.py")):
        # Relative parts only. Matched against the absolute path, a repo that
        # merely LIVES under a directory named `venv` or `node_modules` had
        # every one of its files skipped, so `find` answered "not found" for
        # every symbol in it.
        if _SKIP.intersection(path.relative_to(root).parts):
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(body.split("\n"), 1):
            if symbol in line:
                hits.append(f"{path.relative_to(root)}:{number}")
                if len(hits) >= 20:
                    return "\n".join(hits)
    return "\n".join(hits) or f"'{symbol}' not found"
