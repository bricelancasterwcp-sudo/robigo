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
    rung: int | None = None
    """The ladder rung (1-4) the most recently rendered prompt actually
    used, or None if no prompt was ever rendered (a turn-1 refusal before
    any fit). Invariant 7: history grows and the scope is re-resolved
    mid-loop, so the rung a run needs can differ turn to turn -- this is
    the rung the run last needed, so plan 03's profiler can tell a run
    that stayed at rung 1 apart from one that silently degraded to rung 4."""


def _result(
    outcome: str,
    turns: int,
    branch: str | None,
    detail: str,
    undo: UndoInfo | None = None,
    rung: int | None = None,
) -> RunResult:
    return RunResult(outcome, turns, OUTCOMES[outcome], branch, detail, undo, rung)


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
    reserve_out = getattr(client, "num_predict", 0)
    history: tuple[Turn, ...] = ()
    seen: set[str] = set()
    stalls = 0
    last_rung: int | None = None
    for turn in range(1, turn_cap + 1):
        try:
            prompt, rung = _select_rung(
                scope, diag, history, codec, root, window, reserve_out
            )
        except BudgetExhausted as exc:
            # Same evidence gate as the ContextOverflowError branch below,
            # for the same reason (invariant 5): with at least one attempt
            # already made this is a session RESULT, with none there is
            # nothing to preserve. `last_rung` carries whatever rung the
            # PREVIOUS turn actually used -- there is no rung for THIS
            # turn, since nothing fit it.
            outcome = "budget_exhausted" if turn > 1 else "refused"
            return _result(outcome, turn, branch, str(exc), undo, rung=last_rung)
        last_rung = rung
        try:
            gen = client.generate(prompt, seed=turn)
        except ContextOverflowError as exc:
            # Law 3, the evidence gate: with at least one attempt already
            # made this is a session RESULT and the work so far stands;
            # with none, there is nothing to preserve and it is a refusal.
            # Which check caught it does not matter -- only whether
            # evidence exists.
            recorder.turn(prompt, f"<no reply: {exc}>", diag.raw)
            outcome = "budget_exhausted" if turn > 1 else "refused"
            return _result(outcome, turn, branch, str(exc), undo, rung=last_rung)
        except ModelError as exc:
            recorder.turn(prompt, f"<no reply: {exc}>", diag.raw)
            return _result("infrastructure", turn, branch, str(exc), undo, rung=last_rung)

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
                return _result("infrastructure", turn, branch, str(exc), undo, rung=last_rung)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                return _result(
                    "infrastructure", turn, branch, f"git failed: {exc}", undo,
                    rung=last_rung,
                )
            if diag.passed:
                return _result("pass", turn, branch, "tests pass", undo, rung=last_rung)
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
                return _result("refused", turn, branch, str(exc), undo, rung=last_rung)

        key = f"{action_text}\n{gen.text}"
        stalls = stalls + 1 if key in seen else 0
        seen.add(key)
        if stalls >= stall_cap - 1:
            return _result(
                "stalled", turn, branch, "no progress; repeating", undo, rung=last_rung
            )

    return _result(
        "stalled", turn_cap, branch, f"turn cap {turn_cap} reached", undo, rung=last_rung
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
    Arithmetic proposes a rung; measurement decides the stopping point
    (task 2, invariant 4, amended 2026-08-09 from measurement). `fit`
    degrades the scope using its own seated `Budget` -- exact for the
    scope it was measured against, but `estimate_tokens` is
    `int(len/CHARS_PER_TOKEN) + 1`, not additive, so a DIFFERENT rung's
    actual rendered length can round differently by +/-1 token. Measured
    at ~5% of cases, always in the direction that matters: the seated
    arithmetic says a rung fits when its real rendered prompt is one
    token over.

    So the candidate `fit` proposes is rendered here and checked against
    the real window; a rung that fails the check steps down to the next
    one and is checked again, refusing only once rung `MAX_STEP - 1` --
    the smallest -- still does not fit. Rendering a candidate is cheap
    (its files are already read) and there are at most four, so this
    never re-measures the seated system/diagnostic/history costs -- only
    `fit`'s single seated `Budget` is built, via `measure` -- against the
    ORIGINAL, undegraded scope; only the scope section changes rung to
    rung, which is exactly the piece this re-renders and re-checks for
    real."""
    budget = measure(scope, diag, history, codec, root, window, reserve_out)
    candidate, step = fit(scope, budget, root)
    while True:
        prompt = render(candidate, diag, history, codec, root)
        if estimate_tokens(prompt) + reserve_out <= window:
            return prompt, step
        if step >= MAX_STEP - 1:
            # `fit`'s own arithmetic already refuses BEFORE this point
            # whenever every rung fails ITS estimate; reaching here means
            # arithmetic accepted rung `step` but the real render did not,
            # and there is no rung further down the ladder to try.
            over = estimate_tokens(prompt) + reserve_out - window
            raise BudgetExhausted(
                f"scope cannot fit the window: even rung {step} (of "
                f"{MAX_STEP - 1}), the smallest, renders a prompt costing "
                f"{estimate_tokens(prompt)} tokens, which plus the "
                f"{reserve_out}-token output reserve exceeds the "
                f"{window}-token window by {over}.\n"
                f"  window {window}   reserve {reserve_out}   "
                f"system {budget.system}   diagnostic {budget.diagnostic}"
                f"   history {budget.history}\n"
                f"Narrow it with --scope, or use a model with a larger "
                f"window."
            )
        step += 1
        candidate = scope.degrade(step)


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
