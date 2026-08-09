# src/robigo/loop.py
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from robigo.action.codec import PatchError
from robigo.action.verbs import ActionParseError, parse
from robigo.adapters.base import Adapter, AdapterError
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
from robigo.context.render import Turn, render
from robigo.context.scope import Scope, ScopeError, explicit, resolve
from robigo.model.client import ContextOverflowError, Generation, ModelClient, ModelError
from robigo.paths import OutsideRepo, contain
from robigo.record import RunRecorder, new_recorder, slug

_READ_CAP = 4000
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


def _result(
    outcome: str,
    turns: int,
    branch: str | None,
    detail: str,
    undo: UndoInfo | None = None,
) -> RunResult:
    return RunResult(outcome, turns, OUTCOMES[outcome], branch, detail, undo)


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

    history: tuple[Turn, ...] = ()
    seen: set[str] = set()
    stalls = 0
    for turn in range(1, turn_cap + 1):
        prompt = render(scope, diag, history, codec, root)
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
            return _result(outcome, turn, branch, str(exc), undo)
        except ModelError as exc:
            recorder.turn(prompt, f"<no reply: {exc}>", diag.raw)
            return _result("infrastructure", turn, branch, str(exc), undo)

        action_text, result_text, applied, target = _take_turn(
            gen, root, scope, adapter, codec, allow_test_edits
        )
        recorder.turn(prompt, gen.text, diag.raw)
        history = (history + (Turn(action_text, result_text),))[-2:]

        if applied:
            try:
                # Only the path actually written is staged. Staging the
                # whole tree would fold a concurrent hand-edit into a
                # commit titled as the model's patch; a bare `run` action
                # writes nothing, so it commits nothing.
                if target is not None:
                    commit_all(root, f"robigo: {action_text}", [target])
                diag = adapter.run(root, None)
            except (subprocess.CalledProcessError, FileNotFoundError, AdapterError) as exc:
                return _result("infrastructure", turn, branch, f"git failed: {exc}", undo)
            if diag.passed:
                return _result("pass", turn, branch, "tests pass", undo)
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
                return _result("refused", turn, branch, str(exc), undo)

        key = f"{action_text}\n{gen.text}"
        stalls = stalls + 1 if key in seen else 0
        seen.add(key)
        if stalls >= stall_cap - 1:
            return _result("stalled", turn, branch, "no progress; repeating", undo)

    return _result("stalled", turn_cap, branch, f"turn cap {turn_cap} reached", undo)


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
        text = path.read_text(encoding="utf-8", errors="replace")
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
