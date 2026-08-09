# src/robigo/loop.py
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from robigo.action.codec import PatchError
from robigo.action.verbs import ActionParseError, parse
from robigo.adapters.base import Adapter, AdapterError
from robigo.apply.patch import apply_patch
from robigo.apply.safety import RefusedError, commit_all, ensure_repo, snapshot, start_branch
from robigo.context.render import Turn, render
from robigo.context.scope import ScopeError, resolve
from robigo.model.client import ContextOverflowError, ModelError

OUTCOMES: dict[str, int] = {
    "pass": 0,
    "stalled": 1,
    "budget_exhausted": 2,
    "refused": 3,
    "infrastructure": 4,
}


@dataclass(frozen=True)
class RunResult:
    outcome: str
    turns: int
    exit_code: int
    branch: str | None
    detail: str


def _result(outcome: str, turns: int, branch: str | None, detail: str) -> RunResult:
    return RunResult(outcome, turns, OUTCOMES[outcome], branch, detail)


def run(
    task: str,
    root: Path,
    client,
    adapter: Adapter,
    *,
    codec: str,
    turn_cap: int = 8,
    allow_test_edits: bool = False,
    use_git: bool = True,
    stall_cap: int = 3,
) -> RunResult:
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
        scope = resolve(diag, adapter, root)
        branch = None
        if use_git:
            branch = start_branch(root, _slug(task))
            snapshot(root, "robigo: snapshot before first patch", scope.full)
    except (RefusedError, ScopeError) as exc:
        return _result("refused", 0, None, str(exc))
    except (ModelError, AdapterError) as exc:
        # AdapterError means the project's tests cannot be run at all --
        # infrastructure, never a model result (Task 4's amendment).
        return _result("infrastructure", 0, None, str(exc))
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        # git can be missing from PATH entirely, or any of the git helpers
        # above (ensure_repo, start_branch, snapshot) can fail -- an
        # infrastructure problem, never a model result.
        return _result("infrastructure", 0, None, f"git failed: {exc}")

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
            outcome = "budget_exhausted" if turn > 1 else "refused"
            return _result(outcome, turn, branch, str(exc))
        except ModelError as exc:
            return _result("infrastructure", turn, branch, str(exc))

        action_text, result_text, applied, target = _take_turn(
            gen, root, scope, adapter, codec, allow_test_edits
        )
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
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                return _result("infrastructure", turn, branch, f"git failed: {exc}")
            if diag.passed:
                return _result("pass", turn, branch, "tests pass")
            # Mid-loop re-resolution can fail where the first one could not:
            # a timed-out or unanchorable run returns file=None, and resolve
            # refuses that. Keep the scope we already have and let the model
            # see the new diagnostic — aborting here would throw away a
            # recoverable turn (and, unguarded, crash out of the loop).
            try:
                scope = resolve(diag, adapter, root)
            except ScopeError:
                pass

        key = f"{action_text}\n{gen.text}"
        stalls = stalls + 1 if key in seen else 0
        seen.add(key)
        if stalls >= stall_cap - 1:
            return _result("stalled", turn, branch, "no progress; repeating")

    return _result("stalled", turn_cap, branch, f"turn cap {turn_cap} reached")


def _take_turn(gen, root, scope, adapter, codec, allow_test_edits):
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
    path = (root / arg.split()[0]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return f"cannot read '{arg}': no such file in this repository"
    return path.read_text(encoding="utf-8")[:4000]


def _find(root: Path, symbol: str) -> str:
    hits: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if ".git" in path.parts:
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").split("\n"), 1
        ):
            if symbol in line:
                hits.append(f"{path.relative_to(root)}:{number}")
                if len(hits) >= 20:
                    return "\n".join(hits)
    return "\n".join(hits) or f"'{symbol}' not found"


def _slug(task: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:24] or "run"
