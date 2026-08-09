# src/robigo/apply/patch.py
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from robigo.action.codec import CODECS, PatchError
from robigo.action.verbs import Action
from robigo.adapters.base import Adapter
from robigo.apply.safety import check_target
from robigo.context.scope import Scope


def apply_patch(
    action: Action,
    root: Path,
    scope: Scope,
    adapter: Adapter,
    codec: str,
    allow_test_edits: bool = False,
) -> Path:
    """Verify, then write. Nothing reaches the working tree until the
    result is known to parse -- a patch that lands and breaks the file
    costs a turn AND corrupts the scope."""
    target = check_target(action.arg, root, scope, allow_test_edits)
    original = target.read_text(encoding="utf-8")
    new_text = CODECS[codec](original, action.payload or "")
    if not adapter.syntax_ok(new_text):
        raise PatchError(
            f"the result of this patch has invalid {adapter.name} syntax, "
            f"so it was not written. Check brackets and indentation in the "
            f"replacement lines."
        )
    write_atomic(target, new_text)
    return target


def write_atomic(path: Path, text: str) -> None:
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=".robigo-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
