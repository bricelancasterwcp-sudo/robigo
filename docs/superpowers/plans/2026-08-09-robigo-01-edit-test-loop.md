# robigo 01 — Edit→Test Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working edit→test loop that repairs a single-defect Python failure using a locally-served model, with every write atomic, branch-scoped, and reversible.

**Architecture:** A stateless pipeline around one stateful loop. The adapter runs pytest and reduces a failure to a `Diagnostic`; `scope` turns that diagnostic into a small set of files; `render` builds a prompt; the model emits exactly one action in a markdown envelope; `verbs.parse` turns it into an `Action`; `codec` turns a patch payload into new file text; `apply` writes it atomically inside a git branch. Everything except `loop.py` is a pure function over fixtures and is tested without a model.

**Tech Stack:** Python 3.12+, stdlib only at runtime. pytest + pytest-cov as dev dependencies. Model access over HTTP to `llama-server` or the Ollama daemon.

## Global Constraints

- **Runtime dependencies: none.** Standard library only. pytest is dev-only.
- `requires-python = ">=3.12"`.
- Package root is `src/robigo/`; import paths are `robigo.<module>`.
- Files stay in the 200–400 line band; 800 is a hard ceiling.
- Type annotations on every **non-test** function signature. pytest test functions are exempt: `-> None` there carries no information, and the constraint exists to pin production interfaces. `from __future__ import annotations` at the top of every module.
- **Ollama requests MUST send `truncate: false` at the TOP level of the request body, never inside `options`** — nested there the daemon ignores it and silently front-truncates (spec §9 law 5).
- **Never set a context window above the model's training context** (spec §9 law 1).
- Infrastructure failures and model results are never conflated in either direction (spec §9 law 10).
- Error messages raised by `verbs.parse`, `codec`, and `apply` are **prompt surface**: they are fed back to the model, so they name both ends of the problem and suggest the repair (spec §2.4).
- Commit messages: `<type>: <subject>`, single line, no body, no trailers.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | packaging, `robigo` console script, pytest config |
| `src/robigo/action/verbs.py` | the five verbs, `Action`, strict envelope parsing |
| `src/robigo/action/codec.py` | `search_replace` and `whole_file` payload → new file text |
| `src/robigo/adapters/base.py` | `Diagnostic`, the `Adapter` protocol |
| `src/robigo/adapters/python_.py` | pytest runner, `--tb=line` parsing, `ast` imports, `ast` syntax check |
| `src/robigo/context/scope.py` | `Scope`, anchor + import hops, signature extraction |
| `src/robigo/context/render.py` | prompt assembly from scope + diagnostic + history |
| `src/robigo/apply/patch.py` | atomic write, post-write syntax verification |
| `src/robigo/apply/safety.py` | git branch/snapshot/commit, path and anchor refusals |
| `src/robigo/model/client.py` | `Generation`, Ollama and llama.cpp clients, overflow classification |
| `src/robigo/loop.py` | the turn loop, terminal states, the evidence gate |
| `src/robigo/record.py` | `.robigo/runs/<id>/` — verbatim prompts, replies, adapter output |
| `src/robigo/cli.py` | argv, exit codes |

Deferred to plan 02: `model/geometry.py`, `context/budget.py`. This plan takes the window as an explicit `--window` integer so the loop can be built and tested before the arithmetic exists.

Deferred to plan 03: `profile/`.

Deferred deliberately, and named here so their absence is not mistaken for an omission:

- **The `udiff` codec** (spec §2.2). It exists mainly so the profiler can demonstrate cheaply that it is bad, which is a plan-03 concern; `reserve_for("udiff", …)` in plan 02 is arithmetic for a codec not yet registered, and `--codec` does not offer it.
- **Envelope Level 1, the two-step split** (spec §2.3). Plan 01 implements Level 0 — no grammar, stop sequences only. Plan 03 *measures* which level a family needs; consuming that measurement is plan 05.
- **Payload-corruption measurement** (spec §2.3) — stage 3, plan 05.

---

### Task 1: Project skeleton and the five verbs

**Files:**
- Create: `pyproject.toml`
- Create: `CORRECTIONS.md`
- Create: `src/robigo/__init__.py`, `src/robigo/action/__init__.py`
- Create: `src/robigo/action/verbs.py`
- Test: `tests/test_verbs.py`

`CORRECTIONS.md` is created empty in the very first commit, per spec §7. It is the single most credibility-building file the project will have, and creating it up front is a commitment rather than an afterthought. Its content is exactly:

```markdown
# Corrections

Claims this project has made and later withdrawn, with the arithmetic that
killed them. Nothing is quietly deleted; a withdrawn number stays here with
its replacement.

*(No corrections yet.)*
```

**Interfaces:**
- Consumes: nothing.
- Produces: `Action(verb: str, arg: str, payload: str | None, lang: str | None)` (frozen dataclass); `ActionParseError(Exception)`; `parse(text: str) -> Action`; `VERBS: tuple[str, ...]`; `PAYLOAD_VERBS: tuple[str, ...]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verbs.py
from __future__ import annotations

import pytest

from robigo.action.verbs import Action, ActionParseError, parse

PATCH = """patch src/fog.py
```python
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
```
"""


def test_parses_a_payload_verb_and_keeps_the_payload_verbatim():
    action = parse(PATCH)
    assert action.verb == "patch"
    assert action.arg == "src/fog.py"
    assert action.lang == "python"
    # Verbatim: no strip, no re-indent, no normalisation. The payload is
    # matched byte-for-byte against file contents downstream.
    assert action.payload == (
        "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n"
    )


def test_parses_a_bare_verb():
    assert parse("run") == Action(verb="run", arg="", payload=None, lang=None)
    assert parse("read src/fog.py 10:40").arg == "src/fog.py 10:40"


def test_ignores_prose_before_the_action():
    action = parse("I'll fix the radius.\n\nread src/fog.py\n")
    assert (action.verb, action.arg) == ("read", "src/fog.py")


def test_rejects_an_unknown_verb():
    with pytest.raises(ActionParseError) as e:
        parse("edit src/fog.py")
    assert "edit" in str(e.value)
    # The message is prompt surface: it must list what IS available.
    assert "patch" in str(e.value)


def test_rejects_a_patch_with_no_payload():
    with pytest.raises(ActionParseError) as e:
        parse("patch src/fog.py\n")
    assert "fenced" in str(e.value)


def test_rejects_a_second_action():
    # One action per turn is an invariant. Taking the first and ignoring
    # the rest would let the model believe both applied -- a silent
    # failure, which is worse than a rejected turn.
    with pytest.raises(ActionParseError) as e:
        parse("read src/fog.py\nrun\n")
    assert "one action" in str(e.value)


def test_a_verb_word_inside_a_payload_is_not_a_second_action():
    text = "patch a.py\n```\nrun the thing\n```\n"
    assert parse(text).payload == "run the thing\n"


def test_an_indented_fence_is_still_a_payload():
    # Both fence checks must tolerate indentation equally. Small models
    # indent erratically, and a payload rejected as "no fenced payload"
    # teaches the model the wrong lesson while burning a turn.
    text = "patch a.py\n  ```\n  x = 1\n  ```\n"
    assert parse(text).payload == "  x = 1\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_verbs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "robigo"
version = "0.0.1"
description = "A coding agent for local models, designed against a VRAM budget"
requires-python = ">=3.12"
dependencies = []

[project.scripts]
robigo = "robigo.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-m 'not live'"
markers = ["live: requires a running model daemon"]
testpaths = ["tests"]
```

```python
# src/robigo/action/verbs.py
from __future__ import annotations

import re
from dataclasses import dataclass

VERBS: tuple[str, ...] = ("read", "find", "patch", "run", "done")
PAYLOAD_VERBS: tuple[str, ...] = ("patch",)

_HEADER = re.compile(rf"^({'|'.join(VERBS)})(?:\s+(.*))?$")
_FENCE = re.compile(r"^```(\w+)?\s*$")


class ActionParseError(Exception):
    """A malformed action. The message is prompt surface: it is fed back
    to the model verbatim, so it names both what was wrong and what to
    emit instead (spec section 2.4)."""


@dataclass(frozen=True)
class Action:
    verb: str
    arg: str
    payload: str | None
    lang: str | None


def parse(text: str) -> Action:
    lines = text.replace("\r\n", "\n").split("\n")
    verb, arg, index = _find_header(lines)
    payload, lang, end = _read_payload(verb, arg, lines, index + 1)
    _reject_second_action(lines, end)
    return Action(verb=verb, arg=arg, payload=payload, lang=lang)


def _find_header(lines: list[str]) -> tuple[str, str, int]:
    for i, line in enumerate(lines):
        match = _HEADER.match(line.strip())
        if match:
            return match.group(1), (match.group(2) or "").strip(), i
        first = line.strip().split(" ", 1)[0]
        if first and first.isalpha() and _looks_like_an_attempt(lines, i):
            raise ActionParseError(
                f"'{first}' is not a verb. Emit one of: "
                f"{', '.join(VERBS)}. Only 'patch' takes a fenced payload."
            )
    raise ActionParseError(
        f"no action found. Emit exactly one of: {', '.join(VERBS)}, "
        f"as a line of its own."
    )


def _looks_like_an_attempt(lines: list[str], i: int) -> bool:
    """A lone lowercase word plus an argument on its own line is a verb
    attempt. Free prose is not, and must not raise -- models preface
    actions with explanation and that is allowed (spec section 2)."""
    parts = lines[i].strip().split()
    return len(parts) == 2 and parts[0].islower() and "." in parts[1]


def _read_payload(
    verb: str, arg: str, lines: list[str], start: int
) -> tuple[str | None, str | None, int]:
    open_at = _next_fence(lines, start)
    if verb not in PAYLOAD_VERBS:
        if open_at is not None:
            raise ActionParseError(
                f"'{verb}' takes no payload but a fenced block followed it. "
                f"Only 'patch' takes one."
            )
        return None, None, start
    if open_at is None:
        raise ActionParseError(
            f"'patch {arg}' needs a fenced payload immediately after the "
            f"header line, opened and closed with ```."
        )
    # Stripped for the same reason as _next_fence: matching the raw line
    # here returns None for an indented fence and the .group(1) below
    # raises AttributeError instead of parsing.
    lang = _FENCE.match(lines[open_at].strip()).group(1)
    for close_at in range(open_at + 1, len(lines)):
        if lines[close_at].strip() == "```":
            body = "\n".join(lines[open_at + 1 : close_at])
            return (body + "\n" if body else ""), lang, close_at + 1
    raise ActionParseError(
        f"the fenced payload for 'patch {arg}' is never closed. "
        f"Close it with ``` on a line of its own."
    )


def _next_fence(lines: list[str], start: int) -> int | None:
    for i in range(start, len(lines)):
        # Stripped, to match the closing-fence check in _read_payload.
        # Asymmetry here rejects an indented payload with a message
        # claiming no payload exists.
        if _FENCE.match(lines[i].strip()):
            return i
        if lines[i].strip():
            return None
    return None


def _reject_second_action(lines: list[str], start: int) -> None:
    for line in lines[start:]:
        if _HEADER.match(line.strip()):
            raise ActionParseError(
                "more than one action in a single reply. Emit exactly one "
                "action per turn and wait for its result."
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_verbs.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml CORRECTIONS.md src/robigo tests/test_verbs.py
git commit -m "feat: five-verb action envelope with strict parsing"
```

---

### Task 2: The `search_replace` codec and its fuzzy-locate diagnostic

**Files:**
- Create: `src/robigo/action/codec.py`
- Test: `tests/test_codec.py`

**Interfaces:**
- Consumes: `robigo.action.verbs.Action`.
- Produces: `PatchError(Exception)`; `apply_search_replace(original: str, payload: str) -> str`; `CODECS: dict[str, Callable[[str, str], str]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_codec.py
from __future__ import annotations

import pytest

from robigo.action.codec import CODECS, PatchError, apply_search_replace

FILE = "def radius(t):\n    r = compute(t)\n    return r\n"


def _block(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}=======\n{replace}>>>>>>> REPLACE\n"


def test_replaces_an_exact_match():
    out = apply_search_replace(FILE, _block("    r = compute(t)\n", "    r = compute(t) * 2\n"))
    assert out == "def radius(t):\n    r = compute(t) * 2\n    return r\n"


def test_applies_several_blocks_in_order():
    payload = _block("def radius(t):\n", "def radius(t, k):\n") + _block(
        "    return r\n", "    return r * k\n"
    )
    out = apply_search_replace(FILE, payload)
    assert out == "def radius(t, k):\n    r = compute(t)\n    return r * k\n"


def test_a_missing_search_block_names_the_closest_line_and_the_difference():
    # The characteristic small-model failure: a transcription slip. The
    # diagnostic must name BOTH ends -- what is in the file and what was
    # sent -- or the model repairs blind (spec section 2.4).
    payload = _block("    r = compute(t);\n", "    r = compute(t) * 2\n")
    with pytest.raises(PatchError) as e:
        apply_search_replace(FILE, payload)
    message = str(e.value)
    assert "    r = compute(t)" in message      # the file's line
    assert "    r = compute(t);" in message     # what the model sent
    assert "copied exactly" in message


def test_an_ambiguous_search_block_is_refused():
    doubled = "x = 1\nx = 1\n"
    with pytest.raises(PatchError) as e:
        apply_search_replace(doubled, _block("x = 1\n", "x = 2\n"))
    assert "2 times" in str(e.value)


def test_a_payload_with_no_blocks_is_refused():
    with pytest.raises(PatchError) as e:
        apply_search_replace(FILE, "just some code\n")
    assert "SEARCH" in str(e.value)


def test_codecs_registry_exposes_search_replace():
    assert CODECS["search_replace"] is apply_search_replace
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_codec.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.action.codec'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/action/codec.py
from __future__ import annotations

import difflib
import re
from typing import Callable

_BLOCK = re.compile(
    r"^<<<<<<< SEARCH\n(?P<search>.*?)^=======\n(?P<replace>.*?)^>>>>>>> REPLACE\s*$",
    re.MULTILINE | re.DOTALL,
)


class PatchError(Exception):
    """A payload that cannot be applied. The message is prompt surface
    and must name both ends of the mismatch (spec section 2.4)."""


def apply_search_replace(original: str, payload: str) -> str:
    blocks = list(_BLOCK.finditer(payload))
    if not blocks:
        raise PatchError(
            "no SEARCH/REPLACE block found. The payload must contain "
            "<<<<<<< SEARCH, then the exact existing lines, then =======, "
            "then the replacement, then >>>>>>> REPLACE."
        )
    text = original
    for block in blocks:
        text = _apply_one(text, block.group("search"), block.group("replace"))
    return text


def _apply_one(text: str, search: str, replace: str) -> str:
    count = text.count(search)
    if count == 1:
        return text.replace(search, replace, 1)
    if count > 1:
        raise PatchError(
            f"this SEARCH block matches {count} times, so the edit is "
            f"ambiguous. Include more surrounding lines to make it unique."
        )
    raise PatchError(_miss_message(text, search))


def _miss_message(text: str, search: str) -> str:
    sent = search.splitlines()[0] if search.strip() else ""
    close = difflib.get_close_matches(sent, text.splitlines(), n=1, cutoff=0.6)
    if not close:
        return (
            "SEARCH block not found in the file, and nothing in the file "
            "resembles its first line. Re-read the file and copy the "
            "target lines exactly."
        )
    return (
        "SEARCH block not found in the file.\n"
        f"  closest line in file   {close[0]!r}\n"
        f"  your SEARCH line       {sent!r}\n"
        "Re-emit the SEARCH block copied exactly from the file, including "
        "indentation and punctuation."
    )


CODECS: dict[str, Callable[[str, str], str]] = {
    "search_replace": apply_search_replace,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codec.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/action/codec.py tests/test_codec.py
git commit -m "feat: search/replace codec with a both-ends miss diagnostic"
```

---

### Task 3: The `whole_file` codec

**Files:**
- Modify: `src/robigo/action/codec.py`
- Modify: `tests/test_codec.py`

**Interfaces:**
- Produces: `apply_whole_file(original: str, payload: str) -> str`; `CODECS["whole_file"]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_codec.py
from robigo.action.codec import apply_whole_file


def test_whole_file_replaces_everything_and_ignores_the_original():
    assert apply_whole_file(FILE, "x = 1\n") == "x = 1\n"


def test_whole_file_refuses_an_empty_payload():
    # An empty emission would silently gut the file. The likeliest real
    # data loss in the whole design (spec section 6).
    with pytest.raises(PatchError) as e:
        apply_whole_file(FILE, "   \n")
    assert "empty" in str(e.value)


def test_whole_file_strips_a_nested_fence_if_the_model_adds_one():
    # Models re-fence habitually; the envelope already consumed the outer
    # fence, so an inner one is content the file must not receive.
    assert apply_whole_file(FILE, "```python\nx = 1\n```\n") == "x = 1\n"


def test_codecs_registry_exposes_whole_file():
    assert CODECS["whole_file"] is apply_whole_file
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_codec.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_whole_file'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/robigo/action/codec.py, above CODECS
def apply_whole_file(original: str, payload: str) -> str:
    """The whole file, re-emitted. ``original`` is unused by design and
    kept in the signature so both codecs share one call shape."""
    text = _strip_nested_fence(payload)
    if not text.strip():
        raise PatchError(
            "the payload is empty, which would delete the file's contents. "
            "Emit the complete new file, or use a SEARCH/REPLACE edit."
        )
    return text if text.endswith("\n") else text + "\n"


def _strip_nested_fence(payload: str) -> str:
    lines = payload.split("\n")
    if lines and lines[0].startswith("```"):
        end = len(lines) - 1
        while end > 0 and not lines[end].strip():
            end -= 1
        if lines[end].strip() == "```":
            return "\n".join(lines[1:end]) + "\n"
    return payload
```

Then extend the registry:

```python
CODECS: dict[str, Callable[[str, str], str]] = {
    "search_replace": apply_search_replace,
    "whole_file": apply_whole_file,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codec.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/action/codec.py tests/test_codec.py
git commit -m "feat: whole-file codec, refusing an emptying payload"
```

---

### Task 4: The pytest adapter

**Files:**
- Create: `src/robigo/adapters/__init__.py`, `src/robigo/adapters/base.py`
- Create: `src/robigo/adapters/python_.py`
- Test: `tests/test_adapter_python.py`

**Interfaces:**
- Produces: `Diagnostic(passed: bool, file: str | None, line: int | None, message: str, raw: str)` (frozen); `Adapter` protocol with `name`, `test_command`, `run(root, filt) -> Diagnostic`, `imports(path, root) -> list[Path]`, `syntax_ok(text) -> bool`; `PythonAdapter`; `DIAGNOSTIC_CHAR_CAP: int`.

`run` uses `pytest --tb=line -q --no-header`, whose failure lines are exactly `"/abs/path.py:42: AssertionError: msg"` — chosen over `--tb=short` because it is one deterministic line per failure with no block parsing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_python.py
from __future__ import annotations

from pathlib import Path

import pytest

from robigo.adapters.python_ import DIAGNOSTIC_CHAR_CAP, PythonAdapter


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "fog.py").write_text("def radius(t):\n    return t\n")
    (tmp_path / "src" / "grid.py").write_text("CELL = 5\n")
    (tmp_path / "tests" / "test_fog.py").write_text(
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        "from fog import radius\n\n"
        "def test_radius():\n"
        "    assert radius(2) == 4\n"
    )
    return tmp_path


def test_run_reports_a_failure_with_file_and_line(repo: Path):
    diag = PythonAdapter().run(repo, None)
    assert diag.passed is False
    assert diag.file == "tests/test_fog.py"
    assert diag.line == 6
    assert "assert" in diag.message.lower()


def test_run_reports_a_pass(repo: Path):
    (repo / "src" / "fog.py").write_text("def radius(t):\n    return t * 2\n")
    diag = PythonAdapter().run(repo, None)
    assert diag.passed is True
    assert diag.file is None


def test_raw_output_is_capped(repo: Path):
    diag = PythonAdapter().run(repo, None)
    assert len(diag.raw) <= DIAGNOSTIC_CHAR_CAP


def test_imports_resolves_local_modules_only(repo: Path):
    found = PythonAdapter().imports(repo / "tests" / "test_fog.py", repo)
    # `fog` resolves inside the repo; `sys` is stdlib and must not appear.
    assert found == [repo / "src" / "fog.py"]


def test_syntax_ok_distinguishes_valid_from_broken():
    adapter = PythonAdapter()
    assert adapter.syntax_ok("x = 1\n") is True
    assert adapter.syntax_ok("def f(\n") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapter_python.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.adapters'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/adapters/base.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DIAGNOSTIC_CHAR_CAP = 2400
"""Roughly 600 tokens. A single bad turn must not be able to eat the
window (spec section 3)."""


@dataclass(frozen=True)
class Diagnostic:
    passed: bool
    file: str | None
    line: int | None
    message: str
    raw: str


class Adapter(Protocol):
    name: str
    test_command: str

    def run(self, root: Path, filt: str | None) -> Diagnostic: ...
    def imports(self, path: Path, root: Path) -> list[Path]: ...
    def syntax_ok(self, text: str) -> bool: ...
```

```python
# src/robigo/adapters/python_.py
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

from robigo.adapters.base import DIAGNOSTIC_CHAR_CAP, Diagnostic

_FAIL_LINE = re.compile(r"^(?P<file>/[^:]+\.py):(?P<line>\d+): (?P<msg>.+)$")
_TIMEOUT_S = 300


class PythonAdapter:
    name = "python"
    test_command = "pytest --tb=line -q --no-header"

    def run(self, root: Path, filt: str | None) -> Diagnostic:
        argv = ["python", "-m", "pytest", "--tb=line", "-q", "--no-header", "-p",
                "no:cacheprovider"]
        if filt:
            argv += ["-k", filt]
        proc = subprocess.run(
            argv, cwd=root, capture_output=True, text=True, timeout=_TIMEOUT_S
        )
        raw = (proc.stdout + proc.stderr)[-DIAGNOSTIC_CHAR_CAP:]
        if proc.returncode == 0:
            return Diagnostic(True, None, None, "all tests passed", raw)
        return self._first_failure(raw, root)

    def _first_failure(self, raw: str, root: Path) -> Diagnostic:
        for line in raw.split("\n"):
            match = _FAIL_LINE.match(line.strip())
            if not match:
                continue
            path = Path(match.group("file"))
            try:
                rel = str(path.relative_to(root))
            except ValueError:
                rel = str(path)
            return Diagnostic(
                False, rel, int(match.group("line")), match.group("msg"), raw
            )
        return Diagnostic(False, None, None, "tests failed", raw)

    def imports(self, path: Path, root: Path) -> list[Path]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return []
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        found: list[Path] = []
        for name in names:
            for candidate in self._candidates(name, root):
                if candidate.is_file() and candidate not in found:
                    found.append(candidate)
                    break
        return found

    def _candidates(self, name: str, root: Path) -> list[Path]:
        parts = name.split(".")
        return [
            root / "src" / Path(*parts).with_suffix(".py"),
            root / Path(*parts).with_suffix(".py"),
            root / "src" / Path(*parts) / "__init__.py",
            root / Path(*parts) / "__init__.py",
        ]

    def syntax_ok(self, text: str) -> bool:
        try:
            ast.parse(text)
        except SyntaxError:
            return False
        return True
```

Also create empty `src/robigo/adapters/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapter_python.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/adapters tests/test_adapter_python.py
git commit -m "feat: pytest adapter with deterministic tb=line diagnostics"
```

#### Amendment (ruled 2026-08-09): the interpreter must be discovered, not assumed

`argv = ["python", ...]` above is a defect. `"python"` resolves from the
subprocess `PATH`, so the suite only passes with an activated venv — and in
production it is worse: installed via `uv tool install`, robigo lives in its
own isolated venv, so `sys.executable` cannot see the *target project's*
pytest either. The interpreter that runs a project's tests is the project's,
not robigo's.

Add to `adapters/base.py`:

```python
class AdapterError(Exception):
    """The adapter cannot run the project's tests at all. Infrastructure,
    never a model result: it maps to exit code 4 (spec section 6.1)."""
```

In `adapters/python_.py`, take the interpreter as constructor state,
discover it per-root, and probe it once:

```python
    def __init__(self, python: str | None = None) -> None:
        self._python = python

    def _interpreter(self, root: Path) -> str:
        """The project's interpreter, not robigo's. Checked in order, so a
        repo with its own venv needs nothing activated."""
        if self._python:
            return self._python
        for candidate in (root / ".venv/bin/python", root / "venv/bin/python"):
            if candidate.is_file():
                return str(candidate)
        return "python"

    def _preflight(self, python: str) -> None:
        """Refuse loudly rather than fail per-run with a confusing
        ModuleNotFoundError from inside a subprocess."""
        try:
            proc = subprocess.run(
                [python, "-m", "pytest", "--version"],
                capture_output=True, text=True, timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AdapterError(f"cannot execute {python!r}: {exc}") from exc
        if proc.returncode != 0:
            raise AdapterError(
                f"{python} cannot import pytest. Activate the project's "
                f"virtualenv, or pass --python <path> to name the "
                f"interpreter that has the project's test dependencies."
            )
```

`run` resolves and probes before invoking, and uses the resolved path:

```python
        python = self._interpreter(root)
        self._preflight(python)
        argv = [python, "-m", "pytest", "--tb=line", "-q", "--no-header",
                "-p", "no:cacheprovider"]
```

**The tests change accordingly.** Fixture repos under `tmp_path` have no
venv, so they would fall through to PATH `python` and refuse. Every
construction in `tests/test_adapter_python.py` becomes
`PythonAdapter(python=sys.executable)` (add `import sys`) — the interpreter
running the suite always has pytest, so the suite passes with or without
activation. Add two tests:

```python
def test_a_project_venv_is_preferred_over_path(tmp_path: Path):
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    assert PythonAdapter()._interpreter(tmp_path) == str(venv_python)


def test_an_interpreter_without_pytest_is_refused_loudly(tmp_path: Path):
    from robigo.adapters.base import AdapterError

    fake = tmp_path / "fake-python"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    with pytest.raises(AdapterError) as e:
        PythonAdapter(python=str(fake)).run(tmp_path, None)
    assert "--python" in str(e.value)
```

Task 10's loop must catch `AdapterError` alongside `ModelError` and return
the `infrastructure` outcome for it.

#### Amendment 2 (ruled 2026-08-09): the anchor must be inside the repo

The brief's rationale for `--tb=line` — "one deterministic line per failure"
— is **false for collection errors**. A broken import or a `SyntaxError` in an
imported module makes pytest print a full traceback, so "the first
`path:line:` match anywhere" lands on framework internals. Reproduced:

```
ModuleNotFoundError → /usr/lib/python3.14/importlib/__init__.py:88
SyntaxError         → .venv/.../_pytest/python.py:508
```

Both discard the real failure, and broken imports and syntax errors are
among the most common things a model produces mid-loop. Since Task 5 anchors
scope on `diag.file`, a location outside the repo shows the model code it
cannot edit — worse than no anchor at all.

**The rule: an anchor must be a path inside the repo that is not under a
virtualenv.** Filter candidates rather than guessing at traceback shape, and
prefer pytest's own `E   <Type>: <message>` line for the message when one is
present, since that names the real failure.

```python
_FAIL_LINE = re.compile(r"^(?P<file>[^\s:][^:]*\.py):(?P<line>\d+):\s*(?P<msg>.*)$")
_ERROR_LINE = re.compile(r"^E\s+(?P<msg>\S.*)$")
_EXCLUDED = ("site-packages", "dist-packages", "/.venv/", "/venv/")


    def _first_failure(self, raw: str, root: Path) -> Diagnostic:
        root = root.resolve()
        summary = self._error_summary(raw)
        for line in raw.split("\n"):
            match = _FAIL_LINE.match(line.strip())
            if not match:
                continue
            rel = self._in_repo(match.group("file"), root)
            if rel is None:
                continue
            return Diagnostic(
                False, rel, int(match.group("line")),
                summary or match.group("msg"), raw,
            )
        return Diagnostic(False, None, None, summary or "tests failed", raw)

    def _in_repo(self, candidate: str, root: Path) -> str | None:
        """Repo-relative path, or None when the location is outside the
        project. An anchor the model cannot edit is worse than none."""
        if any(fragment in candidate for fragment in _EXCLUDED):
            return None
        path = Path(candidate)
        resolved = path if path.is_absolute() else root / path
        try:
            return str(resolved.resolve().relative_to(root))
        except (ValueError, OSError):
            return None

    def _error_summary(self, raw: str) -> str | None:
        """pytest's own `E   <Type>: <message>` line, which names the real
        failure when a traceback replaces --tb=line's one-liner."""
        for line in raw.split("\n"):
            match = _ERROR_LINE.match(line.rstrip())
            if match:
                return match.group("msg")
        return None
```

Note the path group no longer requires a leading `/`: pytest emits relative
paths in traceback frames, and `_in_repo` resolves them against the root.

**A timeout is a model result.** The main subprocess had no
`TimeoutExpired` handling, so a model-written infinite loop propagated an
uncaught exception — neither a model result nor an infrastructure error,
violating the classification rule outright. The cause is almost always the
patch just applied, so it is fed back:

```python
        try:
            proc = subprocess.run(
                argv, cwd=root, capture_output=True, text=True,
                timeout=_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return Diagnostic(
                False, None, None,
                f"tests timed out after {_TIMEOUT_S}s — the last patch may "
                f"not terminate", "",
            )
```

**Three tests, covering exactly the reproduced cases:**

```python
def test_a_broken_import_anchors_in_the_repo_not_in_importlib(repo: Path):
    (repo / "tests" / "test_bad.py").write_text("import nonexistent_xyz\n")
    diag = PythonAdapter(python=sys.executable).run(repo, None)
    assert diag.passed is False
    assert diag.file == "tests/test_bad.py"
    assert "nonexistent_xyz" in diag.message


def test_a_syntax_error_anchors_in_the_repo_not_in_pytest(repo: Path):
    (repo / "src" / "fog.py").write_text("def radius(t:\n")
    diag = PythonAdapter(python=sys.executable).run(repo, None)
    assert diag.file is not None
    assert "site-packages" not in diag.file
    assert diag.file.startswith(("src/", "tests/"))


def test_a_hanging_suite_is_a_model_result_not_a_crash(repo: Path, monkeypatch):
    monkeypatch.setattr("robigo.adapters.python_._TIMEOUT_S", 3)
    (repo / "tests" / "test_fog.py").write_text(
        "def test_spin():\n    while True:\n        pass\n"
    )
    diag = PythonAdapter(python=sys.executable).run(repo, None)
    assert diag.passed is False
    assert "timed out" in diag.message
```

#### Amendment 3 (ruled 2026-08-09): the anchor must be untrusted input

Amendment 2 dropped the leading-`/` requirement so relative traceback frames
could match. That made **captured stdout eligible**, and `_in_repo` never
checked the path exists. Reproduced:

```python
print("nonexistent_config.py:999: totally unrelated fake path text")
assert 1 == 2, "the real failure"
# → Diagnostic(file="nonexistent_config.py", line=999)
```

The model authors the test code, so model output could steer where the tool
believes the failure lives — attacking the property Amendment 2 restored. A
second defect from the same code: `_error_summary` scanned all of `raw`, so a
multi-failure run could staple one failure's message onto another's anchor.

**Treat pytest's output as untrusted.** Three guards:

```python
_CAPTURED = re.compile(r"^-+ Captured .* -+$")


    def _first_failure(self, raw: str, root: Path) -> Diagnostic:
        root = root.resolve()
        lines = raw.split("\n")
        anchor = self._anchor(lines, root)
        if anchor is None:
            summary = self._error_summary(lines, 0)
            return Diagnostic(False, None, None, summary or "tests failed", raw)
        index, rel, number, tail = anchor
        summary = self._error_summary(lines, index)
        return Diagnostic(False, rel, number, summary or tail, raw)

    def _anchor(self, lines: list[str], root: Path) -> tuple[int, str, int, str] | None:
        """The first in-repo, on-disk location outside captured output.
        Captured output is model-authored: a test that prints
        'fake.py:999: x' must never redirect where the failure is."""
        captured = False
        for index, line in enumerate(lines):
            if _CAPTURED.match(line.strip()):
                captured = True
                continue
            if line.startswith(("=", "_")):
                captured = False
            if captured:
                continue
            match = _FAIL_LINE.match(line.strip())
            if not match:
                continue
            rel = self._in_repo(match.group("file"), root)
            if rel is not None:
                return index, rel, int(match.group("line")), match.group("msg")
        return None

    def _in_repo(self, candidate: str, root: Path) -> str | None:
        if any(fragment in candidate for fragment in _EXCLUDED):
            return None
        path = Path(candidate)
        resolved = path if path.is_absolute() else root / path
        try:
            resolved = resolved.resolve()
            relative = resolved.relative_to(root)
        except (ValueError, OSError):
            return None
        if not resolved.is_file():
            return None          # a fabricated path is not an anchor
        return str(relative)

    def _error_summary(self, lines: list[str], start: int) -> str | None:
        """pytest's `E   <Type>: <message>`, searched FORWARD from the
        anchor, so a message can never be attached to a different
        failure's location."""
        for line in lines[start:]:
            match = _ERROR_LINE.match(line.rstrip())
            if match:
                return match.group("msg")
        return None
```

Searching forward is correct for the shape pytest actually emits: the
in-repo frame precedes its own `E` line in a traceback.

**Three tests:**

```python
def test_model_authored_stdout_cannot_hijack_the_anchor(repo: Path):
    (repo / "tests" / "test_fog.py").write_text(
        "def test_x():\n"
        "    print('nonexistent_config.py:999: totally unrelated fake path')\n"
        "    assert 1 == 2, 'the real failure'\n"
    )
    diag = PythonAdapter(python=sys.executable).run(repo, None)
    assert diag.file == "tests/test_fog.py"
    assert "nonexistent_config" not in (diag.file or "")


def test_a_path_that_does_not_exist_is_not_an_anchor(tmp_path: Path):
    assert PythonAdapter()._in_repo("nonexistent_config.py", tmp_path) is None


def test_the_summary_is_paired_with_the_anchor_not_the_first_failure():
    lines = [
        "E   AssertionError: an earlier unrelated failure",
        "tests/test_b.py:2: AssertionError",
        "E   AssertionError: the failure that belongs here",
    ]
    assert PythonAdapter()._error_summary(lines, 1) == (
        "AssertionError: the failure that belongs here"
    )
```

#### Amendment 4 (ruled 2026-08-09): delete the captured-output state machine

Amendment 3's `_anchor` state machine is wrong and cannot be fixed by
tuning. pytest emits its `--tb=line` one-liner **directly after** a
`Captured ... call` section with **zero** separator lines when there is one
line of captured content, so `captured` never resets and the real anchor is
skipped — a false negative. Two layout-shaped heuristics have now failed, and
pytest offers no reliable delimiter for the end of captured output.

The empirical finding that settles it: `_in_repo`'s existence check already
rejects the fabricated path on its own. The state machine was never what
protected the anchor.

**So: remove `_CAPTURED` and the captured tracking entirely**, and bound the
location instead — a file the model merely *printed* is not a failure site,
so the file must exist and the line must be inside it. `_in_repo` takes the
line number:

```python
    def _anchor(self, lines: list[str], root: Path) -> tuple[int, str, int, str] | None:
        for index, line in enumerate(lines):
            match = _FAIL_LINE.match(line.strip())
            if not match:
                continue
            number = int(match.group("line"))
            rel = self._in_repo(match.group("file"), root, number)
            if rel is not None:
                return index, rel, number, match.group("msg")
        return None

    def _in_repo(self, candidate: str, root: Path, number: int) -> str | None:
        """Repo-relative path for a location that could plausibly be a real
        failure site, or None.

        An anchor the model cannot edit is worse than none, and a location
        the model merely PRINTED is not a failure site — so the file must
        exist and the line must fall inside it.

        Residual, accepted: a model that prints "src/real.py:12: ..." —
        naming a real file at a plausible line — can still misdirect the
        anchor. Bounding captured output by layout was tried twice and
        failed twice; the cost here is one wasted turn, and Task 5 refuses
        an anchor that is not a real file.
        """
        if any(fragment in candidate for fragment in _EXCLUDED):
            return None
        path = Path(candidate)
        resolved = path if path.is_absolute() else root / path
        try:
            resolved = resolved.resolve()
            relative = resolved.relative_to(root)
            if not resolved.is_file():
                return None
            body = resolved.read_text(encoding="utf-8", errors="replace")
            if number < 1 or number > len(body.splitlines()):
                return None
        except (ValueError, OSError):
            return None
        return str(relative)
```

Tests: keep the paired-summary test; rewrite the hijack test to assert the
fabricated path is rejected and the real test file wins; update the
existence test for the new signature; add the line-bound case.

```python
def test_model_authored_stdout_cannot_hijack_the_anchor(repo: Path):
    (repo / "tests" / "test_fog.py").write_text(
        "def test_x():\n"
        "    print('nonexistent_config.py:999: totally unrelated fake path')\n"
        "    assert 1 == 2, 'the real failure'\n"
    )
    diag = PythonAdapter(python=sys.executable).run(repo, None)
    assert diag.file == "tests/test_fog.py"
    assert "nonexistent_config" not in (diag.file or "")


def test_a_path_that_does_not_exist_is_not_an_anchor(tmp_path: Path):
    assert PythonAdapter()._in_repo("nonexistent_config.py", tmp_path, 1) is None


def test_a_real_file_at_an_impossible_line_is_not_an_anchor(tmp_path: Path):
    (tmp_path / "short.py").write_text("x = 1\n")
    adapter = PythonAdapter()
    assert adapter._in_repo("short.py", tmp_path, 1) == "short.py"
    assert adapter._in_repo("short.py", tmp_path, 999) is None
```

#### Amendment 5 (ruled 2026-08-09): prefer the anchor's own tail

Amendment 3's forward search rests on a false premise. pytest 9.1.1 emits
`E   <Type>: <message>` **before** its own crash line in a `FAILURES` block,
so searching forward from the anchor skips the anchor's message and takes the
*next* failure's. Reproduced:

```
E   AssertionError: FAILURE-A-MESSAGE
assert 1 == 2
tests/test_a.py:2: AssertionError: FAILURE-A-MESSAGE
E   AssertionError: FAILURE-B-MESSAGE
tests/test_b.py:2: ...
→ shipped ('tests/test_a.py', 2, 'AssertionError: FAILURE-B-MESSAGE')
```

Scanning all of `raw` was *correct* on this shape, so Amendment 3 was a
regression. And the right message was already in hand: the anchor line's own
tail carries it.

The two shapes order oppositely, so discriminate by **shape, not direction**:

- `FAILURES` (`--tb=line`): the tail is the message. Use it.
- `ERRORS` (traceback): the in-repo frame's tail is a bare frame marker like
  `in <module>`, and the real message follows. Fall back to the forward search.

```python
_FRAME_TAIL = re.compile(r"^in\s")


        index, rel, number, tail = anchor
        if tail and not _FRAME_TAIL.match(tail):
            message = tail
        else:
            message = self._error_summary(lines, index) or tail or "tests failed"
        return Diagnostic(False, rel, number, message, raw)
```

`_error_summary` stays exactly as it is — it is now the collection-error path
only.

**Delete `test_the_summary_is_paired_with_the_anchor_not_the_first_failure`.**
It hand-builds a three-line list in the *inverse* of pytest's real ordering,
so it passes while the defect is live. A synthetic ordering test is worse than
none here. Replace it with a real end-to-end multi-failure run:

```python
def test_a_multi_failure_run_pairs_the_message_with_its_own_anchor(repo: Path):
    (repo / "tests" / "test_fog.py").write_text(
        "def test_a():\n    assert 1 == 2, 'FAILURE-A'\n"
    )
    (repo / "tests" / "test_grid.py").write_text(
        "def test_b():\n    assert 3 == 4, 'FAILURE-B'\n"
    )
    diag = PythonAdapter(python=sys.executable).run(repo, None)
    assert diag.file == "tests/test_fog.py"
    assert "FAILURE-A" in diag.message
    assert "FAILURE-B" not in diag.message
```

The existing broken-import test already guards the other branch: it asserts
`"nonexistent_xyz" in diag.message`, which only passes via the `E`-line
fallback.

---

### Task 5: Test-anchored scope resolution

**Files:**
- Create: `src/robigo/context/__init__.py`, `src/robigo/context/scope.py`
- Test: `tests/test_scope.py`

**Interfaces:**
- Consumes: `Diagnostic`, `PythonAdapter`.
- Produces: `Scope(anchor: Path, full: tuple[Path, ...], signatures: tuple[Path, ...])` (frozen); `resolve(diag: Diagnostic, adapter: Adapter, root: Path, hops: int = 2) -> Scope`; `signatures_of(text: str) -> str`; `ScopeError(Exception)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scope.py
from __future__ import annotations

from pathlib import Path

import pytest

from robigo.adapters.base import Diagnostic
from robigo.adapters.python_ import PythonAdapter
from robigo.context.scope import Scope, ScopeError, resolve, signatures_of


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "grid.py").write_text("CELL = 5\n")
    (tmp_path / "src" / "fog.py").write_text("import grid\n\ndef radius(t):\n    return t\n")
    (tmp_path / "tests" / "test_fog.py").write_text("import fog\n\ndef test_r():\n    assert 0\n")
    return tmp_path


def _diag(file: str) -> Diagnostic:
    return Diagnostic(False, file, 4, "AssertionError", "raw")


def test_anchor_is_the_diagnostic_file_and_hop_one_is_full_text(repo: Path):
    scope = resolve(_diag("tests/test_fog.py"), PythonAdapter(), repo, hops=1)
    assert scope.anchor == repo / "tests" / "test_fog.py"
    assert scope.full == (repo / "tests" / "test_fog.py", repo / "src" / "fog.py")
    assert scope.signatures == ()


def test_hop_two_arrives_as_signatures_only(repo: Path):
    scope = resolve(_diag("tests/test_fog.py"), PythonAdapter(), repo, hops=2)
    assert scope.signatures == (repo / "src" / "grid.py",)
    # grid.py must NOT also be in full -- paying for it twice is the bug.
    assert repo / "src" / "grid.py" not in scope.full


def test_repo_size_cannot_affect_scope(repo: Path):
    for i in range(200):
        (repo / "src" / f"noise{i}.py").write_text("x = 1\n")
    scope = resolve(_diag("tests/test_fog.py"), PythonAdapter(), repo, hops=2)
    assert len(scope.full) + len(scope.signatures) == 3


def test_a_diagnostic_with_no_file_is_refused(repo: Path):
    with pytest.raises(ScopeError) as e:
        resolve(Diagnostic(False, None, None, "tests failed", "raw"), PythonAdapter(), repo)
    assert "anchor" in str(e.value)


def test_signatures_of_keeps_definitions_and_drops_bodies():
    out = signatures_of("import os\n\ndef f(a, b):\n    return a\n\nclass K:\n    pass\n")
    assert "def f(a, b):" in out
    assert "class K:" in out
    assert "return a" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scope.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.context'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/context/scope.py
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from robigo.adapters.base import Adapter, Diagnostic


class ScopeError(Exception):
    """Scope cannot be resolved. Refused before any generation, because
    a session with no anchor can only fabricate a result (spec 3, step 5)."""


@dataclass(frozen=True)
class Scope:
    anchor: Path
    full: tuple[Path, ...]
    signatures: tuple[Path, ...]


def resolve(
    diag: Diagnostic, adapter: Adapter, root: Path, hops: int = 2
) -> Scope:
    if not diag.file:
        raise ScopeError(
            "the test output named no file, so there is no anchor to scope "
            "from. Run with --scope to supply one explicitly."
        )
    anchor = (root / diag.file).resolve()
    if not anchor.is_file():
        raise ScopeError(f"anchor {diag.file} does not exist under {root}")

    full: list[Path] = [anchor]
    for path in adapter.imports(anchor, root):
        if path not in full:
            full.append(path)
    signatures: list[Path] = []
    if hops >= 2:
        for parent in full[1:]:
            for path in adapter.imports(parent, root):
                if path not in full and path not in signatures:
                    signatures.append(path)
    return Scope(anchor=anchor, full=tuple(full), signatures=tuple(signatures))


def signatures_of(text: str) -> str:
    """Definition lines only. Hop-2 files are for orientation, not
    reading, and their bodies are the single largest avoidable cost in a
    small window."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    lines = text.split("\n")
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(lines[node.lineno - 1].rstrip())
    return "\n".join(out) + ("\n" if out else "")
```

Also create empty `src/robigo/context/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scope.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/context tests/test_scope.py
git commit -m "feat: test-anchored scope with signature-only second hop"
```

#### Amendment (ruled 2026-08-09): `signatures_of` must be readable, and refusals actionable

Three defects in the reference code above, two verified by reproduction:

1. **`ast.walk` is breadth-first, so definitions come out scrambled.**
   `class Foo: method_a` / `class Bar: method_b` emits
   `Foo, Bar, method_a, method_b`. Any file with two top-level definitions
   where one has nested content — the common case — gets a misleading outline.
2. **Decorators are dropped.** `node.lineno` points at `def`, not at the
   decorator, so `@property` reads as a plain method.
3. **The missing-anchor `ScopeError` names the problem but not the remedy**,
   contradicting the plan's own constraint that these messages say what to do.

Sort by position, carry decorator lines, and add a containment check — an
anchor must never resolve outside the repo:

```python
def signatures_of(text: str) -> str:
    """Definition lines only, in source order, decorators included. Hop-2
    files are for orientation, and an outline whose order does not match
    the file is worse than no outline."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    lines = text.split("\n")
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    out: list[str] = []
    for node in sorted(nodes, key=lambda item: item.lineno):
        start = node.lineno
        if node.decorator_list:
            start = min(decorator.lineno for decorator in node.decorator_list)
        out.extend(lines[number - 1].rstrip() for number in range(start, node.lineno + 1))
    return "\n".join(out) + ("\n" if out else "")
```

and in `resolve`, before the existence check:

```python
    anchor = (root / diag.file).resolve()
    if not anchor.is_relative_to(root.resolve()):
        raise ScopeError(
            f"anchor {diag.file} resolves outside {root}. Scope never leaves "
            f"the repository; pass --scope to set it explicitly."
        )
    if not anchor.is_file():
        raise ScopeError(
            f"anchor {diag.file} does not exist under {root}. Check the path "
            f"is repo-relative and spelled correctly, or pass --scope to name "
            f"the files to work in explicitly."
        )
```

Four tests:

```python
def test_signatures_keep_source_order_across_nested_definitions():
    out = signatures_of(
        "class Foo:\n    def method_a(self):\n        pass\n\n"
        "class Bar:\n    def method_b(self):\n        pass\n"
    )
    assert out.split("\n")[:4] == [
        "class Foo:",
        "    def method_a(self):",
        "class Bar:",
        "    def method_b(self):",
    ]


def test_signatures_keep_decorators():
    out = signatures_of(
        "class K:\n    @property\n    def size(self):\n        return 1\n"
    )
    assert "@property" in out
    assert "return 1" not in out


def test_an_anchor_outside_the_repo_is_refused(repo: Path):
    (repo.parent / "escape.py").write_text("x = 1\n")
    with pytest.raises(ScopeError) as e:
        resolve(_diag("../escape.py"), PythonAdapter(), repo)
    assert "outside" in str(e.value)


def test_a_missing_anchor_says_what_to_do(repo: Path):
    with pytest.raises(ScopeError) as e:
        resolve(_diag("tests/nope.py"), PythonAdapter(), repo)
    assert "--scope" in str(e.value)
```

---

### Task 6: Prompt rendering

**Files:**
- Create: `src/robigo/context/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `Scope`, `Diagnostic`, `signatures_of`.
- Produces: `SYSTEM: str`; `Turn(action: str, result: str)` (frozen); `render(scope, diag, history, codec, root) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_render.py
from __future__ import annotations

from pathlib import Path

from robigo.adapters.base import Diagnostic
from robigo.context.render import SYSTEM, Turn, render
from robigo.context.scope import Scope


def _scope(root: Path) -> Scope:
    (root / "a.py").write_text("def f():\n    return 1\n")
    (root / "b.py").write_text("def g(x):\n    return x\n")
    return Scope(root / "a.py", (root / "a.py",), (root / "b.py",))


def test_prompt_contains_the_verbs_the_scope_and_the_diagnostic(tmp_path: Path):
    diag = Diagnostic(False, "a.py", 2, "AssertionError: 1 != 2", "raw tail")
    out = render(_scope(tmp_path), diag, (), "search_replace", tmp_path)
    for verb in ("read", "find", "patch", "run", "done"):
        assert verb in out
    assert "def f():" in out                 # full text for hop 0/1
    assert "def g(x):" in out                # signature for hop 2
    assert "return x" not in out             # but not its body
    assert "AssertionError: 1 != 2" in out
    assert "a.py:2" in out


def test_only_the_selected_codec_is_described(tmp_path: Path):
    diag = Diagnostic(False, "a.py", 2, "boom", "raw")
    out = render(_scope(tmp_path), diag, (), "whole_file", tmp_path)
    # Describing codecs the profile did not select wastes window and
    # invites the model to use the wrong one.
    assert "SEARCH" not in out
    assert "complete new file" in out


def test_history_is_included_oldest_first(tmp_path: Path):
    diag = Diagnostic(False, "a.py", 2, "boom", "raw")
    history = (Turn("patch a.py", "SEARCH not found"), Turn("run", "still failing"))
    out = render(_scope(tmp_path), diag, history, "search_replace", tmp_path)
    assert out.index("SEARCH not found") < out.index("still failing")


def test_system_prompt_stays_small():
    # The window is the scarce resource; the fixed cost must stay bounded.
    assert len(SYSTEM) < 1800
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.context.render'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/context/render.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robigo.adapters.base import Diagnostic
from robigo.context.scope import Scope, signatures_of

SYSTEM = """You are repairing one failing test in an existing codebase.

Emit exactly ONE action per reply, as a line of its own, then stop:

  read <path>        show a file you have not been given
  find <symbol>      locate a symbol elsewhere in the repo
  patch <path>       change a file (needs a fenced payload)
  run                re-run the tests
  done <summary>     the test passes and you are finished

Rules:
- One action per reply. Never two.
- Only `patch` takes a fenced payload.
- You may not edit the failing test itself.
- Do not explain at length; the action is what matters.
"""

_CODEC_HELP = {
    "search_replace": (
        "For `patch`, the payload is one or more blocks:\n"
        "<<<<<<< SEARCH\n<exact existing lines>\n=======\n"
        "<replacement lines>\n>>>>>>> REPLACE\n"
        "The SEARCH lines must match the file byte-for-byte."
    ),
    "whole_file": (
        "For `patch`, the payload is the complete new file, top to "
        "bottom. Do not abbreviate or elide any part of it."
    ),
}


@dataclass(frozen=True)
class Turn:
    action: str
    result: str


def render(
    scope: Scope,
    diag: Diagnostic,
    history: tuple[Turn, ...],
    codec: str,
    root: Path,
) -> str:
    parts = [SYSTEM, _CODEC_HELP[codec], ""]
    for path in scope.full:
        parts.append(f"--- {_rel(path, root)} ---")
        parts.append(path.read_text(encoding="utf-8"))
    for path in scope.signatures:
        parts.append(f"--- {_rel(path, root)} (signatures only) ---")
        parts.append(signatures_of(path.read_text(encoding="utf-8")))
    where = f"{diag.file}:{diag.line}" if diag.line else str(diag.file)
    parts += ["--- failing test ---", f"{where}  {diag.message}", ""]
    for turn in history:
        parts.append(f"you: {turn.action}\nresult: {turn.result}")
    parts.append("Your action:")
    return "\n".join(parts)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_render.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/context/render.py tests/test_render.py
git commit -m "feat: prompt rendering with codec-specific help only"
```

#### Amendment (ruled 2026-08-09): nothing in rendering may crash or say "None"

Two defects in the reference code above.

**1. Unhandled reads.** `path.read_text(encoding="utf-8")` raises
`UnicodeDecodeError` on a non-UTF-8 file and `FileNotFoundError` on one
deleted between scope resolution and rendering. Either is an uncaught
exception, which the global constraints forbid. A file that cannot be read is
neither a model result nor an infrastructure failure — so report it in place,
in the prompt, and carry on.

**2. `diag.file is None` renders the word "None".** The location line branches
only on `diag.line`, so a `Diagnostic(False, None, None, …)` produces
`"None  tests timed out after 300s"`. That case is **reachable**: Task 4's
timeout path and its no-anchor fallback both return `file=None`.

```python
_UNREADABLE = "<unreadable or not valid UTF-8; not shown>\n"


def _read(path: Path) -> str | None:
    """None when the file cannot be read. Callers substitute a marker: a
    file vanishing or being binary must not end the run."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
```

Both loops substitute the marker, and the signatures loop must do so
*instead of* calling `signatures_of` — passing the marker through `ast.parse`
would swallow it and emit nothing:

```python
    for path in scope.full:
        text = _read(path)
        parts.append(f"--- {_rel(path, root)} ---")
        parts.append(text if text is not None else _UNREADABLE)
    for path in scope.signatures:
        text = _read(path)
        parts.append(f"--- {_rel(path, root)} (signatures only) ---")
        parts.append(signatures_of(text) if text is not None else _UNREADABLE)
```

and the location line branches on both fields:

```python
    if diag.file and diag.line:
        where = f"{diag.file}:{diag.line}"
    elif diag.file:
        where = diag.file
    else:
        where = "(location unknown)"
```

Three tests:

```python
def test_an_unreadable_file_is_reported_in_place_not_raised(tmp_path: Path):
    scope = _scope(tmp_path)
    (tmp_path / "a.py").write_bytes(b"\xff\xfe not utf-8\n")
    diag = Diagnostic(False, "a.py", 2, "boom", "raw")
    out = render(scope, diag, (), "search_replace", tmp_path)
    assert "unreadable" in out


def test_a_missing_signature_file_is_reported_in_place(tmp_path: Path):
    scope = _scope(tmp_path)
    (tmp_path / "b.py").unlink()
    diag = Diagnostic(False, "a.py", 2, "boom", "raw")
    out = render(scope, diag, (), "search_replace", tmp_path)
    assert "unreadable" in out


def test_a_diagnostic_with_no_file_never_renders_the_word_None(tmp_path: Path):
    # Reachable: the adapter returns file=None for a timed-out suite and
    # for a failure it could not anchor in the repo.
    diag = Diagnostic(False, None, None, "tests timed out after 300s", "")
    out = render(_scope(tmp_path), diag, (), "search_replace", tmp_path)
    assert "location unknown" in out
    assert "None:" not in out
```

---

### Task 7: Safety — path refusals and the git envelope

**Files:**
- Create: `src/robigo/apply/__init__.py`, `src/robigo/apply/safety.py`
- Test: `tests/test_safety.py`

**Interfaces:**
- Consumes: `Scope`.
- Produces: `RefusedError(Exception)`; `check_target(arg, root, scope, allow_test_edits=False) -> Path`; `ensure_repo(root) -> None`; `start_branch(root, slug) -> str`; `snapshot(root, message) -> None`; `commit_all(root, message) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_safety.py
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from robigo.apply.safety import (
    RefusedError,
    check_target,
    commit_all,
    ensure_repo,
    snapshot,
    start_branch,
)
from robigo.context.scope import Scope


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src.py").write_text("x = 1\n")
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert 0\n")
    (tmp_path / "outside.py").write_text("y = 1\n")
    for argv in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", *argv], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], cwd=tmp_path, check=True)
    return tmp_path


def _scope(repo: Path) -> Scope:
    return Scope(repo / "test_x.py", (repo / "test_x.py", repo / "src.py"), ())


def test_a_file_in_scope_is_allowed(repo: Path):
    assert check_target("src.py", repo, _scope(repo)) == (repo / "src.py").resolve()


def test_the_anchor_test_is_read_only_by_default(repo: Path):
    with pytest.raises(RefusedError) as e:
        check_target("test_x.py", repo, _scope(repo))
    assert "failing test" in str(e.value)


def test_the_anchor_test_can_be_opted_into(repo: Path):
    assert check_target("test_x.py", repo, _scope(repo), allow_test_edits=True)


def test_a_file_outside_scope_is_refused(repo: Path):
    with pytest.raises(RefusedError) as e:
        check_target("outside.py", repo, _scope(repo))
    assert "scope" in str(e.value)


@pytest.mark.parametrize("bad", ["../escape.py", "/etc/passwd"])
def test_paths_leaving_the_repo_are_refused(repo: Path, bad: str):
    with pytest.raises(RefusedError):
        check_target(bad, repo, _scope(repo))


def test_ensure_repo_refuses_a_non_repo(tmp_path: Path):
    with pytest.raises(RefusedError) as e:
        ensure_repo(tmp_path)
    assert "--no-git" in str(e.value)


def test_snapshot_commits_a_dirty_tree_so_nothing_is_lost(repo: Path):
    (repo / "src.py").write_text("x = 999\n")
    branch = start_branch(repo, "fog")
    snapshot(repo, "robigo: snapshot before first patch")
    assert branch.startswith("robigo/fog-")
    out = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                         capture_output=True, text=True).stdout
    assert out.strip() == ""


def test_commit_all_records_each_applied_patch(repo: Path):
    start_branch(repo, "fog")
    (repo / "src.py").write_text("x = 2\n")
    commit_all(repo, "robigo: patch src.py")
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo,
                         capture_output=True, text=True).stdout
    assert "robigo: patch src.py" in log
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_safety.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.apply'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/apply/safety.py
from __future__ import annotations

import subprocess
from pathlib import Path

from robigo.context.scope import Scope

_GIT_ID = ("-c", "user.email=robigo@localhost", "-c", "user.name=robigo")


class RefusedError(Exception):
    """A refusal, not a failure. Raised before anything is written."""


def check_target(
    arg: str, root: Path, scope: Scope, allow_test_edits: bool = False
) -> Path:
    target = (root / arg).resolve()
    root = root.resolve()
    if not target.is_relative_to(root):
        raise RefusedError(
            f"'{arg}' resolves outside the repository. Patch only files "
            f"you were shown."
        )
    if target == scope.anchor.resolve() and not allow_test_edits:
        raise RefusedError(
            f"'{arg}' is the failing test itself and is read-only. Fix the "
            f"code under test, not the test. (--allow-test-edits overrides.)"
        )
    if target not in {p.resolve() for p in scope.full}:
        raise RefusedError(
            f"'{arg}' is outside the current scope. Use `read {arg}` first, "
            f"or re-run with --scope to widen it."
        )
    return target


def ensure_repo(root: Path) -> None:
    if not (root / ".git").is_dir():
        raise RefusedError(
            f"{root} is not a git repository, so a run could not be undone. "
            f"Run `git init`, or pass --no-git to accept unreversible edits."
        )


def start_branch(root: Path, slug: str) -> str:
    """The first UNUSED name, not a count. Counting collides as soon as any
    earlier branch is deleted — two branches minus one deleted still counts
    1, and `git checkout -b` on a name that already exists aborts the run
    under check=True."""
    existing = set(
        subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.split()
    )
    number = 1
    while f"robigo/{slug}-{number}" in existing:
        number += 1
    branch = f"robigo/{slug}-{number}"
    subprocess.run(["git", "checkout", "-q", "-b", branch], cwd=root, check=True)
    return branch


def snapshot(root: Path, message: str) -> None:
    """Commit whatever is in the tree BEFORE the first patch, dirty or
    not, so a pre-existing uncommitted change can never be lost."""
    _commit(root, message, allow_empty=True)


def commit_all(root: Path, message: str) -> None:
    _commit(root, message, allow_empty=False)


def _commit(root: Path, message: str, allow_empty: bool) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    argv = ["git", *_GIT_ID, "commit", "-qm", message]
    if allow_empty:
        argv.append("--allow-empty")
    subprocess.run(argv, cwd=root, check=True)
```

Also create empty `src/robigo/apply/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_safety.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/apply tests/test_safety.py
git commit -m "feat: path refusals and the git snapshot envelope"
```

---

### Task 8: Atomic apply with post-write verification

**Files:**
- Create: `src/robigo/apply/patch.py`
- Test: `tests/test_apply_patch.py`

**Interfaces:**
- Consumes: `Action`, `CODECS`, `PatchError`, `Adapter`, `check_target`.
- Produces: `apply_patch(action, root, scope, adapter, codec, allow_test_edits=False) -> Path`; `write_atomic(path, text) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_apply_patch.py
from __future__ import annotations

from pathlib import Path

import pytest

from robigo.action.codec import PatchError
from robigo.action.verbs import Action
from robigo.adapters.python_ import PythonAdapter
from robigo.apply.patch import apply_patch, write_atomic
from robigo.context.scope import Scope

ORIGINAL = "def f():\n    return 1\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text(ORIGINAL)
    (tmp_path / "test_a.py").write_text("def test_a():\n    assert 0\n")
    return tmp_path


def _scope(repo: Path) -> Scope:
    return Scope(repo / "test_a.py", (repo / "test_a.py", repo / "a.py"), ())


def _patch(payload: str) -> Action:
    return Action("patch", "a.py", payload, "python")


def test_a_valid_patch_is_written(repo: Path):
    payload = "<<<<<<< SEARCH\n    return 1\n=======\n    return 2\n>>>>>>> REPLACE\n"
    apply_patch(_patch(payload), repo, _scope(repo), PythonAdapter(), "search_replace")
    assert (repo / "a.py").read_text() == "def f():\n    return 2\n"


def test_a_patch_producing_broken_syntax_is_rejected_and_nothing_is_written(repo: Path):
    payload = "<<<<<<< SEARCH\n    return 1\n=======\n    return (\n>>>>>>> REPLACE\n"
    with pytest.raises(PatchError) as e:
        apply_patch(_patch(payload), repo, _scope(repo), PythonAdapter(), "search_replace")
    assert "syntax" in str(e.value).lower()
    assert (repo / "a.py").read_text() == ORIGINAL


def test_write_atomic_leaves_no_partial_file_and_no_temp_files(repo: Path):
    write_atomic(repo / "a.py", "x = 1\n")
    assert (repo / "a.py").read_text() == "x = 1\n"
    assert list(repo.glob("*.tmp*")) == []


def test_a_patch_to_the_anchor_test_is_refused_before_any_write(repo: Path):
    from robigo.apply.safety import RefusedError

    before = (repo / "test_a.py").read_text()
    payload = "<<<<<<< SEARCH\n    assert 0\n=======\n    assert 1\n>>>>>>> REPLACE\n"
    with pytest.raises(RefusedError):
        apply_patch(Action("patch", "test_a.py", payload, None), repo,
                    _scope(repo), PythonAdapter(), "search_replace")
    assert (repo / "test_a.py").read_text() == before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_apply_patch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.apply.patch'`

- [ ] **Step 3: Write minimal implementation**

```python
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
            f"the result of this patch does not parse as valid "
            f"{adapter.name}, so it was not written. Check brackets and "
            f"indentation in the replacement lines."
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_apply_patch.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/apply/patch.py tests/test_apply_patch.py
git commit -m "feat: atomic apply, verified to parse before it lands"
```

---

### Task 9: Model client with mandatory `truncate: false`

**Files:**
- Create: `src/robigo/model/__init__.py`, `src/robigo/model/client.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Produces: `Generation(text, tokens_in, tokens_out, truncated)` (frozen); `ModelError`; `ContextOverflowError(ModelError)`; `ServerContextOverflowError(ContextOverflowError)`; `parse_http_error(exc) -> dict | None`; `raise_if_context_overflow(model, exc) -> dict | None`; `OllamaClient(model, *, window, num_predict, host, stop, retries, sleep)`; `LlamaCppClient(...)`; both with `.generate(prompt, *, seed) -> Generation`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_client.py
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from robigo.model.client import (
    ContextOverflowError,
    Generation,
    ModelError,
    OllamaClient,
    ServerContextOverflowError,
    parse_http_error,
)

OVERFLOW = {"error": {"type": "exceed_context_size_error",
                      "message": "request (6648 tokens) exceeds the available "
                                 "context size (2048 tokens)"}}
MALFORMED = {"error": {"type": "invalid_request_error",
                       "message": "'messages' is required"}}


def _http_error(body: dict) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://x", code=400, msg="Bad Request", hdrs=None,
        fp=io.BytesIO(json.dumps(body).encode()),
    )


def _ollama_error(body: dict) -> urllib.error.HTTPError:
    """Ollama proxies the same object as a JSON *string* one level deeper."""
    return _http_error({"error": json.dumps(body)})


class _FakeHTTP:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict | None]] = []

    def __call__(self, url, payload=None, timeout_s=120):
        self.calls.append((url, payload))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(monkeypatch, http) -> OllamaClient:
    monkeypatch.setattr("robigo.model.client._request", http)
    return OllamaClient("m", window=2048, sleep=lambda _s: None)


def _reply(text="ok", done_reason="stop"):
    return {"message": {"content": text}, "done_reason": done_reason,
            "prompt_eval_count": 12, "eval_count": 3}


def test_generate_sends_truncate_false_at_the_top_level(monkeypatch):
    http = _FakeHTTP(_reply())
    _client(monkeypatch, http).generate("hi", seed=1)
    payload = http.calls[0][1]
    # Without this the daemon accepts an oversized prompt, discards the
    # FRONT of it -- the system prompt and verb list -- and answers
    # anyway. Measured: 3160 tokens into a 256-token window returned 200
    # with prompt_eval_count 130 (spec section 9 law 5).
    assert payload["truncate"] is False
    # Top-level ONLY: nested in options the daemon silently ignores it.
    assert "truncate" not in payload["options"]


def test_generate_returns_a_populated_generation(monkeypatch):
    gen = _client(monkeypatch, _FakeHTTP(_reply("hello"))).generate("hi", seed=1)
    assert gen == Generation(text="hello", tokens_in=12, tokens_out=3, truncated=False)


def test_a_generation_stopped_at_the_cap_is_marked_truncated(monkeypatch):
    gen = _client(monkeypatch, _FakeHTTP(_reply(done_reason="length"))).generate("hi", seed=1)
    assert gen.truncated is True


def test_parse_http_error_unwraps_both_wire_shapes():
    assert parse_http_error(_http_error(OVERFLOW)) == OVERFLOW["error"]
    assert parse_http_error(_ollama_error(OVERFLOW)) == OVERFLOW["error"]


def test_parse_http_error_keeps_a_plain_string_error_as_a_message():
    assert parse_http_error(_http_error({"error": "model not found"})) == {
        "message": "model not found"
    }


def test_an_overflow_400_raises_the_subclass_without_retrying(monkeypatch):
    http = _FakeHTTP(_ollama_error(OVERFLOW))
    with pytest.raises(ServerContextOverflowError):
        _client(monkeypatch, http).generate("hi", seed=1)
    assert len(http.calls) == 1


def test_a_non_overflow_400_is_retried_and_surfaces_the_server_message(monkeypatch):
    http = _FakeHTTP(*[_ollama_error(MALFORMED) for _ in range(3)])
    with pytest.raises(ModelError) as e:
        _client(monkeypatch, http).generate("hi", seed=1)
    assert not isinstance(e.value, ContextOverflowError)
    assert len(http.calls) == 3
    assert "'messages' is required" in str(e.value)


def test_a_malformed_200_is_infrastructure_not_an_empty_generation(monkeypatch):
    # An empty Generation would be parsed, fail, and be recorded as a
    # model failure -- infrastructure misread as a result.
    with pytest.raises(ModelError):
        _client(monkeypatch, _FakeHTTP({"message": None})).generate("hi", seed=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.model'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/model/client.py
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Sequence

OLLAMA_HOST = "http://127.0.0.1:11434"
LLAMACPP_HOST = "http://127.0.0.1:8081"


class ModelError(Exception):
    """Infrastructure failure, and nothing else. A model that rambles or
    stops at the cap is a RESULT (spec section 9 law 10)."""


class ContextOverflowError(ModelError):
    """Prompt plus reserved generation exceeds the window."""


class ServerContextOverflowError(ContextOverflowError):
    """The server's real tokenizer rejected a prompt. Distinct so it is
    never retried -- deterministic once rejected -- and so records can
    say which check caught it."""


@dataclass(frozen=True)
class Generation:
    text: str
    tokens_in: int
    tokens_out: int
    truncated: bool


def _request(url: str, payload: dict | None = None, timeout_s: int = 120) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode())


def parse_http_error(exc: urllib.error.HTTPError) -> dict | None:
    """The server's error object. llama.cpp nests a dict under "error";
    Ollama proxies the same object as a JSON *string* one level deeper,
    which a bare isinstance-dict test silently discards."""
    try:
        body = json.loads(exc.read().decode("utf-8"))
    except Exception:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, str):
        try:
            inner = json.loads(error)
        except Exception:
            return {"message": error}
        error = inner.get("error", inner) if isinstance(inner, dict) else None
    return error if isinstance(error, dict) else None


def raise_if_context_overflow(
    model: str, exc: urllib.error.HTTPError
) -> dict | None:
    error = parse_http_error(exc)
    if error is not None and error.get("type") == "exceed_context_size_error":
        raise ServerContextOverflowError(
            f"{model}: the server rejected the prompt as exceeding its "
            f"window ({error.get('message') or 'context size exceeded'})."
        ) from exc
    return error


class _HTTPClient:
    """Shared transport. HTTPError is caught ahead of URLError because it
    IS a URLError subclass; listing it second would retry an overflow
    three times and then misreport it as a transport failure."""

    def __init__(
        self,
        model: str,
        *,
        window: int,
        num_predict: int = 1024,
        host: str = "",
        stop: Sequence[str] = (),
        temperature: float = 0.2,
        timeout_s: int = 300,
        retries: int = 3,
        backoff_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.window = window
        self.num_predict = num_predict
        self.host = (host or self.default_host).rstrip("/")
        self.stop = list(stop)
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self._sleep = sleep

    default_host = OLLAMA_HOST

    def _call(self, url: str, payload: dict | None = None) -> dict:
        last: Exception | str | None = None
        for attempt in range(self.retries):
            try:
                return _request(url, payload, self.timeout_s)
            except urllib.error.HTTPError as exc:
                error = raise_if_context_overflow(self.model, exc)
                message = error.get("message") if error else None
                last = f"{exc} ({message})" if message else exc
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last = exc
            if attempt < self.retries - 1:
                self._sleep(self.backoff_s * (2**attempt))
        raise ModelError(f"{self.model}: {self.retries} attempts failed: {last}")


class OllamaClient(_HTTPClient):
    default_host = OLLAMA_HOST

    def generate(self, prompt: str, *, seed: int) -> Generation:
        body = self._call(
            f"{self.host}/api/chat",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # TOP-LEVEL, never inside options: nested there the daemon
                # ignores it and front-truncates silently.
                "truncate": False,
                "options": {
                    "temperature": self.temperature,
                    "seed": seed,
                    "num_predict": self.num_predict,
                    "num_ctx": self.window,
                    "stop": self.stop,
                },
            },
        )
        message = body.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or body.get("error"):
            raise ModelError(f"{self.model}: malformed 200 response: {body!r}")
        return Generation(
            text=content,
            tokens_in=int(body.get("prompt_eval_count", 0)),
            tokens_out=int(body.get("eval_count", 0)),
            truncated=body.get("done_reason") == "length",
        )


class LlamaCppClient(_HTTPClient):
    default_host = LLAMACPP_HOST

    def generate(self, prompt: str, *, seed: int) -> Generation:
        body = self._call(
            f"{self.host}/v1/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "seed": seed,
                "temperature": self.temperature,
                "max_tokens": self.num_predict,
                "stop": self.stop,
            },
        )
        choices = body.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ModelError(f"{self.model}: malformed 200 response: {body!r}")
        usage = body.get("usage") or {}
        return Generation(
            text=content,
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            truncated=first.get("finish_reason") == "length",
        )
```

Also create empty `src/robigo/model/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_client.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/model tests/test_client.py
git commit -m "feat: model clients with mandatory top-level truncate:false"
```

---

### Task 10: The turn loop and its terminal states

**Files:**
- Create: `src/robigo/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `RunResult(outcome: str, turns: int, exit_code: int, branch: str | None, detail: str)` (frozen); `OUTCOMES: dict[str, int]`; `run(task, root, client, adapter, *, codec, turn_cap=8, allow_test_edits=False, use_git=True, stall_cap=3) -> RunResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loop.py
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from robigo.adapters.python_ import PythonAdapter
from robigo.loop import OUTCOMES, run
from robigo.model.client import Generation


class _ScriptedClient:
    """A model whose replies are fixed. The loop must be testable with no
    GPU, or it cannot be tested at all."""

    def __init__(self, *replies: str, truncated: bool = False) -> None:
        self.replies = list(replies)
        self.truncated = truncated
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, seed: int) -> Generation:
        self.prompts.append(prompt)
        text = self.replies.pop(0) if self.replies else "done nothing left"
        return Generation(text, 10, 5, self.truncated)


FIX = """patch src/fog.py
```python
<<<<<<< SEARCH
    return t
=======
    return t * 2
>>>>>>> REPLACE
```
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "fog.py").write_text("def radius(t):\n    return t\n")
    (tmp_path / "tests" / "test_fog.py").write_text(
        "import sys\nsys.path.insert(0, 'src')\nfrom fog import radius\n\n"
        "def test_radius():\n    assert radius(2) == 4\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def test_a_correct_patch_reaches_pass(repo: Path):
    result = run("make the failing test pass", repo,
                 _ScriptedClient(FIX), PythonAdapter(), codec="search_replace")
    assert (result.outcome, result.exit_code) == ("pass", 0)
    assert result.turns == 1
    assert (repo / "src" / "fog.py").read_text().endswith("return t * 2\n")


def test_a_truncated_generation_is_never_applied(repo: Path):
    before = (repo / "src" / "fog.py").read_text()
    client = _ScriptedClient(FIX, FIX, FIX, truncated=True)
    result = run("fix", repo, client, PythonAdapter(),
                 codec="search_replace", turn_cap=3)
    assert result.outcome == "stalled"
    assert (repo / "src" / "fog.py").read_text() == before


def test_a_parse_failure_is_fed_back_and_costs_a_turn(repo: Path):
    client = _ScriptedClient("edit src/fog.py", FIX)
    result = run("fix", repo, client, PythonAdapter(), codec="search_replace")
    assert result.outcome == "pass"
    assert result.turns == 2
    # The diagnostic must reach the model, or it repairs blind.
    assert "not a verb" in client.prompts[1]


def test_repeating_an_identical_failing_patch_stalls(repo: Path):
    miss = FIX.replace("    return t\n", "    return t;\n")
    result = run("fix", repo, _ScriptedClient(miss, miss, miss, miss),
                 PythonAdapter(), codec="search_replace", stall_cap=3)
    assert (result.outcome, result.exit_code) == ("stalled", OUTCOMES["stalled"])


def test_the_turn_cap_ends_the_run(repo: Path):
    result = run("fix", repo, _ScriptedClient("run", "run", "run", "run"),
                 PythonAdapter(), codec="search_replace", turn_cap=2)
    assert result.turns == 2
    assert result.outcome == "stalled"


def test_a_passing_suite_refuses_before_any_generation(repo: Path):
    (repo / "src" / "fog.py").write_text("def radius(t):\n    return t * 2\n")
    client = _ScriptedClient(FIX)
    result = run("fix", repo, client, PythonAdapter(), codec="search_replace")
    assert (result.outcome, result.exit_code) == ("refused", OUTCOMES["refused"])
    assert "failing test" in result.detail
    assert client.prompts == []


def test_overflow_with_evidence_is_budget_exhausted_not_infrastructure(repo: Path):
    # Law 3: with at least one attempt already made, running out of window
    # is a SESSION RESULT with the work preserved -- not an abort. Mapping
    # it to infrastructure would discard real evidence and misreport a
    # model-side limit as a broken daemon.
    from robigo.model.client import ServerContextOverflowError

    class _OverflowsOnTurnTwo(_ScriptedClient):
        def generate(self, prompt: str, *, seed: int) -> Generation:
            if seed > 1:
                raise ServerContextOverflowError("prompt exceeds the window")
            return super().generate(prompt, seed=seed)

    result = run("fix", repo, _OverflowsOnTurnTwo("read src/fog.py"),
                 PythonAdapter(), codec="search_replace")
    assert (result.outcome, result.exit_code) == ("budget_exhausted", 2)
    assert result.turns == 2


def test_overflow_with_no_evidence_is_refused(repo: Path):
    # Zero attempts submitted: nothing to preserve, so it is a loud
    # refusal rather than a fabricated result (law 3, other branch).
    from robigo.model.client import ServerContextOverflowError

    class _OverflowsImmediately(_ScriptedClient):
        def generate(self, prompt: str, *, seed: int) -> Generation:
            raise ServerContextOverflowError("prompt exceeds the window")

    result = run("fix", repo, _OverflowsImmediately(), PythonAdapter(),
                 codec="search_replace")
    assert (result.outcome, result.exit_code) == ("refused", 3)


def test_a_run_is_branch_scoped_and_snapshots_first(repo: Path):
    result = run("fix", repo, _ScriptedClient(FIX), PythonAdapter(),
                 codec="search_replace")
    assert result.branch is not None and result.branch.startswith("robigo/")
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo,
                         capture_output=True, text=True).stdout
    assert "snapshot" in log
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.loop'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/loop.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from robigo.action.codec import PatchError
from robigo.action.verbs import ActionParseError, parse
from robigo.adapters.base import Adapter
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
    except (RefusedError, ScopeError) as exc:
        return _result("refused", 0, None, str(exc))
    except (ModelError, AdapterError) as exc:
        # AdapterError means the project's tests cannot be run at all --
        # infrastructure, never a model result (Task 4's amendment).
        return _result("infrastructure", 0, None, str(exc))

    branch = None
    if use_git:
        branch = start_branch(root, _slug(task))
        snapshot(root, "robigo: snapshot before first patch")

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

        action_text, result_text, applied = _take_turn(
            gen, root, scope, adapter, codec, allow_test_edits
        )
        history = (history + (Turn(action_text, result_text),))[-2:]

        if applied:
            commit_all(root, f"robigo: {action_text}")
            diag = adapter.run(root, None)
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
    """→ (action label, result text fed back, whether a file changed)."""
    try:
        action = parse(gen.text)
    except ActionParseError as exc:
        return "<unparseable>", f"ACTION PARSE FAILED\n{exc}", False

    label = f"{action.verb} {action.arg}".strip()
    if action.verb == "patch":
        if gen.truncated:
            # The likeliest real data loss in the design: a whole-file
            # emission cut at the cap would faithfully write a gutted
            # file (spec section 6).
            return label, (
                "REJECTED: your reply was cut off at the token limit, so "
                "the patch was not applied. Send a smaller edit."
            ), False
        try:
            apply_patch(action, root, scope, adapter, codec, allow_test_edits)
        except (PatchError, RefusedError) as exc:
            return label, f"PATCH REJECTED\n{exc}", False
        return label, "applied", True
    if action.verb == "read":
        return label, _read(root, action.arg), False
    if action.verb == "find":
        return label, _find(root, action.arg), False
    if action.verb == "done":
        return label, "the tests do not pass yet; keep going", False
    return label, "tests re-run", True


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_loop.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add src/robigo/loop.py tests/test_loop.py
git commit -m "feat: turn loop with distinct terminal states and a stall guard"
```

---

### Task 11: CLI, exit codes, and a live smoke test

**Files:**
- Create: `src/robigo/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `run`, `OUTCOMES`, `OllamaClient`, `LlamaCppClient`, `PythonAdapter`.
- Produces: `main(argv: list[str] | None = None) -> int`; `build_client(args)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from robigo.cli import build_client, main
from robigo.model.client import LlamaCppClient, OllamaClient


def test_backend_selection_and_window_pass_through():
    ollama = build_client(_args(backend="ollama", model="m", window=4096))
    llama = build_client(_args(backend="llamacpp", model="m", window=8192))
    assert isinstance(ollama, OllamaClient) and ollama.window == 4096
    assert isinstance(llama, LlamaCppClient) and llama.window == 8192


def _args(**kw):
    import argparse

    defaults = dict(backend="ollama", model="m", window=8192, host=None,
                    num_predict=1024)
    return argparse.Namespace(**{**defaults, **kw})


def test_exit_code_is_3_when_the_suite_already_passes(tmp_path: Path, capsys):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert 1\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    code = main(["--root", str(tmp_path), "--model", "m", "fix it"])
    assert code == 3
    assert "failing test" in capsys.readouterr().out


def test_exit_code_is_3_outside_a_git_repo(tmp_path: Path):
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert 0\n")
    assert main(["--root", str(tmp_path), "--model", "m", "fix it"]) == 3


@pytest.mark.live
def test_live_one_real_repair(tmp_path: Path):
    """One real generation end to end. Asserts the plumbing works, NOT
    that the model succeeds -- a failure here is a valid result."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("def double(x):\n    return x\n")
    (tmp_path / "test_m.py").write_text(
        "import sys\nsys.path.insert(0, 'src')\nfrom m import double\n\n"
        "def test_double():\n    assert double(2) == 4\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    code = main([
        "--root", str(tmp_path), "--model", "qwen2.5-coder:7b-instruct-q8_0",
        "--window", "8192", "make the failing test pass",
    ])
    assert code in (0, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/cli.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robigo.adapters.python_ import PythonAdapter
from robigo.loop import run
from robigo.model.client import LlamaCppClient, OllamaClient

_STOP = ("\nread ", "\nfind ", "\nrun\n", "\ndone ")


def build_client(args: argparse.Namespace):
    kind = LlamaCppClient if args.backend == "llamacpp" else OllamaClient
    return kind(
        args.model,
        window=args.window,
        num_predict=args.num_predict,
        host=args.host or "",
        stop=_STOP,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robigo")
    parser.add_argument("task")
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("ollama", "llamacpp"), default="ollama")
    parser.add_argument("--host", default=None)
    # Explicit for now. Plan 02 computes it from model geometry and free
    # VRAM, and the advertised context length is never trusted.
    parser.add_argument("--window", type=int, default=8192)
    parser.add_argument("--num-predict", dest="num_predict", type=int, default=1024)
    parser.add_argument("--codec", choices=("search_replace", "whole_file"),
                        default="search_replace")
    parser.add_argument("--turn-cap", dest="turn_cap", type=int, default=8)
    parser.add_argument("--allow-test-edits", dest="allow_test_edits",
                        action="store_true")
    parser.add_argument("--no-git", dest="use_git", action="store_false")
    args = parser.parse_args(argv)

    result = run(
        args.task,
        Path(args.root).resolve(),
        build_client(args),
        PythonAdapter(),
        codec=args.codec,
        turn_cap=args.turn_cap,
        allow_test_edits=args.allow_test_edits,
        use_git=args.use_git,
    )
    print(f"{result.outcome}  turns={result.turns}  {result.detail}")
    if result.branch:
        print(f"branch {result.branch} — `git checkout -` to undo everything")
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v` then `pytest -q`
Expected: 3 passed, 1 deselected (live); full suite green

- [ ] **Step 5: Commit**

```bash
git add src/robigo/cli.py tests/test_cli.py
git commit -m "feat: cli with distinct exit codes and a live smoke test"
```

#### Amendment (ruled 2026-08-09): `--scope` must actually exist

Three refusal messages — Task 5's two `ScopeError`s and the budget-exhaustion
refusal in plan 02 — tell the user to pass `--scope`, and spec §3 and §6 both
reference it, but this task's CLI never defined it. A refusal that names a
nonexistent flag is worse than one that names nothing.

Add the flag, and an explicit scope builder that bypasses import tracing:

```python
    parser.add_argument("--scope", type=Path, nargs="+", default=None,
                        metavar="PATH",
                        help="files or directories to work in, instead of "
                             "tracing imports from the failing test")
```

In `src/robigo/context/scope.py`, add:

```python
def explicit(diag: Diagnostic, root: Path, paths: Sequence[Path]) -> Scope:
    """Scope drawn by the user rather than traced. The anchor still comes
    from the diagnostic — the failing test is what the run is about — but
    nothing is inferred beyond the paths given."""
    if not diag.file:
        raise ScopeError(
            "--scope needs a failing test to anchor on, and the test output "
            "named no file."
        )
    anchor = (root / diag.file).resolve()
    full: list[Path] = [anchor]
    for given in paths:
        target = (root / given).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ScopeError(f"--scope path {given} is outside {root}")
        found = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for path in found:
            if path.is_file() and path not in full:
                full.append(path)
    return Scope(anchor=anchor, full=tuple(full), signatures=())
```

and in `loop.run`, take an optional `scope_paths` and use it:

```python
        scope = (
            explicit(diag, root, scope_paths) if scope_paths
            else resolve(diag, adapter, root)
        )
```

Signature-only hops are deliberately empty here: the user drew the box, so
nothing is added they did not name. If the result is too large, plan 02's
degradation ladder and refusal handle it.

- [ ] **Step 6: Run the live test against a real model**

Run: `pytest tests/test_cli.py -m live -v`
Expected: PASS. If it errors rather than fails, the plumbing is wrong; a
model that simply cannot fix the bug returns exit 1 and the test still passes.

---

### Task 12: Run records

**Files:**
- Create: `src/robigo/record.py`
- Modify: `src/robigo/loop.py`
- Test: `tests/test_record.py`

**Interfaces:**
- Consumes: `RunResult`.
- Produces: `RunRecorder(root: Path, run_id: str)` with `turn(prompt: str, reply: str, adapter_raw: str) -> None` and `finish(result: RunResult, model: str, window: int, codec: str) -> None`; `next_run_id(root: Path, slug: str) -> str`.

Required by spec §6.2. **Raw model outputs are stored verbatim**, which is what makes a user's bug report actionable — and per spec §5.1 these records double as corpus candidates for plan 04, so the format is not incidental.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_record.py
from __future__ import annotations

import json
from pathlib import Path

from robigo.loop import RunResult
from robigo.record import RunRecorder, next_run_id


def test_run_ids_increment_without_collision(tmp_path: Path):
    assert next_run_id(tmp_path, "fog") == "fog-1"
    (tmp_path / ".robigo" / "runs" / "fog-1").mkdir(parents=True)
    assert next_run_id(tmp_path, "fog") == "fog-2"


def test_a_turn_stores_the_reply_byte_for_byte(tmp_path: Path):
    recorder = RunRecorder(tmp_path, "fog-1")
    # Trailing whitespace and CRLF are exactly what breaks SEARCH blocks,
    # so the record must not normalise anything.
    raw = "patch a.py\r\n```\nx = 1   \n```\n"
    recorder.turn("the prompt", raw, "pytest said no")
    stored = (tmp_path / ".robigo" / "runs" / "fog-1" / "turn-01-reply.txt")
    assert stored.read_text(newline="") == raw


def test_finish_writes_machine_readable_meta(tmp_path: Path):
    recorder = RunRecorder(tmp_path, "fog-1")
    recorder.turn("p", "r", "a")
    recorder.finish(RunResult("pass", 1, 0, "robigo/fog-1", "tests pass"),
                    model="m", window=8192, codec="search_replace")
    meta = json.loads((tmp_path / ".robigo" / "runs" / "fog-1" / "meta.json").read_text())
    assert meta["outcome"] == "pass"
    assert meta["turns"] == 1
    assert meta["model"] == "m"
    assert meta["window"] == 8192
    assert meta["branch"] == "robigo/fog-1"


def test_turns_are_numbered_in_order(tmp_path: Path):
    recorder = RunRecorder(tmp_path, "fog-1")
    for i in range(3):
        recorder.turn(f"p{i}", f"r{i}", "a")
    names = sorted(p.name for p in (tmp_path / ".robigo" / "runs" / "fog-1").glob("turn-*-reply.txt"))
    assert names == ["turn-01-reply.txt", "turn-02-reply.txt", "turn-03-reply.txt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_record.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'robigo.record'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/robigo/record.py
from __future__ import annotations

import json
from pathlib import Path


def next_run_id(root: Path, slug: str) -> str:
    """The first unused id, not a count — counting collides as soon as any
    earlier run directory is deleted, and would then overwrite it."""
    runs = root / ".robigo" / "runs"
    number = 1
    while (runs / f"{slug}-{number}").exists():
        number += 1
    return f"{slug}-{number}"


class RunRecorder:
    """Prompts, raw replies, and adapter output, verbatim. Verbatim is the
    point: trailing whitespace and line endings are exactly what breaks a
    SEARCH block, so normalising here would erase the evidence. These
    records are also corpus candidates (spec 5.1)."""

    def __init__(self, root: Path, run_id: str) -> None:
        self.dir = root / ".robigo" / "runs" / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._turns = 0

    def turn(self, prompt: str, reply: str, adapter_raw: str) -> None:
        self._turns += 1
        stem = f"turn-{self._turns:02d}"
        self._write(f"{stem}-prompt.txt", prompt)
        self._write(f"{stem}-reply.txt", reply)
        self._write(f"{stem}-adapter.txt", adapter_raw)

    def finish(
        self, result, model: str, window: int, codec: str
    ) -> None:
        self._write("meta.json", json.dumps({
            "outcome": result.outcome, "turns": result.turns,
            "exit_code": result.exit_code, "branch": result.branch,
            "detail": result.detail, "model": model, "window": window,
            "codec": codec,
        }, indent=2, sort_keys=True))

    def _write(self, name: str, text: str) -> None:
        (self.dir / name).write_text(text, encoding="utf-8", newline="")
```

Then wire it into `src/robigo/loop.py`. Add the import and an optional parameter:

```python
from robigo.record import RunRecorder, next_run_id
```

Add `recorder: RunRecorder | None = None` to `run`'s keyword arguments, create one when `use_git` produced a branch:

```python
    if use_git:
        branch = start_branch(root, _slug(task))
        snapshot(root, "robigo: snapshot before first patch")
    if recorder is None:
        recorder = RunRecorder(root, next_run_id(root, _slug(task)))
```

record each turn immediately after `_take_turn` returns:

```python
        recorder.turn(prompt, gen.text, diag.raw)
```

and call `recorder.finish(...)` on every return path by wrapping the body — simplest correct form is to compute the result into a local and finish once:

```python
    result = _run_turns(...)          # the existing loop body, extracted
    recorder.finish(result, model=getattr(client, "model", "?"),
                    window=getattr(client, "window", 0), codec=codec)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_record.py tests/test_loop.py -v`
Expected: PASS — 4 record tests, 9 loop tests still green

- [ ] **Step 5: Commit**

```bash
git add src/robigo/record.py src/robigo/loop.py tests/test_record.py
git commit -m "feat: verbatim run records that double as corpus candidates"
```

---

## Done when

- `pytest -q` is green, `pytest -m live -v` passes against a served model.
- `robigo --model <m> "make the failing test pass"` repairs a single-defect
  Python failure in a scratch repo, on a branch, reversible with
  `git checkout -`.
- Exit codes 0/1/2/3/4 are all reachable and observed in tests, with 2
  produced only by the evidence gate's with-evidence branch.
- `.robigo/runs/<id>/` holds every prompt and raw reply byte-for-byte.
- No runtime dependency outside the standard library: `pip install .` in a
  clean venv, then `python -c "import robigo"` with nothing else installed.
