# robigo plan 05 — the repair gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build profiler stages 4 (repair) and 5 (loop discipline), plus the two CLI prerequisites they need, then read stage 4's number and apply the §0.2 kill criterion at 33.3%.

**Architecture:** Stage 4 composes existing, already-mutation-tested parts rather than reimplementing them: plan 04's clone/baseline/pytest harness judges the result, and `loop.run` — the shipped tool, unmodified in behaviour — performs the repair. A new `profile/repair.py` owns one attempt and the aggregate over records × seeds. Stage 5 derives two metrics from stage 4's runs with zero extra model calls.

**Tech Stack:** Python 3.12+, zero runtime dependencies, pytest, Ollama at `http://127.0.0.1:11434`.

**Spec:** `docs/superpowers/specs/2026-08-10-robigo-05-repair-gate-design.md`. Read §1 (the four neutral decisions) and §4 (stage 4) before Task 4.

## Global Constraints

- **Python 3.12+, zero runtime dependencies.** Nothing new may enter `pyproject.toml`'s runtime deps.
- **Every new test is mutation-tested.** A test that still passes with the code under test deleted or inverted is a defect, not a test. Record what you mutated and what happened.
- **The offline suite stays green with `socket.socket.connect` patched to raise.** Nothing added here may touch the network in a non-`live` test.
- **`PYTHONPATH` is forced and asserted in any subprocess that imports a clone.** This project is installed editable; without it a subprocess silently imports `~/workspace/robigo/src` and inverts the result. Measured: 8 of 8 mutants falsely surviving.
- **Sanitize the environment being measured in.** `PYTEST_ADDOPTS=-x` exported once turned a correctly-rejected mutant into a kept record.
- **Anything not measured is stated in `dropped`.** Never `None` with no explanation, never `0.0` standing in for "not measured".
- **Do not use `timeout(1)` on this box** — the uutils build SIGSEGVs while reaping multithreaded children; exit 139 is the wrapper, not the program.
- **Baseline before you start:** 626 tests (625 passing + 1 skipped without `OLLAMA_MODELS`; 2 deselected are `live`). Every task must leave the suite green.
- **Branch:** `feat/repair-gate`, already created, spec already committed at `6c635b1`.

### Verified externals (checked 2026-08-10 — do not re-derive, do not assume beyond this list)

```python
# robigo.model.detect
plan_window(backend: str, model: str, host: str, user_cap: int | None, *,
            kv_bits: int = 16, gguf_path: Path | None = None) -> WindowPlan

# robigo.model.geometry
WindowPlan(window, limited_by, free_vram, kv_per_token,
           weights_bytes, overhead_bytes, training_ctx=0)

# robigo.loop
OUTCOMES = {"pass": 0, "stalled": 1, "budget_exhausted": 2,
            "refused": 3, "infrastructure": 4}          # NOTE: "pass", NOT "repaired"
RunResult(outcome: str, turns: int, exit_code: int, branch: str | None,
          detail: str, undo: UndoInfo | None = None, rungs: tuple[int, ...] = ())
run(task: str, root: Path, client: ModelClient, adapter: Adapter, *,
    codec: str, turn_cap: int = 8, allow_test_edits: bool = False,
    use_git: bool = True, stall_cap: int = 3,
    scope_paths: Sequence[Path] | None = None,
    recorder: RunRecorder | None = None) -> RunResult

# robigo.adapters.python_
PythonAdapter(python: str | None = None)

# robigo.cli
build_client(args: argparse.Namespace) -> ModelClient
    # args needs: .backend .model .window .num_predict .host

# robigo.profile.corpus_io
CorpusRecord(name: str, path: Path, line: int, broken: str, fixed: str,
             test_id: str, diagnostic: str, operator: str,
             source_repo: str, source_sha: str)
read_corpus(path: Path) -> tuple[str, tuple[CorpusRecord, ...], tuple[str, ...]]
read_corpus_baseline(path: Path) -> Baseline

# robigo.profile.verify
Runner = Callable[[Path, str], str]
Baseline(broken: int, executed: int, seconds: float)
pytest_runner(repo: Path, package: str) -> str
baseline(repo: Path, runner: Runner) -> Baseline
_primary_package(repo: Path) -> str
_package_name(relative: Path) -> str
_resolve_in_clone(repo: Path, relative: Path) -> Path
_broken_count(text) -> int; _broken_ids(text) -> tuple[str, ...]
_executed_total(text) -> int; _run_did_not_complete(text) -> str | None
_assert_in_clone(text: str, repo: Path) -> None      # raises WrongTreeError

# robigo.profile.fixtures
fixtures_from_corpus(records) -> FixturesFromCorpus(fixtures, dropped)
CORPUS_NAME = "fixtures-v1"

# robigo.profile.schema
Profile(...); CodecResult(lands, attempts, max_file_tokens)
Profile.best_codec() -> str | None
verdict_for(envelope_fidelity: float, codecs: dict, usable_window: int) -> str

# robigo.profile.report
run_profile(client, plan, *, model, quant, family, seeds, mode, corpus,
            kv_bits=16) -> Profile
profile_path(family: str) -> Path
render_table(profile: Profile) -> str
```

### File structure

| File | Responsibility |
|---|---|
| `src/robigo/profile/repair.py` | **new.** One repair attempt; the stage-4 aggregate. |
| `src/robigo/profile/discipline.py` | **new.** Stage 5's two metrics, derived from stage-4 attempt records. |
| `src/robigo/profile/verify.py` | **modify.** Add public `suite_state()`; existing privates unchanged. |
| `src/robigo/loop.py` | **modify.** Add `RunResult.repeats` (total, not consecutive). |
| `src/robigo/profile/schema.py` | **modify.** Four new `Profile` fields + round-trip. |
| `src/robigo/profile/report.py` | **modify.** Stage-4/5 gating, `dropped` rules, table rows. |
| `src/robigo/cli.py` | **modify.** `--window`, `--corpus`, `--repo` on `robigo profile`. |
| `tests/test_repair.py`, `tests/test_discipline.py`, `tests/test_suite_state.py` | **new.** |

---

### Task 1: `--window` on `robigo profile` (prerequisite P2)

Without this the best available family cannot be profiled on this box at all: `qwen2.5-coder:7b` resolves to its full 32768 training context (VRAM never binds — ~7.6 GiB weights + ~1.75 GiB KV against 14,558 MiB free), stage 0 probes past this box's measured ~11.5k daemon ceiling, and the run dies before stage 0 finishes.

**Files:**
- Modify: `src/robigo/cli.py` (`profile_main`, the `plan_window(...)` call passing `None`)
- Test: `tests/test_cli_profile.py`

**Interfaces:**
- Consumes: `plan_window(backend, model, host, user_cap, *, kv_bits, gguf_path)` — `user_cap` is already the 4th positional parameter and already implements `min(training_ctx, vram, user_cap)`. This task only exposes it.
- Produces: `robigo profile --window N`. Task 8 relies on it existing.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_profile.py
import argparse
from pathlib import Path
import pytest
from robigo import cli
from robigo.model.geometry import WindowPlan


def test_window_flag_is_passed_to_plan_window_as_the_user_cap(monkeypatch):
    seen = {}

    def fake_plan_window(backend, model, host, user_cap, *, kv_bits=16, gguf_path=None):
        seen["user_cap"] = user_cap
        return WindowPlan(window=4096, limited_by="user_cap", free_vram=None,
                          kv_per_token=57344, weights_bytes=0, overhead_bytes=0,
                          training_ctx=32768)

    monkeypatch.setattr(cli, "plan_window", fake_plan_window)
    monkeypatch.setattr(cli, "run_profile", lambda *a, **k: pytest.skip("not reached"))
    with pytest.raises(BaseException):
        cli.profile_main(["--model", "m", "--window", "4096"])
    assert seen["user_cap"] == 4096


def test_no_window_flag_still_passes_none(monkeypatch):
    seen = {}

    def fake_plan_window(backend, model, host, user_cap, *, kv_bits=16, gguf_path=None):
        seen["user_cap"] = user_cap
        raise SystemExit(99)

    monkeypatch.setattr(cli, "plan_window", fake_plan_window)
    with pytest.raises(SystemExit):
        cli.profile_main(["--model", "m"])
    assert seen["user_cap"] is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli_profile.py -v`
Expected: FAIL — `unrecognized arguments: --window`.

- [ ] **Step 3: Implement**

In `profile_main`'s parser, beside `--kv-bits`:

```python
    parser.add_argument(
        "--window", type=int, default=None,
        help="cap the window at N tokens; a CEILING only -- it can never "
             "raise the window above what geometry allows (spec 9 law 1). "
             "Needed on any box whose daemon rejects prompts below the "
             "model's training context.",
    )
```

And change the call from `None` to `args.window`:

```python
        plan = plan_window(args.backend, args.model, args.host or "", args.window,
                           kv_bits=args.kv_bits, gguf_path=args.gguf)
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli_profile.py -v`
Expected: PASS.

- [ ] **Step 5: Prove the ceiling invariant against the real function (P2.1)**

This is the invariant, not the flag. `--window` above the training context must change nothing.

```python
def test_window_above_training_ctx_does_not_raise_the_window():
    from robigo.model.geometry import usable_window, Geometry
    g = Geometry(layers=28, kv_heads=4, head_dim=128, training_ctx=4096)
    capped = usable_window(g, free_vram=None, user_cap=999_999, kv_bits=16)
    uncapped = usable_window(g, free_vram=None, user_cap=None, kv_bits=16)
    assert capped.window == uncapped.window
    assert capped.limited_by == "training_ctx"
```

Check `usable_window`'s real signature first (`src/robigo/model/geometry.py`) and adjust the constructor call to match it — do **not** transcribe the line above if it disagrees with the source. If it disagrees, report the true signature in your task report.

- [ ] **Step 6: Mutation-test**

Invert the flag wiring (`args.window` → `None`) and re-run. The first test must fail. If it passes, the test is vacuous — fix it before committing.

- [ ] **Step 7: Commit**

```bash
git add tests/test_cli_profile.py src/robigo/cli.py
git commit -m "feat: --window on robigo profile, unblocking the best family

Without a user cap, qwen2.5-coder:7b resolves to its full 32768 training
context (VRAM never binds here), stage 0 probes past this box's ~11.5k
daemon ceiling, and the profile dies before stage 0 finishes. A ceiling
only: plan_window's min() semantics are unchanged."
```

---

### Task 2: `--corpus` on `robigo profile` (prerequisite P1)

**Files:**
- Modify: `src/robigo/cli.py` (`profile_main`), `src/robigo/profile/report.py` (`run_profile` signature)
- Test: `tests/test_cli_profile.py`

**Interfaces:**
- Consumes: `read_corpus(path) -> (name, records, dropped)`; `fixtures_from_corpus(records) -> FixturesFromCorpus(fixtures, dropped)`.
- Produces: `robigo profile --corpus PATH`; `run_profile(..., fixtures=..., corpus_dropped=...)`. Tasks 7 and 8 rely on these names.

**The invariant that matters (spec P1.2):** `fixtures_from_corpus` drops ~9.2% of real records as unwrappable (measured: 91 of 986 from `src/robigo`). Those are **harness artifacts and are never scored as model failures.** They must be excluded from both numerator and denominator, and named in `dropped`.

- [ ] **Step 1: Write the failing test — the falsification test for P1.2**

```python
# tests/test_cli_profile.py
from pathlib import Path
from robigo.profile.corpus_io import CorpusRecord, write_corpus
from robigo.profile.verify import Baseline
from robigo.profile.fixtures import fixtures_from_corpus


def _record(name, broken, fixed):
    return CorpusRecord(
        name=name, path=Path("src/pkg/mod.py"), line=3, broken=broken,
        fixed=fixed, test_id="tests/test_mod.py::test_x",
        diagnostic="exactly one net new failure", operator="arith",
        source_repo="/tmp/src", source_sha="deadbeef",
    )


def test_unwrappable_records_leave_the_rate_identical_and_are_named(tmp_path):
    """A harness artifact must not reach the model's score. The rate from a
    corpus containing an unwrappable record must equal the rate from a corpus
    where that record is physically absent."""
    good = _record("good", "    return a - b\n", "    return a + b\n")
    # A single physical line cut from a multi-line expression: no wrapping
    # strategy at any indent forms a complete statement from it.
    bad = _record("bad", "        for x in (\n", "        for x in (1,\n")

    both = fixtures_from_corpus([good, bad])
    only_good = fixtures_from_corpus([good])

    assert len(both.fixtures) == len(only_good.fixtures) == 1
    assert any("bad" in note for note in both.dropped)
    assert only_good.dropped == ()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_cli_profile.py::test_unwrappable_records_leave_the_rate_identical_and_are_named -v`
Expected: PASS — this behaviour already exists (plan 04 task 4 built it). **This step is a characterization test, deliberately.** If it FAILS, stop and report BLOCKED: the spec's P1.2 assumption about existing behaviour is wrong, and every rate in this plan is affected.

- [ ] **Step 3: Write the failing test for the CLI wiring**

```python
def test_corpus_flag_routes_records_and_carries_dropped(tmp_path, monkeypatch):
    path = tmp_path / "corpus.json"
    good = _record("good", "    return a - b\n", "    return a + b\n")
    bad = _record("bad", "        for x in (\n", "        for x in (1,\n")
    write_corpus([good, bad], path, name="corpus-under-test",
                 dropped=("gen dropped: target foo abandoned",),
                 baseline=Baseline(broken=0, executed=120, seconds=0.4))

    seen = {}

    def fake_run_profile(client, plan, **kw):
        seen.update(kw)
        raise SystemExit(0)

    monkeypatch.setattr(cli, "run_profile", fake_run_profile)
    monkeypatch.setattr(cli, "plan_window", lambda *a, **k: WindowPlan(
        window=4096, limited_by="training_ctx", free_vram=None,
        kv_per_token=57344, weights_bytes=0, overhead_bytes=0, training_ctx=4096))
    monkeypatch.setattr(cli, "build_client", lambda a: object())

    with pytest.raises(SystemExit):
        cli.profile_main(["--model", "m", "--corpus", str(path)])

    assert seen["corpus"] == "corpus-under-test"       # the file's own name
    assert len(seen["fixtures"]) == 1                  # bad one excluded
    assert any("bad" in n for n in seen["corpus_dropped"])
    assert any("abandoned" in n for n in seen["corpus_dropped"])  # generator's too
```

- [ ] **Step 4: Run to verify it fails**

Expected: FAIL — `unrecognized arguments: --corpus`.

- [ ] **Step 5: Implement**

Parser:

```python
    parser.add_argument(
        "--corpus", type=Path, default=None,
        help="a corpus file from `robigo corpus`; without it the bundled "
             "fixtures-v1 is measured, which is not a publishable result",
    )
```

Then, before the `run_profile` call:

```python
    if args.corpus:
        corpus_name, records, gen_dropped = read_corpus(args.corpus)
        converted = fixtures_from_corpus(records)
        fixtures = converted.fixtures
        # Both sources of loss travel together: what the GENERATOR dropped
        # while mining, and what CONVERSION dropped as unwrappable. Neither
        # is a model failure and neither may be silently absent (P1.2).
        corpus_dropped = tuple(gen_dropped) + converted.dropped
    else:
        corpus_name, fixtures, corpus_dropped = CORPUS_NAME, FIXTURES, ()
```

and pass `corpus=corpus_name, fixtures=fixtures, corpus_dropped=corpus_dropped`.

In `report.run_profile`, add the two keyword parameters and thread `fixtures` into `stage2_codecs(client, seeds, fixtures=fixtures)`; extend `dropped` with every entry of `corpus_dropped`.

- [ ] **Step 6: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green, count ≥ 626.

- [ ] **Step 7: Mutation-test**

Delete the `corpus_dropped` extension of `dropped` in `run_profile`. The `abandoned`/`bad` assertions must fail.

- [ ] **Step 8: Commit**

```bash
git add tests/test_cli_profile.py src/robigo/cli.py src/robigo/profile/report.py
git commit -m "feat: --corpus on robigo profile, retiring fixtures-v1 as the live default

Carries BOTH loss channels into Profile.dropped: what the generator
dropped while mining and what conversion dropped as unwrappable (~9.2%,
measured 91 of 986). Neither is a model failure; scoring either as one
would put a harness artifact into the number the kill gate reads."
```

---

### Task 3: `suite_state()` — the public judging primitive

Stage 4 must ask "is this suite green, and did the run actually complete?" Every parser it needs already exists in `verify.py` as a module-private. Reaching into privates from a new module would create a second, drifting copy of the same knowledge — the exact failure mode plan 01's process lesson 2 names. One new public function instead.

**Files:**
- Modify: `src/robigo/profile/verify.py`
- Test: `tests/test_suite_state.py`

**Interfaces:**
- Consumes: `_broken_count`, `_broken_ids`, `_executed_total`, `_run_did_not_complete`, `_assert_in_clone` (all existing privates in the same module).
- Produces:

```python
@dataclass(frozen=True)
class SuiteState:
    broken: int
    executed: int
    broken_ids: tuple[str, ...]
    incomplete: str | None   # None when the run completed normally

def suite_state(repo: Path, runner: Runner, package: str) -> SuiteState: ...
```

Tasks 4 and 5 rely on these exact names.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_suite_state.py
from pathlib import Path
import pytest
from robigo.profile.verify import SuiteState, suite_state, WrongTreeError

CLEAN = ("MODULE_UNDER_TEST=/clone/src/pkg/__init__.py\n"
         "120 passed\nEXIT_CODE=0\n")
ONE_BAD = ("MODULE_UNDER_TEST=/clone/src/pkg/__init__.py\n"
           "FAILED tests/test_mod.py::test_x\n"
           "119 passed, 1 failed\nEXIT_CODE=1\n")
INTERRUPTED = ("MODULE_UNDER_TEST=/clone/src/pkg/__init__.py\n"
               "Interrupted: 1 error during collection\n"
               "1 error\nEXIT_CODE=2\n")


def test_clean_suite(tmp_path):
    s = suite_state(tmp_path, lambda repo, pkg: CLEAN.replace("/clone", str(tmp_path)), "pkg")
    assert s.broken == 0 and s.executed == 120
    assert s.broken_ids == () and s.incomplete is None


def test_one_failure_is_identified_by_id(tmp_path):
    s = suite_state(tmp_path, lambda repo, pkg: ONE_BAD.replace("/clone", str(tmp_path)), "pkg")
    assert s.broken == 1
    assert s.broken_ids == ("tests/test_mod.py::test_x",)
    assert s.incomplete is None


def test_an_interrupted_run_is_reported_not_counted(tmp_path):
    s = suite_state(tmp_path, lambda repo, pkg: INTERRUPTED.replace("/clone", str(tmp_path)), "pkg")
    assert s.incomplete is not None
    assert "Interrupted" in s.incomplete


def test_a_run_outside_the_clone_raises(tmp_path):
    outside = CLEAN.replace("/clone", "/somewhere/else")
    with pytest.raises(WrongTreeError):
        suite_state(tmp_path, lambda repo, pkg: outside, "pkg")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_suite_state.py -v`
Expected: FAIL — `cannot import name 'SuiteState'`.

- [ ] **Step 3: Implement in `verify.py`**

```python
@dataclass(frozen=True)
class SuiteState:
    """One suite run, parsed. `incomplete` is non-None when the run did not
    finish normally (a collection error, an `-x` early exit, INTERNALERROR)
    -- in which case `broken` and `executed` describe a run that never ran
    everything, and no caller may compare them against a baseline as though
    they did. Plan 04's process lesson 2: a tool that reports a count must
    check the run finished."""

    broken: int
    executed: int
    broken_ids: tuple[str, ...]
    incomplete: str | None


def suite_state(repo: Path, runner: Runner, package: str) -> SuiteState:
    """`repo`'s current suite state, whatever is on disk right now. Asserts
    invariant 7 (the tests imported the CLONE, not an installed package)
    before any number it returns is trusted."""
    text = runner(repo, package)
    _assert_in_clone(text, repo)
    return SuiteState(
        broken=_broken_count(text),
        executed=_executed_total(text),
        broken_ids=_broken_ids(text),
        incomplete=_run_did_not_complete(text),
    )
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_suite_state.py -v`
Expected: PASS. If `_assert_in_clone`'s real signature or raising behaviour differs from the Verified Externals block, report BLOCKED with the true behaviour rather than adapting the test to whatever it does.

- [ ] **Step 5: Mutation-test**

Replace `incomplete=_run_did_not_complete(text)` with `incomplete=None`. `test_an_interrupted_run_is_reported_not_counted` must fail. Then replace `_assert_in_clone(text, repo)` with `pass`; the fourth test must fail.

- [ ] **Step 6: Commit**

```bash
git add tests/test_suite_state.py src/robigo/profile/verify.py
git commit -m "feat: public suite_state() for stage 4's judging step

One public accessor over verify.py's existing parsers rather than a
second copy of the same knowledge in a new module."
```

---

### Task 4: One repair attempt

**Files:**
- Create: `src/robigo/profile/repair.py`
- Test: `tests/test_repair.py`

**Interfaces:**
- Consumes: `suite_state`, `SuiteState`, `baseline`, `pytest_runner`, `_resolve_in_clone`, `_package_name` (verify); `run`, `RunResult` (loop); `CorpusRecord`; `Baseline`.
- Produces:

```python
@dataclass(frozen=True)
class Attempt:
    record: str          # CorpusRecord.name
    seed: int
    passed: bool
    outcome: str         # RunResult.outcome, or "" when the loop never ran
    turns: int
    repeats: int         # filled in by Task 6; 0 here
    excluded: str | None # non-None => infrastructure, NOT a model failure

def attempt_repair(record, repo, client, *, seed, codec, base,
                   turn_cap=8, runner=pytest_runner) -> Attempt: ...
```

Task 5 and Task 6 rely on these exact names.

**Read spec §4 before writing this.** Two rules decide whether the number is real:

1. **§4.2 — the task string names only the failing test id.** Never `record.path`, `record.line`, or `record.fixed`. A number produced by telling the model where the bug is describes a tool nobody has.
2. **§4.3.4 — an infrastructure failure is not a model failure.** It sets `excluded`, and Task 5 removes it from both numerator and denominator.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_repair.py
import hashlib
from pathlib import Path
import pytest
from robigo.loop import RunResult
from robigo.profile.corpus_io import CorpusRecord
from robigo.profile.verify import Baseline, SuiteState
from robigo.profile import repair as R


def _record(**over):
    base = dict(
        name="off_by_one", path=Path("src/pkg/mod.py"), line=2,
        broken="    return len(items) - 1\n", fixed="    return len(items)\n",
        test_id="tests/test_mod.py::test_len", diagnostic="exactly one",
        operator="arith", source_repo="/tmp/src", source_sha="deadbeef",
    )
    base.update(over)
    return CorpusRecord(**base)


def _repo(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "pkg" / "mod.py").write_text(
        "def n(items):\n    return len(items)\n", encoding="utf-8")
    (tmp_path / "tests" / "test_mod.py").write_text(
        "from pkg.mod import n\ndef test_len():\n    assert n([1]) == 1\n",
        encoding="utf-8")
    return tmp_path


def test_the_task_string_never_leaks_the_defect_location():
    """Falsification test for spec 4.2. One token away from silently
    inflating the headline figure."""
    r = _record()
    task = R.task_for(r)
    assert r.test_id in task
    assert str(r.path) not in task
    assert "2" not in task.replace(r.test_id, "")   # no line number
    assert r.fixed.strip() not in task
    assert r.broken.strip() not in task


def test_a_clean_repair_passes(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="pass", turns=2, exit_code=0, branch=None, detail="tests pass"))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=0, executed=1, broken_ids=(), incomplete=None))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is True and a.excluded is None and a.turns == 2


def test_greening_the_target_while_breaking_a_neighbour_fails(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=2, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="pass", turns=1, exit_code=0, branch=None, detail="tests pass"))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=1, executed=2, broken_ids=("tests/test_other.py::test_y",),
        incomplete=None))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False and a.excluded is None


def test_editing_the_anchor_test_file_fails_even_if_the_suite_is_green(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)

    def sneaky(*a, **k):
        (repo / "tests" / "test_mod.py").write_text(
            "def test_len():\n    assert True\n", encoding="utf-8")
        return RunResult(outcome="pass", turns=1, exit_code=0,
                         branch=None, detail="tests pass")

    monkeypatch.setattr(R, "run", sneaky)
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=0, executed=1, broken_ids=(), incomplete=None))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False


def test_an_incomplete_suite_run_is_excluded_not_failed(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="pass", turns=1, exit_code=0, branch=None, detail="tests pass"))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=1, executed=0, broken_ids=(), incomplete="Interrupted: collection error"))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False
    assert a.excluded is not None and "Interrupted" in a.excluded


def test_an_executed_total_mismatch_is_excluded_not_failed(tmp_path, monkeypatch):
    """A mutant that breaks a module's import makes pytest report `1 error`
    while three tests never ran. Plan 04 lesson 2."""
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=120, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="pass", turns=1, exit_code=0, branch=None, detail="tests pass"))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=0, executed=117, broken_ids=(), incomplete=None))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.excluded is not None


def test_a_loop_infrastructure_outcome_is_excluded_not_failed(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="infrastructure", turns=0, exit_code=4, branch=None,
        detail="daemon unreachable"))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False and a.excluded is not None


def test_a_stalled_run_is_a_real_model_failure_not_an_exclusion(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    base = Baseline(broken=0, executed=1, seconds=0.1)
    monkeypatch.setattr(R, "run", lambda *a, **k: RunResult(
        outcome="stalled", turns=8, exit_code=1, branch=None,
        detail="turn cap 8 reached"))
    monkeypatch.setattr(R, "suite_state", lambda *a, **k: SuiteState(
        broken=1, executed=1, broken_ids=("tests/test_mod.py::test_len",),
        incomplete=None))
    a = R.attempt_repair(_record(), repo, client=object(), seed=0,
                         codec="search_replace", base=base)
    assert a.passed is False and a.excluded is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_repair.py -v`
Expected: FAIL — no module `robigo.profile.repair`.

- [ ] **Step 3: Implement `repair.py`**

```python
# src/robigo/profile/repair.py
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from robigo.adapters.python_ import PythonAdapter
from robigo.loop import RunResult, run
from robigo.profile.corpus_io import CorpusRecord
from robigo.profile.verify import (
    Baseline, Runner, SuiteState, _package_name, _resolve_in_clone,
    pytest_runner, suite_state,
)

_INFRA_OUTCOMES = frozenset({"infrastructure"})
"""Only `infrastructure` is excluded. `stalled`, `refused` and
`budget_exhausted` are REAL model failures -- a model that cannot get a
patch past the safety layer, or that burns its turn cap, failed to repair,
and the gate's number must say so. Excluding those would be scoring the
tool on the subset of tasks it already handles."""


@dataclass(frozen=True)
class Attempt:
    """One (record, seed) repair attempt. `excluded` non-None means this
    attempt never gave the model a fair chance (spec 4.3.4) and belongs in
    NEITHER the numerator nor the denominator of any rate."""

    record: str
    seed: int
    passed: bool
    outcome: str
    turns: int
    repeats: int
    excluded: str | None


def task_for(record: CorpusRecord) -> str:
    """The task the model is given. Names ONLY the failing test (spec 4.2).
    The record also carries `path`, `line` and `fixed`; putting any of them
    here would measure a tool nobody has."""
    return f"the test {record.test_id} fails; make it pass"


def _anchor_path(record: CorpusRecord, repo: Path) -> Path:
    """The test file the anchor hash guards -- the file part of the pytest
    node id, resolved inside the clone."""
    return _resolve_in_clone(repo, Path(record.test_id.split("::")[0]))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reset_clone(repo: Path) -> None:
    """Discard everything the previous attempt did. Runs before EVERY
    attempt, not once per record (spec 4.3.3)."""
    subprocess.run(["git", "checkout", "--", "."], cwd=repo, check=True,
                   capture_output=True)
    subprocess.run(["git", "clean", "-fdq"], cwd=repo, check=True,
                   capture_output=True)


def break_it(record: CorpusRecord, repo: Path) -> None:
    """Write `record.broken` at `record.line`, reproducing the corpus's
    defective tree. `line` is 1-based and `broken` carries its own line
    ending, so `splitlines(keepends=True)` is the only correct split."""
    target = _resolve_in_clone(repo, record.path)
    lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[record.line - 1] = record.broken
    target.write_text("".join(lines), encoding="utf-8")


def attempt_repair(
    record: CorpusRecord,
    repo: Path,
    client,
    *,
    seed: int,
    codec: str,
    base: Baseline,
    turn_cap: int = 8,
    runner: Runner = pytest_runner,
) -> Attempt:
    def excluded(why: str, outcome: str = "", turns: int = 0) -> Attempt:
        return Attempt(record.name, seed, False, outcome, turns, 0, why)

    try:
        reset_clone(repo)
        break_it(record, repo)
    except (OSError, subprocess.CalledProcessError, IndexError) as exc:
        return excluded(f"could not stage the defect: {exc}")

    anchor = _anchor_path(record, repo)
    before = _sha(anchor)

    # The SHIPPED tool: real turn cap, real codec, git on, test edits off
    # (spec 4.3.1). A defect on any of those paths counts against the tool,
    # because a user meets it.
    result: RunResult = run(
        task_for(record), repo, client, PythonAdapter(),
        codec=codec, turn_cap=turn_cap, allow_test_edits=False, use_git=True,
    )
    if result.outcome in _INFRA_OUTCOMES:
        return excluded(f"loop infrastructure: {result.detail}",
                        result.outcome, result.turns)

    try:
        state: SuiteState = suite_state(repo, runner, _package_name(record.path))
    except Exception as exc:
        return excluded(f"suite did not run: {exc}", result.outcome, result.turns)

    if state.incomplete is not None:
        return excluded(f"suite run incomplete: {state.incomplete}",
                        result.outcome, result.turns)
    if state.executed != base.executed:
        return excluded(
            f"executed total {state.executed} != baseline {base.executed}",
            result.outcome, result.turns)

    anchor_intact = _sha(anchor) == before
    passed = (
        result.outcome == "pass"
        and state.broken == 0
        and record.test_id not in state.broken_ids
        and anchor_intact
    )
    return Attempt(record.name, seed, passed, result.outcome, result.turns,
                   0, None)
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_repair.py -v`
Expected: PASS. Note `run` and `suite_state` are imported into `repair`'s namespace so the tests' `monkeypatch.setattr(R, ...)` works — do not change to `from robigo import loop; loop.run(...)`.

- [ ] **Step 5: Mutation-test each rule**

Run each and confirm the named test fails:

| mutation | must break |
|---|---|
| `passed = result.outcome == "pass"` (drop the rest) | neighbour-break, anchor-edit |
| `anchor_intact = True` | anchor-edit |
| drop the `state.executed != base.executed` guard | executed-mismatch |
| `_INFRA_OUTCOMES = frozenset()` | loop-infrastructure |
| `_INFRA_OUTCOMES` add `"stalled"` | stalled-is-a-real-failure |
| `task_for` returns `f"... in {record.path}:{record.line}"` | task-string-leak |

- [ ] **Step 6: Commit**

```bash
git add tests/test_repair.py src/robigo/profile/repair.py
git commit -m "feat: stage 4's single repair attempt

Runs the shipped loop against a staged defect and judges strictly:
target test green AND suite green AND anchor byte-unchanged AND
outcome == pass. Infrastructure failures set `excluded` and leave both
sides of every rate, so a harness fault can never read as a model
failure. The task string names only the failing test id."
```

---

### Task 5: The stage-4 aggregate

**Files:**
- Modify: `src/robigo/profile/repair.py`
- Test: `tests/test_repair.py`

**Interfaces:**
- Consumes: `Attempt`, `attempt_repair`.
- Produces:

```python
@dataclass(frozen=True)
class Stage4:
    rate: float | None            # None => not measured
    attempts: int                 # scored attempts (excluded ones removed)
    records: int                  # records with >= 1 scored attempt
    per_record: dict[str, tuple[int, int]]   # name -> (passes, scored)
    dropped: tuple[str, ...]

def stage4_repair(records, repo, client, *, seeds, codec, base,
                  turn_cap=8, runner=pytest_runner) -> Stage4: ...
```

Tasks 6 and 7 rely on these names. `per_record` is what makes the record-level confidence interval computable (spec §4.5) — without it the gate can only report an attempt-level interval, which would overclaim precision by roughly √10.

- [ ] **Step 1: Write the failing tests**

```python
def test_rate_is_attempt_level_and_per_record_is_kept(monkeypatch, tmp_path):
    calls = []

    def fake_attempt(record, repo, client, *, seed, **kw):
        calls.append((record.name, seed))
        # record "a" passes on even seeds; record "b" never passes
        ok = record.name == "a" and seed % 2 == 0
        return R.Attempt(record.name, seed, ok, "pass" if ok else "stalled",
                         1, 0, None)

    monkeypatch.setattr(R, "attempt_repair", fake_attempt)
    recs = [_record(name="a"), _record(name="b")]
    s = R.stage4_repair(recs, tmp_path, object(), seeds=4,
                        codec="search_replace",
                        base=Baseline(broken=0, executed=1, seconds=0.1))
    assert len(calls) == 8
    assert s.attempts == 8 and s.records == 2
    assert s.rate == pytest.approx(2 / 8)
    assert s.per_record == {"a": (2, 4), "b": (0, 4)}


def test_excluded_attempts_leave_both_sides_of_the_rate(monkeypatch, tmp_path):
    def fake_attempt(record, repo, client, *, seed, **kw):
        if seed == 0:
            return R.Attempt(record.name, seed, False, "", 0, 0, "daemon died")
        return R.Attempt(record.name, seed, True, "pass", 1, 0, None)

    monkeypatch.setattr(R, "attempt_repair", fake_attempt)
    s = R.stage4_repair([_record(name="a")], tmp_path, object(), seeds=3,
                        codec="search_replace",
                        base=Baseline(broken=0, executed=1, seconds=0.1))
    assert s.attempts == 2            # not 3
    assert s.rate == pytest.approx(1.0)   # 2/2, NOT 2/3
    assert any("daemon died" in d for d in s.dropped)


def test_a_record_with_every_attempt_excluded_is_not_counted(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "attempt_repair", lambda record, repo, client, *, seed, **kw:
                        R.Attempt(record.name, seed, False, "", 0, 0, "clone broken"))
    s = R.stage4_repair([_record(name="a")], tmp_path, object(), seeds=2,
                        codec="search_replace",
                        base=Baseline(broken=0, executed=1, seconds=0.1))
    assert s.attempts == 0 and s.records == 0
    assert s.rate is None       # not 0.0 -- nothing was measured
```

- [ ] **Step 2: Run to verify they fail**

Expected: FAIL — `stage4_repair` not defined.

- [ ] **Step 3: Implement**

```python
@dataclass(frozen=True)
class Stage4:
    rate: float | None
    attempts: int
    records: int
    per_record: dict[str, tuple[int, int]]
    dropped: tuple[str, ...]


def stage4_repair(
    records, repo: Path, client, *, seeds: int, codec: str,
    base: Baseline, turn_cap: int = 8, runner: Runner = pytest_runner,
) -> Stage4:
    """Every record against every seed. `rate` is attempt-level (spec 4.5:
    it is the quantity the attempts-to-success arithmetic is about);
    `per_record` is kept because the CONFIDENCE INTERVAL is record-level,
    and an attempt-level interval would claim ~sqrt(seeds) more precision
    than this design has."""
    per_record: dict[str, tuple[int, int]] = {}
    dropped: list[str] = []
    attempts: list[Attempt] = []

    for record in records:
        for seed in range(seeds):
            a = attempt_repair(record, repo, client, seed=seed, codec=codec,
                               base=base, turn_cap=turn_cap, runner=runner)
            attempts.append(a)
            if a.excluded is not None:
                dropped.append(f"{a.record} seed {a.seed}: {a.excluded}")
                continue
            passes, scored = per_record.get(a.record, (0, 0))
            per_record[a.record] = (passes + int(a.passed), scored + 1)

    scored_total = sum(scored for _, scored in per_record.values())
    passes_total = sum(passes for passes, _ in per_record.values())
    return Stage4(
        rate=(passes_total / scored_total) if scored_total else None,
        attempts=scored_total,
        records=len(per_record),
        per_record=per_record,
        dropped=tuple(dropped),
    )
```

Also store `attempts` on the module for Task 6 — return them by adding `all_attempts: tuple[Attempt, ...]` to `Stage4` and populating it with `tuple(attempts)`. Task 6 reads it; add the field now so no later task changes this dataclass's shape.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_repair.py -v`
Expected: PASS.

- [ ] **Step 5: Mutation-test**

Change `rate=(passes_total / scored_total) if scored_total else None` to `... else 0.0`. The third test must fail. Then change `continue` to `pass` after the `dropped.append`; the second test must fail.

- [ ] **Step 6: Commit**

```bash
git add tests/test_repair.py src/robigo/profile/repair.py
git commit -m "feat: stage 4 aggregate over records x seeds

rate is attempt-level; per_record is retained so the gate's confidence
interval can be computed at RECORD level, where the correlation actually
lives. rate is None (not 0.0) when nothing was scored."
```

---

### Task 6: `RunResult.repeats`, and stage 5

The loop already detects repeats, but `stalls` is a **consecutive** counter that resets to zero on any non-repeat (`loop.py:280`), and it is local — never exposed. Stage 5 needs a **total**. This is a real, small change to production code, not a free read.

**Files:**
- Modify: `src/robigo/loop.py` (`RunResult`, `_result`, `_execute`)
- Modify: `src/robigo/profile/repair.py` (populate `Attempt.repeats`)
- Create: `src/robigo/profile/discipline.py`
- Test: `tests/test_discipline.py`, `tests/test_loop.py`

**Interfaces:**
- Produces:

```python
# loop.py
RunResult(..., rungs=(), repeats: int = 0)

# discipline.py
@dataclass(frozen=True)
class Stage5:
    turns_to_green_median: float | None
    repeat_rate: float | None

def stage5_discipline(attempts: Sequence[Attempt]) -> Stage5: ...
```

Task 7 relies on these names.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_discipline.py
import pytest
from robigo.profile.repair import Attempt
from robigo.profile.discipline import Stage5, stage5_discipline


def _a(passed, turns, repeats, excluded=None):
    return Attempt("r", 0, passed, "pass" if passed else "stalled",
                   turns, repeats, excluded)


def test_median_uses_only_passing_attempts():
    s = stage5_discipline([_a(True, 1, 0), _a(True, 3, 0), _a(False, 8, 0)])
    assert s.turns_to_green_median == 2.0    # median(1, 3), the 8 excluded


def test_no_passes_means_none_not_zero():
    """Invariant 7.2: one value must not mean both 'not applicable' and
    'measured zero'."""
    s = stage5_discipline([_a(False, 8, 0), _a(False, 8, 2)])
    assert s.turns_to_green_median is None


def test_repeat_rate_is_repeats_over_turns_across_scored_attempts():
    s = stage5_discipline([_a(False, 8, 2), _a(True, 2, 0)])
    assert s.repeat_rate == pytest.approx(2 / 10)


def test_excluded_attempts_contribute_to_neither_metric():
    s = stage5_discipline([_a(True, 1, 0), _a(False, 9, 9, excluded="daemon died")])
    assert s.turns_to_green_median == 1.0
    assert s.repeat_rate == pytest.approx(0.0)


def test_no_turns_at_all_means_none():
    s = stage5_discipline([])
    assert s.repeat_rate is None and s.turns_to_green_median is None
```

```python
# tests/test_loop.py  (add)
def test_repeats_counts_every_repeat_not_just_consecutive_ones():
    """`stalls` resets on any non-repeat, so it cannot answer 'how often did
    this run re-emit something it already tried'. A run emitting A, B, A, C, A
    has one consecutive-stall streak of zero and THREE repeats."""
    # Drive _execute with a client replying A, B, A, C, A and a never-green
    # adapter; assert result.repeats == 3 and result.turns == 5.
    # Follow the construction used by the existing loop tests in this file --
    # reuse their fake client/adapter helpers rather than inventing new ones.
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_discipline.py tests/test_loop.py -v`
Expected: FAIL — no module `robigo.profile.discipline`; `RunResult` has no `repeats`.

- [ ] **Step 3: Implement `repeats` in `loop.py`**

Add the field last, with a default, so no existing constructor call breaks:

```python
    repeats: int = 0
    """How many turns re-emitted an (action, reply) pair this run had
    already tried -- a TOTAL, not the consecutive streak `stall_cap`
    watches. `stalls` resets to 0 on any non-repeat, so a run cycling
    A, B, A, C, A never trips the stall cap yet repeated itself three
    times. Stage 5's identical-failing-patch metric is that total; the
    stall cap is a different question and keeps its own counter."""
```

In `_execute`, beside the existing `stalls` line (`loop.py:280`):

```python
        key = f"{action_text}\n{gen.text}"
        if key in seen:
            repeats += 1
        stalls = stalls + 1 if key in seen else 0
        seen.add(key)
```

Initialise `repeats = 0` beside `stalls = 0` (`loop.py:195`), thread it through every `_result(...)` call via a new keyword, and add it to `_result`'s signature and the `RunResult(...)` construction at `loop.py:89`.

- [ ] **Step 4: Implement `discipline.py`**

```python
# src/robigo/profile/discipline.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from robigo.profile.repair import Attempt


@dataclass(frozen=True)
class Stage5:
    """Spec 5's stage 5, derived entirely from stage 4's runs -- zero
    additional model calls (invariant 7.1). Both fields are None rather
    than 0.0 when nothing was observed (invariant 7.2)."""

    turns_to_green_median: float | None
    repeat_rate: float | None


def stage5_discipline(attempts: Sequence[Attempt]) -> Stage5:
    scored = [a for a in attempts if a.excluded is None]
    green = [a.turns for a in scored if a.passed]
    turns = sum(a.turns for a in scored)
    return Stage5(
        turns_to_green_median=float(median(green)) if green else None,
        repeat_rate=(sum(a.repeats for a in scored) / turns) if turns else None,
    )
```

- [ ] **Step 5: Populate `Attempt.repeats` in `repair.py`**

Replace both `Attempt(...)` constructions' hardcoded `0` with `result.repeats` (the excluded-before-`run` paths keep `0`, since no turn ever happened).

- [ ] **Step 6: Run everything**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green.

- [ ] **Step 7: Prove invariant 7.1 — stage 5 costs nothing**

```python
def test_stage5_never_calls_the_client():
    """Falsification test for invariant 7.1."""
    class Exploding:
        def generate(self, *a, **k):
            raise AssertionError("stage 5 must not call the model")
    stage5_discipline([_a(True, 1, 0), _a(False, 3, 1)])   # no client at all
```

Note `stage5_discipline` takes no client by construction, which is the strongest possible form of this guarantee — state that in the task report.

- [ ] **Step 8: Mutation-test**

`turns_to_green_median=... if green else 0.0` must break `test_no_passes_means_none_not_zero`. Dropping the `excluded is None` filter must break `test_excluded_attempts_contribute_to_neither_metric`. Reverting `repeats` to reuse `stalls` must break the loop test.

- [ ] **Step 9: Commit**

```bash
git add src/robigo/loop.py src/robigo/profile/discipline.py \
        src/robigo/profile/repair.py tests/test_discipline.py tests/test_loop.py
git commit -m "feat: stage 5 loop discipline, and a total repeat counter

The loop's existing `stalls` is a CONSECUTIVE counter that resets on any
non-repeat, so A,B,A,C,A never trips the stall cap despite repeating
three times. RunResult.repeats is the total stage 5 needs. Stage 5 takes
no client at all -- invariant 7.1 by construction, not by assertion."
```

---

### Task 7: Schema and reporter wiring

**Files:**
- Modify: `src/robigo/profile/schema.py`, `src/robigo/profile/report.py`
- Test: `tests/test_profile_report.py`, `tests/test_schema.py`

**Interfaces:**
- Consumes: `Stage4`, `stage4_repair`, `Stage5`, `stage5_discipline`.
- Produces: `Profile` with `repair_rate`, `repair_attempts`, `repair_records`, `turns_to_green_median`; `repeat_rate` finally populated.

**Three honesty rules from spec §8, each of which produces a false profile if missed.**

- [ ] **Step 1: Write the failing tests**

```python
def test_repeat_rates_dropped_line_disappears_once_it_is_measured():
    """Spec 8.1. Leaving it would have the profile declare a field
    unmeasured while carrying its measurement."""
    p = _profile(repeat_rate=0.2)
    assert not any("repeat_rate" in d for d in p.dropped)


def test_repeat_rate_still_states_dropped_when_it_is_none():
    p = _profile(repeat_rate=None)
    assert any("repeat_rate" in d for d in p.dropped)


def test_payload_corruption_stays_dropped_and_names_plan_06():
    p = _profile()
    line = next(d for d in p.dropped if "payload_corruption" in d)
    assert "plan 06" in line


def test_stage_4_does_not_run_without_a_best_codec():
    """Spec 4.4. repair_rate None is 'never measured', which is NOT a pass."""
    p = run_profile(_client_landing_nothing(), _plan(), model="m", quant="q",
                    family="f", seeds=1, mode="quick", corpus="c")
    assert p.repair_rate is None
    assert any("stage 4" in d for d in p.dropped)


def test_none_and_zero_repair_rates_are_distinguishable_in_json():
    never = _profile(repair_rate=None).to_json()
    zero = _profile(repair_rate=0.0).to_json()
    assert never != zero
    assert Profile.from_json(json.loads(never)).repair_rate is None
    assert Profile.from_json(json.loads(zero)).repair_rate == 0.0


def test_every_new_field_round_trips():
    p = _profile(repair_rate=0.31, repair_attempts=1000, repair_records=100,
                 turns_to_green_median=2.0, repeat_rate=0.18)
    assert Profile.from_json(json.loads(p.to_json())) == p
```

- [ ] **Step 2: Run to verify they fail**

Expected: FAIL — `Profile.__init__() got an unexpected keyword argument 'repair_rate'`.

- [ ] **Step 3: Implement**

Add the four fields to `Profile` and to both `to_json` and `from_json`. In `run_profile`:

```python
    repair, discipline = None, None
    best = Profile(...).best_codec() if codecs else None   # or compute directly
    if best is None:
        dropped.append(
            "stage 4: not run, no codec landed anything -- there is nothing "
            "to configure a repair loop around. repair_rate is None, which "
            "means NOT MEASURED, not zero (spec 4.4)."
        )
        dropped.append("stage 5: not run, stage 4 did not run")
    elif repo is None:
        dropped.append(
            "stage 4: not run, no --repo given (a repair needs a working "
            "tree; a corpus record alone is not one)"
        )
        dropped.append("stage 5: not run, stage 4 did not run")
    else:
        repair = stage4_repair(records, repo, client, seeds=seeds, codec=best,
                               base=corpus_baseline, turn_cap=turn_cap)
        discipline = stage5_discipline(repair.all_attempts)
        dropped.extend(repair.dropped)
```

Then the conditional `dropped` lines:

```python
    if discipline is None or discipline.repeat_rate is None:
        dropped.append("repeat_rate: not measured (stage 5 did not run)")
    dropped.append(
        "payload_corruption: not measured -- stage 3 is deferred to plan 06, "
        "which runs only if the gate passes"
    )
```

Do **not** compute `best` by constructing a throwaway `Profile`; call the same `max(...)`-with-floor logic `best_codec` uses, or refactor `best_codec` into a module-level function both call. One definition, not two.

- [ ] **Step 4: Run to verify they pass, then run everything**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green.

- [ ] **Step 5: Add the new rows to `render_table`**

```python
    if profile.repair_rate is None:
        lines.append("  repair        not measured")
    else:
        lines.append(
            f"  repair        {profile.repair_rate:.1%} of "
            f"{profile.repair_attempts} attempts over "
            f"{profile.repair_records} records"
        )
```

plus a `discipline` row for the two stage-5 numbers, each printing `not measured` when `None`.

- [ ] **Step 6: Mutation-test**

Make `repeat_rate`'s dropped line unconditional again — `test_repeat_rates_dropped_line_disappears_once_it_is_measured` must fail. Make `repair_rate` default to `0.0` instead of `None` — the stage-4-gate test and the JSON-distinguishability test must both fail.

- [ ] **Step 7: Commit**

```bash
git add src/robigo/profile/schema.py src/robigo/profile/report.py tests/
git commit -m "feat: stage 4/5 in the profile schema and reporter

repair_rate None means NOT MEASURED and is not a pass at the gate.
repeat_rate's dropped line now disappears exactly when it is measured;
payload_corruption's stays and names plan 06."
```

---

### Task 8: CLI wiring — `--repo`, and end-to-end exercise

**Files:**
- Modify: `src/robigo/cli.py`
- Test: `tests/test_cli_profile.py`

- [ ] **Step 1: Write the failing test**

```python
def test_repo_flag_is_required_for_stage_4_and_says_so_when_missing(tmp_path, capsys):
    """Spec: a message must never advise a flag the user just passed, and
    must always name the one they need. Plan 01 shipped two that did the
    opposite."""
    # run profile with --corpus but no --repo; assert the dropped line names
    # --repo explicitly and the exit code is not an alias of a run outcome.
```

- [ ] **Step 2: Implement**

```python
    parser.add_argument(
        "--repo", type=Path, default=None,
        help="a git clone of the corpus's source repo, checked out at the "
             "corpus's source_sha. Stage 4 needs a real working tree; "
             "without it stages 4 and 5 are dropped, not failed.",
    )
```

Thread `repo=args.repo` and the corpus's records/baseline (`read_corpus_baseline`) into `run_profile`.

- [ ] **Step 3: Verify `--repo`'s SHA matches the corpus (a real trap)**

Every record carries `source_sha`. A `--repo` at a *different* commit stages defects at line numbers that no longer mean what the corpus recorded — silently producing failures that are harness artifacts.

```python
def test_a_repo_at_the_wrong_sha_is_refused_not_measured():
    """Line numbers are only meaningful at the recorded commit. A mismatched
    repo produces failures that are harness artifacts, exactly the class
    P1.2 exists to keep out of the number."""
```

Implement: read `git rev-parse HEAD` in `--repo`, compare against the records' `source_sha`, and refuse with a message naming both SHAs if they differ.

- [ ] **Step 4: Run the tool, twice**

Project law — the worst user-facing defect so far was invisible to 300 passing tests and appeared on the *second* consecutive run.

```bash
.venv/bin/robigo profile --model qwen2.5-coder:7b --window 8192 --seeds 1
.venv/bin/robigo profile --model qwen2.5-coder:7b --window 8192 --seeds 1
```

Both must complete and print a coherent table. Record both outputs in the task report. A `window 0`, a `not measured` where something was measured, or a message naming a flag that was passed is a defect to fix now, not to report.

- [ ] **Step 5: Commit**

```bash
git add src/robigo/cli.py tests/test_cli_profile.py
git commit -m "feat: --repo on robigo profile, with a source_sha guard

A repo at the wrong commit stages defects at line numbers the corpus
never recorded, manufacturing failures that are harness artifacts."
```

---

### Task 9: Select and freeze the corpus — **RATIFICATION GATE**

**Stop and get Brice's sign-off before generating anything.** Spec §5: selection made after seeing a number can only be a rationalization.

- [ ] **Step 1: Shortlist three candidate repos against the spec §5 criteria**

Pure Python, no compiled extensions; suite green at HEAD (`baseline.broken == 0`); baseline wall clock under ~30 s; no network in tests; permissive licence; enough source for ~100 records; not robigo.

For each candidate actually clone it and measure — do not judge from reputation:

```bash
git clone --no-hardlinks --depth 1 <url> /tmp/cand && cd /tmp/cand
.venv/bin/python -m pytest -q 2>&1 | tail -5
```

Record for each: URL, licence, HEAD SHA, test count, wall clock, `broken` count.

- [ ] **Step 2: Present the table to Brice and STOP**

Do not proceed to Step 3 without an explicit choice. This is a human gate, not a checkpoint.

- [ ] **Step 3: Generate the corpus at ~120 candidates to yield ~100 records**

```bash
.venv/bin/robigo corpus --repo /path/to/chosen --out docs/corpus/<name>.json \
    --max-records 120 --time-budget 5400
```

- [ ] **Step 4: Verify the generated corpus before trusting it**

- every record's `source_sha` identical and equal to the clone's HEAD
- `read_corpus_baseline(path).broken == 0` — spec §1.2's pass definition depends on it
- `fixtures_from_corpus(records).dropped` length recorded; report the conversion loss as a percentage
- record count ≥ 100 after conversion loss

- [ ] **Step 5: Commit the frozen corpus**

```bash
git add docs/corpus/<name>.json
git commit -m "corpus: freeze the gate's corpus at <sha>

Selected against spec section 5's criteria and ratified BEFORE any
stage-4 run. Under the anti-tuning law this file is not regenerated: if
the number disappoints, the corpus does not change."
```

---

### Task 10: Small-N dry run

Spec §10: ~12 h of unattended GPU time should run once, correctly.

- [ ] **Step 1: Run the full pipeline at N=5 records, 2 seeds**

```bash
.venv/bin/robigo profile --model qwen2.5-coder:7b --window 8192 \
    --corpus docs/corpus/<name>.json --repo /path/to/clone --seeds 2
```

- [ ] **Step 2: Check every one of these, and report each explicitly**

- Does `repair_rate` exist and sit in `[0, 1]`?
- Does `repair_attempts` equal `5 × 2` minus exclusions, and do the exclusions appear in `dropped`?
- Is `per_record` populated for every record that was scored?
- Did the anchor test file survive every attempt?
- Did the clone reset cleanly between attempts — is `git status` clean at the end?
- Is `turns_to_green_median` `None` if nothing passed, rather than `0.0`?
- Time one attempt. Multiply by 100 × 10. **If the extrapolation exceeds 16 h, stop and report before the real run.**

- [ ] **Step 3: Fix whatever this surfaces, then re-run the dry run**

Do not proceed to Task 11 with a known defect. The dry run exists precisely so the 12-hour run is not the thing that discovers it.

- [ ] **Step 4: Commit any fixes**

---

### Task 11: The real run, and the gate

- [ ] **Step 1: Run the publishable profile**

```bash
.venv/bin/robigo profile --model qwen2.5-coder:7b --window 8192 \
    --corpus docs/corpus/<name>.json --repo /path/to/clone --full \
    --record docs/transcripts/qwen7b-stage4.jsonl
```

`--full` fixes seeds at 10 and mode at `full` — the only publishable mode, and the mode the gate reads (spec §1.1). Run it detached; it is ~12 h.

- [ ] **Step 2: Compute the record-level confidence interval**

Spec §4.5 and §6.1: the point estimate is attempt-level, the interval is **record**-level. From `per_record`, take each record's pass proportion, then a 95% interval on the record-level mean. Do **not** compute a binomial interval over `repair_attempts` — that would claim roughly √10 more precision than this design has.

- [ ] **Step 3: Apply the gate**

```
repair_rate >= 0.333  ->  ships; build order proceeds to step 7 (memory)
repair_rate <  0.333  ->  benchmark-and-findings repo; the agent is not
                          shipped as a tool
repair_rate is None   ->  NOT a pass. Not measured. Find out why first.
```

Pre-registered. No extension, no re-run, no corpus change (spec §6.2).

- [ ] **Step 4: Amend the design spec (spec §2)**

In `docs/superpowers/specs/2026-08-09-robigo-design.md` §0.2:
1. *"quick-profile stage 4"* → *"full-profile stage 4"*
2. *"below **40% strict**"* → *"below **33.3% strict**"*, with the attempts-to-success derivation inline and a footnote recording the 2026-08-10 correction from 40% and why.

Neither silently. The old value stays visible in history and in the debt file.

- [ ] **Step 5: Write the result up honestly**

Whatever the number is, report: the rate, the record-level interval, the corpus (repo, SHA, record count, conversion loss), the window cap and that it was a cap, stage 5's two diagnostics, and every `dropped` line. Spec §10's three biases all belong here — single-repo generality, the ~11.5k daemon ceiling, and stage 0's ~0.55 probe-density pessimism, which biases *against* the thesis.

A null result with its own precision stated is the publishable artifact §0.2 promises. A bare percentage is not.

- [ ] **Step 6: Carry the debt forward**

Append a plan 05 section to `docs/CARRIED-DEBT.md`: what was deferred with rulings, what stage 3 (plan 06) inherits, and the process lessons. Every prior plan did this and it is the only durable record.

- [ ] **Step 7: Commit and open the PR**

```bash
git add -A && git commit -m "feat: the repair gate — stage 4, stage 5, and the number"
gh pr create --title "Plan 05: the repair gate" --body "..."
```

---

## Self-Review

**Spec coverage:** §1.1 → Task 11 step 1 (`--full`). §1.2 → Task 4. §1.3 → Task 9. §1.4 → Task 11 step 4. §2 → Task 11 step 4. §3 P1 → Task 2, P2 → Task 1. §4.1 → Task 4. §4.2 → Task 4 step 1. §4.3.1–4 → Task 4. §4.4 → Task 7. §4.5 → Tasks 5, 11. §5 → Task 9. §6.1–6.2 → Task 11. §6.3 → Task 7 (`verdict_for` untouched). §7 → Task 6. §8.1–8.3 → Task 7. §9 → every task's mutation step + Task 8 step 4. §10 → Task 10, Task 11 step 5.

**Known gaps, stated rather than hidden:**

- **Task 6's `test_repeats_counts_every_repeat_not_just_consecutive_ones` and Task 8's two tests are described, not written out.** They depend on fake-client/adapter helpers that already exist in `tests/test_loop.py` and `tests/test_cli_profile.py`, and transcribing invented versions here would produce exactly the "plan text as dominant defect source" failure this project has recorded twice. The implementer must read the existing helpers and follow them. Every *assertion* is specified.
- **Task 7 step 3's `best` computation is deliberately not final code** — it says what must hold (one definition shared with `best_codec`, not two) and leaves the refactor shape to the implementer.
- **Task 9's repo is unnamed by design.** Naming it here would pre-empt the ratification gate.

**Type consistency:** `Attempt` (7 fields incl. `repeats`) is created in Task 4, extended by Task 6, consumed by Tasks 5 and 6. `Stage4` gains `all_attempts` in Task 5 and is not reshaped later. `SuiteState` is defined in Task 3 and used unchanged in Task 4. `RunResult.repeats` is added in Task 6 with a default, so Task 4's construction of `RunResult(...)` in tests stays valid. `stage4_repair`/`stage5_discipline`/`suite_state`/`task_for` names are identical everywhere they appear.
