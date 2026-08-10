# tests/test_loop_budget.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from robigo.adapters.python_ import PythonAdapter
from robigo.context.budget import estimate_tokens
from robigo.context.render import render
from robigo.context.scope import resolve
from robigo.loop import OUTCOMES, run
from robigo.model.client import Generation

FIX = """patch src/fog.py
```python
<<<<<<< SEARCH
    return t
=======
    return t * 2
>>>>>>> REPLACE
```
"""


def _padded(prefix: str, count: int) -> str:
    """`count` trivial, syntactically-real functions -- enough real `def`
    lines that dropping hop-2 signatures (rung 2) and collapsing a file to
    its own outline (rung 3) each measurably shrink the rendered prompt,
    not just in principle."""
    return "\n".join(f"def {prefix}{i}():\n    pass\n" for i in range(count)) + "\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real two-hop import chain -- test_fog.py imports fog.py imports
    helper.py -- traced by `resolve()` exactly as the CLI's default (no
    `--scope`) path does, so these tests exercise the ladder wiring against
    the scope-building code a real run actually uses, not a hand-built
    `Scope`. `fog.py` and `helper.py` are padded well past what the fixed
    costs (preamble/diagnostic/history) could ever be, so degrading the
    scope is what visibly shrinks the prompt, not noise in the fixed
    terms."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "helper.py").write_text(_padded("helper_fn", 40))
    (tmp_path / "src" / "fog.py").write_text(
        "import helper\n\n\ndef radius(t):\n    return t\n\n\n" + _padded("pad_fn", 60)
    )
    (tmp_path / "tests" / "test_fog.py").write_text(
        "import sys\nsys.path.insert(0, 'src')\nfrom fog import radius\n\n"
        "def test_radius():\n    assert radius(2) == 4\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


class _ScriptedClient:
    """A model whose replies, window, and output reserve are all fixed by
    the test -- the loop must be testable with no GPU, and the ladder must
    be drivable to a specific rung with no daemon in the loop."""

    def __init__(
        self, *replies: str, window: int, num_predict: int, truncated: bool = False,
    ) -> None:
        self.replies = list(replies)
        self.truncated = truncated
        self.window = window
        self.num_predict = num_predict
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, seed: int) -> Generation:
        self.prompts.append(prompt)
        text = self.replies.pop(0) if self.replies else "done nothing left"
        return Generation(text, 10, 5, self.truncated)


class _AssertingClient(_ScriptedClient):
    """Invariant 4, checked at the one place that matters: the moment a
    prompt is actually about to be sent. A step-down bug that lets an
    oversized prompt through fails loudly here, inside the run, rather
    than needing a separate length assertion bolted on afterwards."""

    def generate(self, prompt: str, *, seed: int) -> Generation:
        cost = estimate_tokens(prompt)
        assert cost + self.num_predict <= self.window, (
            f"invariant 4 violated: prompt costs {cost} tokens, "
            f"+ reserve {self.num_predict} > window {self.window}"
        )
        return super().generate(prompt, seed=seed)


def _rung_costs(repo: Path) -> tuple[list[int], object]:
    """The REAL rendered cost (preamble + scope + diagnostic, empty
    history) of each rung, computed with the same `render`/`estimate_
    tokens` the loop itself uses -- so the windows these tests pick are
    derived from the fixture's actual shape, not hand-typed magic numbers
    that silently stop meaning anything the day the fixture changes."""
    adapter = PythonAdapter(python=sys.executable)
    diag = adapter.run(repo, None)
    scope = resolve(diag, adapter, repo)
    costs = [
        estimate_tokens(render(scope.degrade(step), diag, (), "search_replace", repo))
        for step in (1, 2, 3, 4)
    ]
    return costs, diag


def test_a_generous_window_renders_the_full_scope_at_rung_one(repo: Path):
    client = _ScriptedClient("read src/fog.py", window=200_000, num_predict=128)
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace", turn_cap=1)
    assert result.rungs == (1,)
    # The full scope really did render: hop-2's own signature line and
    # fog.py's own padding both survive only at rung 1.
    assert "def helper_fn0():" in client.prompts[0]
    assert "def pad_fn0():" in client.prompts[0]


def test_a_tight_window_degrades_the_scope_and_records_the_rung(repo: Path):
    costs, diag = _rung_costs(repo)
    cost1, cost2, cost3, cost4 = costs
    reserve_out = 64
    # Sanity on the fixture's own assumption: rung 3 must be strictly
    # smaller than rung 2, or a window between them would prove nothing.
    assert cost3 < cost2 < cost1
    window = cost3 + reserve_out + 20
    assert window < cost2 + reserve_out  # rung 2 must NOT fit this window

    client = _ScriptedClient("read src/fog.py", window=window, num_predict=reserve_out)
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace", turn_cap=1)

    assert result.rungs == (3,)
    full = render(resolve(diag, PythonAdapter(python=sys.executable), repo),
                  diag, (), "search_replace", repo)
    assert len(client.prompts[0]) < len(full)
    assert estimate_tokens(client.prompts[0]) + reserve_out <= window


def test_an_impossible_window_refuses_before_any_generation(repo: Path):
    client = _ScriptedClient("read src/fog.py", window=1, num_predict=0)
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace")
    assert (result.outcome, result.exit_code) == ("refused", OUTCOMES["refused"])
    # "It refused" and "it refused before generating" are different claims;
    # only zero calls proves the second one.
    assert client.prompts == []
    # Whole-branch review finding 2 (ruled 2026-08-09): nothing was
    # generated, so this counts zero turns, not the turn (1) `_select_rung`
    # was attempting -- and the rung sequence is empty for the same reason,
    # not `(None,)` or any placeholder for a rung that was never selected.
    assert result.turns == 0
    assert result.rungs == ()
    # Invariant 6: the arithmetic reaches the user, not a summary of it.
    assert "window 1" in result.detail
    assert "reserve" in result.detail


def test_budget_exhaustion_mid_run_is_budget_exhausted_and_preserves_branch(repo: Path):
    costs, _ = _rung_costs(repo)
    cost4 = costs[3]
    reserve_out = 64
    # Tight enough that turn 1 (empty history) just fits at the smallest
    # rung. Turn 2's history is what actually changes between turns --
    # `read src/fog.py`'s own big result, fed back by the loop itself --
    # so the SAME window that fit turn 1 exhausts every rung on turn 2.
    # This is the mechanism invariant 7 names: the rung, and whether one
    # exists at all, is a per-turn question.
    window = cost4 + reserve_out + 10

    client = _ScriptedClient(
        "read src/fog.py", "read src/fog.py", window=window, num_predict=reserve_out,
    )
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace", turn_cap=5)

    assert (result.outcome, result.exit_code) == ("budget_exhausted", OUTCOMES["budget_exhausted"])
    # Whole-branch review finding 2 (ruled 2026-08-09): turn 2 generated
    # NOTHING (`_select_rung` raised before `client.generate` was ever
    # called for it), so only turn 1 counts -- `1`, not `2`. A single real
    # generate() call (asserted below) is the actual evidence for this.
    assert result.turns == 1
    # Finding 3: the rung sequence has exactly one entry -- turn 1's --
    # matching `turns`, not a scalar "last rung" that could just as easily
    # describe a run that degraded across many turns.
    assert len(result.rungs) == 1 == result.turns
    assert result.branch is not None and result.branch.startswith("robigo/")
    assert result.undo is not None
    # The client really was called on turn 1 -- this is budget_exhausted,
    # not the turn-1 refused path.
    assert len(client.prompts) == 1


def test_invariant_four_holds_for_every_prompt_in_a_generous_multi_turn_run(repo: Path):
    client = _AssertingClient(
        "read src/fog.py", "find radius", FIX,
        window=50_000, num_predict=200,
    )
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace", turn_cap=5)
    assert result.outcome == "pass"
    assert len(client.prompts) == 3


def test_invariant_four_holds_across_a_degraded_multi_turn_run(repo: Path):
    costs, _ = _rung_costs(repo)
    cost3 = costs[2]
    reserve_out = 64
    window = cost3 + reserve_out + 30
    client = _AssertingClient(
        "read src/fog.py", "read src/helper.py", "find radius",
        window=window, num_predict=reserve_out,
    )
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace", turn_cap=5)
    # Every prompt actually sent already satisfied invariant 4 (checked
    # inside `_AssertingClient.generate`); this just proves the run was
    # not vacuous -- history growth is expected to exhaust the budget
    # before all three scripted turns land, but at least one must.
    assert len(client.prompts) >= 1
    assert result.outcome in ("pass", "stalled", "budget_exhausted")


def test_measurement_overrides_a_falsely_refusing_fit(repo: Path, monkeypatch):
    """Whole-branch review finding 1 (ruled 2026-08-09), the direction the
    task-2 amendment missed: `fit`'s seated arithmetic can REFUSE a rung
    whose real rendered prompt actually fits -- reproduced by the review
    at window 608/reserve 64, where rung 4's real prompt costs 544 and
    `544 + 64 == 608` fits exactly, while the seated arithmetic said
    `291 > 290`. Forced deterministically here by monkeypatching `fit` to
    ALWAYS raise `BudgetExhausted`, regardless of the budget it is given.

    This is also the acceptance mutation for the fix: reverting
    `_select_rung` to the old "fit decides the stopping point" design
    (calling `fit` first and letting its exception propagate unguarded)
    makes this go RED, because that design refuses the instant the lying
    `fit` raises, without ever rendering rung 1 for real -- exactly the
    false refusal this finding reproduces. Measurement, walked ascending
    from rung 1 and checked against the real window, must never consult
    `fit`'s verdict on the accept path at all; it must proceed and send
    rung 1, which genuinely fits this generous window."""
    import robigo.loop as loop_module
    from robigo.context.budget import BudgetExhausted

    def lying_fit(sc, budget, root):
        raise BudgetExhausted("fake: arithmetic always refuses")

    monkeypatch.setattr(loop_module, "fit", lying_fit)

    client = _ScriptedClient("read src/fog.py", window=200_000, num_predict=128)
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace", turn_cap=1)

    assert client.prompts, (
        "rung 1 fits by real measurement and should have been sent, "
        "regardless of what the (lying) fit() says"
    )
    assert result.rungs == (1,)


def test_measurement_finds_the_largest_fitting_rung_even_when_fit_underproposes(
    repo: Path, monkeypatch,
):
    """The "milder twin" finding 1 also names: because `fit`'s seated
    system/diagnostic/history are measured against the ORIGINAL, undegraded
    scope, its arithmetic can propose a rung LOWER than necessary --
    dropping a file's body for a 1-token artifact -- and a step-down-only
    search can never step back up from that. Forced deterministically by
    monkeypatching `fit` to always claim the SMALLEST rung (4) is needed,
    even though rung 1 genuinely fits this generous window. Walking the
    ladder ascending from rung 1 by real measurement, ignoring `fit`'s
    verdict entirely on the accept path, must still find and send rung 1
    -- the largest rung that actually fits, not whatever `fit` proposed."""
    import robigo.loop as loop_module

    def underproposing_fit(sc, budget, root):
        return sc.degrade(4), 4

    monkeypatch.setattr(loop_module, "fit", underproposing_fit)

    client = _ScriptedClient("read src/fog.py", window=200_000, num_predict=128)
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace", turn_cap=1)

    assert result.rungs == (1,)
    assert "def helper_fn0():" in client.prompts[0]
    assert "def pad_fn0():" in client.prompts[0]


def test_a_rung_no_real_render_fits_refuses_even_when_fit_disagrees(
    repo: Path, monkeypatch,
):
    """Invariant 4's inequality has TWO terms on the left --
    `estimate_tokens(prompt) + reserve_out <= window` -- not one. A rung
    whose rendered prompt fits under the raw window but leaves no room for
    `reserve_out` tokens of output must still be rejected. This also
    exercises `_select_rung`'s defensive fallback: `fit` is monkeypatched
    to always claim rung 3 fits and never raise, even after real
    measurement has already rejected every rung -- measurement is the
    authority in BOTH directions (finding 1), so a `fit` that disagrees
    even at the refusal step must not be trusted either. Window is picked
    so rung 3's real cost clears the raw window but not window-minus-
    reserve, and this fixture's rung 4 costs the SAME as rung 3 (its short
    anchor makes windowing a no-op -- see `_rung_costs`'s table), so there
    is no smaller rung that could rescue it."""
    import robigo.loop as loop_module

    costs, _ = _rung_costs(repo)
    cost3, cost4 = costs[2], costs[3]
    assert cost4 == cost3  # sanity: this fixture's rung 4 saves nothing further
    reserve_out = 64
    window = cost3 + 10  # cost3 <= window, but cost3 + reserve_out > window

    def lying_fit(sc, budget, root):
        return sc.degrade(3), 3

    monkeypatch.setattr(loop_module, "fit", lying_fit)

    client = _ScriptedClient("read src/fog.py", window=window, num_predict=reserve_out)
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace", turn_cap=1)

    assert client.prompts == [], (
        "rung 3 fits the window but not window-minus-reserve, and rung 4 "
        "saves nothing further on this fixture -- nothing should have "
        "been sent"
    )
    assert (result.outcome, result.exit_code) == ("refused", OUTCOMES["refused"])
    assert result.turns == 0
    assert result.rungs == ()


def test_the_rung_sequence_reaches_the_persisted_run_record(repo: Path):
    """Invariant 7, at the actual durable artefact: `RunResult.rungs`
    living only in memory is not enough for plan 03's profiler to read
    after the process has exited. `RunRecorder.finish` writes
    `.robigo/runs/*/meta.json`; this reads that file back rather than
    trusting `result.rungs` directly, so a regression that dropped
    `rungs` from the JSON payload (while leaving `RunResult.rungs` itself
    correct) would still be caught."""
    import json

    client = _ScriptedClient("read src/fog.py", window=200_000, num_predict=128)
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace", turn_cap=1)
    assert result.rungs == (1,)
    metas = list((repo / ".robigo" / "runs").glob("*/meta.json"))
    assert len(metas) == 1
    meta = json.loads(metas[0].read_text())
    assert meta["rungs"] == [1]


def test_the_rung_sequence_accumulates_across_turns_not_just_the_last(
    repo: Path, monkeypatch,
):
    """Whole-branch review finding 3 (ruled 2026-08-09), reproduced with
    the review's own numbers: a scalar "last rung" cannot tell a run that
    silently degraded apart from one that never left rung 1, because only
    the last turn's rung survives -- the review's own reproduction used
    per-turn rungs `[1, 2, 3, 1]`, recorded by the pre-fix code as the
    misleading scalar `1`, identical to a run that never degraded at all.

    `_select_rung` itself is left real (still called, against the real
    scope/history, so a genuinely valid prompt is sent every turn) --
    only the RUNG it reports is substituted, so this isolates the
    accumulation bug from rung-selection correctness, which the other
    tests in this file already cover. This is also the acceptance
    mutation for finding 3: recording only the last turn's rung
    (`rungs = (rung,)` instead of `rungs = rungs + (rung,)`) makes this go
    RED, collapsing `(1, 2, 3, 1)` down to `(1,)`."""
    import robigo.loop as loop_module

    scripted_rungs = iter([1, 2, 3, 1])
    real_select_rung = loop_module._select_rung

    def scripted_select_rung(scope, diag, history, codec, root, window, reserve_out):
        prompt, _real_rung = real_select_rung(
            scope, diag, history, codec, root, window, reserve_out
        )
        return prompt, next(scripted_rungs)

    monkeypatch.setattr(loop_module, "_select_rung", scripted_select_rung)

    client = _ScriptedClient(
        "read src/fog.py", "read src/fog.py", "read src/fog.py", "read src/fog.py",
        window=200_000, num_predict=128,
    )
    # stall_cap raised: the four scripted replies are identical, and the
    # default `stall_cap=3` would end the run at turn 3 -- before the
    # fourth, rung-1-again turn -- for a reason unrelated to what this
    # test checks.
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace", turn_cap=4, stall_cap=10)

    assert result.rungs == (1, 2, 3, 1)
    assert result.turns == 4


def test_a_client_missing_num_predict_fails_loudly_not_silently(repo: Path):
    """Whole-branch review finding 5 (ruled 2026-08-09): a client
    conforming to everything else `ModelClient` declared but missing
    `num_predict` used to silently reserve 0 output tokens, via
    `getattr(client, "num_predict", 0)` -- formally satisfying invariant 4
    while leaving no room for a reply at all (the review reproduced a real
    prompt filling 1730 of a 1730-token window this way). `num_predict` is
    now declared on the Protocol and read directly (`client.num_predict`,
    no default), so a client missing it fails loudly instead: an
    uncaught `AttributeError`. `run`'s own wrapper deliberately RE-RAISES
    after recording an infrastructure result (its docstring: "so the
    traceback survives for debugging; `cli.main` is what turns it into
    exit 4") -- calling `run` directly, as this test does, therefore sees
    the raised exception itself, not a returned `RunResult`. `generate`
    must never be reached: the failure is at attribute access, before any
    prompt is even built."""

    class _MissingNumPredict:
        model = "m"
        window = 8192

        def generate(self, prompt: str, *, seed: int) -> Generation:
            raise AssertionError("unreachable: num_predict access fails first")

    with pytest.raises(AttributeError, match="num_predict"):
        run("fix", repo, _MissingNumPredict(), PythonAdapter(python=sys.executable),
            codec="search_replace", turn_cap=1)
    # The infrastructure record was still written before the re-raise
    # (`run`'s own contract), so this failure is not just loud -- it is
    # also not lost.
    metas = list((repo / ".robigo" / "runs").glob("*/meta.json"))
    assert len(metas) == 1
    import json
    meta = json.loads(metas[0].read_text())
    assert meta["outcome"] == "infrastructure"
