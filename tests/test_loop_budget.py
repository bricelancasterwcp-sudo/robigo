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
    assert result.rung == 1
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

    assert result.rung == 3
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
    assert result.turns == 2
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


def test_arithmetic_accepting_a_rung_that_does_not_actually_fit_steps_down(
    repo: Path, monkeypatch,
):
    """Invariant 4's amendment, pinned directly rather than left to chance:
    the amendment's own sweep found the seated arithmetic accepts a rung
    whose real rendered prompt is over budget at ~5% of cases. That is
    forced HERE, deterministically, by monkeypatching `fit` to always
    claim the full, undegraded scope (rung 1) fits -- exactly the failure
    shape the amendment describes -- while the real window only rung 3
    actually fits. Without the render-and-step-down check, this either
    sends the oversized rung-1 prompt (violating invariant 4) or -- with
    `fit` lying the way it does here -- never refuses either, so the only
    way this test passes is the loop catching its own arithmetic being
    wrong."""
    import robigo.loop as loop_module

    costs, _ = _rung_costs(repo)
    cost3 = costs[2]
    reserve_out = 64
    window = cost3 + reserve_out + 20

    def lying_fit(sc, budget, root):
        return sc, 1

    monkeypatch.setattr(loop_module, "fit", lying_fit)

    client = _ScriptedClient("read src/fog.py", window=window, num_predict=reserve_out)
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace", turn_cap=1)

    assert client.prompts, "a fitting rung existed and should have been sent"
    assert estimate_tokens(client.prompts[0]) + reserve_out <= window
    assert result.rung is not None and result.rung > 1


def test_the_step_down_check_honours_the_output_reserve_not_just_the_window(
    repo: Path, monkeypatch,
):
    """Invariant 4's inequality has TWO terms on the left --
    `estimate_tokens(prompt) + reserve_out <= window` -- not one. A rung
    whose rendered prompt fits under the raw window but leaves no room for
    `reserve_out` tokens of output must still be rejected. Pinned by
    forcing `fit` to claim rung 3 always fits (this fixture's rung 4 costs
    the SAME as rung 3, since its short anchor makes windowing a no-op --
    see `_rung_costs`'s table -- so there is nowhere lower to step down
    to), with a window picked so rung 3's real cost clears the window
    alone but not window-minus-reserve. A correct implementation has
    nothing left to send and refuses."""
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


def test_the_rung_reaches_the_persisted_run_record(repo: Path, monkeypatch):
    """Invariant 7, at the actual durable artefact: `RunResult.rung` living
    only in memory is not enough for plan 03's profiler to read after the
    process has exited. `RunRecorder.finish` writes `.robigo/runs/*/
    meta.json`; this reads that file back rather than trusting `result.rung`
    directly, so a regression that dropped `rung` from the JSON payload
    (while leaving `RunResult.rung` itself correct) would still be caught."""
    import json

    client = _ScriptedClient("read src/fog.py", window=200_000, num_predict=128)
    result = run("fix", repo, client, PythonAdapter(python=sys.executable),
                 codec="search_replace", turn_cap=1)
    assert result.rung == 1
    metas = list((repo / ".robigo" / "runs").glob("*/meta.json"))
    assert len(metas) == 1
    meta = json.loads(metas[0].read_text())
    assert meta["rung"] == 1
