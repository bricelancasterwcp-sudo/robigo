# tests/test_discipline.py
from __future__ import annotations

import pytest

from robigo.profile.discipline import Stage5, stage5_discipline
from robigo.profile.repair import Attempt


def _a(passed: bool, turns: int, repeats: int, excluded: str | None = None) -> Attempt:
    return Attempt(
        "r", 0, passed, "pass" if passed else "stalled", turns, repeats, excluded
    )


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


def test_stage5_never_calls_the_client():
    """Falsification test for invariant 7.1, for the record -- the real
    guarantee is that `stage5_discipline` takes no `client` parameter at
    all (see its own docstring), so there is no object here for a mutant
    to even reach `.generate` on. This test cannot construct an
    `Exploding` client and hand it in; if it could, `stage5_discipline`
    would already have failed the invariant by accepting one."""

    class Exploding:
        def generate(self, *a, **k):
            raise AssertionError("stage 5 must not call the model")

    result = stage5_discipline([_a(True, 1, 0), _a(False, 3, 1)])   # no client at all
    assert isinstance(result, Stage5)
