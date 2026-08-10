# src/robigo/profile/fixtures/__init__.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fixture:
    """One single-defect edit with a known-correct target. A stopgap for
    stage 2 until plan 04's mutation generator replaces it; the interface
    it presents to stages.py does not change when that happens.

    `original` is the exact line(s) as they exist before the fix; `expect`
    is the line(s) after. Both are wrapped into a tiny, self-contained
    function body by `stages.fixture_body` before being shown to a model
    -- neither field is a complete file on its own."""

    name: str
    filename: str
    original: str
    expect: str


FIXTURES: tuple[Fixture, ...] = (
    Fixture("off_by_one", "src/counter.py",
            "    return len(items) - 1\n", "    return len(items)\n"),
    Fixture("wrong_operator", "src/scale.py",
            "    return value + factor\n", "    return value * factor\n"),
    Fixture("swapped_args", "src/clamp.py",
            "    return max(high, min(low, value))\n",
            "    return max(low, min(high, value))\n"),
    Fixture("missing_return", "src/total.py",
            "    sum(values)\n", "    return sum(values)\n"),
    Fixture("inverted_test", "src/gate.py",
            "    if not ready:\n", "    if ready:\n"),
)
