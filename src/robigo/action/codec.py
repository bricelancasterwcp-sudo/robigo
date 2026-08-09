# src/robigo/action/codec.py
from __future__ import annotations

import difflib
import re
from typing import Callable

_BLOCK = re.compile(
    r"^<<<<<<< SEARCH\n(?P<search>.*?)^=======\n(?P<replace>.*?)^>>>>>>> REPLACE\s*$",
    re.MULTILINE | re.DOTALL,
)
_OPEN_LINE = re.compile(r"^<<<<<<< SEARCH$", re.MULTILINE)


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
    # Before anything is applied: `finditer` silently discards whatever falls
    # outside a matched span, so a payload whose last block is missing its
    # `=======` or `>>>>>>> REPLACE` used to apply one of two intended edits
    # and report success -- a half-edit, written and committed.
    _refuse_dangling_markers(payload, blocks)
    text = original
    for block in blocks:
        text = _apply_one(text, block.group("search"), block.group("replace"))
    return text


def _refuse_dangling_markers(payload: str, blocks: list[re.Match[str]]) -> None:
    """A SEARCH marker on its own line but outside every matched block means
    an edit the regex silently dropped. Counted line-anchored and outside the
    matched spans, so neither a marker quoted inside a body nor one appearing
    mid-line is mistaken for a dropped edit — a file carrying conflict-marker
    text must still be patchable."""
    spans = [block.span() for block in blocks]
    dangling = [
        match
        for match in _OPEN_LINE.finditer(payload)
        if not any(start <= match.start() < end for start, end in spans)
    ]
    if dangling:
        raise PatchError(
            f"{len(dangling)} SEARCH marker(s) in this payload are not part of "
            f"a complete block, so that many edit(s) would have been silently "
            f"dropped. Every block needs all three markers on their own lines: "
            f"<<<<<<< SEARCH, =======, and >>>>>>> REPLACE."
        )


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


CODECS: dict[str, Callable[[str, str], str]] = {
    "search_replace": apply_search_replace,
    "whole_file": apply_whole_file,
}
