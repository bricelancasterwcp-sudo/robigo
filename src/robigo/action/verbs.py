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
    # Stripped for the same reason as _next_fence: the opening fence may
    # be indented, and the language tag still needs to be read off it.
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
