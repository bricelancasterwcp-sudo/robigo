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
