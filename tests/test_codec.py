# tests/test_codec.py
from __future__ import annotations

import pytest

from robigo.action.codec import CODECS, PatchError, apply_search_replace, apply_whole_file

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


def test_a_dangling_trailing_marker_is_refused_rather_than_half_applied():
    # Verified: a well-formed first block plus a second one missing its
    # ======= and >>>>>>> REPLACE applied one of two intended edits and
    # returned success. finditer drops whatever falls outside a match.
    payload = (
        _block("    r = compute(t)\n", "    r = compute(t) * 2\n")
        + "<<<<<<< SEARCH\n    return r\n"
    )
    with pytest.raises(PatchError) as e:
        apply_search_replace(FILE, payload)
    message = str(e.value)
    assert "2 SEARCH markers but only 1 complete blocks" in message
    assert "1 edit(s) would have been silently dropped" in message


def test_a_payload_with_no_blocks_is_refused():
    with pytest.raises(PatchError) as e:
        apply_search_replace(FILE, "just some code\n")
    assert "SEARCH" in str(e.value)


def test_codecs_registry_exposes_search_replace():
    assert CODECS["search_replace"] is apply_search_replace


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
