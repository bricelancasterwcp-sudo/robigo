# tests/test_client.py
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from robigo.model.client import (
    ContextOverflowError,
    Generation,
    LlamaCppClient,
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


def test_a_response_missing_the_token_counts_is_infrastructure_not_a_zero(
    monkeypatch,
):
    # Whole-branch review (ruled 2026-08-10): measured live -- a real,
    # reproducible daemon response with valid `content` but no
    # `prompt_eval_count`/`eval_count`/`done_reason` at all (`"done":
    # false`). `int(body.get("prompt_eval_count", 0))` used to silently
    # read that absence as "0 tokens were evaluated" -- the identical shape
    # as plan 02's `.get("size", 0)` bug, now load-bearing because stage 0
    # trusts `tokens_in` as authoritative. Fails if the `.get(..., 0)`
    # default ever comes back: a response this malformed must raise, not
    # produce a Generation at all.
    body = {"message": {"content": " token token token"}, "done": False}
    with pytest.raises(ModelError) as e:
        _client(monkeypatch, _FakeHTTP(body)).generate("hi", seed=1)
    assert "prompt_eval_count" in str(e.value)
    assert "eval_count" in str(e.value)


def test_a_response_with_an_explicit_null_token_count_is_also_infrastructure(
    monkeypatch,
):
    # The other shape "absent" can take: the key IS present but its value
    # is JSON null, which `.get(key, 0)` would NOT have substituted for
    # (only a missing key triggers a `.get` default) -- so this needs its
    # own check, not just "key not in body". `int(None)` would otherwise
    # escape as a raw TypeError instead of a named ModelError.
    body = {"message": {"content": "hi"}, "prompt_eval_count": None,
            "eval_count": 3}
    with pytest.raises(ModelError) as e:
        _client(monkeypatch, _FakeHTTP(body)).generate("hi", seed=1)
    # "carries no X" names exactly the missing one -- eval_count (present,
    # not None) does appear elsewhere in the message (the repr'd body, for
    # debugging), so the assertion is on the naming phrase, not blanket
    # absence of the substring.
    assert "carries no prompt_eval_count" in str(e.value)
    assert "carries no eval_count" not in str(e.value)


def test_a_response_with_only_eval_count_missing_is_also_infrastructure(
    monkeypatch,
):
    # The other half of the OR: eval_count (tokens_out) matters exactly as
    # much as prompt_eval_count -- both are measurements a caller can
    # trust, or neither is.
    body = {"message": {"content": "hi"}, "prompt_eval_count": 12}
    with pytest.raises(ModelError) as e:
        _client(monkeypatch, _FakeHTTP(body)).generate("hi", seed=1)
    assert "carries no eval_count" in str(e.value)
    assert "carries no prompt_eval_count" not in str(e.value)


def _llama_reply(content="OK", finish_reason="stop", prompt=32, completion=2):
    """A real llama-server /v1/chat/completions body, trimmed of `timings`."""
    return {
        "choices": [{"finish_reason": finish_reason, "index": 0,
                     "message": {"role": "assistant", "content": content}}],
        "created": 1786273256, "model": "local", "object": "chat.completion",
        "system_fingerprint": "b1-4988f6e", "id": "chatcmpl-test",
        "usage": {"completion_tokens": completion, "prompt_tokens": prompt,
                  "total_tokens": prompt + completion},
    }


def _llama(monkeypatch, http) -> LlamaCppClient:
    monkeypatch.setattr("robigo.model.client._request", http)
    return LlamaCppClient("local", window=2048, sleep=lambda _s: None)


def test_llamacpp_returns_a_populated_generation(monkeypatch):
    gen = _llama(monkeypatch, _FakeHTTP(_llama_reply("hello"))).generate("hi", seed=1)
    assert gen == Generation(text="hello", tokens_in=32, tokens_out=2, truncated=False)


def test_llamacpp_marks_a_length_stop_as_truncated(monkeypatch):
    # Verified live: a generation cut at max_tokens reports "length".
    gen = _llama(monkeypatch, _FakeHTTP(_llama_reply(
        "Count slowly from one", finish_reason="length", prompt=38, completion=4
    ))).generate("hi", seed=1)
    assert gen.truncated is True


def test_llamacpp_sends_stop_top_level_and_max_tokens(monkeypatch):
    http = _FakeHTTP(_llama_reply())
    monkeypatch.setattr("robigo.model.client._request", http)
    LlamaCppClient("local", window=2048, num_predict=64, stop=("\nread ",),
                   sleep=lambda _s: None).generate("hi", seed=7)
    payload = http.calls[0][1]
    assert payload["stop"] == ["\nread "]
    assert payload["max_tokens"] == 64
    assert payload["seed"] == 7


@pytest.mark.parametrize("body", [
    {"choices": []},
    {"choices": [{"message": None}]},
    {"choices": [{"message": "just a string"}]},
    {"object": "chat.completion"},
])
def test_llamacpp_refuses_a_malformed_200(monkeypatch, body):
    with pytest.raises(ModelError):
        _llama(monkeypatch, _FakeHTTP(body)).generate("hi", seed=1)


def test_an_overflow_400_from_llamacpp_raises_without_retrying(monkeypatch):
    # llama.cpp nests the error object directly; Ollama adds a layer.
    http = _FakeHTTP(_http_error(OVERFLOW))
    with pytest.raises(ServerContextOverflowError):
        _llama(monkeypatch, http).generate("hi", seed=1)
    assert len(http.calls) == 1


def test_a_200_carrying_a_top_level_error_is_infrastructure(monkeypatch):
    # The other half of the OR guard: valid content beside a truthy error.
    body = {"message": {"content": "looks fine"}, "error": "model unloaded"}
    with pytest.raises(ModelError):
        _client(monkeypatch, _FakeHTTP(body)).generate("hi", seed=1)
