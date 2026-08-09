# tests/test_client.py
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from robigo.model.client import (
    ContextOverflowError,
    Generation,
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
