# src/robigo/model/client.py
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Sequence

OLLAMA_HOST = "http://127.0.0.1:11434"
LLAMACPP_HOST = "http://127.0.0.1:8081"


class ModelError(Exception):
    """Infrastructure failure, and nothing else. A model that rambles or
    stops at the cap is a RESULT (spec section 9 law 10)."""


class ContextOverflowError(ModelError):
    """Prompt plus reserved generation exceeds the window."""


class ServerContextOverflowError(ContextOverflowError):
    """The server's real tokenizer rejected a prompt. Distinct so it is
    never retried -- deterministic once rejected -- and so records can
    say which check caught it."""


@dataclass(frozen=True)
class Generation:
    text: str
    tokens_in: int
    tokens_out: int
    truncated: bool


def _request(url: str, payload: dict | None = None, timeout_s: int = 120) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode())


def parse_http_error(exc: urllib.error.HTTPError) -> dict | None:
    """The server's error object. llama.cpp nests a dict under "error";
    Ollama proxies the same object as a JSON *string* one level deeper,
    which a bare isinstance-dict test silently discards."""
    try:
        body = json.loads(exc.read().decode("utf-8"))
    except Exception:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, str):
        try:
            inner = json.loads(error)
        except Exception:
            return {"message": error}
        error = inner.get("error", inner) if isinstance(inner, dict) else None
    return error if isinstance(error, dict) else None


def raise_if_context_overflow(
    model: str, exc: urllib.error.HTTPError
) -> dict | None:
    error = parse_http_error(exc)
    if error is not None and error.get("type") == "exceed_context_size_error":
        raise ServerContextOverflowError(
            f"{model}: the server rejected the prompt as exceeding its "
            f"window ({error.get('message') or 'context size exceeded'})."
        ) from exc
    return error


class _HTTPClient:
    """Shared transport. HTTPError is caught ahead of URLError because it
    IS a URLError subclass; listing it second would retry an overflow
    three times and then misreport it as a transport failure."""

    def __init__(
        self,
        model: str,
        *,
        window: int,
        num_predict: int = 1024,
        host: str = "",
        stop: Sequence[str] = (),
        temperature: float = 0.2,
        timeout_s: int = 300,
        retries: int = 3,
        backoff_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.window = window
        self.num_predict = num_predict
        self.host = (host or self.default_host).rstrip("/")
        self.stop = list(stop)
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self._sleep = sleep

    default_host = OLLAMA_HOST

    def _call(self, url: str, payload: dict | None = None) -> dict:
        last: Exception | str | None = None
        for attempt in range(self.retries):
            try:
                return _request(url, payload, self.timeout_s)
            except urllib.error.HTTPError as exc:
                error = raise_if_context_overflow(self.model, exc)
                message = error.get("message") if error else None
                last = f"{exc} ({message})" if message else exc
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last = exc
            if attempt < self.retries - 1:
                self._sleep(self.backoff_s * (2**attempt))
        raise ModelError(f"{self.model}: {self.retries} attempts failed: {last}")


class OllamaClient(_HTTPClient):
    default_host = OLLAMA_HOST

    def generate(self, prompt: str, *, seed: int) -> Generation:
        body = self._call(
            f"{self.host}/api/chat",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # TOP-LEVEL, never inside options: nested there the daemon
                # ignores it and front-truncates silently.
                "truncate": False,
                "options": {
                    "temperature": self.temperature,
                    "seed": seed,
                    "num_predict": self.num_predict,
                    "num_ctx": self.window,
                    "stop": self.stop,
                },
            },
        )
        message = body.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or body.get("error"):
            raise ModelError(f"{self.model}: malformed 200 response: {body!r}")
        return Generation(
            text=content,
            tokens_in=int(body.get("prompt_eval_count", 0)),
            tokens_out=int(body.get("eval_count", 0)),
            truncated=body.get("done_reason") == "length",
        )


class LlamaCppClient(_HTTPClient):
    default_host = LLAMACPP_HOST

    def generate(self, prompt: str, *, seed: int) -> Generation:
        body = self._call(
            f"{self.host}/v1/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "seed": seed,
                "temperature": self.temperature,
                "max_tokens": self.num_predict,
                "stop": self.stop,
            },
        )
        choices = body.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ModelError(f"{self.model}: malformed 200 response: {body!r}")
        usage = body.get("usage") or {}
        return Generation(
            text=content,
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            truncated=first.get("finish_reason") == "length",
        )
