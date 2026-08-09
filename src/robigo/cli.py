# src/robigo/cli.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robigo.adapters.python_ import PythonAdapter
from robigo.loop import _slug, run
from robigo.model.client import LlamaCppClient, ModelClient, OllamaClient
from robigo.record import RunRecorder, next_run_id

_STOP = ("\nread ", "\nfind ", "\nrun\n", "\ndone ")


def build_client(args: argparse.Namespace) -> ModelClient:
    kind = LlamaCppClient if args.backend == "llamacpp" else OllamaClient
    return kind(
        args.model,
        window=args.window,
        num_predict=args.num_predict,
        host=args.host or "",
        stop=_STOP,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robigo")
    parser.add_argument("task")
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("ollama", "llamacpp"), default="ollama")
    parser.add_argument("--host", default=None)
    # Explicit for now. Plan 02 computes it from model geometry and free
    # VRAM, and the advertised context length is never trusted.
    parser.add_argument("--window", type=int, default=8192)
    parser.add_argument("--num-predict", dest="num_predict", type=int, default=1024)
    parser.add_argument("--codec", choices=("search_replace", "whole_file"),
                        default="search_replace")
    parser.add_argument("--turn-cap", dest="turn_cap", type=int, default=8)
    parser.add_argument("--allow-test-edits", dest="allow_test_edits",
                        action="store_true")
    parser.add_argument("--no-git", dest="use_git", action="store_false")
    parser.add_argument("--python", type=Path, default=None,
                        help="interpreter holding the project's test "
                             "dependencies; defaults to the project's "
                             ".venv/bin/python, then venv/bin/python, then PATH")
    parser.add_argument("--scope", type=Path, nargs="+", default=None,
                        metavar="PATH",
                        help="files or directories to work in, instead of "
                             "tracing imports from the failing test")
    args = parser.parse_args(argv)

    adapter = PythonAdapter(python=str(args.python) if args.python else None)
    root = Path(args.root).resolve()
    recorder = RunRecorder(root, next_run_id(root, _slug(args.task)))
    result = run(
        args.task,
        root,
        build_client(args),
        adapter,
        codec=args.codec,
        turn_cap=args.turn_cap,
        allow_test_edits=args.allow_test_edits,
        use_git=args.use_git,
        scope_paths=args.scope,
        recorder=recorder,
    )
    print(f"{result.outcome}  turns={result.turns}  {result.detail}")
    if result.branch:
        print(f"branch {result.branch} — `git checkout -` to undo everything")
    if recorder.error:
        print(f"run record unavailable: {recorder.error}")
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
