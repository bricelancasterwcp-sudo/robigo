# src/robigo/cli.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robigo.adapters.python_ import PythonAdapter
from robigo.loop import OUTCOMES, run
from robigo.model.client import LlamaCppClient, ModelClient, OllamaClient
from robigo.model.detect import plan_window
from robigo.model.geometry import GeometryError
from robigo.record import new_recorder

# No stop sequences. They were matched against the whole stream, payload
# included, and all four of the old ones ("\nread ", "\nfind ", "\nrun\n",
# "\ndone ") match ordinary Python at column 0 -- `done = False`,
# `read = open(path)`, a bare `run`. Generation then stopped mid-payload with
# finish_reason "stop", so the truncation veto could not fire and the reply
# reached the parser as an unclosed fence, every turn, forever, for that file.
# `verbs._reject_second_action` already refuses a multi-action reply, so the
# stops were a token-saving optimisation; spec 2.3's Level 0 sanctions
# unconstrained decoding explicitly.

# EX_USAGE from sysexits.h, deliberately outside the five contract codes.
_EX_USAGE = 64


def build_client(args: argparse.Namespace) -> ModelClient:
    kind = LlamaCppClient if args.backend == "llamacpp" else OllamaClient
    return kind(
        args.model,
        window=args.window,
        num_predict=args.num_predict,
        host=args.host or "",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robigo")
    parser.add_argument("task")
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("ollama", "llamacpp"), default="ollama")
    parser.add_argument("--host", default=None)
    parser.add_argument("--window", default="auto",
                        help="'auto' (default) computes it from model "
                             "geometry and free VRAM; an integer caps it")
    parser.add_argument("--kv-bits", dest="kv_bits", type=int,
                        choices=(16, 8), default=16)
    parser.add_argument("--gguf", type=Path, default=None,
                        help="GGUF path, required with --backend llamacpp "
                             "when --window is auto")
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
                             "tracing imports from the failing test. Greedy: "
                             "put it AFTER the task, or it swallows the task")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse exits 2 on a usage error, and 2 is `budget_exhausted` in
        # the contract. A harness-level mistake must not alias a run outcome.
        return _EX_USAGE if exc.code else 0

    if args.window == "auto":
        cap = None
    else:
        try:
            cap = int(args.window)
        except ValueError:
            # Not GeometryError/OSError below: a malformed --window is a
            # harness-level usage mistake, same class as the SystemExit(2)
            # case above, and must not alias a run outcome either.
            print(f"--window must be 'auto' or an integer, got {args.window!r}")
            return _EX_USAGE
    # Runs for BOTH 'auto' and an explicit int: the never-exceed-training-
    # context law (usable_window always seats training_ctx as a limit) must
    # bind on a user-supplied cap too, not only on the computed default.
    try:
        plan = plan_window(args.backend, args.model, args.host or "", cap,
                           kv_bits=args.kv_bits, gguf_path=args.gguf)
    except (GeometryError, OSError) as exc:
        print(f"cannot determine the usable window: {exc}")
        return OUTCOMES["infrastructure"]
    args.window = plan.window
    print(f"window {plan.window} (limited by {plan.limited_by}, "
          f"{plan.kv_per_token // 1024} KiB/token)")

    adapter = PythonAdapter(python=str(args.python) if args.python else None)
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"--root {args.root} is not a directory")
        return OUTCOMES["refused"]
    recorder = new_recorder(root, args.task)
    try:
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
    except Exception as exc:
        # KeyboardInterrupt is deliberately not caught: the user asking to
        # stop is not an infrastructure failure.
        print(f"infrastructure  turns=0  internal error: {exc!r}")
        return OUTCOMES["infrastructure"]
    print(f"{result.outcome}  turns={result.turns}  {result.detail}")
    if result.branch:
        print(f"branch {result.branch}", end="")
        undo = result.undo
        if undo and undo.original_branch:
            print(f" (from {undo.original_branch})")
            print(f"  to undo:  git checkout {undo.original_branch}")
            if undo.started_dirty and undo.snapshot_sha:
                print(
                    f"            git checkout {undo.snapshot_sha} -- ."
                    f"   # your tree was dirty; this restores it"
                )
        else:
            print()
    if recorder.error:
        print(f"run record unavailable: {recorder.error}")
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
