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
from robigo.profile.report import profile_path, render_table, run_profile
from robigo.profile.transcript import CallRecorder, CallReplayer
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
    dispatch = sys.argv[1:] if argv is None else argv
    if dispatch and dispatch[0] == "profile":
        # Collision, considered and accepted: a normal `robigo` invocation
        # whose TASK positional is the single literal word "profile" (e.g.
        # `robigo profile --model m` meant as "fix the thing named
        # profile", not the profiler subcommand) is swallowed by this
        # dispatch and routed to `profile_main` instead. The flat parser
        # below has no subcommand concept to disambiguate the two without
        # a bigger parser rework, and "profile" alone is not a plausible
        # free-form task description in practice, so the collision is left
        # in place rather than papered over.
        return profile_main(list(dispatch[1:]))
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
                        choices=(16, 8), default=16,
                        help="the KV cache precision the SERVER is already "
                             "running -- Ollama's OLLAMA_KV_CACHE_TYPE or "
                             "llama.cpp's --cache-type-k, both set at "
                             "server launch. robigo cannot set this over "
                             "the API, so this only DESCRIBES the "
                             "server's existing configuration to size the "
                             "window correctly; it does not request it. A "
                             "value that does not match what the server "
                             "actually runs overcommits VRAM by that "
                             "ratio (e.g. claiming 8 against an actual 16 "
                             "doubles real usage)")
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
    if plan.window <= 0:
        # No degradation rung can rescue this: the five-step ladder shrinks
        # the SCOPE, not the KV cache, so a 0-token window has no scope
        # small enough to fit it. Refused, not infrastructure -- nothing in
        # the environment is broken, the model simply does not fit this
        # card, and exit 4 is reserved for a harness that could not run at
        # all. Stops here, before adapter/root setup, rather than printing
        # this line and going on to build a prompt against a 0-token budget.
        if plan.limited_by == "vram":
            # limited_by can only be "vram" when free_vram was measured
            # (usable_window only adds that limit when it is not None), so
            # this division is safe by construction, not by a runtime check.
            mib = 1024 * 1024
            print(
                # "free+resident", not "free" (whole-branch review finding
                # 4, ruled 2026-08-09): `plan.free_vram` is `nvidia-smi`'s
                # free reading PLUS this model's own residency credited
                # back (`plan_window`'s `resident_bytes` correction), so it
                # is not the number `nvidia-smi` itself reports, and a
                # second consecutive run against an already-loaded model
                # would otherwise print a "free VRAM" figure larger than
                # what any tool on this box would show.
                f"refused  turns=0  window 0: free+resident "
                f"{plan.free_vram // mib} MiB - weights "
                f"{plan.weights_bytes // mib} MiB - margin "
                f"{plan.overhead_bytes // mib} MiB leaves no room for a "
                f"single token at {plan.kv_per_token // 1024} KiB/token. "
                f"Pick a smaller quantisation, a smaller model, or free VRAM."
            )
        else:
            print(
                f"refused  turns=0  window 0 (limited by {plan.limited_by}): "
                f"nothing can run with a zero-token window."
            )
        return OUTCOMES["refused"]

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


def profile_main(argv: list[str]) -> int:
    """`robigo profile` -- run stages 0-2 against one model and write the
    result to `profile_path(family)`. Nothing reads that file yet (plan 04
    wires it into `robigo run`); this only produces and records it."""
    parser = argparse.ArgumentParser(prog="robigo profile")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("ollama", "llamacpp"), default="ollama")
    parser.add_argument("--host", default=None)
    parser.add_argument("--gguf", type=Path, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--full", action="store_true",
                        help="all stages at 10 seeds; the only publishable mode")
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--replay", type=Path, default=None)
    parser.add_argument("--kv-bits", dest="kv_bits", type=int,
                        choices=(16, 8), default=16)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # Same reasoning as main()'s own usage-error handling above: 2 is
        # budget_exhausted in the run-outcome contract, so a harness-level
        # argparse mistake must not alias it.
        return _EX_USAGE if exc.code else 0

    # --full is the ONLY publishable mode (spec 5.5): a quick run's
    # provenance (seeds, mode) travels with every number it produced so it
    # can never be quoted as a result later, but it must actually be
    # DIFFERENT provenance, not a "full" flag that gets ignored downstream.
    seeds = 10 if args.full else args.seeds
    mode = "full" if args.full else "quick"
    try:
        plan = plan_window(args.backend, args.model, args.host or "", None,
                           kv_bits=args.kv_bits, gguf_path=args.gguf)
    except (GeometryError, OSError) as exc:
        print(f"cannot determine the usable window: {exc}")
        return OUTCOMES["infrastructure"]

    client: ModelClient = (
        CallReplayer(args.replay) if args.replay else build_client(
            argparse.Namespace(backend=args.backend, model=args.model,
                               window=plan.window, host=args.host,
                               num_predict=1024)
        )
    )
    if args.record:
        client = CallRecorder(client, args.record)

    family = args.model.replace(":", "-").replace("/", "-")
    profile = run_profile(client, plan, model=args.model, quant=_quant(args.model),
                          family=family, seeds=seeds, mode=mode,
                          kv_bits=args.kv_bits)
    print(render_table(profile))
    path = profile_path(family)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.to_json(), encoding="utf-8")
    print(f"written to {path}")
    return 0 if profile.verdict != "UNUSABLE" else OUTCOMES["refused"]


def _quant(model: str) -> str:
    """A rough quant tag pulled from the model's own tag string (e.g. the
    `q8_0` in `qwen2.5-coder:7b-instruct-q8_0`) -- there is no registry to
    ask, so this is a naming-convention heuristic, not a measurement, and
    `"unknown"` when the tail does not look like one is the honest answer
    for a model whose tag does not follow that convention."""
    tail = model.rsplit("-", 1)[-1]
    return tail if tail.lower().startswith("q") else "unknown"


if __name__ == "__main__":
    sys.exit(main())
