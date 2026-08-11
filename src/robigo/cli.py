# src/robigo/cli.py
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from robigo.adapters.python_ import PythonAdapter
from robigo.loop import OUTCOMES, run
from robigo.model.client import LlamaCppClient, ModelClient, OllamaClient
from robigo.model.detect import plan_window
from robigo.model.geometry import GeometryError
from robigo.profile.corpus_io import read_corpus, read_corpus_baseline, write_corpus
from robigo.profile.fixtures import CORPUS_NAME, FIXTURES, fixtures_from_corpus
from robigo.profile.generate import GenerationResult, generate_corpus, render_report
from robigo.profile.repair import CorruptedCloneError, InterpreterMismatchError
from robigo.profile.report import profile_path, render_table, run_profile
from robigo.profile.transcript import CallRecorder, CallReplayer
from robigo.profile.verify import (
    WrongTreeError,
    baseline as measure_baseline,
    pytest_runner,
    sentinel_ok,
)
from robigo.profile.verify import _is_test_path as is_test_shaped_path
from robigo.profile.verify import _source_files as discover_source_files
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
# EX_DATAERR from sysexits.h -- also deliberately outside the five contract
# codes, and deliberately DIFFERENT from _EX_USAGE: fix round 1 (2026-08-10)
# found `CorruptedCloneError` propagating uncaught out of `profile_main`
# and being turned into exit 1 by Python's own top-level handler -- bitwise
# identical to `OUTCOMES["stalled"]`, so a script checking `$?` after a
# long `--full` run would misread a corrupted-clone abort (a defect in the
# shared --repo clone itself, not a model result at all) as "the model
# stalled". `CorruptedCloneError`'s own name is the closest sysexits.h
# category: the on-disk clone -- the "data" this run was handed -- was not
# in the state this run needed it to be in.
_EX_CORRUPTED_CLONE = 65


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
    if dispatch and dispatch[0] == "corpus":
        # Same collision, same reasoning, same acceptance: a task literally
        # named "corpus" is swallowed by this dispatch too. See the
        # "profile" branch above.
        return corpus_main(list(dispatch[1:]))
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
    parser.add_argument(
        "--window", type=int, default=None,
        help="cap the window at N tokens; a CEILING only -- it can never "
             "raise the window above what geometry allows (spec 9 law 1). "
             "Needed on any box whose daemon rejects prompts below the "
             "model's training context.",
    )
    parser.add_argument(
        "--corpus", type=Path, default=None,
        help="a corpus file from `robigo corpus`; without it the bundled "
             "fixtures-v1 is measured, which is not a publishable result",
    )
    parser.add_argument(
        "--repo", type=Path, default=None,
        help="a git clone of the corpus's source repo, checked out at the "
             "corpus's source_sha. Stage 4 needs a real working tree; "
             "without it stages 4 and 5 are dropped, not failed.",
    )
    parser.add_argument(
        "--python", type=Path, default=None,
        help="the interpreter stage 4 runs BOTH the model's edits and the "
             "judging test suite under (fix round 1, 2026-08-10). Defaults "
             "to sys.executable -- this process's own interpreter -- NOT "
             "PythonAdapter's usual .venv/venv/PATH search relative to "
             "--repo: `robigo corpus` measured the corpus's own Baseline "
             "under sys.executable too (both go through the same `robigo` "
             "entry point), and stage 4's executed-test comparison against "
             "that Baseline is only meaningful if the interpreter that "
             "reads it back agrees. Pass this only to point at a DIFFERENT "
             "interpreter than the one running `robigo` itself.",
    )
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

    family = args.model.replace(":", "-").replace("/", "-")
    if args.corpus:
        # P1.2 (plan 05 design, spec §3): a real, mined corpus replaces the
        # bundled fixtures-v1 default -- `corpus_name` is the FILE's own
        # identity (`write_corpus`'s `name=`), never the literal
        # "fixtures-v1", the exact mislabelling plan 03's old kwarg default
        # risked (see `report.run_profile`'s own docstring).
        corpus_name, records, gen_dropped = read_corpus(args.corpus)
        converted = fixtures_from_corpus(records)
        fixtures = converted.fixtures
        # Both sources of loss travel together: what the GENERATOR dropped
        # while mining (`read_corpus`'s third return value -- a target
        # abandoned as barren, a candidate a time budget cut short) and
        # what CONVERSION dropped as unwrappable (`FixturesFromCorpus.
        # dropped`, I4 -- a mutant whose wrapped body is not valid Python
        # at any indent, ~9.2% of real records, measured 91 of 986 from
        # src/robigo). Neither is a model failure and neither may be
        # silently absent from the profile that decides whether this
        # project ships (P1.2) -- both are concatenated into one tuple so
        # `run_profile` cannot thread one through `dropped` and forget the
        # other.
        corpus_dropped = tuple(gen_dropped) + converted.dropped
        # Task 8: the same file's own recorded `Baseline` (`write_corpus`'s
        # `baseline=` keyword, I1) -- `stage4_repair`'s judgement compares
        # a repair attempt's executed-test total against it, and there is
        # no safe baseline `run_profile` could assume on this caller's
        # behalf (see `report.run_profile`'s own docstring, gate 4).
        corpus_baseline = read_corpus_baseline(args.corpus)
    else:
        corpus_name, fixtures, corpus_dropped = CORPUS_NAME, FIXTURES, ()
        records = ()
        corpus_baseline = None

    if args.repo is not None and records:
        # The guard task 8 exists to add (a real trap, not a formality):
        # every `CorpusRecord` carries the exact commit its `line` was
        # read from (`corpus_io.py` invariant 9). A `--repo` sitting at a
        # DIFFERENT commit still LOOKS like a valid working tree -- it
        # clones, it has tests, `attempt_repair` will happily break and
        # patch a line at the recorded `record.line` -- but that line no
        # longer means what the corpus recorded, so every failure that
        # produces is a harness artifact, not a model failure, and would
        # silently corrupt the repair-rate number this whole plan exists
        # to produce. Checked HERE, before `plan_window` or `build_client`
        # ever run, so a wrong `--repo` is refused before this command
        # dials a model daemon at all -- not after burning a real window
        # probe or a real generation call on a run that was always going
        # to be thrown away.
        #
        # Every ORDINARY corpus file has exactly one `source_sha` across
        # every record (one `robigo corpus` run mines one repo at one
        # commit -- verified directly against the frozen 94-record
        # `docs/corpus/boltons-gate-v1.json`), but nothing in `corpus_io.py`
        # actually ENFORCES that on disk: a hand-assembled or merged
        # corpus file could carry more than one, and trusting `records[0]`
        # alone as representative would pass this guard on record 0 while
        # silently mis-staging every OTHER record at line numbers pinned
        # to a commit that was never even checked. Checked structurally
        # here (fix round 1, cheapest correct option per the review)
        # rather than merely documented as an assumption -- a set over up
        # to ~100 records costs nothing next to the daemon calls this
        # guard already exists to skip on failure.
        source_shas = {record.source_sha for record in records}
        if len(source_shas) > 1:
            print(
                f"{args.corpus} is not usable with --repo: its records name "
                f"{len(source_shas)} different source_sha values "
                f"({', '.join(sorted(source_shas))}) -- a single --repo can "
                f"only be checked out at one commit, so no one commit could "
                f"make every record's line numbers meaningful at once. This "
                f"corpus was likely hand-assembled or merged from more than "
                f"one `robigo corpus` run; mine (or split) a corpus with "
                f"one source_sha per file instead."
            )
            return OUTCOMES["refused"]
        try:
            repo_sha = _source_sha(args.repo)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            print(f"could not read --repo {args.repo}'s current commit: {exc}")
            return OUTCOMES["infrastructure"]
        corpus_sha = next(iter(source_shas))
        if repo_sha != corpus_sha:
            # Names BOTH shas, explicitly, and never tells the user to
            # pass a flag they already passed (Plan 01 shipped two
            # messages that did exactly that) -- the fix here is to check
            # out the SHA the corpus already names, not to add --repo
            # again.
            print(
                f"--repo {args.repo} is checked out at {repo_sha}, but "
                f"{args.corpus} was mined at {corpus_sha} -- refusing "
                f"rather than staging stage 4's defects at line numbers "
                f"the corpus never recorded. Check out {corpus_sha} in "
                f"--repo (git -C {args.repo} checkout {corpus_sha}), or "
                f"mine a fresh corpus against --repo's current commit."
            )
            return OUTCOMES["refused"]

    try:
        # P2 (2026-08-10 design, spec §9 invariant P2.1): without a user
        # cap, qwen2.5-coder:7b -- the best-measured family -- resolves to
        # its full 32768 training context, because VRAM never binds on this
        # box (~7.6 GiB weights + ~1.75 GiB KV against 14,558 MiB free).
        # Stage 0 then probes past this box's Ollama daemon's measured
        # ~11.5k prompt-token ceiling and the run dies before stage 0
        # finishes -- the best family could not be profiled here at all.
        # `args.window` is a CEILING only: `plan_window` -> `usable_window`
        # still takes `min(training_ctx, vram, user_cap)`, so passing it
        # through can never raise the window above what geometry allows.
        plan = plan_window(args.backend, args.model, args.host or "", args.window,
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

    # `repo`/`records`/`corpus_baseline` are NOT wrapped in a broad
    # `except Exception` here (task 8's own constraint, unchanged): a
    # `CorruptedCloneError` from `run_profile` -> `stage4_repair` names a
    # defect in the shared `repo` clone itself, not in one record, and
    # swallowing it the way that catch swallows a per-record surprise
    # would silently convert one loud, immediate abort into ~940
    # individually excluded attempts, scoring nothing and reporting a
    # corpus-shaped `dropped` list instead of failing where the actual
    # problem is: `repo`. The `except CorruptedCloneError` below is
    # DIFFERENT from that: it catches this ONE named exception, ONLY it,
    # and does not try to keep the run going -- it exists purely to fix
    # the EXIT CODE (fix round 1, confirmed live 2026-08-10), not to
    # rescue the run.
    python = str(args.python) if args.python else sys.executable
    try:
        profile = run_profile(client, plan, model=args.model, quant=_quant(args.model),
                              family=family, seeds=seeds, mode=mode,
                              corpus=corpus_name, fixtures=fixtures,
                              corpus_dropped=corpus_dropped, kv_bits=args.kv_bits,
                              repo=args.repo, records=records,
                              corpus_baseline=corpus_baseline, python=python)
    except CorruptedCloneError:
        # Kept loud (the full traceback still prints, exactly as an
        # uncaught exception would) -- only the EXIT CODE changes.
        # Confirmed live: left uncaught, this propagated through
        # `profile_main` and Python's own top-level handler exited 1 --
        # bitwise identical to `OUTCOMES["stalled"]`, so a script checking
        # `$?` after a long `--full` run would misread a corrupted-clone
        # abort (a defect in the clone, not a model result at all) as
        # "the model stalled". `_EX_CORRUPTED_CLONE` is deliberately
        # outside the five contract codes, same reasoning as `_EX_USAGE`
        # above, and deliberately DIFFERENT from `_EX_USAGE` too -- this
        # is not a usage mistake, it is a harness-level abort of a
        # different, specific kind.
        traceback.print_exc()
        return _EX_CORRUPTED_CLONE
    except InterpreterMismatchError as exc:
        # Fix round 2's own finding: without this pre-flight check,
        # `stage4_repair` would spend the WHOLE ~940-attempt grid
        # discovering `python` was wrong one attempt at a time, landing
        # on `repair_rate: None` -- the identical shape as fix round 1's
        # bug, under a different message. `InterpreterMismatchError` is
        # raised BEFORE that grid ever starts (see its own docstring), so
        # this is a genuine infrastructure/configuration refusal, not a
        # model result -- OUTCOMES["infrastructure"], the same code this
        # function already returns for "cannot determine the usable
        # window" and "could not read --repo's current commit" above; no
        # new dedicated code needed, unlike CorruptedCloneError, because
        # nothing here risks being misread as a MODEL outcome.
        print(exc)
        return OUTCOMES["infrastructure"]
    print(render_table(profile))
    path = profile_path(family)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.to_json(), encoding="utf-8")
    print(f"written to {path}")
    return 0 if profile.verdict != "UNUSABLE" else OUTCOMES["refused"]


_CLONE_TIMEOUT = 60
_DEFAULT_MAX_RECORDS = 50
"""The plan's own example: "a 50-record corpus at a 10% keep rate is ~2
hours of compute" -- a sane default budget for an unattended run, not a
figure derived from any measurement."""
_DEFAULT_TIME_BUDGET = 1800.0
"""30 minutes. Bounds a barren target (or a barren repo) to a coffee break,
never an unattended hour, per invariant 13's own complaint."""


def _clone_repo(repo: Path, dest: Path) -> None:
    """A real, independent clone of `repo` at `dest` -- never the working
    tree itself, and never a plain file copy. `verify.py`'s own module
    docstring (invariant 5, measured 2026-08-10) is why: a `git archive`
    copy has no `.git`, and this project's own git-dependent tests
    baseline at 6 broken without one -- a real `git clone` keeps `.git`
    and gets an honest baseline instead. `--local` skips a network round
    trip for a same-filesystem source (this module never touches the
    network otherwise).

    `--no-hardlinks` is required, not cosmetic: `dest` is a `tempfile`
    directory, which on this box (and generally) is not guaranteed to sit
    on the same filesystem as `repo` -- measured directly: plain `--local`
    against a `/tmp` destination raised `fatal: failed to create link ...
    Invalid cross-device link` the first time this was run for real
    (invariant 14's own end-to-end run). Forcing a real copy of `.git`'s
    objects, rather than hardlinking them, is also what keeps the
    property "never mutate the working tree" honest regardless of
    filesystem layout: the WORKING TREE files git checks out are already
    ordinary files either way (never hardlinked to `repo`'s own working
    tree, only `.git/objects/` blobs ever are), so a mutation applied
    inside `dest` could never reach back into `repo` -- but a clone that
    silently failed here would abort loudly (`CalledProcessError`) rather
    than leaving `dest` half-populated, which is the outcome that
    actually matters."""
    subprocess.run(
        ["git", "clone", "--local", "--no-hardlinks", "--quiet", str(repo), str(dest)],
        check=True, timeout=_CLONE_TIMEOUT, capture_output=True, text=True,
    )


def _source_sha(clone: Path) -> str:
    """The commit `clone`'s working tree is checked out at -- `CorpusRecord.
    source_sha`'s value (invariant 9): every record this run produces names
    exactly which commit its line was read from."""
    result = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"],
        check=True, timeout=_CLONE_TIMEOUT, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _resolve_targets(
    repo: Path, clone: Path, given: list[Path] | None
) -> tuple[Path, ...]:
    """The one integration seam tasks 1-3 left open: `verify()` requires
    `Mutant.path` relative to the clone and refuses an absolute one, while
    `candidates()` will happily hand back a `Mutant` carrying whatever
    `path` it is given, and `--target` is a CLI flag a user will
    reasonably paste an absolute path into (an editor's "copy path"
    command, say). Converting here -- once, before any target ever reaches
    `candidates()`/`verify()` -- is that conversion; an absolute
    `--target` outside `repo` raises `ValueError` rather than silently
    resolving to the wrong file the way `_resolve_in_clone`'s own
    docstring warns pathlib's `/` operator would.

    With no `--target` given, discovers every real source file the clone
    itself offers, via `robigo.profile.verify._source_files` -- the SAME
    rule `sentinel_ok`'s own general search already uses to decide "which
    files are source", reused rather than a second, independently
    maintained copy of that filtering logic.

    An EXPLICIT `--target` is checked against that same rule too (I2,
    whole-branch review 2026-08-10) and refused with `ValueError` if it is
    test-shaped: `_sentinel_via_search`'s own docstring explains why the
    general discovery path excludes test files -- pytest collects a test
    file straight out of the clone regardless of whether `import <package>`
    resolves there at all, so a mutation to one proves nothing about
    invariant 7, and a "kept" record cut from one would really be a
    sabotaged assertion wearing a repair task's shape. Before this fix,
    `discover_source_files` applied the filter but an explicit `--target`
    bypassed it entirely -- the exact gap I2 closes."""
    if given is None:
        return discover_source_files(clone)
    repo_resolved = repo.resolve()
    resolved: list[Path] = []
    for target in given:
        if not target.is_absolute():
            relative = target
        else:
            try:
                relative = target.resolve().relative_to(repo_resolved)
            except ValueError:
                raise ValueError(f"--target {target} is not inside --repo {repo}") from None
        if is_test_shaped_path(relative):
            raise ValueError(
                f"--target {target} ({relative}) is test-shaped -- a mutation "
                f"to a test file proves nothing about invariant 7 and cannot "
                f"become a corpus record (I2, whole-branch review 2026-08-10)"
            )
        resolved.append(relative)
    return tuple(resolved)


def corpus_main(argv: list[str]) -> int:
    """`robigo corpus` -- generate a mutation corpus from a real repo's
    source, verified against that repo's own test suite (plan 04, task 4).
    Retires plan 03's hand-written `fixtures-v1`: a corpus is now mined,
    never hand-authored, and this is the command that mines one.

    `--repo` is read, never written -- this command clones it into a
    throwaway temp directory before applying a single mutation, and every
    mutation happens inside that clone alone (see `_clone_repo`'s
    docstring for why that is what actually keeps `--repo`'s own working
    tree, which may well be the repo this very command is running from,
    untouched even if a run is interrupted mid-mutation).

    Aborts before generating anything if `sentinel_ok` cannot prove the
    harness can see a real break in this repo (invariant 4) -- a corpus
    "generated" by a blind harness would be full of false survivors, not a
    corpus with zero records, and reporting the difference is exactly what
    this abort exists to guarantee ("Done when": "A sentinel-blind harness
    aborts instead of reporting survivors")."""
    parser = argparse.ArgumentParser(prog="robigo corpus")
    parser.add_argument("--repo", type=Path, required=True,
                        help="the repo to mine mutants from")
    parser.add_argument("--out", type=Path, required=True,
                        help="where to write the corpus JSON file")
    parser.add_argument("--max-records", dest="max_records", type=int,
                        default=_DEFAULT_MAX_RECORDS)
    parser.add_argument("--time-budget", dest="time_budget", type=float,
                        default=_DEFAULT_TIME_BUDGET,
                        help="wall-clock seconds before generation stops "
                             "with whatever it has produced so far")
    parser.add_argument("--target", type=Path, nargs="+", default=None,
                        help="files to mutate, relative to --repo (or "
                             "absolute, inside it); default: every real "
                             "source file the repo offers")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return _EX_USAGE if exc.code else 0

    if not args.repo.is_dir():
        print(f"--repo {args.repo} is not a directory")
        return OUTCOMES["refused"]

    with tempfile.TemporaryDirectory(prefix="robigo-corpus-") as tmp:
        clone = Path(tmp) / "clone"
        try:
            _clone_repo(args.repo, clone)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            print(f"could not clone {args.repo}: {exc}")
            return OUTCOMES["infrastructure"]

        try:
            harness_proven = sentinel_ok(clone, pytest_runner)
        except subprocess.TimeoutExpired as exc:
            print(f"sentinel check timed out: {exc}")
            return OUTCOMES["infrastructure"]
        if not harness_proven:
            print(
                "sentinel failed: this harness cannot prove it can see a "
                "real break in this repo -- aborting rather than reporting "
                "survivors it never actually detected (invariant 4)"
            )
            return OUTCOMES["infrastructure"]

        try:
            base = measure_baseline(clone, pytest_runner)
        except (WrongTreeError, ValueError, subprocess.TimeoutExpired) as exc:
            print(f"could not measure a baseline: {exc}")
            return OUTCOMES["infrastructure"]

        try:
            targets = _resolve_targets(args.repo, clone, args.target)
        except ValueError as exc:
            print(str(exc))
            return _EX_USAGE

        if not targets:
            print(f"{args.repo} offers no source file to mutate")
            return OUTCOMES["refused"]

        source_sha = _source_sha(clone)
        try:
            result: GenerationResult = generate_corpus(
                clone, targets, base, pytest_runner,
                max_records=args.max_records, time_budget=args.time_budget,
                source_repo=str(args.repo), source_sha=source_sha,
            )
        except subprocess.TimeoutExpired as exc:
            print(f"generation aborted: a test run timed out: {exc}")
            return OUTCOMES["infrastructure"]

    name = f"{args.repo.resolve().name}-v1"
    write_corpus(result.records, args.out, name=name, dropped=result.dropped, baseline=base)
    print(render_report(result, name=name))
    print(f"written to {args.out}  ({len(result.records)} records)")
    return 0


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
