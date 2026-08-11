# tests/test_cli_profile.py
"""P2 (2026-08-10 design, docs/superpowers/specs/2026-08-10-robigo-05-repair-gate-design.md
§3 P2): `robigo profile` had no way to cap the window it asks the daemon for.
`plan_window` already accepted a `user_cap` fourth positional argument and
already folded it into `min(training_ctx, vram, user_cap)`
(`src/robigo/model/geometry.py::usable_window`) -- `cli.profile_main` simply
never passed anything but `None` through. On this box that is not a cosmetic
gap: `qwen2.5-coder:7b`, the best-measured family, resolves to its full 32768
training context because VRAM never binds here, stage 0 then probes past this
box's Ollama daemon's measured ~11.5k prompt-token ceiling, and the run dies
before stage 0 finishes -- the best family cannot be profiled at all without
an explicit ceiling. These tests cover the plumbing (the flag reaches
`plan_window` as `user_cap`, and its absence still passes `None` rather than
some other default) and the invariant the flag exists to preserve (a cap
above the training context changes nothing -- it is a ceiling, never a
floor)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from robigo import cli
from robigo.loop import OUTCOMES
from robigo.model.client import Generation
from robigo.model.geometry import Geometry, WindowPlan, usable_window
from robigo.profile.corpus_io import CorpusRecord, write_corpus
from robigo.profile.fixtures import FIXTURES, fixtures_from_corpus
from robigo.profile.repair import CorruptedCloneError, InterpreterMismatchError
from robigo.profile.verify import Baseline


def test_window_flag_is_passed_to_plan_window_as_the_user_cap(monkeypatch):
    """`--window 4096` must arrive at `plan_window` as its 4th positional
    argument, `user_cap` -- that positional slot, not a keyword, is what
    `plan_window(backend, model, host, user_cap, *, kv_bits, gguf_path)`
    exposes (confirmed against `src/robigo/model/detect.py` before writing
    this test; the brief's assertion about the 4th positional was correct).
    `run_profile` is stubbed to raise rather than return, so this test never
    depends on stage 0-2 actually running -- it only proves the cap reaches
    `plan_window`, nothing downstream of it."""
    seen = {}

    def fake_plan_window(backend, model, host, user_cap, *, kv_bits=16, gguf_path=None):
        seen["user_cap"] = user_cap
        return WindowPlan(window=4096, limited_by="user_cap", free_vram=None,
                          kv_per_token=57344, weights_bytes=0, overhead_bytes=0,
                          training_ctx=32768)

    monkeypatch.setattr(cli, "plan_window", fake_plan_window)
    monkeypatch.setattr(cli, "run_profile", lambda *a, **k: pytest.skip("not reached"))
    with pytest.raises(BaseException):
        cli.profile_main(["--model", "m", "--window", "4096"])
    assert seen["user_cap"] == 4096


def test_no_window_flag_still_passes_none(monkeypatch):
    """Absent `--window`, the default must stay `None` -- not `0`, not some
    other sentinel that `usable_window` would treat as a real cap. `0` in
    particular would look like a legitimate (if useless) user cap rather
    than "no cap given", so this is not a redundant restatement of the test
    above; it pins the default's identity, not just its truthiness."""
    seen = {}

    def fake_plan_window(backend, model, host, user_cap, *, kv_bits=16, gguf_path=None):
        seen["user_cap"] = user_cap
        raise SystemExit(99)

    monkeypatch.setattr(cli, "plan_window", fake_plan_window)
    with pytest.raises(SystemExit):
        cli.profile_main(["--model", "m"])
    assert seen["user_cap"] is None


def test_window_above_training_ctx_does_not_raise_the_window():
    """P2.1, proved against the real `usable_window`, not a stub: a cap
    above the training context must change nothing, because `usable_window`
    takes `min(training_ctx, vram, user_cap)` and `--window` only ever adds
    a fourth term to that `min`, never replaces it.

    The brief transcribed this call from memory as
    `Geometry(layers=28, kv_heads=4, head_dim=128, training_ctx=4096)` and
    `usable_window(g, free_vram=None, user_cap=999_999, kv_bits=16)`. Both
    are wrong against the real source (`src/robigo/model/geometry.py`):
    `Geometry` is `(arch, layers, kv_heads, key_dim, value_dim,
    training_ctx)` -- there is no `head_dim` field, and `key_dim`/`value_dim`
    are tracked separately because they differ on some architectures
    (`Geometry.kv_bytes_per_token`'s own docstring). `arch` has no default,
    so it must be supplied. And `usable_window`'s `weights_bytes` is a
    required keyword-only parameter with no default at all -- omitting it,
    as the brief's call does, raises `TypeError` before the invariant is
    even exercised. `weights_bytes=0` here is deliberate: with `free_vram`
    also `None`, the vram limit never enters the `min()` (see
    `usable_window`'s `if free_vram is not None` guard), so the zero is
    inert and only the training_ctx/user_cap comparison this test cares
    about is live.
    """
    g = Geometry(arch="test", layers=28, kv_heads=4, key_dim=128, value_dim=128,
                 training_ctx=4096)
    capped = usable_window(g, free_vram=None, weights_bytes=0, user_cap=999_999,
                           kv_bits=16)
    uncapped = usable_window(g, free_vram=None, weights_bytes=0, user_cap=None,
                             kv_bits=16)
    assert capped.window == uncapped.window
    assert capped.limited_by == "training_ctx"


def _record(name, broken, fixed, *, source_sha="deadbeef"):
    """A minimal, valid `CorpusRecord` -- every field `CorpusRecord`
    requires (none carry a default; see `corpus_io.py`'s invariant 9) with
    plausible values, so tests below only need to vary `name`/`broken`/
    `fixed`, the three fields that actually decide each test's outcome.
    `source_sha` is keyword-only with the same `"deadbeef"` default every
    existing call site already relied on positionally -- added (fix round
    1) only so the multi-sha guard test below can build two records that
    deliberately disagree, without a second, hand-rolled `CorpusRecord`
    construction drifting from this one."""
    return CorpusRecord(
        name=name, path=Path("src/pkg/mod.py"), line=3, broken=broken,
        fixed=fixed, test_id="tests/test_mod.py::test_x",
        diagnostic="exactly one net new failure", operator="arith",
        source_repo="/tmp/src", source_sha=source_sha,
    )


def test_unwrappable_records_leave_the_rate_identical_and_are_named(tmp_path):
    """Characterization test for P1.2 (plan 05 design §3): a harness
    artifact -- a record `fixtures_from_corpus` cannot wrap into valid
    Python (I4, `robigo.profile.fixtures`) -- must not reach the model's
    score. The rate from a corpus containing an unwrappable record must
    equal the rate from a corpus where that record is physically absent,
    proving the drop changes neither numerator nor denominator of
    anything downstream, not merely that SOME note gets appended.

    This behaviour already exists (plan 04 task 4's I4 fix,
    `fixtures_from_corpus`'s `ast.parse` check) -- this test is
    deliberately a characterization test, pinning existing behaviour
    rather than driving new code, per the task brief's Step 2. `tmp_path`
    is accepted but unused: no corpus file is written here, only
    `CorpusRecord`s built directly, matching the brief's own test body
    exactly."""
    good = _record("good", "    return a - b\n", "    return a + b\n")
    # A single physical line cut from a multi-line expression: no wrapping
    # strategy at any indent forms a complete statement from it.
    bad = _record("bad", "        for x in (\n", "        for x in (1,\n")

    both = fixtures_from_corpus([good, bad])
    only_good = fixtures_from_corpus([good])

    assert len(both.fixtures) == len(only_good.fixtures) == 1
    assert any("bad" in note for note in both.dropped)
    assert only_good.dropped == ()


def test_corpus_flag_routes_records_and_carries_dropped(tmp_path, monkeypatch):
    """`--corpus PATH` must reach `run_profile` as three things: the
    corpus file's OWN name (never the bundled `fixtures-v1` constant),
    `fixtures` built only from records `fixtures_from_corpus` could wrap
    (the unwrappable `bad` record excluded), and `corpus_dropped` carrying
    BOTH loss channels named in the brief -- what `read_corpus`'s third
    return value (the GENERATOR's own drops, written into the file by
    `write_corpus`'s `dropped=` keyword) reports, and what conversion
    itself dropped (`FixturesFromCorpus.dropped`). Losing either channel
    would let a harness artifact go unnamed in the profile that decides
    whether this project ships (P1.2)."""
    path = tmp_path / "corpus.json"
    good = _record("good", "    return a - b\n", "    return a + b\n")
    bad = _record("bad", "        for x in (\n", "        for x in (1,\n")
    write_corpus([good, bad], path, name="corpus-under-test",
                 dropped=("gen dropped: target foo abandoned",),
                 baseline=Baseline(broken=0, executed=120, seconds=0.4))

    seen = {}

    def fake_run_profile(client, plan, **kw):
        seen.update(kw)
        raise SystemExit(0)

    monkeypatch.setattr(cli, "run_profile", fake_run_profile)
    monkeypatch.setattr(cli, "plan_window", lambda *a, **k: WindowPlan(
        window=4096, limited_by="training_ctx", free_vram=None,
        kv_per_token=57344, weights_bytes=0, overhead_bytes=0, training_ctx=4096))
    monkeypatch.setattr(cli, "build_client", lambda a: object())

    with pytest.raises(SystemExit):
        cli.profile_main(["--model", "m", "--corpus", str(path)])

    assert seen["corpus"] == "corpus-under-test"       # the file's own name
    assert len(seen["fixtures"]) == 1                  # bad one excluded
    assert any("bad" in n for n in seen["corpus_dropped"])
    assert any("abandoned" in n for n in seen["corpus_dropped"])  # generator's too
    # Task 8: the same call must also carry --repo's default (None, since
    # this test passes no --repo), the corpus's own RAW records (both
    # good and bad -- stage4_repair, unlike fixtures_from_corpus, has no
    # reason to drop the unwrappable one; it repairs against `record.line`
    # in a real working tree, never against the wrapped Python body
    # fixtures_from_corpus could not build), and the file's own recorded
    # Baseline, read back via read_corpus_baseline rather than guessed.
    assert seen["repo"] is None
    assert {r.name for r in seen["records"]} == {"good", "bad"}
    assert seen["corpus_baseline"] == Baseline(broken=0, executed=120, seconds=0.4)
    # Fix round 1: absent --python, the call must carry sys.executable --
    # NOT None, NOT PythonAdapter's own .venv/venv/PATH search -- because
    # sys.executable is the interpreter that measured this corpus's own
    # Baseline (both `robigo profile` and `robigo corpus` go through the
    # same entry point), and stage 4's executed-total comparison against
    # that Baseline only means anything if the interpreter reading it
    # back agrees with the one that wrote it.
    assert seen["python"] == sys.executable


class _Landing:
    """The one working `ModelClient` this file needs, trimmed to just the
    three prompt shapes `robigo profile`'s DEFAULT (no `--corpus`) run
    sends: stage 0's plain filler probe (the catch-all branch), stage 1's
    fixed envelope probe (`"read src/target.py"`, `stages.py` line 232),
    and stage 2's per-fixture SEARCH/REPLACE prompt (one per bundled
    `FIXTURES` entry). Mirrors `tests/test_profile_report.py::_Good`
    exactly (that class is already exercised by ~20 tests there) rather
    than inventing new prompt-matching logic -- this is a second, minimal
    COPY of already-proven behaviour, not a new implementation of it,
    kept local so this file does not depend on another test module's
    private helpers staying named or shaped the way they are today."""

    model = "m"
    window = 8192
    num_predict = 512

    def generate(self, prompt: str, *, seed: int) -> Generation:
        if "read src/target.py" in prompt:
            return Generation("read src/target.py", 5, 2, False)
        fixture = next((f for f in FIXTURES if f.filename in prompt), None)
        if fixture:
            return Generation(
                f"patch {fixture.filename}\n```python\n<<<<<<< SEARCH\n"
                f"{fixture.original}=======\n{fixture.expect}>>>>>>> REPLACE\n```\n",
                20, 10, False,
            )
        return Generation("ok", self.window, 1, False)


def test_repo_flag_is_required_for_stage_4_and_says_so_when_missing(
    tmp_path, monkeypatch, capsys
):
    """Spec: a message must never advise a flag the user just passed, and
    must always name the one they need. Plan 01 shipped two that did the
    opposite. Here `--repo` genuinely IS missing (the SHA-mismatch test
    below covers the opposite case, where it is present but wrong), so
    together the two tests cover both directions of that rule.

    `_Landing` clears stage 0/1/2 for real (a genuine `run_profile` call,
    not a stubbed one) so the printed table carries report.py's REAL
    "stage 4: not run, no --repo given" line -- not a line this test
    invented and could therefore get wrong in the same way the code under
    test might. A client that could not land a codec would instead
    produce report.py's OTHER stage-4 line ("no codec ever landed a
    single edit"), which does not name --repo at all -- so a weaker fake
    would make this test pass for the wrong reason.
    """
    plan = WindowPlan(window=8192, limited_by="vram", free_vram=None,
                      kv_per_token=56 * 1024, weights_bytes=0,
                      overhead_bytes=0, training_ctx=32768)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))  # never touch the real ~/.config
    monkeypatch.setattr(cli, "plan_window", lambda *a, **k: plan)
    monkeypatch.setattr(cli, "build_client", lambda a: _Landing())

    code = cli.profile_main(["--model", "m", "--seeds", "1"])

    out = capsys.readouterr().out
    assert code == 0                    # a real, successful profile -- not
                                         # _EX_USAGE(64), not any OUTCOMES
                                         # value; missing --repo is not a
                                         # usage error, it is the ordinary,
                                         # expected, fully-supported case
                                         # of not having wired stage 4 up.
    assert any(
        "stage 4" in line and "--repo" in line for line in out.splitlines()
    )


def _git_repo(tmp_path: Path) -> tuple[Path, str]:
    """A real, minimal git repo with one commit, returning `(repo, sha)`.
    `cli._source_sha` (reused here, unchanged, for the --repo guard) shells
    out to real `git rev-parse HEAD` -- a bare directory with no `.git` at
    all would fail before the guard's own SHA comparison is ever reached,
    which is not what this test is checking."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*argv: str) -> str:
        result = subprocess.run(
            ["git", *argv], cwd=repo, check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()

    run("init", "-q")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "Test")
    (repo / "f.txt").write_text("x\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-q", "-m", "initial")
    return repo, run("rev-parse", "HEAD")


def _unreachable(*_args, **_kwargs):
    """A stub that fails loudly, not quietly, if it is ever called -- used
    below for `plan_window`/`run_profile`, neither of which the SHA guard
    may reach: a wrong --repo must be refused before this command dials a
    model daemon at all, not discovered partway through a run that was
    always going to be thrown away and re-scored as a harness artifact."""
    raise AssertionError("must not be called: the SHA guard must refuse first")


def test_a_repo_at_the_wrong_sha_is_refused_not_measured(tmp_path, monkeypatch, capsys):
    """Line numbers are only meaningful at the recorded commit. A mismatched
    repo produces failures that are harness artifacts, exactly the class
    P1.2 exists to keep out of the number."""
    path = tmp_path / "corpus.json"
    good = _record("good", "    return a - b\n", "    return a + b\n")
    write_corpus([good], path, name="corpus-under-test", dropped=(),
                 baseline=Baseline(broken=0, executed=1, seconds=0.1))
    # good's own source_sha ("deadbeef", see _record) can never collide
    # with a real repo's real 40-hex-char HEAD.
    repo, repo_sha = _git_repo(tmp_path)

    monkeypatch.setattr(cli, "plan_window", _unreachable)
    monkeypatch.setattr(cli, "run_profile", _unreachable)

    code = cli.profile_main(
        ["--model", "m", "--corpus", str(path), "--repo", str(repo)]
    )

    out = capsys.readouterr().out
    assert "deadbeef" in out   # the corpus's own recorded source_sha
    assert repo_sha in out     # --repo's actual, real HEAD
    assert code == OUTCOMES["refused"]


def test_records_with_disagreeing_source_shas_are_refused_not_staged(
    tmp_path, monkeypatch, capsys
):
    """MINOR (fix round 1, review's own wording): `records[0].source_sha`
    alone is not structurally guaranteed to represent the WHOLE corpus --
    nothing in `corpus_io.py` enforces that every record in a file shares
    one `source_sha`, only the fact that `robigo corpus` happens to always
    produce files that do. A hand-assembled or merged corpus could carry
    more than one.

    `good`'s `source_sha` is deliberately set to `--repo`'s OWN real HEAD
    -- the exact trap a `records[0]`-only check falls into: record 0
    would MATCH, the guard would see agreement and let the run proceed,
    and `other` (a genuinely different `source_sha`) would be silently
    staged at line numbers pinned to a commit that was never checked
    against `--repo` at all. Checking the full set closes that gap; this
    test fails if the guard reads only `records[0]`.

    Fix round 2 (MINOR, review's own finding): deleting the multi-sha
    check entirely (`if False and len(source_shas) > 1:`) still made this
    test PASS about 60% of the time in the original version -- with the
    check gone, `corpus_sha = next(iter(source_shas))` picks
    nondeterministically (`PYTHONHASHSEED`-dependent set iteration order)
    from `{repo_sha, "deadbeef"}`, and when it happens to pick
    `"deadbeef"` the OTHER, single-sha mismatch check refuses too --
    satisfying every assertion that only checked substrings ("2" is a
    substring of any string containing a digit 2 somewhere in a hex sha;
    `"deadbeef"`/`repo_sha` both appear in THAT message too) by the WRONG
    check, for the wrong reason. Fixed by asserting the message's own
    SHAPE, not just its content: the multi-sha check's message contains
    the phrase "different source_sha values", which the single-sha
    check's message (`"--repo ... is checked out at ..."`) can never
    produce, and vice versa -- these two assertions are unsatisfiable by
    the wrong check regardless of which element a nondeterministic
    `next(iter(...))` happens to pick, so deleting the real check now
    fails deterministically (verified directly across
    `PYTHONHASHSEED=0..9`, ten separate runs, not sampled once)."""
    path = tmp_path / "corpus.json"
    repo, repo_sha = _git_repo(tmp_path)
    good = _record("good", "    return a - b\n", "    return a + b\n",
                   source_sha=repo_sha)
    other = _record("other", "    return a * b\n", "    return a / b\n")
    # other's source_sha stays "deadbeef" (see _record's default) --
    # deliberately different from good's real repo_sha above.
    write_corpus([good, other], path, name="mixed-corpus", dropped=(),
                 baseline=Baseline(broken=0, executed=1, seconds=0.1))

    monkeypatch.setattr(cli, "plan_window", _unreachable)
    monkeypatch.setattr(cli, "run_profile", _unreachable)

    code = cli.profile_main(
        ["--model", "m", "--corpus", str(path), "--repo", str(repo)]
    )

    out = capsys.readouterr().out
    # The decisive pair: only the multi-sha check can produce the first
    # phrase, and it never produces the second (the single-sha check's
    # own, different, message shape) -- together they rule out the
    # fallback satisfying this test by accident, regardless of which
    # element a broken multi-sha check's `next(iter(...))` might pick.
    assert "different source_sha values" in out
    assert "is checked out at" not in out
    # Real content, not just shape: both disagreeing values named, and
    # how many.
    assert "2" in out
    assert "deadbeef" in out
    assert repo_sha in out
    assert code == OUTCOMES["refused"]


def test_python_flag_reaches_run_profile_as_a_string(monkeypatch):
    """`--python /some/interpreter` (fix round 1) must reach `run_profile`
    as that exact string -- `type=Path` on the argparse flag means
    `args.python` itself is a `Path`, and `attempt_repair`/`PythonAdapter`/
    `pytest_runner` all type their own `python` parameter as `str`, so the
    CLI must convert, not pass the `Path` object straight through."""
    seen = {}

    def fake_run_profile(client, plan, **kw):
        seen.update(kw)
        raise SystemExit(0)

    monkeypatch.setattr(cli, "run_profile", fake_run_profile)
    monkeypatch.setattr(cli, "plan_window", lambda *a, **k: WindowPlan(
        window=4096, limited_by="training_ctx", free_vram=None,
        kv_per_token=57344, weights_bytes=0, overhead_bytes=0, training_ctx=4096))
    monkeypatch.setattr(cli, "build_client", lambda a: object())

    with pytest.raises(SystemExit):
        cli.profile_main(["--model", "m", "--python", "/custom/python"])

    assert seen["python"] == "/custom/python"
    assert isinstance(seen["python"], str)


def test_no_python_flag_defaults_to_sys_executable(monkeypatch):
    """Absent --python, the default reaching `run_profile` must be
    `sys.executable` specifically -- not `None`, and not `PythonAdapter`'s
    own `.venv`/`venv`/`PATH` search (see the corpus-flag test above for
    why that distinction is load-bearing: `sys.executable` is what
    `robigo corpus` was itself running under when it measured whatever
    corpus's `Baseline`, since both go through the same `robigo` entry
    point)."""
    seen = {}

    def fake_run_profile(client, plan, **kw):
        seen.update(kw)
        raise SystemExit(0)

    monkeypatch.setattr(cli, "run_profile", fake_run_profile)
    monkeypatch.setattr(cli, "plan_window", lambda *a, **k: WindowPlan(
        window=4096, limited_by="training_ctx", free_vram=None,
        kv_per_token=57344, weights_bytes=0, overhead_bytes=0, training_ctx=4096))
    monkeypatch.setattr(cli, "build_client", lambda a: object())

    with pytest.raises(SystemExit):
        cli.profile_main(["--model", "m"])

    assert seen["python"] == sys.executable


def test_a_corrupted_clone_error_exits_a_code_that_does_not_alias_stalled(
    monkeypatch, capsys
):
    """IMPORTANT (fix round 1): confirmed live -- left uncaught,
    `CorruptedCloneError` propagated through `profile_main` and Python's
    own top-level handler exited 1, bitwise identical to
    `OUTCOMES["stalled"]`. A script checking `$?` after a long `--full`
    run would misread a corrupted-clone abort (a defect in the shared
    `--repo` clone itself, not a model result) as "the model stalled".
    This must stay LOUD (the traceback still prints) while the exit code
    moves off every contract code."""
    monkeypatch.setattr(cli, "plan_window", lambda *a, **k: WindowPlan(
        window=4096, limited_by="training_ctx", free_vram=None,
        kv_per_token=57344, weights_bytes=0, overhead_bytes=0, training_ctx=4096))
    monkeypatch.setattr(cli, "build_client", lambda a: object())

    def fake_run_profile(client, plan, **kw):
        raise CorruptedCloneError("repo is already on a robigo/* branch")

    monkeypatch.setattr(cli, "run_profile", fake_run_profile)

    code = cli.profile_main(["--model", "m"])

    err = capsys.readouterr().err
    assert code == cli._EX_CORRUPTED_CLONE
    assert code not in OUTCOMES.values()   # never aliases a run outcome
    assert code != cli._EX_USAGE           # and is distinct from _EX_USAGE too
    assert "CorruptedCloneError" in err    # still loud: the traceback printed
    assert "robigo/* branch" in err


def test_an_interpreter_mismatch_error_is_infrastructure_not_a_traceback(
    monkeypatch, capsys
):
    """Fix round 2's IMPORTANT 1, wired at the CLI: `InterpreterMismatchError`
    is a genuine, anticipated infrastructure refusal (a `--python`
    misconfiguration caught BEFORE stage 4's grid starts), not a defect in
    `--repo` itself -- unlike `CorruptedCloneError`, it must land on
    `OUTCOMES["infrastructure"]`, the same code this function already
    returns for "cannot determine the usable window" and "could not read
    --repo's current commit", not a new dedicated code: nothing here risks
    being misread as a MODEL outcome the way exit 1 aliasing "stalled"
    did."""
    monkeypatch.setattr(cli, "plan_window", lambda *a, **k: WindowPlan(
        window=4096, limited_by="training_ctx", free_vram=None,
        kv_per_token=57344, weights_bytes=0, overhead_bytes=0, training_ctx=4096))
    monkeypatch.setattr(cli, "build_client", lambda a: object())

    def fake_run_profile(client, plan, **kw):
        raise InterpreterMismatchError(
            "--python /usr/bin/python3 does not reproduce this corpus's "
            "own baseline: executed 7, corpus was mined with executed=468"
        )

    monkeypatch.setattr(cli, "run_profile", fake_run_profile)

    code = cli.profile_main(["--model", "m"])

    out = capsys.readouterr().out
    assert code == OUTCOMES["infrastructure"]
    assert code != cli._EX_CORRUPTED_CLONE   # distinct refusal, not that one
    assert "468" in out and "7" in out       # the message reached the user
