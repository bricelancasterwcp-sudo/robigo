# tests/test_window_auto.py
from __future__ import annotations

from pathlib import Path

import pytest

from robigo.model.detect import detect_geometry, plan_window, weights_bytes
from robigo.model.geometry import GeometryError, WindowPlan

# Repeated rather than imported from tests/test_geometry.py: cross-test
# imports need tests/ to be an importable package, which it is not.
QWEN7B = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 28,
    "qwen2.attention.head_count": 28,
    "qwen2.attention.head_count_kv": 4,
    "qwen2.attention.key_length": 128,
    "qwen2.context_length": 32768,
    "qwen2.embedding_length": 3584,
}

# Shape of GET /api/tags for a single locally-present model, verified against
# the live daemon: capabilities, details, digest, model, modified_at, name,
# size. Only `name` and `size` matter to weights_bytes.
_TAGS_M = {"models": [{"name": "m", "size": 8 * 1024**3}]}


def test_ollama_geometry_comes_from_api_show(monkeypatch):
    monkeypatch.setattr(
        "robigo.model.detect._show",
        lambda model, host: {"model_info": QWEN7B, "size": 8 * 1024**3},
    )
    geometry = detect_geometry("ollama", "qwen2.5-coder:7b", "")
    assert (geometry.layers, geometry.kv_heads) == (28, 4)


def test_plan_window_reports_what_bound_it(monkeypatch):
    monkeypatch.setattr(
        "robigo.model.detect._show",
        lambda model, host: {"model_info": QWEN7B, "size": 8 * 1024**3},
    )
    # /api/show carries no `size` at all (the amendment below); plan_window's
    # weights figure comes from /api/tags instead, so that is what has to be
    # injected here for this to run with no daemon and no network. The
    # brief's original version of this test mocked only `_show` -- correct
    # against its own (buggy) reference `weights_bytes`, but not against the
    # amendment's, which this module implements.
    monkeypatch.setattr("robigo.model.detect._tags", lambda host: _TAGS_M)
    monkeypatch.setattr("robigo.model.detect.free_vram_bytes", lambda: 15 * 1024**3)
    plan = plan_window("ollama", "m", "", user_cap=None)
    assert (plan.window, plan.limited_by) == (32768, "training_ctx")


def test_a_user_cap_above_the_training_context_is_clamped_not_honoured(monkeypatch):
    monkeypatch.setattr(
        "robigo.model.detect._show",
        lambda model, host: {"model_info": QWEN7B, "size": 8 * 1024**3},
    )
    monkeypatch.setattr("robigo.model.detect._tags", lambda host: _TAGS_M)
    monkeypatch.setattr("robigo.model.detect.free_vram_bytes", lambda: 15 * 1024**3)
    # Asking for 65536 on a 32768-trained model must NOT be granted:
    # Ollama would accept it silently and rope-degrade (law 1).
    plan = plan_window("ollama", "m", "", user_cap=65536)
    assert plan.window == 32768


def test_plan_window_reads_free_vram_before_any_network_call(monkeypatch):
    """usable_window's own precondition is that free_vram is measured
    BEFORE anything could load the model. detect_geometry's /api/show and
    weights_bytes's /api/tags are both confirmed (against the live daemon)
    not to trigger a load, but plan_window reads free VRAM first regardless,
    so the guarantee holds even if that ever stops being true."""
    calls: list[str] = []
    monkeypatch.setattr(
        "robigo.model.detect.free_vram_bytes",
        lambda: calls.append("free_vram") or 15 * 1024**3,
    )
    monkeypatch.setattr(
        "robigo.model.detect._show",
        lambda model, host: calls.append("show") or {"model_info": QWEN7B},
    )
    monkeypatch.setattr(
        "robigo.model.detect._tags",
        lambda host: calls.append("tags") or _TAGS_M,
    )
    plan_window("ollama", "m", "", user_cap=None)
    assert calls[0] == "free_vram"


def test_cli_accepts_the_word_auto(monkeypatch, tmp_path: Path):
    from robigo.cli import main
    import subprocess

    (tmp_path / "test_x.py").write_text("def test_x():\n    assert 0\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.setattr("robigo.cli.plan_window",
                        lambda *a, **k: pytest.importorskip("robigo.model.geometry")
                        .WindowPlan(4096, "vram", None, 56 * 1024,
                                    8 * 1024**3, 256 * 1024**2))
    # The model does not exist, so this must end as infrastructure (4),
    # proving the window was resolved and the loop was entered.
    assert main(["--root", str(tmp_path), "--model", "nope",
                 "--window", "auto", "fix"]) == 4


def test_window_rejects_a_non_auto_non_integer_value(capsys):
    from robigo.cli import main

    # A harness-level usage mistake, not a run outcome: the string is
    # neither "auto" nor an int, so cap = int(args.window) would raise
    # ValueError if uncaught. Exit 64 keeps it out of the five contract
    # codes, same as the pre-existing --scope-swallows-task usage error.
    assert main(["--model", "m", "--window", "not-a-number", "fix"]) == 64
    assert "--window" in capsys.readouterr().out


# --- Amendment 2 (ruled 2026-08-09): a zero window must refuse, and two --
# --- messages are wrong. -----------------------------------------------


def test_cli_refuses_on_a_zero_window_with_arithmetic(monkeypatch, tmp_path: Path,
                                                       capsys):
    """window 0 means the weights plus margin already exceed free VRAM: not
    one token fits, and no degradation rung can help, because the ladder
    shrinks the SCOPE, not the KV cache. Must refuse (exit 3), not
    infrastructure (exit 4) -- nothing is broken in the environment, the
    model simply does not fit this card."""
    from robigo.cli import main

    monkeypatch.setattr(
        "robigo.cli.plan_window",
        lambda *a, **k: WindowPlan(
            0, "vram", 14571 * 1024**2, 56 * 1024, 14540 * 1024**2, 256 * 1024**2
        ),
    )
    code = main(["--root", str(tmp_path), "--model", "m", "fix"])
    assert code == 3
    out = capsys.readouterr().out
    assert "refused" in out
    # The arithmetic itself, not just that something was printed: a bare
    # "window 0" is not actionable, per the amendment.
    assert "free 14571 MiB" in out
    assert "weights 14540 MiB" in out
    assert "margin 256 MiB" in out
    assert "56 KiB/token" in out


def test_cli_zero_window_never_reaches_run(monkeypatch, tmp_path: Path):
    """The regression this amendment exists to fix: today it prints the
    line and continues into adapter setup. Guards specifically against that
    by making `run` explode if it is ever reached."""
    import robigo.cli as cli_module

    def _boom(*args, **kwargs):
        raise AssertionError("run() must not be called for a zero window")

    monkeypatch.setattr(
        cli_module, "plan_window",
        lambda *a, **k: WindowPlan(0, "vram", 1, 1024, 1, 1),
    )
    monkeypatch.setattr(cli_module, "run", _boom)
    code = cli_module.main(["--root", str(tmp_path), "--model", "m", "fix"])
    assert code == 3


def test_cli_refuses_on_a_zero_window_from_a_non_vram_cause(monkeypatch,
                                                             tmp_path: Path,
                                                             capsys):
    """A window of 0 is unusable regardless of which of the three limits
    produced it (e.g. an explicit `--window 0`, which becomes user_cap=0).
    The vram-specific arithmetic line only applies when vram is actually
    why, so this checks the generic fallback names the real cause instead
    of fabricating VRAM numbers that were not the reason.

    `git init` first and the exact "zero-token window" phrase are both
    load-bearing: without them, this test also passes when the zero-window
    check is deleted entirely, because a non-git tmp_path refuses anyway
    (via the unrelated "not a git repository" check) and its message
    happens to contain both "refused" and "limited by user_cap" (from the
    window line printed regardless) -- caught by mutation testing below.
    """
    import subprocess

    from robigo.cli import main

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.setattr(
        "robigo.cli.plan_window",
        lambda *a, **k: WindowPlan(0, "user_cap", None, 1024, 8 * 1024**3,
                                    256 * 1024**2),
    )
    code = main(["--root", str(tmp_path), "--model", "m", "--window", "0", "fix"])
    assert code == 3
    out = capsys.readouterr().out
    assert "nothing can run with a zero-token window" in out
    assert "limited by user_cap" in out
    assert "free" not in out  # no VRAM arithmetic fabricated for a non-vram cause


def test_cli_refuses_end_to_end_when_weights_exceed_free_vram(monkeypatch,
                                                               tmp_path: Path,
                                                               capsys):
    """Reproduces the amendment's own motivating example through the real
    plan_window -> usable_window pipeline (not a stubbed WindowPlan): a
    14.2 GB-class model on a card with 14571 MiB free must refuse with the
    exact arithmetic the amendment quotes."""
    from robigo.cli import main

    monkeypatch.setattr(
        "robigo.model.detect._show",
        lambda model, host: {"model_info": QWEN7B},
    )
    monkeypatch.setattr(
        "robigo.model.detect._tags",
        lambda host: {"models": [{"name": "m", "size": 14540 * 1024**2}]},
    )
    monkeypatch.setattr("robigo.model.detect.free_vram_bytes",
                        lambda: 14571 * 1024**2)
    code = main(["--root", str(tmp_path), "--model", "m", "fix"])
    assert code == 3
    out = capsys.readouterr().out
    assert "window 0 (limited by vram" in out
    assert "free 14571 MiB" in out
    assert "weights 14540 MiB" in out
    assert "margin 256 MiB" in out


def test_detect_geometry_llamacpp_message_offers_window_when_no_cap_given():
    with pytest.raises(GeometryError) as e:
        detect_geometry("llamacpp", "m", "", None)
    msg = str(e.value)
    assert "--gguf" in msg
    assert "--window <int>" in msg


def test_detect_geometry_llamacpp_message_does_not_recommend_the_flag_already_given():
    """The exact defect: with --backend llamacpp --window 4096 and no
    --gguf, the old message said 'Pass --gguf <path> ... or --window <int>'
    -- recommending exactly what was just supplied."""
    with pytest.raises(GeometryError) as e:
        detect_geometry("llamacpp", "m", "", None, user_cap=4096)
    msg = str(e.value)
    assert "--gguf" in msg
    assert "--window <int>" not in msg
    assert "4096" in msg


def test_weights_bytes_llamacpp_message_does_not_recommend_the_flag_already_given():
    """Same defect, same fix, in weights_bytes' sibling message -- reached
    directly by a caller that skips detect_geometry, even though the CLI's
    real call order means detect_geometry's message is the one a user
    actually sees first."""
    with pytest.raises(GeometryError) as e:
        weights_bytes("llamacpp", "m", "", None, user_cap=4096)
    msg = str(e.value)
    assert "--gguf" in msg
    assert "--window <int>" not in msg
    assert "4096" in msg


def test_plan_window_forwards_user_cap_into_the_llamacpp_no_gguf_message():
    with pytest.raises(GeometryError) as e:
        plan_window("llamacpp", "m", "", 4096)
    msg = str(e.value)
    assert "--window <int>" not in msg
    assert "4096" in msg


def test_cli_llamacpp_no_gguf_message_does_not_recommend_window_already_given(
    monkeypatch, capsys, tmp_path: Path
):
    """The coordinator's exact repro, through the real CLI: --backend
    llamacpp --window 4096 with no --gguf must not tell the user to pass
    --window <int>."""
    from robigo.cli import main

    monkeypatch.setattr("robigo.model.detect.free_vram_bytes", lambda: None)
    code = main(["--root", str(tmp_path), "--model", "m", "--backend", "llamacpp",
                 "--window", "4096", "fix"])
    assert code == 4  # infrastructure: GeometryError caught by cli.main
    out = capsys.readouterr().out
    assert "--gguf" in out
    assert "--window <int>" not in out
    assert "4096" in out


# --- Amendment (ruled 2026-08-09): /api/show does not return `size`. ------


def test_weights_come_from_tags_not_show(monkeypatch):
    """/api/show has no `size` field at all. A .get(..., 0) default here
    reports a 0-byte model and hands back the largest window in the table."""
    monkeypatch.setattr(
        "robigo.model.detect._show",
        lambda model, host: {"model_info": QWEN7B},  # no "size" key anywhere
    )
    monkeypatch.setattr(
        "robigo.model.detect._tags",
        lambda host: {"models": [{"name": "qwen2.5-coder:7b-instruct-q8_0",
                                   "size": 8098539207}]},
    )
    assert weights_bytes(
        "ollama", "qwen2.5-coder:7b-instruct-q8_0", "", None
    ) == 8098539207


def test_weights_bytes_falls_back_to_the_latest_tag(monkeypatch):
    # 12 of 30 names on the reference box end in ":latest", so a bare model
    # argument may need that suffix appended to match.
    monkeypatch.setattr(
        "robigo.model.detect._tags",
        lambda host: {"models": [{"name": "codegemma:latest", "size": 5 * 1024**3}]},
    )
    assert weights_bytes("ollama", "codegemma", "", None) == 5 * 1024**3


def test_an_unknown_model_raises_rather_than_reporting_zero_weights(monkeypatch):
    monkeypatch.setattr(
        "robigo.model.detect._tags",
        lambda host: {"models": [{"name": "other:latest", "size": 123}]},
    )
    with pytest.raises(GeometryError) as e:
        weights_bytes("ollama", "missing-model", "", None)
    assert "missing-model" in str(e.value)


def test_a_malformed_tags_size_raises_geometry_error_not_a_raw_type_error(monkeypatch):
    """Task 5's --window fallback catches only GeometryError; a raw
    TypeError from int("not-a-number") would escape it entirely, the same
    class of defect Task 1 fixed three rounds over in geometry.py."""
    monkeypatch.setattr(
        "robigo.model.detect._tags",
        lambda host: {"models": [{"name": "m", "size": "not-a-number"}]},
    )
    with pytest.raises(GeometryError) as e:
        weights_bytes("ollama", "m", "", None)
    assert "m" in str(e.value)


def test_weights_bytes_on_llamacpp_needs_a_gguf_path():
    with pytest.raises(GeometryError) as e:
        weights_bytes("llamacpp", "m", "", None)
    assert "--gguf" in str(e.value)


def test_weights_bytes_on_llamacpp_measures_the_real_file(tmp_path: Path):
    path = tmp_path / "m.gguf"
    path.write_bytes(b"x" * 4096)
    assert weights_bytes("llamacpp", "m", "", path) == 4096


@pytest.mark.live
def test_real_api_show_still_lacks_size():
    """Regression sentinel for the workaround this amendment introduced: if
    Ollama ever starts returning `size` from /api/show, weights_bytes' more
    roundabout /api/tags lookup becomes unnecessary, and this test starts
    failing -- which is the point, so the simplification gets noticed and
    made, rather than the workaround being carried forever unexamined.

    Picks whatever model is actually present rather than a hardcoded name,
    so it does not skip-forever on a box whose local models differ from the
    one this was written against (the exact defect Task 2's real-blob test
    had before its round 3 fix).
    """
    import urllib.error

    from robigo.model.detect import _show, _tags

    try:
        tags = _tags("")
    except (urllib.error.URLError, OSError):
        pytest.skip("ollama daemon not reachable")
    models = tags.get("models") or []
    if not models:
        pytest.skip("no local models to probe")
    info = _show(models[0]["name"], "")
    assert "size" not in info
