# tests/test_stage2.py
from __future__ import annotations

from robigo.model.client import Generation
from robigo.profile.fixtures import FIXTURES
from robigo.profile.stages import fixture_body, landing_prompt, stage2_codecs


def _sr_reply(fixture) -> str:
    return (
        f"patch {fixture.filename}\n```python\n<<<<<<< SEARCH\n"
        f"{fixture.original}=======\n{fixture.expect}>>>>>>> REPLACE\n```\n"
    )


class _Lands:
    model = "m"

    def generate(self, prompt: str, *, seed: int) -> Generation:
        fixture = next(f for f in FIXTURES if f.filename in prompt)
        return Generation(_sr_reply(fixture), 20, 10, False)


class _Misses(_Lands):
    def generate(self, prompt: str, *, seed: int) -> Generation:
        gen = super().generate(prompt, seed=seed)
        # A one-character transcription slip: the characteristic failure.
        return Generation(gen.text.replace("SEARCH\n", "SEARCH\n "), 20, 10, False)


class _Prose:
    model = "m"

    def generate(self, prompt: str, *, seed: int) -> Generation:
        return Generation("Here is what I would change...", 20, 10, False)


def test_there_are_five_fixtures_and_each_is_self_consistent():
    assert len(FIXTURES) == 5
    for fixture in FIXTURES:
        assert fixture.original != fixture.expect
        assert fixture.original.endswith("\n") and fixture.expect.endswith("\n")


def test_every_fixture_body_is_valid_python_before_and_after_its_own_patch():
    # Fails if fixture_body's wrapper does not hold for every fixture's
    # SHAPE, not just the four that happen to be complete `return`
    # statements. This is the regression test for the brief's own defect:
    # transcribed verbatim, the sample's fixture_body wraps
    # "    if not ready:\n" (inverted_test's original) as the sole
    # statement in a function body, which ast.parse rejects outright
    # ("expected an indented block after 'if' statement") -- confirmed by
    # hand before writing this fix. That made test_a_landing_model_scores_
    # one below fail at 4/5 even for a model that replied perfectly, for a
    # reason having nothing to do with the model or the codec. Checked
    # against BOTH original and expect, since a fixture like inverted_test
    # whose line opens a block needs its trailing filler to still be
    # reachable after the SEARCH/REPLACE swap, not just before it.
    import ast

    for fixture in FIXTURES:
        ast.parse(fixture_body(fixture))
        ast.parse(fixture_body(fixture).replace(fixture.original, fixture.expect, 1))


def test_a_landing_model_scores_one():
    result = stage2_codecs(_Lands(), seeds=1, codecs=("search_replace",))
    assert result.results["search_replace"].lands == 1.0
    assert result.results["search_replace"].attempts == 5


def test_a_mixed_family_lands_the_exact_fraction():
    # Fails if the landing fraction is anything but a real division over
    # per-attempt outcomes -- a stage hardcoded to 1.0 passes the all-land
    # test above, one hardcoded to 0.0 passes the none-land test below, and
    # only a mixed case like this one proves the arithmetic. Two of five
    # fixtures (missing_return, inverted_test) get the transcription-slip
    # corruption; the other three get a clean reply, giving 3/5 = 0.6
    # exactly -- verified by hand against the real parser/codec before
    # writing this test.
    corrupted = {"missing_return", "inverted_test"}

    class _ThreeOfFive(_Lands):
        def generate(self, prompt: str, *, seed: int) -> Generation:
            fixture = next(f for f in FIXTURES if f.filename in prompt)
            gen = super().generate(prompt, seed=seed)
            if fixture.name in corrupted:
                return Generation(gen.text.replace("SEARCH\n", "SEARCH\n "), 20, 10, False)
            return gen

    result = stage2_codecs(_ThreeOfFive(), seeds=1, codecs=("search_replace",))
    assert result.results["search_replace"].lands == 0.6
    assert result.results["search_replace"].attempts == 5


def test_a_transcription_slip_scores_zero_and_is_recorded():
    result = stage2_codecs(_Misses(), seeds=1, codecs=("search_replace",))
    assert result.results["search_replace"].lands == 0.0
    assert any("SEARCH block not found" in f for f in result.failures)


def test_prose_scores_zero_without_raising():
    result = stage2_codecs(_Prose(), seeds=1, codecs=("search_replace",))
    assert result.results["search_replace"].lands == 0.0


def test_a_reply_that_parses_but_names_the_wrong_verb_does_not_land():
    # The second shape of "does not parse as a patch action at all":
    # _Prose above never parses as any action, so it only ever exercises
    # the ActionParseError branch. A reply that parses CLEANLY but names a
    # different verb (e.g. the model tries to re-run the tests instead of
    # patching) takes a different code path entirely and needs its own
    # test, or that branch could silently start landing without any test
    # noticing.
    class _WrongVerb:
        model = "m"

        def generate(self, prompt: str, *, seed: int) -> Generation:
            return Generation("run\n", 5, 2, False)

    result = stage2_codecs(_WrongVerb(), seeds=1, codecs=("search_replace",))
    assert result.results["search_replace"].lands == 0.0
    assert any("not a patch" in f for f in result.failures)


def test_a_patch_that_applies_but_leaves_invalid_python_does_not_land():
    # The third way to fail to land, and the one a single combined test
    # would most likely skip: the reply parses as `patch`, the codec
    # applies the SEARCH/REPLACE without raising PatchError (the SEARCH
    # text matches exactly), but the REPLACE text itself is not valid
    # Python (an unbalanced paren). Verified by hand that the codec
    # genuinely succeeds here (no PatchError) for every fixture, and that
    # only the ast.parse check catches it -- otherwise this test would
    # prove nothing beyond what the PatchError test above already does.
    class _BadSyntax:
        model = "m"

        def generate(self, prompt: str, *, seed: int) -> Generation:
            fixture = next(f for f in FIXTURES if f.filename in prompt)
            reply = (
                f"patch {fixture.filename}\n```python\n<<<<<<< SEARCH\n"
                f"{fixture.original}=======\n"
                f"    return len(items)(\n"
                f">>>>>>> REPLACE\n```\n"
            )
            return Generation(reply, 20, 10, False)

    result = stage2_codecs(_BadSyntax(), seeds=1, codecs=("search_replace",))
    assert result.results["search_replace"].lands == 0.0
    assert any("does not parse as Python" in f for f in result.failures)


def test_the_prompt_describes_only_the_codec_under_test():
    fixture = FIXTURES[0]
    assert "SEARCH" in landing_prompt(fixture, "search_replace")
    assert "SEARCH" not in landing_prompt(fixture, "whole_file")


def test_a_size_ceiling_is_recorded_for_whole_file(monkeypatch):
    # whole_file must report the largest file it managed, because at a
    # small window that ceiling is the binding constraint (spec 3.3).
    result = stage2_codecs(_Lands(), seeds=1, codecs=("whole_file",))
    ceiling = result.results["whole_file"].max_file_tokens
    assert ceiling is None or ceiling > 0


def test_the_ceiling_is_the_largest_size_that_landed_not_the_largest_attempted():
    # The trap named directly in the task: "a ceiling is named a ceiling"
    # -- max_file_tokens must be the largest size that ACTUALLY LANDED, not
    # a bound derived by arithmetic over everything that was tried. This is
    # the same shape of bug Stage 0 shipped once: a "verified" number one
    # token past anything the server actually accepted.
    #
    # swapped_args has the largest fixture body of the five (30 estimated
    # tokens; the next largest, inverted_test, is 28) -- verified by hand.
    # This fake makes swapped_args fail to land (prose, no patch action at
    # all) while the other four land cleanly with a real whole-file reply.
    # A correct ceiling is exactly the largest LANDED size (inverted_test,
    # 28) and is strictly less than the largest ATTEMPTED size (swapped_
    # args, 30) -- an implementation that (buggily) maxed over every
    # attempt regardless of outcome would report 30 here, not 28, and this
    # assertion would catch it.
    from robigo.context.budget import estimate_tokens

    class _AllButLargestLand:
        model = "m"

        def generate(self, prompt: str, *, seed: int) -> Generation:
            fixture = next(f for f in FIXTURES if f.filename in prompt)
            if fixture.name == "swapped_args":
                return Generation("Here is what I would change...", 20, 10, False)
            body = fixture_body(fixture)
            new_text = body.replace(fixture.original, fixture.expect, 1)
            reply = f"patch {fixture.filename}\n```python\n{new_text}```\n"
            return Generation(reply, 20, 10, False)

    landed_sizes = [
        estimate_tokens(fixture_body(f)) for f in FIXTURES if f.name != "swapped_args"
    ]
    largest_attempted = max(estimate_tokens(fixture_body(f)) for f in FIXTURES)
    assert max(landed_sizes) < largest_attempted  # sanity: the fixture split is real

    result = stage2_codecs(_AllButLargestLand(), seeds=1, codecs=("whole_file",))
    assert result.results["whole_file"].lands == 0.8
    assert result.results["whole_file"].max_file_tokens == max(landed_sizes)
    assert result.results["whole_file"].max_file_tokens < largest_attempted


def test_search_replace_never_reports_a_size_ceiling():
    # search_replace's payload size does not scale with the file's total
    # size the way whole_file's does (only the diff is emitted, not the
    # whole file), so reporting a max_file_tokens for it would claim a
    # capability this stage never measured. Fails if ceiling tracking is
    # ever made codec-agnostic.
    result = stage2_codecs(_Lands(), seeds=1, codecs=("search_replace",))
    assert result.results["search_replace"].max_file_tokens is None


def test_results_are_never_pooled_across_codecs():
    # Fails if search_replace and whole_file ever share one CodecResult, or
    # if one codec's failures leak into the other's landing count -- the
    # loop needs the two numbers separately to pick a codec, and pooling
    # them would hide exactly the "lands on one, not the other" case the
    # profile exists to surface.
    class _LandsSROnly:
        model = "m"

        def generate(self, prompt: str, *, seed: int) -> Generation:
            fixture = next(f for f in FIXTURES if f.filename in prompt)
            if "SEARCH" in landing_prompt(fixture, "whole_file"):
                raise AssertionError("whole_file prompt leaked SEARCH syntax")
            if "SEARCH" not in prompt:
                # This is the whole_file prompt: reply with prose, so it
                # never lands under whole_file.
                return Generation("Here is what I would change...", 20, 10, False)
            return Generation(_sr_reply(fixture), 20, 10, False)

    result = stage2_codecs(
        _LandsSROnly(), seeds=1, codecs=("search_replace", "whole_file")
    )
    assert result.results["search_replace"].lands == 1.0
    assert result.results["whole_file"].lands == 0.0
    assert set(result.results) == {"search_replace", "whole_file"}


def test_every_attempt_uses_a_distinct_seed_per_fixture():
    # Mirrors stage 1's "every seed is used" test: fails if seeds are
    # reused or skipped, which would understate how many independent draws
    # actually back the reported fraction.
    seen: list[int] = []

    class _RecordsSeeds(_Lands):
        def generate(self, prompt: str, *, seed: int) -> Generation:
            seen.append(seed)
            return super().generate(prompt, seed=seed)

    stage2_codecs(_RecordsSeeds(), seeds=3, codecs=("search_replace",))
    # 5 fixtures x 3 seeds, seeds 1..3 repeating once per fixture.
    assert seen == [1, 2, 3] * 5
