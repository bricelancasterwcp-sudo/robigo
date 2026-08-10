# robigo 04 — Mutation Corpus Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate single-defect repair tasks by mutating real code, keeping only mutants that are mechanically verified — the broken form fails with the intended diagnostic and nothing else, and the reverse patch makes the suite green again. This replaces plan 03's hand-written `fixtures-v1` and is what makes stage 4's number, and therefore the §0.2 kill criterion, mean anything.

**Architecture:** A candidate generator proposes one-line mutations; a verifier runs the target repo's suite in an isolated clone and keeps a candidate only if it breaks *exactly one* test relative to a measured baseline; a writer emits a corpus record carrying the diagnostic and the reverse patch as ground truth. The verifier validates its own harness with a sentinel before trusting any survival result.

**Tech Stack:** Python 3.12+, stdlib only. Builds on plans 01–03.

## Global Constraints

- **Runtime dependencies: none.** Standard library only.
- `requires-python = ">=3.12"`; `from __future__ import annotations`; type annotations on every **non-test** function signature; pytest test functions are exempt.
- **A record may not enter the corpus unless the broken form fails with the intended diagnostic and nothing else, and the reference patch makes the test pass** (spec §5.1). That property is inherited unchanged and is the whole value of the corpus.
- **Never mutate the working tree.** Mutate an isolated clone; a generator that writes into the repo it is being developed in would corrupt its own source while its tests still pass.
- **Anything dropped is stated as dropped.** A candidate rejected for time, or a target skipped, is recorded — never silently absent.
- Commit messages: `<type>: <subject>`, single line, no body, no trailers.

## Measured before planning (2026-08-10)

These were measured on this box against robigo's own suite, and they change the design. Do not re-derive them; do check them if a number looks wrong.

**1. A verification cycle costs ~15 s.** One full `pytest -q` run of robigo's 406 tests. Every candidate needs at least one, so the corpus's size is bounded by wall-clock: a 50-record corpus at a 10% keep rate is ~2 hours of compute. Budget for it, and report what was dropped for time.

**2. Mutating well-covered central code almost never breaks exactly one test.** Sampling seven one-line mutations in `context/scope.py`:

| mutant | tests broken | new failures |
|---|---|---|
| `<=` → `<` | 6 | **0 — survived** |
| `==` → `!=` | 10 | 4 |
| `==` → `!=` | 18 | 12 |
| `if not` → `if` | 52 | **45** |

Zero of seven yielded exactly one. A hot function has many tests through it, so the spec's criterion rejects nearly everything there. **Target selection is therefore the core problem of this plan, not an afterthought** — prefer code with narrow coverage, and measure the keep rate per target so a barren target is abandoned rather than ground through.

**3. Count failures *and* collection errors, relative to a measured baseline.** A syntax-breaking mutant is reported by pytest as an *error*, not a failure; counting only `"N failed"` scores it as a clean run. And the baseline is not always zero — in a `git archive` copy it was **6**, because the copy has no `.git` and the git-dependent tests fail. Use a real clone so `.git` exists, measure the baseline anyway, and compare against it.

**4. The editable-install trap is real and it silently inverts the result.** robigo is installed editable, so a subprocess launched in a copied tree imports `/home/brice/workspace/robigo/src` unless `PYTHONPATH` is forced. Measured: without it, `robigo.__file__` resolves to the real repo and **every mutant appears to survive** — 8 of 8, a perfect false negative. `CARRIED-DEBT.md` warns about exactly this and I still hit it.

*Invariant:* the verifier asserts the code under test is the mutated copy before believing any result — check the resolved module path, do not assume the environment.

**5. A survival result is worthless unless the harness is proven able to detect a break.** After forcing the path, a deliberate sentinel (`estimate_tokens` returning 0) produced 18 failures; before, a malformed sentinel produced 0 and would have validated a blind harness. **The verifier must run a sentinel and abort if it detects nothing.** This is the check that caught both mistakes above.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/robigo/profile/corpus.py` | `Mutant`, `CorpusRecord`, candidate generation, the mutation operators |
| `src/robigo/profile/verify.py` | isolated clone, sentinel validation, baseline, keep/reject decision |
| `src/robigo/profile/corpus_io.py` | corpus JSON read/write, provenance |
| `src/robigo/cli.py` | *modified* — `robigo corpus` subcommand |

---

### Task 1: Mutation operators and candidate generation

**Files:**
- Create: `src/robigo/profile/corpus.py`
- Test: `tests/test_corpus_candidates.py`

**Interfaces:**
- Produces: `Mutant(path: Path, line: int, original: str, mutated: str, operator: str)` (frozen); `OPERATORS: tuple[str, ...]`; `candidates(source: str, path: Path) -> tuple[Mutant, ...]`.

Operators are named, one-line, and reversible — the reverse patch is ground truth, so every mutation must be exactly invertible by construction. Start with the five shapes `fixtures-v1` hand-wrote, because those were chosen to look like real defects: an off-by-one, a flipped comparison, swapped arguments, a dropped `return`, and an inverted condition.

**Invariant 1 — every mutant is exactly one line and exactly reversible.** Applying the mutation then its reverse yields the original file byte-for-byte. Test this as a round-trip over every candidate the generator produces from a non-trivial source file, asserting byte equality — not "looks the same".

**Invariant 2 — a candidate never changes what the file means to Python.** A mutant that does not parse is not a defect, it is a broken file; the corpus is for repair tasks, and a `SyntaxError` teaches nothing about repair. Reject non-parsing candidates at generation, and pin it with a candidate set that includes a line where a naive textual substitution would produce invalid Python.

**Invariant 3 — comments, docstrings and blank lines are never mutated.** A mutation inside a docstring cannot fail a test, so it wastes a 15-second verification. Test with a source file whose comments contain the operators' trigger patterns.

**Steps:** write the round-trip test first and watch it fail; implement; commit.

---

### Task 2: The verifier — sentinel, baseline, and the exactly-one rule

**Files:**
- Create: `src/robigo/profile/verify.py`
- Test: `tests/test_corpus_verify.py`

**Interfaces:**
- Consumes: `Mutant`.
- Produces: `Baseline(broken: int, seconds: float)` (frozen); `Verdict(kept: bool, failures: int, test_id: str | None, reason: str)` (frozen); `sentinel_ok(repo: Path, runner) -> bool`; `baseline(repo: Path, runner) -> Baseline`; `verify(mutant, repo, baseline, runner) -> Verdict`.

`runner` is injected so every test is offline and instant; the real runner shells out to pytest.

**Invariant 4 — the harness proves it can see a break before any survival is believed.** `sentinel_ok` applies a known-fatal change and requires the run to report breakage. If it does not, verification aborts rather than reporting a corpus full of survivors. This is not defensive decoration: measured, a blind harness reported 8 of 8 mutants surviving, a perfect false negative.

**Invariant 5 — breakage counts failures *and* errors, measured against the baseline.** `broken = failures + errors`, and a mutant is judged on `broken - baseline.broken`. A baseline of zero is not assumed. Test a runner whose output contains only `"1 error"` and no `"failed"` at all — that case scored as clean in my own measurement.

**Invariant 6 — "exactly one" means exactly one, and the test's identity is recorded.** A kept mutant has `broken - baseline == 1` and `Verdict.test_id` names the test that failed. Without the id there is no intended diagnostic to verify against later, and the spec's property becomes uncheckable. Test the 0, 1, and many cases separately, and assert the id is captured — not merely that the count was right.

**Invariant 7 — the code under test is the mutated copy.** Before believing a result, assert the resolved module path lies inside the clone. Test the failure mode: a runner that reports on the wrong tree must be rejected, not trusted.

**Steps:** tests first, each failing for its stated reason; implement; commit.

---

### Task 3: Corpus records, ground truth, and provenance

**Files:**
- Create: `src/robigo/profile/corpus_io.py`
- Test: `tests/test_corpus_io.py`

**Interfaces:**
- Consumes: `Mutant`, `Verdict`.
- Produces: `CorpusRecord(name, path, line, broken, fixed, test_id, diagnostic, operator, source_repo, source_sha)` (frozen) with `to_json()`/`from_json()`; `write_corpus(records, path, *, name, dropped) -> None`; `read_corpus(path) -> tuple[str, tuple[CorpusRecord, ...], tuple[str, ...]]`.

**Invariant 8 — the reverse patch is stored, not derived at read time.** `fixed` carries the original line. A consumer must be able to check ground truth without re-running the generator.

**Invariant 9 — every record names where it came from.** `source_repo` and `source_sha` pin the code the mutant was cut from. A corpus whose provenance is a default is not a corpus; plan 03 shipped `corpus="fixtures-v1"` as a kwarg default and it would have mislabelled every profile — that is carried debt this plan must not repeat.

**Invariant 10 — the round-trip is checked structurally as well as by equality.** Plan 03 learned this the hard way: whole-object equality misses a field added with a default, because both sides get the default. Assert every dataclass field name appears in the serialised payload, walking nesting, *and* assert `from_json(to_json(r)) == r`.

**Invariant 11 — the corpus states what was dropped.** Candidates rejected, targets abandoned as barren, records lost to a time budget — all recorded in the file. A corpus of 30 records that silently discarded 400 candidates is not interpretable.

---

### Task 4: `robigo corpus`, and retiring `fixtures-v1`

**Files:**
- Modify: `src/robigo/cli.py`, `src/robigo/profile/stages.py`
- Test: `tests/test_corpus_cli.py`

**Interfaces:**
- Produces: `corpus_main(argv) -> int`; CLI dispatch on a leading `corpus` argument, with `--repo`, `--out`, `--max-records`, `--time-budget`, `--target`.

Dispatch the way `profile` does — on a leading argv element, leaving the flat parser intact so `robigo "<task>"` keeps working.

**Invariant 12 — stage 2 consumes a corpus file, and `fixtures-v1` becomes one of them.** Plan 03 promised this module's interface would not change when the corpus arrived. Honour it: `stage2_codecs` keeps its signature, and the bundled fixtures are expressible as a corpus file. **Also fix the two carried items on this path** — `Profile.corpus` must be derived from the corpus actually used rather than defaulting, and `best_codec()` must not return a codec that landed 0%.

**Invariant 13 — the keep rate is reported per target.** Measured: mutating `context/scope.py` kept 0 of 7. A generator that grinds through a barren target for an hour without saying so is wasting the operator's time; report the rate and abandon a target that is not producing.

**Invariant 14 — a real end-to-end run produces a real corpus.** Generate against a live repo and report: candidates proposed, kept, rejected with reasons, wall-clock, and the keep rate per target. If the rate is too low to build a useful corpus from robigo's own suite, **that is a finding to report, not a number to massage** — and it means the corpus should come from a repo with narrower tests, which is a design answer, not a failure.

---

## Done when

- `pytest -q` green, and green with `socket.socket.connect` patched to raise.
- A sentinel-blind harness aborts instead of reporting survivors.
- Every kept mutant breaks exactly one test, names it, and is exactly reversible.
- The corpus file carries provenance, ground truth, and what was dropped.
- `Profile.corpus` names the corpus actually measured.
- A real run reports its keep rate per target, and the number is stated whatever it is.
