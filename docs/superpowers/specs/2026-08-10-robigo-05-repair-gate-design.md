# robigo plan 05 — the repair gate (profiler stages 4 and 5)

Design for build-order step 5 (partial) and step 6. Written 2026-08-10, before
any stage-4 measurement exists anywhere. That timing is load-bearing: this
document changes the kill threshold, and it may only be read as a correction
rather than a moved goalpost because no data existed to move it toward.

Supersedes nothing. Amends two sentences of
`2026-08-09-robigo-design.md` (§0.2), listed in §2 below.

---

## 0. What this settles, and what it deliberately does not

Plan 05 builds the instrument that produces **the one number this project
exists to produce**, and then reads it.

- **Builds:** two CLI prerequisites, profiler stage 4 (repair), profiler
  stage 5 (loop discipline).
- **Then does:** applies the §0.2 kill criterion to stage 4's result.
- **Does not build:** stage 3 (payload corruption). It is deferred to plan 06
  and runs only if the gate passes.

Stage 3 is deferred on the spec's own reasoning (§8: *"Planning past a gate
whose outcome could cancel the remaining steps is how the wasted-effort
scenario above actually happens"*). Stage 3 feeds nothing the gate reads, and
costs a separate measured pass with its own model calls. Stage 5, by contrast,
is built now because it costs **zero** additional model calls — its two metrics
are observations of the runs stage 4 already performs — and because it is the
data that makes a low number explicable rather than merely disappointing.

---

## 1. Four decisions, taken while neutral

Recorded with their reasoning because each one moves the number, and a reader
in six months must be able to check that none of them was chosen after seeing
it.

### 1.1 The gate reads a `--full` run, at 10 seeds

The original §0.2 said *"quick-profile stage 4"*. §5 says quick profile runs
**stages 0–2** and `--full` runs all six at 10 seeds, and that `--full` is
*"the only mode whose numbers may be published"*. As written, the gate
therefore read a number from a mode that never ran stage 4 — and had stage 4
been added to quick mode, the spec's own honesty rules would have forbidden
publishing the very number that decides the project's fate, while §0.2 calls a
null result *"the most valuable artifact in the niche"*.

**Ruling:** the gate reads stage 4 from `robigo profile --full` (10 seeds).

### 1.2 "Strict" is defined

§5 describes stage 4 loosely as *"does the test go green?"*. §0.2's operative
word is **strict**, which appeared nowhere in the repo. One stage-4 attempt is
a **pass** if and only if all four hold:

```
pass  ⇔  the recorded failing test (record.test_id) goes green
      ∧  the whole target suite is green      (broken == 0; baseline.broken is 0)
      ∧  the anchor test file is byte-unchanged
      ∧  RunResult.outcome == "pass"          within the turn cap
```

`"pass"` is the literal string `loop.OUTCOMES` uses (`loop.py:257`). An earlier
draft of this document said `"repaired"`, which exists nowhere in the codebase
— caught while verifying externals for the plan. Recorded rather than silently
fixed, because plan 01's first process lesson is that plan text naming things
that do not exist is this project's dominant defect source.

Anything else is a fail, including a run that greens `test_id` while breaking a
neighbour. The corpus already carries the ground truth each clause needs:
records are kept only when the mutant breaks **exactly one** test against a
baseline recorded as `broken == 0`.

Two rejected alternatives, and why:

- **Target test green only** would score a model that greens one test by
  breaking two others as a success, overstating the tool.
- **Must match the stored reference patch** would fail genuinely correct
  alternative fixes, measuring imitation rather than repair — understating the
  tool against its own thesis.

### 1.3 The corpus is one external repo, frozen before any run

Corpus difficulty moves this number more than the model does. Measured in plan
04: robigo's own `src/` kept **0 of 7** and **0 of 10** mutants at ~15 s per
verification, while a narrow-test repo kept **4 of 5** and **6 of 10** at
~0.1 s.

**Ruling:** mine one real third-party Python repo, freeze the corpus, commit
it. Selection criteria and protocol are §5. Mining robigo itself was rejected
as measured-infeasible; pooling several repos was rejected as cost the gate
does not need, at the price of the per-repo generality noted in §8.

### 1.4 The threshold is 33.3%, not 40%

§0.2 asked for this explicitly: *"This number is the one figure in this
document chosen by the author rather than derived; it wants a sanity check
before implementation starts."* This is that sanity check.

The stated rationale was *"below 40%, a fix needs more than three attempts on
average."* Expected attempts to first success is `1/p`:

| pass rate | average attempts |
|---|---|
| 40% | 2.5 |
| **33.3%** | **3.0** |
| 25% | 4.0 |

The rationale does not produce 40%; it produces 33.3%. At 39% a fix still
averages 2.6 attempts, not "more than three". The number and its justification
disagreed by about seven points.

**Ruling (Brice, 2026-08-10, no stage-4 data in existence):** the *reasoning*
was the real commitment and the number was a mis-transcription of it. The
threshold becomes **33.3%**.

This lowers the bar by seven points, and that is the shape of a moved goalpost
even when it is done honestly. Three things are what make it a correction
instead, and all three must survive into any write-up:

1. It was taken before the instrument existed, let alone a measurement.
2. It restores the document's own stated argument rather than replacing it.
3. It is recorded here, in public, with the arithmetic and the prior value.

§0.2's prose must be rewritten to state 33.3% and the three-attempt derivation
together, so the two can never drift apart again.

---

## 2. Required amendments to the design spec

Both in `2026-08-09-robigo-design.md` §0.2, to be made as part of plan 05:

1. *"quick-profile stage 4"* → *"full-profile stage 4"* (§1.1).
2. *"below **40% strict**"* → *"below **33.3% strict**"*, with the
   attempts-to-success derivation stated inline and a footnote recording the
   2026-08-10 correction from 40% and why (§1.4).

Neither amendment may be made silently. The old value stays visible in the
document's own history and in `CARRIED-DEBT.md`.

---

## 3. Prerequisites — two CLI flags, both blockers

### P1 · `robigo profile --corpus PATH`

Known, recorded in `CARRIED-DEBT.md`. `stage2_codecs` already accepts a
converted corpus and `fixtures_from_corpus` converts one; nothing selects a
generated file from the CLI, so a live profile still measures `fixtures-v1`.

**Invariant P1.1** — the corpus's identity travels with the profile.
`Profile.corpus` names the file actually used, never a default. `run_profile`
already refuses a defaulted corpus name; the CLI must pass the loaded corpus's
own `name`, not a literal.

**Invariant P1.2 — the one that matters.** `fixtures_from_corpus` drops ~9.2%
of real records as unwrappable (measured: 91 of 986 from `src/robigo`). Those
are **harness artifacts and are never scored as model failures.** Its `dropped`
list must be carried into `Profile.dropped` verbatim, and dropped records must
be excluded from both the numerator and the denominator of **stage 2's** rate.

**Scope correction (found while freezing the corpus, before any stage-4 run).**
This originally said "every rate". It applies to stage 2 only. `fixtures_from_
corpus` wraps a record's line into a small synthetic function body, and a line
like a lone `(` cannot parse standing alone — that is a limitation of the
*wrapper*, not of the record. **Stage 4 writes `record.broken` into the real
file at `record.path:record.line`, where the surrounding lines supply the
context the wrapper lacks, so every record is usable there.** The two stages
therefore have different denominators, both correct. Measured on the frozen
corpus: 94 records, of which 90 convert for stage 2 (4.3% loss) and all 94 are
stage-4 material.

*Falsification test:* build a corpus containing a known-unwrappable record;
assert the resulting `Profile.repair_records` excludes it, that
`Profile.dropped` names it, and that the rate computed is identical to the rate
from a corpus with that record physically absent. If those two rates differ, a
harness artifact is inside the gate's number — which is exactly the plan-03
stage-2 defect (measuring the harness rather than the model), landing this time
in the figure that decides the project.

### P2 · `robigo profile --window N`

Newly found while designing this plan; not previously recorded anywhere.

`plan_window` already accepts a user cap. `cli.profile_main` passes `None`. So
for `qwen2.5-coder:7b` — the best-measured family, at 100%/100% codec landing —
`plan.window` resolves to its full **32768** training context, because VRAM
never binds on this box (~7.6 GiB weights + ~1.75 GiB KV against 14,558 MiB
free). Stage 0's probe therefore targets 32768, and this box's Ollama returns a
200 with no `done`/stats above **~11.5k prompt tokens** (measured: 11500 fine,
11800+ fails 100%). The run dies before stage 0 finishes. This is why
`qwen7b.jsonl` is committed as a single `outcome: "error"` row.

**Consequence, stated plainly: without P2 the best available family cannot be
profiled on this box at all, and the gate cannot be run.**

**Invariant P2.1** — `--window N` is a *ceiling*, never a floor.
`plan_window`'s existing `min(training_ctx, vram, user_cap)` semantics are
unchanged; the flag only adds a fourth term. It can never raise a window above
what geometry allows (spec §9 law 1).

**Invariant P2.2** — a capped window is visible in the profile.
`window_limited_by` reads `user_cap` when the flag binds, so a profile produced
under a cap can never be read as the family's full capability.

*Falsification test:* profile a family with `--window` above its training
context and assert the resulting `usable_window` is unchanged from the uncapped
run and `window_limited_by` still names `training_ctx`.

---

## 4. Stage 4 — repair

### 4.1 Shape

For each `(record, seed)` pair, against a clone of the corpus's `source_repo`
checked out at its recorded `source_sha`:

```
1. reset    git checkout <pristine-branch>            (captured ONCE, see below)
            git reset --hard <pristine-sha>
            git clean -fdq
            delete any robigo/* branches the loop created
2. break    write record.broken at record.path:record.line
3. repair   loop.run(task, root=clone, client, adapter=python_,
                     codec=<stage 2's best_codec()>, turn_cap=8,
                     allow_test_edits=False, use_git=True)
4. judge    pytest_runner + compare against Baseline
5. anchor   sha256 of the test file, before vs after
6. pass  ⇔  outcome == "pass" ∧ step 4 clean ∧ step 5 unchanged
```

Steps 2, 4 and 5 are plan 04's already-mutation-tested harness —
`pytest_runner`'s sanitized environment and forced `PYTHONPATH`,
`Baseline.executed`, `_assert_in_clone`'s `MODULE_UNDER_TEST=` marker. Stage 4
substitutes "run the loop, *then* run pytest" for "run pytest". It is mostly
composition, and it must stay that way: a second, parallel implementation of
any of those three is the drift class this project has paid for repeatedly.

**Step 1 was wrong in this document's first draft, and the error was
project-killing.** It read `git checkout -- . && git clean -fdq`. That does not
isolate an attempt, because §4.3.1 mandates `use_git=True`: the loop checks out
a `robigo/*` branch and `git add -A` snapshots the tree — **committing the
staged defect** — so `git checkout -- .` restores to *that branch's* index, not
to the pristine tree, and never returns to the original branch. Measured during
task 4's review:

```
PRISTINE    branch=master                    other.py='VALUE = 1\n'
AFTER ATT1  branch=robigo/the-test-fails-1   other.py='VALUE = 999  # stray edit\n'
AFTER RESET branch=robigo/the-test-fails-1   other.py='VALUE = 999  # stray edit\n'
            mod.py = the defect, still committed;  git status: clean
```

One `repo` is shared by every `(record, seed)` attempt and nothing re-clones, so
record 1's unrepaired defect would persist into every later record,
`state.broken == 0` could never hold again, and essentially no attempt after the
first could pass. **The gate would have read a near-zero repair rate and killed
the project on a harness bug.** The pristine branch and SHA must therefore be
captured ONCE, before the first attempt, and never re-derived from a tree that
may already be corrupted — re-deriving after attempt 1 captures the corruption
as the new "pristine".

§4.3.3's falsification test below is what catches this, and in the first
implementation it was named here and never written. An invariant stated without
its test is how this class of defect reaches a measurement.

### 4.2 The task string

Corpus records carry `path` and `line`, so handing the model the defect
location is one line of code away. **The task names only the failing test id**
— `"the test <test_id> fails; make it pass"` — and scope resolution must locate
the file the way it would for a real user. A number produced by telling the
model where the bug is describes a tool nobody has.

*Falsification test:* assert the **task string** (`task_for`) contains neither
`record.path` nor `record.line` nor `record.fixed`. Pin it; this is a one-token
edit away from silently inflating the headline figure.

This originally said "the rendered stage-4 prompt", which is **unsatisfiable and
was wrong**: scope resolution legitimately puts the defective file's contents in
the prompt — that is what the tool does for a real user, who also does not say
where the bug is. The property that matters is that *robigo* was never told the
location, not that the file is absent from the context robigo assembled for
itself. Corrected after task 4's review.

### 4.3 Invariants

- **4.3.1 The run is the shipped tool.** `use_git=True`, the real `turn_cap`,
  the real codec chosen by stage 2. A defect on the git path counts against the
  tool, because a user meets it.
- **4.3.2 `allow_test_edits=False`, and it is checked anyway.** The anchor hash
  in step 5 is belt-and-braces over the loop's own guarantee. If the two ever
  disagree, the loop's guarantee is broken and the profile must say so rather
  than average over it.
- **4.3.3 Every attempt is independently isolated.** Step 1 runs before every
  attempt, not once per record. *Falsification test:* run two attempts where
  the first leaves a stray file; assert the second's baseline comparison is
  unaffected.
- **4.3.4 An infrastructure failure is not a model failure.** A clone that
  will not reset, a suite that will not collect, a daemon error — none of these
  is scored as a failed repair. Each is excluded from the denominator and named
  in `dropped`. This is invariant P1.2's rule applied to runtime rather than
  conversion: the gate's number counts only attempts where the model was
  actually given a fair chance to repair.

### 4.4 Stage 4 is gated, like every stage before it

Stages gate each other cheapest-first, and stage 4 is no exception. It runs
only if `Profile.best_codec()` returns a codec — which already means stage 0
verified a window, stage 1 cleared `ENVELOPE_FIDELITY_MIN`, and stage 2 landed
something above 0%. If it returns `None`, **stage 4 does not run**,
`repair_rate` stays `None`, and `dropped` names which upstream stage closed the
gate.

`repair_rate: None` and `repair_rate: 0.0` are different facts and must stay
distinguishable in the written profile — "never measured" versus "measured and
nothing was repaired". This is the same collapse `dropped` exists to prevent
for `codecs`, one level further down.

**Consequence for the gate, stated so it cannot be fudged later:** a family
whose `repair_rate` is `None` has **not** passed the gate. It has not been
measured. Only a family that reached stage 4 and produced a real rate can be
compared against 33.3%.

### 4.5 What stage 4 produces

`repair_rate` = passes / attempts, over attempts that survived 4.3.4, plus
`repair_records`, `repair_attempts`, and the per-record breakdown needed by §6.

**The rate and its interval are computed at different levels, deliberately.**
`repair_rate` is attempt-level (passes ÷ attempts) because that is the quantity
§1.4's attempts-to-success arithmetic is about. Its confidence interval is
computed at **record** level, because attempts within a record are correlated
and an attempt-level binomial interval would claim roughly √10 more precision
than the design actually has. Reporting an attempt-level CI would be a value
that looks like a measurement but is not — the exact defect class this project
keeps finding. The per-record breakdown exists so the record-level interval can
be computed at all.

---

## 5. Corpus selection protocol

Selection is a plan task with a written rationale, **ratified before generation
runs**, because afterwards it can only be a rationalization.

Criteria:

- pure Python, no compiled extensions — clone and `pytest` must simply work
- suite green at HEAD: `baseline.broken == 0`. §1.2's pass definition depends
  on it.
- baseline wall clock under ~30 s. Plan 04 measured that ~15 s per verification
  is what made mining robigo infeasible.
- no network access in tests
- permissive license
- enough source to mine ~100 records (§6)
- not robigo; and not a repo selected after seeing any number derived from it

Then it freezes: the corpus JSON is committed, with `source_repo` and
`source_sha` on every record.

**The anti-tuning law, extended.** `CARRIED-DEBT.md` already forbids tuning
corpus fixtures to raise a landing rate. That law now covers the gate: **once
committed, the corpus is not regenerated. If the number disappoints, the corpus
does not change.** A second corpus may be mined only as *additional* reported
data, never as a replacement, and never before the first number is published.

Known limitation, stated rather than mitigated: one repo's idioms shape the
result. §8 records it.

---

## 6. The gate

### 6.1 Sample size, and an uncomfortable fact about precision

Seeds within a record are correlated — the same defect, the same file, varying
only the sampling seed. Statistical power therefore comes from **records**, not
attempts, and the interval is computed at record level for the reason given in
§4.5:

| records | 95% CI at p̂ ≈ 0.33 | estimated stage-4 cost |
|---|---|---|
| 50 | ±0.13 | ~6 h |
| **100** | **±0.09** | **~12 h** |
| 350 | ±0.05 | ~44 h |

Cost figures are estimates from ~45 s per loop run, not measurements; the CI
figures are exact for a record-level binomial.

**No feasible sample resolves 33% from 33%.** Stating otherwise would be this
project's own recurring bug — a value that looks like a measurement but is not.

**Ruling:** N = **94 records × 10 seeds** — the frozen corpus (§5), 940 attempts,
~11.75 h. Records buy power; seeds buy reproducibility and are fixed at 10 by the
`--full` contract (§1.1).

#### 6.1.1 The record-level interval is itself optimistic — clustering

The table above treats records as independent Bernoulli trials. **They are not.**
The frozen corpus's 94 records span 19 files and 67 distinct tests, so up to 4
records share a test and several share a file. Records drawn from one module,
exercising one test, against one defect operator are correlated for the same
reason seeds within a record are: they present near-identical work.

So a record-level binomial interval is an *upper bound on precision*, not the
precision. The honest treatment, decided here before any number exists:

- **Report the record-level interval, labelled as a lower bound on the true
  width** — never as "the" confidence interval.
- **Report a cluster-aware interval alongside it**, clustering on `test_id` (67
  clusters, the binding constraint — fewer than the 94 records and more than the
  19 files). A cluster bootstrap over `test_id` is sufficient; nothing here needs
  a closed form.
- **Report the per-operator breakdown** (§5), since operator is the other axis
  along which records resemble each other.

This does not change §6.2's decision rule — the point estimate still decides, and
it is pre-registered. It changes only what may be *claimed* about that estimate's
precision. Quoting ±0.09 from the table as though records were independent would
be the exact defect this section exists to prevent.

### 6.2 The decision rule

1. **The point estimate decides.** `repair_rate >= 0.333` → the tool ships and
   the build order proceeds to step 7. Below → robigo becomes a
   benchmark-and-findings repo, per §0.2.
2. **Pre-registered, no peeking, no extension.** There is no "gather more
   records until it resolves" clause, deliberately: that is the door through
   which a project that should stop finds a reason to continue.
3. **The confidence interval is reported beside the estimate, always.** A null
   result that states its own precision is the publishable artifact §0.2
   promises; a bare "31%" is not.

The a-priori case that rule 1 is safe rather than a coin flip: the thesis
predicts 60–80%, and the counter-evidence it was written against (a 7B scoring
2/20 on Oxide) predicts ~10%. A result landing inside 30–36% is the unlikely
case, not the expected one. If it happens anyway, the honest report is the
interval and the stage-5 diagnostics, not a re-run.

### 6.3 Verdict stays separate from the gate

`verdict_for` is **not** changed. READY / LIMITED / UNUSABLE describe
*instrument fitness* — window, envelope, codec landing. The gate is a separate
judgment on `repair_rate`. Conflating them would let a future tweak to a
verdict threshold silently move the kill criterion.

---

## 7. Stage 5 — loop discipline

Two metrics, both **observations of stage 4's runs**. Zero additional model
calls.

- **turns-to-green:** the turn count on passing attempts. Reported as a median
  with its distribution, never a bare mean — a bimodal "lands on turn 1 or
  never" is the interesting shape and a mean hides it.
- **identical-failing-patch repeat rate:** the fraction of turns emitting a
  patch byte-identical to one already rejected in that same run.

These are why stage 5 is worth building before the gate rather than after: they
distinguish *"never lands an edit"* from *"lands edits but loops on a wrong
one"*. The first is a capability result; the second is an instrument result,
and an instrument result is fixable. A null number without them is merely
negative.

**Invariant 7.1** — stage 5 never causes a model call. *Falsification test:*
count client invocations across a stage-4-only run and a stage-4-plus-5 run
against the same records and seeds; assert equality.

**Invariant 7.2** — stage 5 metrics are `None`, not `0.0`, when no attempt
produced the observation (no passes → no turns-to-green). The
`max_file_tokens=None` collapse already recorded in `CARRIED-DEBT.md` is the
shape to avoid: one value must not mean both "not applicable" and "measured
zero".

---

## 8. Honesty rules for the reporter

Three details that are individually small and each produce a false profile if
missed:

1. **`repeat_rate`'s `dropped` line must be deleted** when stage 5 lands. It is
   currently emitted unconditionally by `run_profile`. Leaving it in would have
   the profile declare a field unmeasured while carrying its measurement.
   *Falsification test:* assert no `dropped` entry mentions `repeat_rate` when
   `repeat_rate is not None`, and that one does when it is `None`.
2. **`payload_corruption` keeps its `dropped` line**, with the text updated to
   name plan 06 — stage 3 is deferred, not forgotten.
3. **The corpus's provenance is reported with the number.** Source repo, source
   SHA, record count, dropped count, and the single-repo limitation from §5.
   A repair rate without the corpus it was measured on is not a result.

`Profile` gains `repair_rate: float | None`, `repair_attempts: int`,
`repair_records: int`, `turns_to_green_median: float | None`. `repeat_rate`
finally gets a producer. `from_json`/`to_json` round-trip every new field —
this project's records are read by future tooling, so the sequence is stored,
not a reduction of it.

---

## 9. Testing

Project law, restated because it is not optional here:

- **Every new test is mutation-tested.** A test that passes with the code
  deleted is a defect, not a test.
- **The offline suite stays green with `socket.socket.connect` patched to
  raise.** Grep finds the stubs you thought of; the socket finds the rest.
- **Sanitize the environment being measured in.** `PYTEST_ADDOPTS` exported as
  `-x` already turned a correctly-rejected mutant into a kept record once.
- **Run the tool, not just the tests.** Twice, on separate invocations — the
  worst user-facing defect of the project so far was invisible to 300 passing
  tests and appeared on the *second* consecutive run.
- **Stage 4/5 transcripts are recorded for replay**, with the limitation stated
  up front: stage-4 replay reproduces the **model** side only, because the loop
  also mutates a filesystem. Full stage-4 replay requires the source repo
  present at `source_sha`. This is a narrower guarantee than §5.3's "reproduce a
  known profile exactly" and must be documented as such rather than quietly
  claimed.

---

## 10. Risks

- **Single-repo generality (§5).** One repo's idioms shape the headline number.
  Accepted, stated, not mitigated. A second corpus is reportable as additional
  data after publication, never as a replacement.
- **The ~11.5k daemon ceiling is unfixed.** P2 works *around* it with a window
  cap; it does not fix it. Any family whose useful window exceeds ~11.5k is
  profiled below its real capability on this box, and the profile must say so
  via `window_limited_by`.
- **Stage 0's probe remains systematically pessimistic** (~0.55 of target token
  density, per `CARRIED-DEBT.md`). It bounds `usable_window`, which stage 4
  runs inside. The gate's number is therefore measured against a *conservative*
  window — biasing against the thesis, which is the safe direction, but it
  should be stated when the number is published.
- **~12 h of unattended GPU time** is the largest single compute commitment in
  the project. It should run once, correctly. That argues for a dry run at
  small N to shake out harness defects before the real run — and for
  `--corpus`/`--window` being exercised end-to-end first.
