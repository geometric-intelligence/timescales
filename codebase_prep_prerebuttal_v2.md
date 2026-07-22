# Codebase Preparation Spec — Pre-Rebuttal Power-Law Initialization Experiments

**Purpose of this document.** Prepare the existing RNN-timescales codebase so that a specific set of
power-law-initialization experiments can be run cleanly, reproducibly, and at scale before and after
the NeurIPS reviews arrive. This is **infrastructure and refactoring work**, not research. The goal
is to make the target experiments runnable with one command, so that the results aggregate without
editing code by hand. When the reviews arrive, we want to execute immediately rather than build
tooling with little time available.

**Scope boundary.** Do **not** implement any learning-dynamics analysis (fixed-point tracking,
bifurcation detection, eigenvalue-through-training). That is a separate project. Where noted, leave
clean extension points, but implement nothing beyond stubs.

**Source of truth for the experimental setup.** The paper's Figure 3 experiment is the reference
configuration. Match it exactly rather than re-deriving:
- Models: linear RNN and Tanh RNN, jointly optimizing `Θ = {W, W_in, W_out, A}` via BPTT + Adam.
- Standard init: `W_ij ~ N(0, 1/N)`, homogeneous time-constants `τ_i = 1`.
- Power-law init: time-constants power-law-distributed over `[τ_min, τ_max] = [1, 200]` timesteps
  with exponent `β` (reference value `β = 1`).
- Recurrent gain `g = 0.9` (currently a single value; this spec sweeps over it).
- Task: heterogeneous 5-bit flip-flop (currently). We are adding heterogeneous sine-wave generation.
- Metrics: validation-loss curve; steps to 90% accuracy.
- Seeds: currently 5; we are increasing this to ~20.

---

## Step 0 — Inventory before changing anything

Before writing code, produce a short **capability report** (`docs/codebase_inventory.md`) mapping
what already exists. Do not refactor until a person reviews this report. Identify and cite file
paths and line ranges for:

1. Model definitions — where the linear and Tanh RNN forward pass lives; how `τ_i` and `A` (leak)
   enter the dynamics; how `g` scales `W`.
2. Initialization logic — **the existing power-law `τ` init used for flip-flop in the paper.** This
   already works; it is the code to reuse everywhere. Note its exact parameterization (random draw
   vs. rank-ordering, scaling to `[1,200]`, how `β` is applied).
3. Task and data generation — the flip-flop generator (heterogeneous timescales), and whether any
   sine-wave generator exists at all.
4. Training loop — optimizer setup, BPTT truncation (if any), validation cadence, the "steps to 90%
   accuracy" computation.
5. Existing sweep and config mechanism — argparse, hydra, yaml, or hardcoded loops? Logging via
   wandb, tensorboard, csv, or pickles?
6. Jacobian and spectrum utilities — anything that already computes eigenvalues, decay and
   oscillation timescales, or the spectral-pinching visualization from Fig. 3C.
7. Plotting — the scripts that produced the Fig. 3 panels (loss curves, steps-to-convergence bars,
   Jacobian spectra, timescale histograms).

End the report with a **gap list**: for each workstream below, what already exists vs. what must be
built.

---

## Target architecture — a config-driven harness

Everything below assumes a single experiment entry point driven by a config, so that a "sweep" is a
set of configs rather than edited source. If a config system already exists, extend it; do not
introduce a second one.

A single run should be fully specified by a config with roughly these fields (names illustrative;
match existing conventions):

```yaml
task:
  name: flip_flop | sine_wave
  n_bits: 5                 # flip-flop
  n_freqs: 2                # sine-wave
  timescale_dist: heterogeneous
  seq_len: ...
  # task-timescale parameters (decay rates / frequencies) — see Workstream A
model:
  type: linear | tanh
  N: ...
  g: 0.9
init:
  scheme: standard | power_law
  tau_min: 1
  tau_max: 200
  beta: 1.0                 # ignored when scheme=standard
  W_std: null               # defaults to 1/sqrt(N)
optim:
  lr: ...
  optimizer: adam
  n_steps: ...
run:
  seed: ...
  val_every: ...
  log_spectrum: true        # snapshot Jacobian spectrum (see Workstream C)
  spectrum_cadence: init_and_end   # init_and_end | periodic:<k>
logging:
  out_dir: ...
  run_id: ...               # deterministic from config hash + seed
```

Requirement: **the run_id and output paths are deterministic from (config, seed)**, so reruns are
idempotent and sweeps are resumable (skip completed runs).

---

## Workstream A — Task parity: sine-wave generation power-law path (TOP PRIORITY)

The flip-flop power-law path already works. The task is to make the **same init and the same
measurement pipeline** run on heterogeneous sine-wave generation, reusing existing code wherever
possible.

**A1. Sine-wave task generator (if absent).** Heterogeneous sine-wave generation: multiple target
output channels, each a sinusoid at a distinct frequency (reference: 2 frequencies, "slow" and
"fast," per the paper's heterogeneous framing). Inputs specify which frequency each channel should
produce. Expose the task timescales (oscillation periods `T_i = 2π/ω_i`) as quantities the code logs
directly, mirroring how flip-flop exposes its decay timescales.

**A2. Convergence metric — this needs a decision, do not guess.** Flip-flop uses "steps to 90%
accuracy," which does not apply to a continuous generation task. Implement a pluggable
`convergence_metric` and **report the choice to the researcher** (see Design Decisions). Implement
these candidate definitions behind a flag, each with an explicit config field so nothing is
hardcoded:
- `mse_threshold`: steps until normalized MSE (or `1 − R²` against target) drops below
  `conv_mse_threshold`.
- `frac_of_final`: steps to reach `conv_final_frac` (e.g., 0.90) of the run's own final performance.
- `phase_amp_tol`: steps until per-frequency phase/amplitude error falls below `conv_phase_tol` /
  `conv_amp_tol`.

Whatever is chosen must be computed identically across init conditions, and must be invariant to the
different loss *scale* of sine-wave vs. flip-flop (this is why the normalized/fractional forms
exist). Log the full validation curve in every case, so the metric choice can be revisited during
aggregation without rerunning.

**A3. Reuse the init and spectrum code unchanged.** The power-law `τ` init and the Jacobian spectrum
utilities must apply to the sine-wave runs with no task-specific branching. If they currently assume
flip-flop, refactor them to be task-agnostic.

**A4. Smoke test.** A tiny smoke config (small N, few steps, 1 seed) that trains sine-wave under both
init schemes end-to-end and emits a curve + spectrum snapshot, so the path is verified before any
large sweep.

**Note (do not code around this):** it is a live possibility that power-law init helps flip-flop but
*not* sine-wave, because the real-axis spectral pinch suits decay tasks, not oscillatory ones. The
harness must report this outcome clearly rather than obscure it — the sine-wave result should be
reported on equal footing whether positive or negative.

---

## Workstream B — Fair-comparison sweep infrastructure

The rebuttal-critical requirement is that the "faster training" claim withstands a skeptical
reviewer. Build the sweep machinery to support that.

**B1. Grid dimensions.** The harness must sweep the Cartesian product of:
- `init.scheme` ∈ {standard, power_law}
- `task` ∈ {flip_flop, sine_wave}
- `model.type` ∈ {linear, tanh}
- `init.beta` ∈ (a range — see Design Decisions), for power_law runs
- `model.g` ∈ (a grid — see Design Decisions)
- `optim.lr` ∈ (a range — see below)
- `run.seed` ∈ 20 seeds

This is large. Implement it as a job generator over configs plus a runner that supports resume (skip
existing run_ids), parallel execution, and per-run checkpointing. Do not hardcode the grid in a
script; read it from a sweep spec file.

**B2. Two-stage LR handling (fair speed claim).** Reviewers attack "faster training" by claiming the
baseline's LR was untuned. Support **per-condition best-LR selection**: for each condition (task ×
model × init scheme × g × β), sweep LR and select the best-performing LR *for that condition* by a
fixed rule set before the runs. Expose this rule as a config field `lr_selection_rule` (default:
best median final validation loss over seeds). The headline comparison is then **best-LR-standard
vs. best-LR-power-law**. The selection must be code, not manual, and the chosen rule and selected LRs
must be logged.

**B3. β robustness — report the conservative (worst-case) gap, backed by the full curve.** Treat the
two swept parameters differently, because "conservative" means opposite things for each:

- **β is the method's own hyperparameter**, so worst-case over β is the *fair, conservative*
  summary: "even at the exponent least favorable to us, power-law still beats homogeneous by at
  least X." The headline statistic is therefore the **minimum benefit over β**, a lower bound on the
  effect. (Reporting the *largest* gap — the best-case β — would be selecting the most favorable
  result; we do not do that.)
- **LR is a training nuisance parameter** (handled in B2), where worst-case is *not* fair: a bad LR
  just means a badly-tuned run. LR is optimized per condition (best-vs-best), not minimized.

Two guards make the minimum-over-β statistic genuinely conservative rather than accidentally
optimistic:
1. **The baseline must be optimally LR-tuned (B2) for the lower bound to be valid.** If the
   homogeneous baseline is under-tuned, *every* gap including the smallest is inflated by an
   under-tuned baseline. B2 (best-vs-best LR) and B3 (min over β) are not alternatives — the min-gap
   claim depends on B2 holding.
2. **Take the minimum over β values at or above a configured `beta_active_min`, not the degenerate
   small-β tail.** As β → 0 the power-law init collapses toward homogeneous, so the gap → 0 by
   construction; a minimum that includes that tail would understate a real benefit for a trivial
   reason. Expose `beta_active_min` as a config field so a person sets where the method is
   meaningfully engaged.

Deliverable: the full **benefit-vs-β curve** (with seed spread), from which the conservative minimum
over `β ≥ beta_active_min` is read off. The curve is what goes to the reviewer — it contains the
lower-bound number *and* shows it is not an artifact of the degenerate tail, and that the benefit is
a plateau rather than a single favorable point (expected to peak near β≈1). Report both: "≥ X for
β ≥ beta_active_min," plus the curve.

**B4. Seed scaling.** Move from 5 to ~20 seeds per condition as a config value, and make seed a
sweep axis so the count is easy to adjust if compute is tight.

---

## Workstream C — Metrics & logging schema

**C1. Per-run structured record.** Every run writes one machine-readable record (row) plus its full
curve. Minimum fields: full config (flattened), `run_id`, seed, per-step validation loss, per-step
convergence metric, `steps_to_convergence`, final loss/accuracy, wall-clock time, and the Jacobian
spectrum snapshot handles. Prefer a columnar store (parquet) for the run-level table and separate
arrays for curves and spectra, so aggregation never re-opens checkpoints.

**C2. Jacobian spectrum snapshots.** At minimum at init and end; optionally periodic (`periodic:k`)
behind the existing flag. For each snapshot store the complex eigenvalues and the derived decay and
oscillation timescales. Also compute the following named **spectral-pinching statistics** at init,
so the "expands the range of available timescales" claim is quantified, not only shown:
- `frac_near_real_axis`: fraction of eigenvalues with `|Im(λ)| < eps_real_axis` (a configured band).
- `max_abs_lambda` and `gap_to_unit_circle` (= `1 − max|λ|`): how close the spectrum sits to the
  stability boundary.
- `decay_timescale_range`: spread (e.g., min/max or a percentile range) of the decay timescales
  available at init.

These statistics feed both the aggregation and Fig. 3C. Define `eps_real_axis` as a config field.

**C3. Determinism & provenance.** Log the resolved config, git commit, library versions, and RNG
seeds with every run. A run must be reproducible from its logged config alone.

---

## Workstream D — Aggregation & statistics

**D1. Cross-seed aggregation.** Given a completed sweep, produce mean ± CI validation curves per
condition and distributions of `steps_to_convergence`.

**D2. Effect size + significance.** For the headline speed comparison, report an effect size and a
significance test appropriate to the data. Step-to-convergence counts are typically non-normal and
possibly heavy-tailed, so default to a rank-based test (Mann–Whitney U) and a rank-based effect size
(Cliff's delta), with Cohen's d as a secondary. Implement the test as a function over the run table,
not a manual step.

**D3. Fair-comparison report.** One function that, given the sweep table, emits: best-LR-per-
condition selection, best-vs-best comparison with effect size and p-value, the benefit-vs-β curve,
and the per-g slices. This report is what we give to a reviewer.

---

## Workstream E — Plotting

Regenerate the Fig. 3 family, parameterized by task/model/g so the sine-wave and per-g variants
require no additional work:
- Validation-loss curves (seeds + mean), per init scheme.
- Steps-to-convergence bars/distributions, per init scheme.
- Jacobian spectra at init (standard vs. power-law), with the pinching visualization.
- Available-timescale histograms at init.
- New: benefit-vs-β curve; per-g panel grid.

Plotting reads only the aggregated tables and arrays from Workstreams C–D, never checkpoints.

---

## Deferred / extension points (leave stubs only, implement nothing)

- **Sequential MNIST.** Leave a task-registry slot and a note; do not implement. (Deferred: harder
  task, contestable task-timescale definition, and not appropriate to add with little rebuttal time.)
- **Rich/lazy regime control.** Expose `init.W_std` (weight magnitude at init) as a config field so
  the rich-vs-lazy question can be probed later; add an optional step-detection hook on the loss
  curve, but do not build the analysis.
- **Learning-dynamics hooks.** The periodic spectrum-snapshot mechanism (C2) is the only interface
  needed later for eigenvalue-through-training work. Do not add fixed-point or bifurcation code.

---

## Design decisions to confirm before large runs

Implement the machinery, but **do not silently pick values** for these; report them to the
researcher for a decision:

1. **Sine-wave convergence metric** (A2) — which definition is the headline metric, and the value of
   its threshold field (`conv_mse_threshold`, `conv_final_frac`, or the phase/amp tolerances).
2. **β grid and `beta_active_min`** (B3) — the range and resolution of the sweep, and the small-β
   boundary below which power-law collapses toward homogeneous. The conservative minimum over β uses
   `β ≥ beta_active_min` only.
3. **g grid** — which values around the reference 0.9.
4. **LR range and `lr_selection_rule`** (B2) — the set to sweep, and the rule for picking the best
   LR per condition.
5. **Seed count / compute budget** — 20 is the target; confirm given the full grid size.
6. **Flip-flop n** — keep 5-bit as the reference, or also vary?
7. **τ range** — confirm `[1, 200]` for sine-wave as well.
8. **`eps_real_axis`** (C2) — the band width defining "near the real axis" for the pinching statistic.

Report the full grid size (product of all axes × seeds) and a rough runtime estimate **before**
launching, so the budget is a deliberate choice.

---

## Acceptance criteria

The prep is done when:
1. `codebase_inventory.md` exists and a person has confirmed the gap list.
2. A single command runs one fully-specified config end-to-end for **both** tasks and **both** model
   types, under both init schemes, emitting the structured record + curve + spectrum snapshot.
3. The smoke sweep (small N, few steps, 2 seeds, both tasks) completes, is resumable, and aggregates
   into the Fig. 3-family plots plus the benefit-vs-β and per-g panels.
4. The fair-comparison report function runs on the smoke sweep and emits best-vs-best with an effect
   size and a p-value.
5. Runs are idempotent and reproducible from logged config alone.

Get the smoke sweep working end-to-end **before** any full-scale run.

---

## Suggested sequencing for the agent

1. Step 0 inventory → gap list.
2. Config harness + deterministic run IDs + resume (Workstream B scaffolding).
3. Sine-wave generator + convergence metric behind a flag; reuse init/spectrum code (Workstream A).
4. Logging schema + spectrum snapshots + pinching statistics (Workstream C).
5. Smoke sweep passing end-to-end (acceptance #2–3).
6. Aggregation + fair-comparison report + stats (Workstream D).
7. Plotting parameterized across task/g/β (Workstream E).
8. Report the design decisions, print grid size + runtime estimate, then hand back for full launch.
