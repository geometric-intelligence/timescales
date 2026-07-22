# Codebase Inventory & Cleanup Plan

**Status:** Step 0 of `codebase_prep_prerebuttal_v2.md`. Capability report + keep/remove
recommendation + gap list.

**Cleanup executed (2026-07-21), decisions confirmed with researcher:**
- *Removal method:* delete outright (incl. the old `archive/` and `sandbox/` dumps). 357 files removed.
- *Path-integration:* deep strip — removed the datamodules, the whole `timescales/analysis/` package,
  `scores.py`, and the `path_integration` / position-decoding branches in `train.py`, `callbacks.py`,
  and `RNNLightning` (`rnn.py`).
- *Kept borderline items:* `null_task` datamodule, `tanh_voltage_grid` sweep, flat `notebooks/<study>/` layout.
- *Also removed:* `coupled` + standalone-`schur` architectures (configs/sweeps/notebooks), teacher-student,
  binary-counter. **Kept** `schur_init.py` (live init helper inside `rnn.py`).
- *Verified:* core `py_compile` clean; core modules + `train.py` import cleanly in the poetry env;
  `MODEL_FACTORIES = {rnn, multitimescale}`; no dangling references outside notebooks (two notebook
  hits are an inert docstring + comment).

The capability map and gap list below describe the **post-cleanup** tree.

**Method:** usage traced by import-graph (`grep` of the `timescales` package), config/sweep
references, and git recency (`git log -1` per file/dir). "Active" below means touched in the
2026-07-21 commits (the current rebuttal work); "stale" means last touched Apr–May 2026 or earlier
and unreferenced by the current flip-flop / sine-wave path.

---

## 1. Capability report — what exists and where

### 1.1 Models (`timescales/rnns/`)

| File | Lines | Role | Status |
|------|------:|------|--------|
| `rnn.py` | 781 | **Core model.** `RNN` + `RNNLightning`. Per-unit time constants `τ` (leak `A`), `g`-scaled `W`, linear/Tanh via `activation`, `dynamics_type` (rate/voltage). This is the Fig. 3 model. | **Active** (07-21) |
| `schur_init.py` | 266 | `compute_W_tilde` — builds `W_rec` from a target power-law spectrum. **Live dependency of `rnn.py`** via `wrec_init="schur_powerlaw"`. Not a separate model. | Active dep |
| `coupled_rnn.py` | 356 | `CoupledRNN` — two coupled populations (nonlinear r + linear s). Separate architecture, exploratory. | Stale (04-12) |
| `schur_rnn.py` | 346 | `SchurRNN` — standalone Schur-decomposed recurrent weights. Separate architecture, **distinct from `schur_init.py`**. | Stale (04-10) |

- **Power-law `τ` init** (the code the spec says to reuse everywhere): `rnn.py` `_generate_time_constants`
  (`rnn.py:328`), `distribution: "powerlaw"` branch at `rnn.py:359` — inverse-CDF draw over
  `[min_time_constant, max_time_constant]` with `exponent`. Applied as fixed τ
  (`time_constants_config`) or as learnable-init (`init_time_constants_config`).
- **`W` init menu** (`rnn.py:434`): `orthogonal`, `normal_scaled` (N(0,1/N)), `levy_stable`,
  `lognormal`, `schur_powerlaw`. The last builds J from a power-law spectrum via `schur_init.py`.

### 1.2 Training / harness (`timescales/`)

| File | Role | Status |
|------|------|--------|
| `train.py` | Unified entry. `create_datamodule` (task factory, 8 tasks), 3 model factories (`rnn`/`coupled`/`schur`), `single_seed()` training loop, RNG-state dump, artifact save. | **Active** |
| `sweep.py` | Config-driven sweep: base-config + `grid`/`experiments` overrides, `Job` objects, `ProcessPoolExecutor` GPU scheduler, `seeds_outermost`, per-exp summary yaml. | **Active** |
| `run_job.py` | Single-job subprocess wrapper called by `sweep.py`. | Active |
| `callbacks.py` | 7 Lightning callbacks (see below). | **Active** |
| `scores.py` | `GridScorer` — grid-cell scoring. **Path-integration only**; imported only by `analysis/spatial.py`. | Stale (dead) |

**Callbacks** (`callbacks.py`): `LossLoggerCallback`, `TauTrajectoryCallback`,
`SpectralSnapshotCallback` (Jacobian spectrum at init/end — **directly relevant to Workstream C**),
`SpectralTrajectoryCallback` (periodic top-k eigenvalue tracking), `GradientStatisticsCallback`,
and `PositionDecodingCallback` + `TrajectoryVisualizationCallback` (**path-integration only**).

### 1.3 Tasks / data (`timescales/datamodules/`)

| DataModule | Task key | Referenced by current config? | Status |
|-----------|----------|:---:|--------|
| `flip_flop.py` | `flip_flop` | ✅ (4 configs, 7 sweeps) | **Core** (04-04) |
| `sine_wave.py` | `sine_wave` | ✅ (2 configs, 6 sweeps) | **Core** (05-01) |
| `signed_flip_flop.py` | `signed_flip_flop` | ✅ (asym_2bit demo) | **Active** (07-21) |
| `null_task.py` | `null` | theory notebook (untrained-spectrum) | Keep? (small) |
| `binary_counter.py` | `binary_counter` | 1 config, no sweep | Stale (01-12) |
| `teacher_student.py` | `teacher_student` | 1 config, 2 sweeps | Stale (04-10) |
| `path_integration.py` | `path_integration` | ❌ **none** | Stale (01-12) |
| `path_integration_1d.py` | `path_integration_1d` | ❌ **none** | Stale (01-31) |

### 1.4 Analysis (`timescales/analysis/`) — essentially dead

`measurements.py`, `spatial.py`, `load_models.py`, `analyses.py`, `sweep_evaluator.py`,
`plotting.py` + 3 `.md` docs + `test_measurements.py`. Only `measurements.PositionDecodingMeasurement`
is imported by core (`train.py`, `callbacks.py`) — **and only on the `path_integration` branch**, which
has no current config. No notebook imports anything from `analysis/`. This whole directory is
path-integration legacy. (The `archive/README.md` claim that these are "actively imported" is stale —
it refers to a `single_run.py` that no longer exists.)

### 1.5 Configs

- `configs/rnn/`: `flip_flop`, `flip_flop_hetero`, `flip_flop_asym_2bit_demo`, `sine_wave`,
  `sine_wave_hetero` → **keep**. `binary_counter`, `teacher_student`, `mts` → stale.
- `configs/coupled/`, `configs/schur/` → stale (their architectures).
- `sweep_configs/rnn/`: `flip_flop_*`, `sine_wave_*`, `schur_init_grid_g05/g09`, `tanh_dyn_grid`,
  `tanh_voltage_grid` → **keep**. `teacher_student_*` → stale.
- `sweep_configs/coupled/`, `sweep_configs/schur/` → stale.

### 1.6 Notebooks (`timescales/notebooks/`, jupytext `.py`)

| Subdir | Depends on | Status |
|--------|-----------|--------|
| `flip_flop/` | core `rnn` | **Active** (07-21) |
| `sine_wave/` | core `rnn` | **Active** (07-21) |
| `schur_init_grid/` | core `rnn` + `wrec_init=schur_powerlaw` (NOT `SchurRNN`) | **Active** (07-21) |
| `tanh_dyn_grid/` | core `rnn` | **Active** (07-21) |
| `theory/` | RMT / linear-network sims; uses `null` task | **Active** (07-21) |
| `figs/` | old generated metrics.csv + fig dumps | Stale (05-04) |
| `schur/` | `SchurRNN` | Stale (05-01) |
| `coupled/` | `CoupledRNN` | Stale (04-10) |
| `teacher_student/` | `teacher_student` | Stale (04-10) |

### 1.7 Already-archived / scratch

- `archive/` — March-2026 refactor dump (20 path-integration notebooks, dead analysis modules,
  legacy shims). Already dead by definition.
- `sandbox/` — 3 exploratory studies (`linear_network_timescales`, `preconditioner_viz`,
  `teacher_student_identifiability`); `results/` is gitignored so only stray config/png remain.

---

## 2. Recommended keep / remove

**Rebuttal core (KEEP):** `rnn.py`, `schur_init.py`, `train.py`, `sweep.py`, `run_job.py`,
`callbacks.py`; datamodules `flip_flop`, `sine_wave`, `signed_flip_flop`; configs & sweeps for
flip-flop / sine-wave / schur_init_grid / tanh_dyn_grid; notebooks `flip_flop`, `sine_wave`,
`schur_init_grid`, `tanh_dyn_grid`, `theory`.

**Proposed to remove (stale, out of rebuttal scope):**

| Group | Paths | Why |
|-------|-------|-----|
| Coupled architecture | `rnns/coupled_rnn.py`, `configs/coupled/`, `sweep_configs/coupled/`, `notebooks/coupled/` | Separate exploratory model; Apr 2026; not in Fig. 3 scope |
| Standalone Schur model | `rnns/schur_rnn.py`, `configs/schur/`, `sweep_configs/schur/`, `notebooks/schur/` | Distinct from the kept `schur_init.py`; Apr–May 2026 |
| Teacher-student | `datamodules/teacher_student.py`, `configs/rnn/teacher_student.yaml`, `sweep_configs/rnn/teacher_student_*`, `notebooks/teacher_student/`, `sandbox/teacher_student_identifiability/` | Exploratory; Apr 2026 |
| Path-integration | `datamodules/path_integration*.py`, entire `timescales/analysis/`, `scores.py`, path-integration branches in `train.py`/`callbacks.py`, `configs/rnn/mts.yaml` | Original task; **no current config references it**; largest single win |
| Binary counter | `datamodules/binary_counter.py`, `configs/rnn/binary_counter.yaml` | Exploratory; Jan 2026 |
| Stale notebooks | `notebooks/figs/` | Superseded generated outputs |
| Scratch | `sandbox/`, `archive/` | Already-dead reference dumps |

**Judgment calls flagged for you** (§ decisions below): `null_task` (keep as cheap
spectrum-at-init helper vs. remove), whether to **delete outright vs. move to `archive/`**, and how
aggressively to strip the path-integration branches from `train.py`/`callbacks.py`.

---

## 3. Gap list — per workstream (exists vs. must build)

- **Harness / config (Target arch, B-scaffolding).** *Exists:* base-config + override sweep,
  GPU pool, seed axis, per-run config/artifact dump, RNG-state capture.
  ✅ **Done (2026-07-22):** deterministic `run_id` = `config_fingerprint(config)_seed<seed>`
  (`timescales/run_ids.py`); deterministic single-run + sweep output dirs; idempotent reruns and
  resumable sweeps via a fingerprinted completion marker (`job_result.yaml`); `sweep.py --no-resume`
  to force. Covered by `tests/test_run_ids.py` + an offline CPU end-to-end smoke.
  *Still missing:* single `scheme: standard|power_law` knob (today expressed as raw
  `time_constants_config` + `wrec_init` combos).
- **A — Sine-wave parity.** *Exists:* `SineWaveDataModule`, hetero configs/sweeps, notebooks; init
  & spectral callbacks are already task-agnostic.
  ✅ **Done (2026-07-22):** pluggable `convergence_metric` (`timescales/convergence.py`) — computes
  every curve-based candidate (`frac_of_final`, `mse_threshold`; `phase_amp_tol` is a stub needing
  signal-level eval) from the always-logged validation curve, stored under
  `result["convergence"]["by_method"]`; headline deferred via `convergence_metric` config field
  (default unset). Tested (`tests/test_convergence.py`) + sine-wave end-to-end smoke.
  *Still missing:* explicit logging of task timescales `T_i = 2π/ω_i`; a committed smoke config (A4).
- **B — Fair-comparison sweep.** *Exists:* Cartesian `grid`. *Missing:* best-LR-per-condition
  selection (`lr_selection_rule`); β-robustness / min-over-β with `beta_active_min`; resume.
- **C — Metrics/logging schema.** *Exists:* `SpectralSnapshotCallback` (init/end eigenvalues),
  `SpectralTrajectoryCallback`, per-run yaml. *Missing:* columnar per-run table (parquet); named
  pinching stats (`frac_near_real_axis`, `gap_to_unit_circle`, `decay_timescale_range`,
  `eps_real_axis`); git-commit/lib-version provenance.
- **D — Aggregation & stats.** *Missing entirely* in core (ad-hoc in notebooks): cross-seed CI
  curves, Mann–Whitney U + Cliff's delta, the fair-comparison report function.
- **E — Plotting.** *Exists:* per-study notebook plotting (loss/acc curves, spectra, steps-to-threshold).
  *Missing:* parameterized Fig. 3 family reading only aggregated tables; benefit-vs-β and per-g panels.

---

## 4. Decisions to confirm before deleting

1. **Delete outright vs. move to `archive/`?** Git history preserves everything either way; `archive/`
   is the repo's existing convention. Recommendation: hard-delete the already-archived `archive/`
   and stale scratch; move freshly-removed architectures to `archive/` only if you want them one
   `git mv` away.
2. **`null_task`** — keep (used by theory notebook for untrained-spectrum; ~small) or remove?
3. **Path-integration depth** — remove just the datamodules/configs, or also strip the
   `path_integration` branches from `train.py`/`callbacks.py` and delete the whole `analysis/`
   package + `scores.py`? (The latter is the real de-clutter but touches the core entry point.)
4. **Notebooks convention** — the active subdirs are already task-named and reasonable. Keep the
   flat `notebooks/<study>/` layout, or regroup (e.g. `figures/` vs `explorations/`)?
