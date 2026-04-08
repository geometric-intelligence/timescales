# Archive

Deprecated, stale, or broken files moved here during the March 2026 refactor.
Everything below was part of the original codebase but is no longer actively used.
Kept for reference — feel free to delete anything you're sure you won't need.

---

## `analysis/`

Dead analysis modules that are no longer imported anywhere in the codebase.

| File | What it did | Why archived |
|------|-------------|--------------|
| `dynamics.py` | PCA, phase-space trajectories, latent-vs-spatial distance analysis for RNN dynamics. ~757 lines. | Never imported under the `timescales` package name. Old notebooks referenced it as `pirnns.analysis.dynamics`. |
| `connectivity.py` | Timescale-group connectivity visualization (W_rec heatmaps, bipartite graphs). ~680 lines. | Never imported. Equivalent logic was duplicated inline in `02_analyze_mt_connectivity.ipynb`. |

**Note**: The remaining `analysis/` files (`measurements.py`, `spatial.py`, `load_models.py`, `analyses.py`, `sweep_evaluator.py`, `plotting.py`) are still in `timescales/analysis/` because they are actively imported by `callbacks.py`, `single_run.py`, or working notebooks. They are path-integration-specific and could be archived later if that task is dropped.

---

## `datamodule.py`

Legacy re-export shim. One-liner that re-exported `PathIntegrationDataModule` from `timescales.datamodules`. Superseded by importing directly from `timescales.datamodules`.

---

## `notebooks/`

20 Jupyter notebooks from the path-integration era. Most are broken due to the `pirnns` → `timescales` rename or missing modules.

| Notebook | Topic | Status |
|----------|-------|--------|
| `01_analyze_vanilla_latents` | Vanilla RNN latent analysis | **Broken**: imports `timescales.rnns.rnn.RNN` (file doesn't exist) |
| `02_analyze_mt_connectivity` | Multi-timescale connectivity analysis | **Broken**: imports `analysis.ood_generalization` (doesn't exist) |
| `03_ood_traj_length` | Out-of-distribution trajectory length generalization | Likely functional but path-integration-specific |
| `04_training_metrics` | Training curves and loss visualization | Likely functional but path-integration-specific |
| `05_speed_timescale_sweep` | Speed × timescale parameter sweep | Path-integration-specific |
| `06_behavioral_vs_neural_timescale_stds` | Behavioral vs neural timescale standard deviations | Path-integration-specific |
| `07_behav_vs_neural_timescale_means` | Behavioral vs neural timescale means | Path-integration-specific |
| `08_learnable_timescales` | Learnable timescale analysis | Path-integration-specific |
| `09_learnable_init_sweep` | Learnable timescale initialization sweep | Path-integration-specific |
| `10_timescale_estimation_from_activations` | Estimating timescales from hidden activations | Path-integration-specific |
| `11_latent_dynamics_analysis` | Latent dynamics and PCA | Path-integration-specific |
| `12_normalization_comparison` | LayerNorm vs no-norm comparison | Path-integration-specific |
| `13_binary_counter_analysis` | Binary counter task analysis | May be functional |
| `14_activation_func` | ReLU vs Tanh activation comparison | Path-integration-specific |
| `15_shared_timescale_sweep` | Shared vs per-unit timescale | Path-integration-specific |
| `16_timescale_distribution_sweep` | Timescale distribution types | Path-integration-specific |
| `17_ablation_robustness` | Ablation / robustness tests | Path-integration-specific |
| `fig_conectivity` | Connectivity figure for paper | **Broken**: imports `pirnns.paper_figs.connectivity` |
| `fig_learning_efficiency` | Learning efficiency figure for paper | **Broken**: imports `pirnns.paper_figs.load_models`, uses `os.chdir("pirnns")` |
| `lognormal_random_autocorrelation` | Lognormal timescale autocorrelation theory | Standalone theory notebook |

---

## `root_notebooks/`

Notebooks that lived at the repository root level.

| Notebook | Status |
|----------|--------|
| `01_train_rnn_2D_path_integation` | **Broken**: imports `pirnns.rnns.rnn` |
| `koop` | Koopman operator exploration. Status unknown. |

---

## `experiments/`

Linear path-integration theory validation experiments.

| File | What it did |
|------|-------------|
| `train.py` | Training script for 1D path integration theory validation |
| `_template.ipynb` | Template notebook for experiment analysis |
| `results/` | JSON + .pt result files for N=1 and N=2 networks with/without self-connections |

Archived because these experiments appear to be completed and the results are stored.

---

## `sandbox_preconditioner_viz/`

*Previously `sandbox/preconditioner_viz/`*

2D RNN demo for visualizing gradient preconditioning effects. Standalone script (`run.py`) with config and saved result figures. Archived as a completed one-off exploration.

---

## `sandbox_teacher_student_identifiability/`

*Previously `sandbox/teacher_student_identifiability/`*

Teacher-student timescale identifiability experiments. Tests whether gradient-based training can recover teacher timescales under different conditions (activations, diagonal constraints, preconditioning). Includes `run.py`, configs, and a README. Archived as a completed exploration.

---

## `sandbox_results/`

*Previously `sandbox/results/`*

Output artifacts (plots, logs, JSON results) from the teacher-student identifiability experiments. Multiple timestamped run directories with vector field visualizations, loss curves, and summaries.

---

## `sandbox_linear_network_timescales/`

*Previously `sandbox/linear_network_timescales/`*

Linear network timescale theory explorations: saddle-node visualization, random linear network simulations, and nullcline analysis. Includes Jupyter notebooks and YAML configs. Archived as completed theoretical exploration.

---

## `sandbox_README.md`

*Previously `sandbox/README.md`*

Original README for the sandbox directory.
