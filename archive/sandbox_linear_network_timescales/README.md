# Linear Network Timescales Experiment

Replicates theoretical results on power-law timescale distributions in linear recurrent networks.

## Overview

This experiment validates the relationship between eigenvalue distributions and emergent timescales in linear recurrent neural networks. It consists of two main experiments:

- **Experiment A (Theoretical Scaling):** Verifies that eigenvalue → timescale mapping produces expected power-law slopes for different eigenvalue distributions.
- **Experiment B (Physics Validation):** Validates that simulated autocorrelation timescales match theoretical predictions from eigenvalue analysis.

## Quick Start

```bash
# Quick test (N=100, 5s simulation)
python run.py --config configs/quick_test.yaml

# Full experiment (N=1000, 50s simulation)
python run.py --config configs/default.yaml
```

## Requirements

- numpy
- scipy
- matplotlib
- pyyaml

## Experiment Details

### Experiment A: Theoretical Scaling

Tests four eigenvalue distributions and their expected power-law exponents:

| Distribution | Implementation | Expected γ |
|-------------|----------------|------------|
| Uniform | `np.random.uniform(0.01, 1.0, N)` | -2.0 |
| Gamma(0.5) | `np.random.gamma(0.5, scale, N)` | -1.5 |
| Gamma(2.0) | `np.random.gamma(2.0, scale, N)` | -3.0 |
| Semi-circle | N×N Gaussian W, λ_eff = 1 - g·Re(λ_W) | -2.5 |

The timescale conversion is: τ = τ_syn / Re(λ)

**Output:** `eigenvalue_distributions.png` - 2×2 grid showing log-log histograms with fitted slopes.

### Experiment B: Physics Validation

Simulates a linear recurrent network and compares theoretical vs empirical timescales.

**Model:** τ_syn · dx/dt = -x + g·W·x + η

**Default Parameters:**
- N = 1000 neurons
- W ~ N(0, 1/N)
- g = 0.99 (near critical)
- τ_syn = 10 ms
- dt = 1 ms, duration = 50s
- noise std = 0.1

**Outputs:**
- `simulation_vs_theory.png` - Overlaid histograms comparing theory and simulation
- `eigenvector_projections.png` - Sanity check showing single-exponential autocorrelations for eigenvector projections

## Configuration

Edit YAML files in `configs/` to customize parameters:

```yaml
run_experiment_a: true
run_experiment_b: true
seed: 42

experiment_a:
  n_eigenvalues: 1000
  tau_syn_values: [1.0, 10.0, 100.0]
  n_bins: 50

experiment_b:
  n_neurons: 1000
  g_coupling: 0.99
  tau_syn: 10.0
  dt: 1.0
  duration: 50000.0  # ms
  noise_std: 0.1
  max_lag: 500
```

## Command Line Options

```
--config CONFIG       Path to configuration file (default: configs/default.yaml)
--output-dir DIR      Output directory (default: results/<timestamp>)
--seed SEED           Random seed (overrides config)
```

## Verification

1. Run quick test: `python run.py --config configs/quick_test.yaml`
2. Check slopes match expected values (within ~0.2)
3. Run full experiment: `python run.py --config configs/default.yaml`
4. Verify simulation vs theory overlap in Experiment B
5. Verify eigenvector projections show clean single exponentials

## Theory

For a linear network with dynamics τ_syn · dx/dt = -x + g·W·x + η, the effective eigenvalues are:

λ_eff = 1 - g·λ_W

where λ_W are eigenvalues of W. The timescales of the system modes are:

τ = τ_syn / λ_eff = τ_syn / (1 - g·Re(λ_W))

The distribution of timescales inherits a power-law structure from the eigenvalue distribution, with the exponent γ related to the eigenvalue distribution's tail behavior.
