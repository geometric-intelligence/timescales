# RNN Preconditioner Visualization

Toy 2D RNN example demonstrating gradient preconditioning for multi-timescale networks.

This script visualizes how heterogeneous timescales create ill-conditioned loss landscapes
and how preconditioning (scaling gradients by 1/α) improves optimization.

## Quick Start

```bash
cd sandbox/preconditioner_viz

# Run with default config
python run.py --config configs/default.yaml

# Run with modified parameters
python run.py --config configs/default.yaml --n_steps 10000 --alpha_fast 0.95

# Run with all defaults (no config file)
python run.py
```

## Output

Results are saved to `results/<timestamp>/`:
- `gradient_fields.png` - Vector fields comparing raw vs preconditioned gradients
- `optimization_trajectories.png` - Multiple trajectories on colored loss landscape + loss curves
- `summary.txt` - Numerical results
- `config.yaml` - Copy of the configuration used

## Configuration

Edit `configs/default.yaml` or pass parameters via command line:

```yaml
# Timescales
alpha_fast: 0.9   # Fast neuron (try 0.95 for more extreme)
alpha_slow: 0.1   # Slow neuron (try 0.05 for more extreme)

# Teacher weights (optimization target)
w12_teacher: 1.0
w21_teacher: 0.5

# Dynamics
rollout_length: 20  # Timesteps per trajectory (try 50-100 for harder problem)

# Optimization
n_steps: 5000
learning_rate: 0.05

# Multiple trajectories
n_trajectories: 8
init_range: [-0.5, 1.5]

# Visualization
grid_resolution: 15
```

### Command-line Overrides

```bash
python run.py --config configs/default.yaml \
    --alpha_fast 0.95 \
    --alpha_slow 0.05 \
    --n_steps 10000 \
    --n_trajectories 12
```

## Key Results

With default settings (8 trajectories, 5000 steps):

| Metric | Standard GD | Preconditioned GD |
|--------|------------|-------------------|
| Final loss (mean) | ~0.0007 | ~0.000008 |
| Improvement | - | **~80-100x** |

The improvement is consistent across different random initializations.

## Connection to Main Codebase

This toy example demonstrates the mechanism behind `precondition_gradients: true` in
`timescales/base_configs/mts.yaml`. The main `MultiTimescaleRNN` implementation uses
the same idea: scaling gradients by `1/(α + eps)` to compensate for ill-conditioning
from heterogeneous neural timescales.

## Key Insight

The loss landscape is **anisotropic** (stretched along the slow neuron's weight axis) because:
- Changes to w12 (into fast neuron, α=0.9) immediately affect dynamics
- Changes to w21 (into slow neuron, α=0.1) take many timesteps to manifest

The preconditioner **A⁻¹ = diag(1/α₁, 1/α₂)** re-scales gradients to make the effective
landscape more isotropic, enabling faster convergence from any starting point.

## Summary

This example cleanly demonstrates:
1. **Why heterogeneous timescales create ill-conditioning**: Loss landscape stretched along slow axis
2. **How the preconditioner fixes it**: A⁻¹ rescales to make landscape isotropic
3. **Robustness**: Multiple trajectories show consistent improvement regardless of initialization
4. **Practical impact**: ~100x faster convergence to lower loss
