# Teacher-Student Timescale Identifiability

Tests whether a student RNN can recover teacher dynamics by changing W_rec
when the student has different timescales (α_i).

**Interpretation:**
- If student **FAILS** to match teacher → timescales are identifiable
- If student **SUCCEEDS** → timescales are NOT identifiable (absorbed into W_rec)

## Quick Start

```bash
cd sandbox/teacher_student_identifiability

# Run with default sweep config
python run.py

# Run with custom sweep config
python run.py --sweep-config configs/sweep_config.yaml

# Run with custom teacher dynamics
python run.py --teacher-config configs/example_teacher.yaml

# Run on specific GPUs
python run.py --gpus 0,1,2,3

# Run sequentially (no parallelism)
python run.py --sequential
```

## Configuration

### Sweep Config (`configs/sweep_config.yaml`)

Controls which experimental conditions to test:

```yaml
# Training hyperparameters
n_epochs: 5000
batch_size: 128
T: 80           # Trajectory length
lr: 0.01
lag: 20         # Burn-in steps

# Sweep dimensions
activations: ["Tanh", "ReLU"]
zero_diags: [false, true]
preconditions: [false, true]
loss_types: ["trajectory", "vector_field"]
vf_normalize_modes: ["none", "cosine"]
num_neurons_list: [2, 4, 8]
```

### Teacher Config (`configs/example_teacher.yaml`)

Custom teacher network parameters:

```yaml
teacher_W: [[0, 1.0], [0.5, 0]]
teacher_b: [0.1, -0.1]
teacher_alphas: [0.8, 0.2]
student_alphas: [0.3, 0.6]
```

## Output

Results are saved to `results/<timestamp>/`:
- `results.json` - Summary of all experiments
- `experiment.log` - Full training log
- `*.png` - Phase portraits, loss curves, eigenvalue plots
- `summary.png` - Bar chart comparing all conditions

## Experimental Variables

| Variable | Values | Description |
|----------|--------|-------------|
| Activation | Tanh, ReLU, Identity | Nonlinearity |
| Zero diagonal | true/false | Self-connections in W_rec |
| Preconditioning | true/false | Scale gradients by 1/α |
| Loss type | trajectory, vector_field | What to optimize |
| VF normalization | none, cosine, angular_mse | For VF loss only |
| Network size | 2, 4, 8, ... | Number of neurons |
