# Training Models

This project provides three ways to train models:

### 1. Single Training Run (`single_run.py`)

To train a single model, use:

```bash
# Train a vanilla RNN
python single_run.py --config vanilla.yaml

# Train a multitimescale RNN  
python single_run.py --config mts.yaml
```

**Output:** `logs/single_runs/{model_type}_{timestamp}/`


### 2. Parameter Sweeps (`run_sweep.py`)

For comparing multiple configurations systematically:

```bash
python run_sweep.py --sweep experiments/<sweep_name>.yaml
```

**Use when:**
- Hyperparameter tuning
- Comparing different model architectures
- Systematic ablation studies
- You want to compare multiple configurations, each with multiple seeds

**Output:** `logs/experiments/{sweep_name}_{timestamp}/{config_name}/seed_{0,1,2...}/`

#### Creating Parameter Sweep Experiments

Create an experiment file (e.g., `experiments/<sweep_name>.yaml`):

```yaml
# Base configuration to inherit from
base_config: "configs/mts.yaml"

# Number of seeds per configuration
n_seeds: 3

# Parameter sweep configurations
experiments:
  - name: "discrete_single"
    overrides:
      timescales_config:
        type: "discrete"
        values: [0.1443]
  
  - name: "discrete_dual" 
    overrides:
      timescales_config:
        type: "discrete"
        values: [0.1, 0.5]
      max_epochs: 25  # Can override multiple parameters
```

This will run 2 configurations × 3 seeds = 6 total training runs.