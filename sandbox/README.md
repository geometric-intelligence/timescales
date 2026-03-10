# Sandbox Experiments

Standalone experiments for exploring ideas related to multi-timescale RNNs.

## Structure

Each experiment lives in its own directory with a consistent structure:

```
sandbox/
├── <experiment_name>/
│   ├── run.py              # Main script
│   ├── README.md           # Documentation
│   ├── configs/            # YAML configuration files
│   │   └── default.yaml
│   └── results/            # Output (timestamped subdirectories)
│       └── YYYYMMDD_HHMMSS/
```

## Experiments

### `preconditioner_viz/`

Toy 2D RNN demonstrating gradient preconditioning for multi-timescale networks.
Shows how heterogeneous timescales create ill-conditioned loss landscapes and
how preconditioning (scaling gradients by 1/α) improves optimization.

```bash
cd preconditioner_viz
python run.py --config configs/default.yaml
```

### `teacher_student_identifiability/`

Tests whether a student RNN can recover teacher dynamics by changing W_rec
when the student has different timescales. Explores conditions under which
timescales are or are not identifiable.

```bash
cd teacher_student_identifiability
python run.py --sweep-config configs/sweep_config.yaml
```

## Adding New Experiments

1. Create a new directory: `mkdir -p <name>/configs <name>/results`
2. Create `run.py` with argparse and YAML config support
3. Create `configs/default.yaml` with documented parameters
4. Create `README.md` with usage instructions
