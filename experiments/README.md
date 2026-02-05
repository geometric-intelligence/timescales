# Experiments

Each experiment lives in its own dated folder.

## Quick Start

1. Copy `_template.ipynb` to a new folder:
   ```
   experiments/YYYY-MM-DD_your_experiment_name/analysis.ipynb
   ```

2. Edit the notebook, run cells, iterate

3. Save figures/results to the `results/` subfolder

## Structure

```
experiments/
├── _template.ipynb                    # Copy this to start
├── 2025-01-28_alpha_preconditioning/
│   ├── analysis.ipynb                 # Your experiment
│   ├── results/                       # Saved figures, metrics
│   └── notes.md                       # Observations (optional)
└── ...
```

## Tips

- Keep experiments isolated - don't modify core code from here
- If you create useful helper functions, move them to `timescales/analysis/`
- Use descriptive folder names: `YYYY-MM-DD_brief_description`
