# Analysis Module

This module contains the tools for analyzing trained RNN models.

Key terms:

- Single run: A single training run of a model (1 configuration, 1 random seed).
- Experiment: The same model configuration, but with multiple random seeds.
- Sweep: Multiple experiments with different configurations., each with multiple seeds.

A sweep can be loaded using the `load_experiment_sweep` function in `load_models.py`, by passing the sweep directory. This will return a dictionary with the following structure:

## Sweep Structure

Expected sweep structure:

```python
        sweep = {
            "experiment_name_1": {
                0 : { # seed 0
                    "model": <model>, # trained RNN
                    "config": {...}, # dictionary of configuration parameters
                    "place_cell_centers": <tensor>
                },
                1: { # seed 1
                    "model": <model>,
                    "config": {...},
                    "place_cell_centers": <tensor>
                },
                ...
            },
            "experiment_name_2": {...},
            ...
        }
```


## Analysis Objects

- Measurement `(measurement.py)`: A class that computes a single scalar value from a given model and dataset.

- Analysis `(analyses.py)`: A class that applies a measurement to a given model and dataset.

- SweepEvaluator `(sweep_evaluator.py)`: A class that applies an analysis to a given sweep.

## Plotting

- Plotting `(plotting.py)`: A module that contains the plotting functions for the analysis results.



