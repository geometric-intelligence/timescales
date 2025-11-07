"""
SweepEvaluator class for applying analyses to entire experiment sweeps.

A sweep contains multiple experiments (configs), each with multiple seeds (runs).
The evaluator applies an analysis+measurement to all models and aggregates results.
"""

import numpy as np
from typing import Any
from dataclasses import dataclass, field

from .analyses import Analysis, AnalysisResult
from .measurements import Measurement


@dataclass
class ExperimentResult:
    """Results for a single experiment (all seeds)."""

    experiment_name: str
    test_conditions: list  # e.g., [25, 50, 100] for trajectory lengths
    condition_name: str  # e.g., "trajectory_length"

    # Per-seed results
    seed_results: dict[int | str, list[float]]  # {seed: [measurements]}

    # Aggregated statistics
    mean_measurements: list[float]  # Mean across seeds for each condition
    std_measurements: list[float]  # Std across seeds for each condition

    # Metadata
    config: dict  # Sample config from first seed
    metadata: dict = field(default_factory=dict)


@dataclass
class SweepResult:
    """Results for entire sweep (all experiments)."""

    experiment_results: dict[str, ExperimentResult]  # {exp_name: ExperimentResult}
    analysis_type: str  # Name of analysis class used
    measurement_type: str  # Name of measurement class used


class SweepEvaluator:
    """
    Evaluates an entire experimental sweep by applying an analysis to all models.

    The evaluator takes Analysis and Measurement *classes* (not instances), along with
    sweep-level parameters. For each seed, it instantiates fresh Analysis and Measurement
    objects using that seed's specific config and place_cell_centers.

    Expected sweep structure:
        sweep = {
            "experiment_name_1": {
                0: {
                    "model": <model>,
                    "config": {...},
                    "place_cell_centers": <tensor>
                },
                1: {
                    "model": <model>,
                    "config": {...},
                    "place_cell_centers": <tensor>
                },
                ...
            },
            "experiment_name_2": {...},
            ...
        }

    Or with string keys: "seed_0", "seed_1", etc.
    """

    def __init__(
        self,
        analysis_class: type[Analysis],
        analysis_params: dict[str, Any],
        measurement_class: type[Measurement],
        measurement_params: dict[str, Any] | None = None,
        verbose: bool = True,
    ):
        """
        Initialize the sweep evaluator.

        Args:
            analysis_class: Analysis class (e.g., OODAnalysis)
            analysis_params: Parameters for analysis (e.g., {"test_lengths": [20, 50, 100]})
                           Note: config and place_cell_centers will be added per-seed
            measurement_class: Measurement class (e.g., PositionDecodingMeasurement)
            measurement_params: Parameters for measurement (e.g., {"decode_k": 256})
            verbose: Whether to print progress
        """
        self.analysis_class = analysis_class
        self.analysis_params = analysis_params
        self.measurement_class = measurement_class
        self.measurement_params = measurement_params or {}
        self.verbose = verbose

    def evaluate(
        self,
        sweep: dict[str, dict[int | str, dict[str, Any]]],
    ) -> SweepResult:
        """
        Evaluate the entire sweep.

        Args:
            sweep: Dictionary of experiments, each containing seeds with models

        Returns:
            SweepResult containing aggregated results for all experiments
        """
        if self.verbose:
            print(f"Evaluating {len(sweep)} experiments")
            print(f"Analysis: {self.analysis_class.__name__}")
            print(f"Measurement: {self.measurement_class.__name__}")

        experiment_results = {}

        for exp_name, seeds in sweep.items():
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Experiment: {exp_name}")
                print(f"  Seeds: {len(seeds)}")

            exp_result = self._evaluate_experiment(exp_name, seeds)
            experiment_results[exp_name] = exp_result

        return SweepResult(
            experiment_results=experiment_results,
            analysis_type=self.analysis_class.__name__,
            measurement_type=self.measurement_class.__name__,
        )

    def _evaluate_experiment(
        self,
        exp_name: str,
        seeds: dict[int | str, dict[str, Any]],
    ) -> ExperimentResult:
        """
        Evaluate a single experiment across all its seeds.

        Args:
            exp_name: Name of the experiment
            seeds: Dictionary of seeds with their models, configs, and place_cell_centers

        Returns:
            ExperimentResult with aggregated statistics
        """
        seed_results = {}
        test_conditions = None
        condition_name = None
        sample_config = None
        metadata = {}

        for seed_key, seed_data in seeds.items():
            if self.verbose:
                print(f"  Processing seed {seed_key}...")

            try:
                model = seed_data["model"]
                config = seed_data["config"]
                place_cell_centers = seed_data["place_cell_centers"]

                # Store sample config from first seed
                if sample_config is None:
                    sample_config = config

                # Instantiate fresh Analysis and Measurement for this seed
                analysis = self.analysis_class(
                    config=config,
                    place_cell_centers=place_cell_centers,
                    **self.analysis_params,
                )
                measurement = self.measurement_class(**self.measurement_params)

                # Run the analysis on this model
                result: AnalysisResult = analysis.run(model, measurement)

                # Store results
                seed_results[seed_key] = result.measurements

                # Store metadata from first seed
                if test_conditions is None:
                    test_conditions = result.test_conditions
                    condition_name = result.condition_name
                    if result.metadata:
                        metadata.update(result.metadata)

                if self.verbose:
                    print(f"    Results: {result.measurements}")

            except Exception as e:
                print(f"  ERROR processing seed {seed_key}: {str(e)}")
                import traceback

                traceback.print_exc()
                continue

        # Compute statistics across seeds
        if seed_results:
            # Convert to array: [n_seeds, n_conditions]
            measurements_array = np.array(list(seed_results.values()))
            mean_measurements = np.mean(measurements_array, axis=0).tolist()
            std_measurements = np.std(measurements_array, axis=0).tolist()

            if self.verbose:
                print(f"  Mean: {mean_measurements}")
                print(f"  Std:  {std_measurements}")
        else:
            # No successful seeds
            n_conditions = len(test_conditions) if test_conditions else 0
            mean_measurements = [np.nan] * n_conditions
            std_measurements = [np.nan] * n_conditions

        return ExperimentResult(
            experiment_name=exp_name,
            test_conditions=test_conditions or [],
            condition_name=condition_name or "unknown",
            seed_results=seed_results,
            mean_measurements=mean_measurements,
            std_measurements=std_measurements,
            config=sample_config or {},
            metadata=metadata,
        )

    def print_summary(self, sweep_result: SweepResult) -> None:
        """Print a summary of the sweep results."""
        print("\n" + "=" * 70)
        print("SWEEP EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Analysis: {sweep_result.analysis_type}")
        print(f"Measurement: {sweep_result.measurement_type}")
        print(f"Number of experiments: {len(sweep_result.experiment_results)}")

        for exp_name, exp_result in sweep_result.experiment_results.items():
            print(f"\n{exp_name}:")
            print(f"  Condition: {exp_result.condition_name}")
            print(f"  Test values: {exp_result.test_conditions}")
            print(f"  Seeds: {len(exp_result.seed_results)}")
            print(
                f"  Mean measurements: {[f'{x:.4f}' for x in exp_result.mean_measurements]}"
            )
            print(
                f"  Std measurements:  {[f'{x:.4f}' for x in exp_result.std_measurements]}"
            )
