"""Pluggable convergence metrics: steps-to-convergence from a validation curve.

Flip-flop reports "steps to 90% accuracy," which does not transfer to a continuous
generation task (sine-wave). This module defines task-agnostic candidates that all
operate on the per-validation curve persisted by ``LossLoggerCallback``
(``training_losses.json``). Because every run always logs its full curve, the
*headline* metric can be chosen (or revisited) at aggregation time without
rerunning — ``compute_convergence`` computes every curve-based candidate up front
and stores them under ``by_method``.

Candidates (see the spec, Workstream A2):
- ``frac_of_final``  : steps to reach ``conv_final_frac`` of the run's own final
  score (scale-free; the natural analogue of "steps to 90%").
- ``mse_threshold``  : steps until the error (normalized MSE / ``1 - R^2`` for
  sine-wave, ``1 - accuracy`` for flip-flop) drops below ``conv_mse_threshold``.
- ``phase_amp_tol``  : per-frequency phase/amplitude tolerance — an extension point
  that needs signal-level evaluation, not just the loss curve. Not yet implemented.

Higher-is-better "score" and lower-is-better "error" series are derived per task so
a single definition is computed identically across init conditions.
"""

from __future__ import annotations

DEFAULT_CONV_FINAL_FRAC = 0.9
DEFAULT_CONV_MSE_THRESHOLD = 0.1

# convergence_metric values that are computed from the loss curve alone.
CURVE_METHODS = ("frac_of_final", "mse_threshold")


def _score_and_error(curve: dict, task: str) -> tuple[list[int], list[float], list[float]]:
    """Return (steps, score, error) for a task.

    ``score`` is higher-is-better (accuracy or R^2); ``error`` is lower-is-better
    (``1 - score``, i.e. misclassification rate or ``1 - R^2``).
    """
    steps = list(curve.get("steps") or [])
    acc = list(curve.get("val_accuracies") or [])
    # val_accuracies holds accuracy for flip-flop tasks and R^2 for sine-wave.
    n = min(len(steps), len(acc))
    steps, score = steps[:n], acc[:n]
    error = [1.0 - s for s in score]
    return steps, score, error


def steps_to_frac_of_final(
    steps: list[int], score: list[float], frac: float, tail: int = 1
) -> int | None:
    """First step whose score reaches ``frac`` of the final score.

    ``final`` is the mean of the last ``tail`` validations. Returns None if the run
    never reaches the target, or if the final score is non-positive (degenerate:
    the fractional target is undefined / meaningless).
    """
    if not score:
        return None
    tail = max(1, min(tail, len(score)))
    final = sum(score[-tail:]) / tail
    if final <= 0:
        return None
    target = frac * final
    for s, v in zip(steps, score, strict=False):
        if v >= target:
            return s
    return None


def steps_to_error_threshold(
    steps: list[int], error: list[float], threshold: float
) -> int | None:
    """First step whose error falls to/below ``threshold`` (None if never)."""
    for s, v in zip(steps, error, strict=False):
        if v <= threshold:
            return s
    return None


def compute_convergence(curve: dict, config: dict) -> dict:
    """Compute every curve-based convergence candidate for one run.

    Returns a record with each candidate under ``by_method`` plus the selected
    headline (``config['convergence_metric']``, or None to defer the choice).
    """
    task = config.get("task", "flip_flop")
    steps, score, error = _score_and_error(curve, task)

    frac = float(config.get("conv_final_frac", DEFAULT_CONV_FINAL_FRAC))
    mse_thr = float(config.get("conv_mse_threshold", DEFAULT_CONV_MSE_THRESHOLD))

    by_method = {
        "frac_of_final": {
            "steps": steps_to_frac_of_final(steps, score, frac),
            "conv_final_frac": frac,
        },
        "mse_threshold": {
            "steps": steps_to_error_threshold(steps, error, mse_thr),
            "conv_mse_threshold": mse_thr,
        },
        # Extension point: needs the model's per-frequency output, not the curve.
        "phase_amp_tol": {"steps": None, "note": "not implemented (needs signal-level eval)"},
    }

    selected = config.get("convergence_metric")  # None -> defer to aggregation
    steps_to_convergence = None
    if selected in by_method:
        steps_to_convergence = by_method[selected]["steps"]

    return {
        "convergence_metric": selected,
        "steps_to_convergence": steps_to_convergence,
        "by_method": by_method,
        "n_validations": len(steps),
        "final_score": score[-1] if score else None,
    }
