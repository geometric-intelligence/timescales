"""Tests for timescales.convergence — steps-to-convergence from a validation curve."""

from timescales import convergence as cv


def test_frac_of_final_basic():
    steps = [0, 10, 20, 30, 40]
    score = [0.0, 0.5, 0.8, 0.9, 1.0]  # final 1.0, 0.9*final=0.9 -> step 30
    assert cv.steps_to_frac_of_final(steps, score, frac=0.9) == 30


def test_frac_of_final_never_reached_returns_none():
    # Monotone but frac target above every value except the last-tail mean.
    steps = [0, 10, 20]
    score = [0.1, 0.2, 0.3]
    # final=0.3, target=0.9*0.3=0.27 -> first >=0.27 is step 20
    assert cv.steps_to_frac_of_final(steps, score, frac=0.9) == 20


def test_frac_of_final_nonpositive_final_is_degenerate():
    steps = [0, 10, 20]
    score = [-1.0, -0.5, -0.2]  # negative R^2 throughout -> undefined target
    assert cv.steps_to_frac_of_final(steps, score, frac=0.9) is None


def test_frac_of_final_uses_tail_mean():
    steps = [0, 10, 20, 30]
    score = [0.0, 0.9, 0.7, 0.9]  # tail=2 mean=0.8, target=0.72 -> step 10
    assert cv.steps_to_frac_of_final(steps, score, frac=0.9, tail=2) == 10


def test_error_threshold_basic():
    steps = [0, 10, 20, 30]
    error = [0.5, 0.3, 0.08, 0.02]  # <=0.1 first at step 20
    assert cv.steps_to_error_threshold(steps, error, threshold=0.1) == 20


def test_error_threshold_never():
    assert cv.steps_to_error_threshold([0, 10], [0.9, 0.8], threshold=0.1) is None


def test_compute_convergence_defers_when_unselected():
    # final=0.8 -> frac target 0.72 reached at step 20; error min 0.2 never <= 0.1.
    curve = {"steps": [0, 10, 20, 30], "val_accuracies": [0.0, 0.5, 0.8, 0.8]}
    rec = cv.compute_convergence(curve, {"task": "sine_wave"})
    assert rec["convergence_metric"] is None
    assert rec["steps_to_convergence"] is None  # deferred
    assert rec["by_method"]["frac_of_final"]["steps"] == 20
    assert rec["by_method"]["mse_threshold"]["steps"] is None  # distinct from frac
    assert rec["by_method"]["phase_amp_tol"]["steps"] is None
    assert rec["final_score"] == 0.8


def test_compute_convergence_selected_headline():
    curve = {"steps": [0, 10, 20, 30], "val_accuracies": [0.0, 0.5, 0.9, 1.0]}
    rec = cv.compute_convergence(
        curve, {"task": "sine_wave", "convergence_metric": "frac_of_final",
                "conv_final_frac": 0.9}
    )
    assert rec["convergence_metric"] == "frac_of_final"
    assert rec["steps_to_convergence"] == 20


def test_compute_convergence_empty_curve():
    rec = cv.compute_convergence({"steps": [], "val_accuracies": []}, {"task": "sine_wave"})
    assert rec["by_method"]["frac_of_final"]["steps"] is None
    assert rec["n_validations"] == 0
    assert rec["final_score"] is None
