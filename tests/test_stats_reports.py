"""Tests for timescales.stats and timescales.reports (delta Sections 1, 7)."""

import numpy as np

from timescales import stats
from timescales.reports import best_vs_best, per_g_table


# ---------------------------------------------------------------------- stats

def test_compare_samples_fully_separated():
    out = stats.compare_samples([1.0, 2.0], [3.0, 4.0])
    assert out["cliffs_delta"] == -1.0          # every x < every y
    assert out["median_x"] == 1.5 and out["median_y"] == 3.5
    assert out["p_value"] is not None


def test_compare_samples_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    out = stats.compare_samples(x, x)
    assert abs(out["cliffs_delta"]) < 1e-9
    assert out["p_value"] > 0.9


def test_compare_samples_significant_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50)
    y = rng.normal(2, 1, 50)
    out = stats.compare_samples(x, y)
    assert out["p_value"] < 1e-6
    assert out["cliffs_delta"] < -0.5
    assert out["cohens_d"] < -1.0


def test_compare_samples_handles_empty_and_nan():
    out = stats.compare_samples([], [1.0])
    assert out["p_value"] is None
    out = stats.compare_samples([float("nan"), 1.0], [2.0])
    assert out["n_x"] == 1  # nan dropped


def test_group_metric():
    rows = [
        {"task": "ff", "m": 1.0}, {"task": "ff", "m": 2.0},
        {"task": "sine", "m": 3.0}, {"task": "sine", "m": None},
        {"task": "sine", "m": "bad"},
    ]
    groups = stats.group_metric(rows, "m", by=("task",))
    assert groups[("ff",)] == [1.0, 2.0]
    assert groups[("sine",)] == [3.0]


# -------------------------------------------------------------------- reports

def _rows():
    """Synthetic run table: powerlaw best at g=0.9, uniform best at g=0.5,
    and best-powerlaw clearly beats best-uniform."""
    rows = []
    per = {
        ("uniform", 0.5): [10.0, 11.0, 12.0],
        ("uniform", 0.9): [20.0, 21.0, 22.0],
        ("powerlaw", 0.5): [8.0, 9.0, 10.0],
        ("powerlaw", 0.9): [1.0, 2.0, 3.0],
    }
    for (scheme, g), vals in per.items():
        for seed, v in enumerate(vals):
            rows.append({
                "cfg.task": "ff", "cfg.trainable": "full", "seed": seed,
                "cfg.tau_init.scheme": scheme, "cfg.recurrent_gain": str(g),
                "final_val_loss": str(v),
            })
    return rows


def test_best_vs_best_picks_best_g_per_scheme():
    report = best_vs_best(_rows(), "final_val_loss", lower_is_better=True)
    assert len(report) == 1
    r = report[0]
    assert r["best_g_x"] == 0.9      # powerlaw's best
    assert r["best_g_y"] == 0.5      # uniform's best (not the weak 0.9 baseline)
    assert r["median_x"] == 2.0 and r["median_y"] == 11.0
    assert r["winner"] == "powerlaw"
    assert r["cliffs_delta"] == -1.0


def test_best_vs_best_missing_scheme_skipped():
    rows = [r for r in _rows() if r["cfg.tau_init.scheme"] == "uniform"]
    assert best_vs_best(rows, "final_val_loss") == []


def test_best_vs_best_tie_reported_as_tie():
    rows = _rows()
    # Make powerlaw identical to uniform at every condition -> exact tie.
    for r in rows:
        if r["cfg.tau_init.scheme"] == "powerlaw":
            r["final_val_loss"] = {"0": "10.0", "1": "11.0", "2": "12.0"}[str(r["seed"])]
            r["cfg.recurrent_gain"] = "0.5"
    report = best_vs_best(rows, "final_val_loss")
    assert report[0]["winner"] == "tie"


def test_per_g_table_medians():
    table = per_g_table(_rows(), metrics=(("final_val_loss", True),))
    assert len(table) == 4  # 2 schemes x 2 g
    row = next(r for r in table if r["scheme"] == "powerlaw" and r["g"] == 0.9)
    assert row["final_val_loss_median"] == 2.0
    assert row["final_val_loss_n"] == 3
    assert row["cfg.task"] == "ff"
