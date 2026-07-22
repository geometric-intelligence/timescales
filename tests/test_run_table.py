"""Tests for timescales.run_table — per-run row collection + serialization."""

import csv
import json
import os

import yaml

from timescales import run_table


def _make_run(seed_dir, seed, final_loss, steps, git="abc123"):
    os.makedirs(seed_dir, exist_ok=True)
    marker = {
        "run_id": f"fp_seed{seed}", "seed": seed, "fingerprint": "fp",
        "final_val_loss": final_loss, "steps_to_convergence": steps,
        "completed_at": "20260722_000000",
        "convergence": {"by_method": {
            "frac_of_final": {"steps": steps},
            "mse_threshold": {"steps": None},
        }},
        "provenance": {"git_commit": git, "git_dirty": False,
                       "packages": {"torch": "2.7.0", "numpy": "2.3.2"}},
    }
    with open(os.path.join(seed_dir, "job_result.yaml"), "w") as f:
        yaml.dump(marker, f)
    with open(os.path.join(seed_dir, f"config_seed{seed}.yaml"), "w") as f:
        yaml.dump({"task": "sine_wave", "recurrent_gain": 0.9,
                   "time_constants_config": {"exponent": 1.0}}, f)
    with open(os.path.join(seed_dir, "spectral_stats_init.json"), "w") as f:
        json.dump({"max_abs_lambda": 0.95, "gap_to_unit_circle": 0.05,
                   "frac_near_real_axis": 0.7}, f)
    # a curve handle file
    with open(os.path.join(seed_dir, "training_losses.json"), "w") as f:
        json.dump({"steps": [0, 1]}, f)


def _sweep(tmp_path):
    sweep = tmp_path / "sweep"
    _make_run(str(sweep / "expA" / "seed_0"), 0, 0.10, 100)
    _make_run(str(sweep / "expA" / "seed_1"), 1, 0.12, 120)
    _make_run(str(sweep / "expB" / "seed_0"), 0, 0.20, None)
    return str(sweep)


def test_collect_rows_count_and_fields(tmp_path):
    rows = run_table.collect_run_rows(_sweep(tmp_path))
    assert len(rows) == 3
    r = next(r for r in rows if r["experiment_name"] == "expA" and r["seed"] == 0)
    assert r["final_val_loss"] == 0.10
    assert r["steps_to_convergence"] == 100
    assert r["conv_frac_of_final_steps"] == 100
    assert r["conv_mse_threshold_steps"] is None
    assert r["git_commit"] == "abc123"
    assert r["pkg_torch"] == "2.7.0"
    assert r["cfg.task"] == "sine_wave"
    assert r["cfg.time_constants_config.exponent"] == 1.0
    assert r["spec_init_max_abs_lambda"] == 0.95
    assert r["curve_path"].endswith("training_losses.json")
    assert r["spectrum_init_path"] is None  # no .pt written in this fixture


def test_flatten_lists_are_stringified(tmp_path):
    sweep = tmp_path / "s"
    seed_dir = str(sweep / "e" / "seed_0")
    _make_run(seed_dir, 0, 0.1, 10)
    with open(os.path.join(seed_dir, "config_seed0.yaml"), "w") as f:
        yaml.dump({"periods": [10.0, 20.0]}, f)
    row = run_table.collect_run_rows(str(sweep))[0]
    assert row["cfg.periods"] == "[10.0, 20.0]"


def test_write_run_table_csv_fallback(tmp_path):
    sweep = _sweep(tmp_path)
    out = run_table.write_run_table(sweep, os.path.join(sweep, "run_table.csv"))
    assert out.endswith(".csv")
    with open(out) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 3
    assert "final_val_loss" in reader.fieldnames
    assert "spec_init_gap_to_unit_circle" in reader.fieldnames
