"""Tests for deterministic run identity and completion markers (timescales.run_ids)."""

import yaml

from timescales import run_ids


BASE = {
    "model_type": "rnn",
    "task": "flip_flop",
    "hidden_size": 128,
    "recurrent_gain": 0.9,
    "time_constants_config": {"type": "discrete", "values": [1.0]},
    "learning_rate": 1e-3,
}


def test_fingerprint_is_deterministic():
    assert run_ids.config_fingerprint(BASE) == run_ids.config_fingerprint(dict(BASE))


def test_fingerprint_ignores_volatile_keys():
    a = dict(BASE)
    b = dict(BASE, seed=7, sweep_dir="/x", experiment_name="e", devices=[0],
             accelerator="gpu", project_name="p", input_size=5, output_size=5)
    assert run_ids.config_fingerprint(a) == run_ids.config_fingerprint(b)


def test_fingerprint_sensitive_to_identity_keys():
    changed = dict(BASE, recurrent_gain=0.5)
    assert run_ids.config_fingerprint(BASE) != run_ids.config_fingerprint(changed)
    changed2 = dict(BASE, time_constants_config={"type": "discrete", "values": [2.0]})
    assert run_ids.config_fingerprint(BASE) != run_ids.config_fingerprint(changed2)


def test_fingerprint_key_order_invariant():
    reordered = dict(reversed(list(BASE.items())))
    assert run_ids.config_fingerprint(BASE) == run_ids.config_fingerprint(reordered)


def test_run_id_includes_seed():
    assert run_ids.run_id_for(BASE, 3).endswith("_seed3")
    assert run_ids.run_id_for(BASE, 3) != run_ids.run_id_for(BASE, 4)


def test_is_complete_requires_marker(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    assert not run_ids.is_complete(str(d))
    fp = run_ids.config_fingerprint(BASE)
    with open(run_ids.marker_path(str(d)), "w") as f:
        yaml.dump({"final_val_loss": 0.1, "fingerprint": fp}, f)
    assert run_ids.is_complete(str(d))
    assert run_ids.is_complete(str(d), fp)


def test_is_complete_fingerprint_mismatch(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    with open(run_ids.marker_path(str(d)), "w") as f:
        yaml.dump({"final_val_loss": 0.1, "fingerprint": "deadbeef00"}, f)
    # Marker exists but for a different config -> not complete for this fingerprint.
    assert run_ids.is_complete(str(d)) is True
    assert run_ids.is_complete(str(d), run_ids.config_fingerprint(BASE)) is False
