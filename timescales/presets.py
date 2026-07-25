"""Config-level presets: trainable-parameter sets and tau-init schemes.

Two high-level knobs that expand deterministically into the low-level config keys
the model factory consumes. Sweeps cross them as clean axes instead of hand-writing
key combinations per condition (post-review delta, Sections 3-4).

``trainable`` — which of {W, W_in, W_out, A} are optimized:
  - ``full``      : {W, W_in, W_out, A}  (learn_time_constants=True)
  - ``fixedA``    : {W, W_in, W_out}     (learn_time_constants=False)
  - ``reservoir`` : {W_out} only — W_rec, W_in, and A frozen at init.

``tau_init`` — the time-constant initialization scheme, independent of whether tau
is trainable (the resolver routes to ``init_time_constants_config`` when A is
trainable vs ``time_constants_config`` when fixed):
  - ``{scheme: uniform, value: 1.0}``
  - ``{scheme: powerlaw, exponent: 1.0, tau_min: ..., tau_max: ...}``
  - ``{scheme: lognormal, median: ..., sigma: ..., tau_min: ..., tau_max: ...}``
  ``tau_min``/``tau_max`` fall back to top-level config keys of the same name, so a
  task variant can set the range while an init variant sets the scheme. Log-normal
  defaults (flagged as a design decision): median = sqrt(tau_min*tau_max) (the
  geometric midpoint) and sigma = ln(tau_max/tau_min)/4, clipped to the same
  [tau_min, tau_max] as power-law for a fair comparison.

Resolution mutates the config in place (the resolved config is what gets saved with
the run). IMPORTANT: run-identity fingerprints are computed on the *as-authored*
config before resolution, so sweep-level resume and single_seed agree.
"""

from __future__ import annotations

import math

TRAINABLE_PRESETS = ("full", "fixedA", "reservoir")
TAU_INIT_SCHEMES = ("uniform", "powerlaw", "lognormal")


def resolve_trainable(config: dict) -> dict:
    """Expand ``config['trainable']`` into learn_time_constants / freeze_* keys."""
    preset = config.get("trainable")
    if preset is None:
        return config
    if preset not in TRAINABLE_PRESETS:
        raise ValueError(
            f"Unknown trainable preset: {preset!r}. Supported: {TRAINABLE_PRESETS}"
        )

    config["learn_time_constants"] = preset == "full"
    if preset == "reservoir":
        config["freeze_wrec"] = True
        config["freeze_win"] = True
    return config


def _lognormal_defaults(tau_min: float, tau_max: float) -> tuple[float, float]:
    """Default (median, sigma) matched to the power-law range: geometric midpoint
    and a spread covering the log-range (~95% of mass within [tau_min, tau_max])."""
    median = math.sqrt(tau_min * tau_max)
    sigma = math.log(tau_max / tau_min) / 4.0
    return median, sigma


def resolve_tau_init(config: dict) -> dict:
    """Expand ``config['tau_init']`` into the low-level time-constant keys."""
    ti = config.get("tau_init")
    if ti is None:
        return config
    scheme = ti.get("scheme")
    if scheme not in TAU_INIT_SCHEMES:
        raise ValueError(
            f"Unknown tau_init scheme: {scheme!r}. Supported: {TAU_INIT_SCHEMES}"
        )

    learn = bool(config.get("learn_time_constants", False))

    if scheme == "uniform":
        value = float(ti.get("value", 1.0))
        if learn:
            config["init_time_constant"] = value
            config.pop("init_time_constants_config", None)
        else:
            config["time_constants_config"] = {"type": "discrete", "values": [value]}
        return config

    # Distribution schemes need a range; task variants may set it at top level.
    tau_min = ti.get("tau_min", config.get("tau_min"))
    tau_max = ti.get("tau_max", config.get("tau_max"))
    if tau_min is None or tau_max is None:
        raise ValueError(
            f"tau_init scheme {scheme!r} needs tau_min/tau_max (in tau_init or "
            f"as top-level config keys)."
        )
    tau_min, tau_max = float(tau_min), float(tau_max)

    if scheme == "powerlaw":
        dist_cfg = {
            "type": "continuous",
            "distribution": "powerlaw",
            "exponent": float(ti.get("exponent", 1.0)),
            "min_time_constant": tau_min,
            "max_time_constant": tau_max,
        }
    else:  # lognormal
        default_median, default_sigma = _lognormal_defaults(tau_min, tau_max)
        dist_cfg = {
            "type": "continuous",
            "distribution": "lognormal",
            "median": float(ti.get("median", default_median)),
            "sigma": float(ti.get("sigma", default_sigma)),
            "min_time_constant": tau_min,
            "max_time_constant": tau_max,
        }

    if learn:
        config["init_time_constants_config"] = dist_cfg
        config.pop("init_time_constant", None)
    else:
        config["time_constants_config"] = dist_cfg
    return config


def resolve_presets(config: dict) -> dict:
    """Resolve all presets in dependency order (trainable before tau_init)."""
    resolve_trainable(config)
    resolve_tau_init(config)
    return config
