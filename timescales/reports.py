"""Fair-comparison report: best-vs-best over the g grid (delta Section 1).

R3's objection: power-law was compared only against the weakest uniform baseline
(g=0.9). The rebuttal comparison is power-law AT ITS BEST vs homogeneous AT ITS
BEST: within each (task, trainable) group, pick each init scheme's best g by
median metric, then compare the two best conditions' seed samples head-to-head
(Mann-Whitney U + Cliff's delta). Reported for BOTH steps-to-convergence and
final validation loss; a per-g medians table backs the headline (final-loss-vs-g
is a first-class output, per R3).

Sign convention: comparisons are powerlaw (x) vs uniform (y); with
lower-is-better metrics, negative Cliff's delta favors power-law.

Runs on the run table (CSV from run_table.py, or collect_run_rows dicts).
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from timescales.stats import compare_samples

SCHEME_KEY = "cfg.tau_init.scheme"
G_KEY = "cfg.recurrent_gain"
GROUP_KEYS = ("cfg.task", "cfg.trainable")

# Metrics reported by default: (name, lower_is_better)
DEFAULT_METRICS = (
    ("final_val_loss", True),
    ("conv_frac_of_final_steps", True),
    ("conv_mse_threshold_steps", True),
)


def load_run_table_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fnum(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _collect(rows: list[dict], metric: str) -> dict[tuple, dict[float, list[float]]]:
    """{(group..., scheme): {g: [seed values]}} for one metric."""
    out: dict[tuple, dict[float, list[float]]] = {}
    for r in rows:
        v = _fnum(r.get(metric))
        g = _fnum(r.get(G_KEY))
        scheme = r.get(SCHEME_KEY)
        if v is None or g is None or not scheme:
            continue
        key = tuple(r.get(k) for k in GROUP_KEYS) + (scheme,)
        out.setdefault(key, {}).setdefault(g, []).append(v)
    return out


def _best_condition(per_g: dict[float, list[float]], lower_is_better: bool):
    """(best_g, values) by median; None when empty."""
    if not per_g:
        return None, []
    pick = min if lower_is_better else max
    best_g = pick(per_g, key=lambda g: float(np.median(per_g[g])))
    return best_g, per_g[best_g]


def best_vs_best(rows: list[dict], metric: str, lower_is_better: bool = True,
                 schemes: tuple[str, str] = ("powerlaw", "uniform")) -> list[dict]:
    """One head-to-head row per (task, trainable): schemes[0] vs schemes[1]."""
    data = _collect(rows, metric)
    groups = sorted({k[:-1] for k in data})
    report = []
    for grp in groups:
        g_x, vals_x = _best_condition(data.get(grp + (schemes[0],), {}), lower_is_better)
        g_y, vals_y = _best_condition(data.get(grp + (schemes[1],), {}), lower_is_better)
        if g_x is None or g_y is None:
            continue
        cmp = compare_samples(vals_x, vals_y)
        row = {
            **{k: v for k, v in zip(GROUP_KEYS, grp, strict=True)},
            "metric": metric,
            "scheme_x": schemes[0], "scheme_y": schemes[1],
            "best_g_x": g_x, "best_g_y": g_y,
            **cmp,
        }
        if cmp["cliffs_delta"] is not None:
            if cmp["cliffs_delta"] == 0:
                row["winner"] = "tie"
            else:
                x_better = (cmp["cliffs_delta"] < 0) == lower_is_better
                row["winner"] = schemes[0] if x_better else schemes[1]
        report.append(row)
    return report


def per_g_table(rows: list[dict],
                metrics: tuple[tuple[str, bool], ...] = DEFAULT_METRICS) -> list[dict]:
    """Median (+ n) of each metric per (task, trainable, scheme, g) — backs the
    headline with the full metric-vs-g curves (final-loss-vs-g per R3)."""
    table: dict[tuple, dict] = {}
    for metric, _ in metrics:
        for key, per_g in _collect(rows, metric).items():
            for g, vals in per_g.items():
                tkey = key + (g,)
                entry = table.setdefault(tkey, {
                    **{k: v for k, v in zip(GROUP_KEYS, key[:-1], strict=True)},
                    "scheme": key[-1], "g": g,
                })
                entry[f"{metric}_median"] = float(np.median(vals))
                entry[f"{metric}_n"] = len(vals)
    return [table[k] for k in sorted(table, key=str)]


def fair_comparison_report(run_table_path: str, out_dir: str | None = None,
                           metrics: tuple[tuple[str, bool], ...] = DEFAULT_METRICS
                           ) -> dict[str, str]:
    """Emit best_vs_best.csv + per_g.csv next to the run table; print the headline."""
    rows = load_run_table_csv(run_table_path)
    out_dir = out_dir or os.path.dirname(os.path.abspath(run_table_path))

    bvb_rows: list[dict] = []
    for metric, lower in metrics:
        bvb_rows.extend(best_vs_best(rows, metric, lower_is_better=lower))
    pg_rows = per_g_table(rows, metrics)

    paths = {}
    for name, data in (("best_vs_best", bvb_rows), ("per_g", pg_rows)):
        path = os.path.join(out_dir, f"{name}.csv")
        fieldnames = sorted({k for r in data for k in r}) if data else []
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        paths[name] = path

    print(f"Best-vs-best ({len(bvb_rows)} comparisons) -> {paths['best_vs_best']}")
    for r in bvb_rows:
        print(f"  [{r['cfg.task']}/{r['cfg.trainable']}] {r['metric']}: "
              f"{r['scheme_x']}(g={r['best_g_x']}) vs {r['scheme_y']}(g={r['best_g_y']}) "
              f"medians {r['median_x']:.4g} vs {r['median_y']:.4g}, "
              f"delta={r['cliffs_delta']:+.2f} p={r['p_value']:.3g} "
              f"-> {r.get('winner', '?')}")
    print(f"Per-g medians -> {paths['per_g']}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Fair-comparison (best-vs-best) report")
    parser.add_argument("--run-table", required=True, help="Path to run_table.csv")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    fair_comparison_report(args.run_table, args.out_dir)


if __name__ == "__main__":
    main()
