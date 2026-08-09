"""Build a self-contained interactive HTML report of the sine-wave trajectory analysis.

Collects the results from notebooks 5 and 7 into one page with interactive 3D views and
switchable colourings (elapsed time, or the phase of any of the three target frequencies).
Plotly's JS is embedded, so the file works offline and can be moved anywhere.

    python 8_interactive_report.py            # writes figs/sine_interactive_report.html

Run from the inner ``timescales`` package directory. Set OUT_HTML to change the target.
"""

import os
import sys

import numpy as np
import plotly.graph_objects as go
import torch
import yaml

for _cand in (os.getcwd(), os.path.abspath(os.path.join(os.getcwd(), ".."))):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _report_common import (  # noqa: E402
    C_GRID,
    C_REF,
    COLOR,
    CYCLIC,
    SEQ,
    base_layout,
    layout3d,
    marker,
    page,
    vis_switcher,
)
from train import _create_rnn_model  # noqa: E402

EXP = os.environ.get("SWEEP_ROOT", os.path.join(os.getcwd(), "logs", "experiments"))

# Which two runs to compare, as paths relative to SWEEP_ROOT. Override with
#   SINE_RUNS="linear=<relpath>,tanh=<relpath>"
DEFAULT_RUNS = {
    "linear": "tau_init_grid/sine_uniform_full_g0.9/seed_0",
    "tanh": "tau_init_grid_tanh_voltage/sine_uniform_full_g0.9/seed_0",
}
_spec = os.environ.get("SINE_RUNS")
RUNS = {k: os.path.join(EXP, v) for k, v in (
    dict(item.split("=", 1) for item in _spec.split(",")) if _spec else DEFAULT_RUNS
).items()}
OUT_HTML = os.environ.get("OUT_HTML", os.path.join(
    os.getcwd(), "notebooks", "tau_init_grid", "figs", "sine_interactive_report.html"))

T_ROLL = int(os.environ.get("T_ROLL", 900))
T_PERTURB, RANK = 300, 6
# PERIODS_TASK / PERIOD are filled in from the configs once the runs are loaded, so the
# report follows whatever periods the sweep actually trained on.
PERIODS_TASK: list[float] = []
PERIOD = 0
# activation is re-read from each config once loaded; this is only the initial guess
NONLIN = {k: ("linear" not in k.lower()) for k in RUNS}


# --------------------------------------------------------------------------- model
def load(name):
    run = RUNS[name]
    cfg = yaml.safe_load(open(os.path.join(run, "config_seed0.yaml")))
    model, _ = _create_rnn_model(cfg)
    model.load_state_dict(torch.load(os.path.join(run, "final_model_seed0.pth"),
                                     map_location="cpu", weights_only=False))
    model.eval()
    return model, cfg


def rollout(model, cfg, T=T_ROLL, h0=None, perturb=None):
    h = (h0 if h0 is not None
         else torch.full((1, cfg["hidden_size"]), float(cfg["init_hidden_value"])))
    z = torch.zeros(1, model.rnn_step.W_in.in_features)
    out = []
    with torch.no_grad():
        for t in range(T):
            if perturb is not None and t == perturb[0]:
                h = h + perturb[1]
            h = model.rnn_step(z, h)
            out.append(h[0].numpy().copy())
    return np.asarray(out)


def jacobian(model, h, nonlinear):
    st = model.rnn_step
    a = st.current_alphas.detach().numpy()
    W = st.W_rec.weight.detach().numpy() * float(st.recurrent_weight_scale)
    d = (1 - np.tanh(h) ** 2) if nonlinear else np.ones_like(h)
    return np.diag(1 - a) + (a[:, None] * W) * d[None, :]


def dmd_basis(H, r=RANK):
    mu = H.mean(0, keepdims=True)
    X = H - mu
    A_, B_ = X[:-1].T, X[1:].T
    U, S, Vt = np.linalg.svd(A_, full_matrices=False)
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r].T
    ev, W = np.linalg.eig(Ur.T @ B_ @ Vr @ np.diag(1 / Sr))
    return ev, B_ @ Vr @ np.diag(1 / Sr) @ W, mu


print("rolling out and computing bases ...")
D = {}
for n in RUNS:
    model, cfg = load(n)
    ref = rollout(model, cfg)
    ev, Phi, mu = dmd_basis(ref[T_PERTURB:])
    osc = sorted([j for j in range(RANK) if np.angle(ev[j]) > 1e-9],
                 key=lambda j: -abs(np.angle(ev[j])))
    pinv = np.linalg.pinv(Phi)
    H = ref[T_PERTURB:]
    mu_p = H.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(H - mu_p, full_matrices=False)
    D[n] = dict(model=model, cfg=cfg, ref=ref, ev=ev, Phi=Phi, mu=mu, pinv=pinv,
                osc=osc, periods=[2 * np.pi / abs(np.angle(ev[j])) for j in osc],
                H=H, pca=((H - mu_p) @ Vt[:3].T),
                evr=(S ** 2 / (S ** 2).sum())[:10])
    # torus coordinates: the phase of each output-coupled mode, one angle per frequency
    b_ref = (pinv @ (H - mu).T).T
    D[n]["torus"] = np.column_stack([np.angle(b_ref[:, j]) % (2 * np.pi) for j in osc])
    NONLIN[n] = cfg["activation"].lower() != "identity"
    print(f"  {n}: DMD periods {[round(p, 2) for p in D[n]['periods']]}")

CFG = D[next(iter(RUNS))]["cfg"]
PERIODS_TASK = [float(x) for x in CFG["periods"]]
PERIOD = int(round(max(PERIODS_TASK)))       # monodromy is taken over one slowest cycle
for n in RUNS:
    if [float(x) for x in D[n]["cfg"]["periods"]] != PERIODS_TASK:
        raise SystemExit(f"run {n!r} trained on different periods "
                         f"({D[n]['cfg']['periods']} vs {PERIODS_TASK}); the panels "
                         "compare frequency-for-frequency and would be meaningless")

for n in RUNS:
    if len(D[n]["osc"]) != len(PERIODS_TASK):
        raise SystemExit(
            f"run {n!r}: rank-{RANK} DMD found {len(D[n]['osc'])} oscillatory modes but the "
            f"task has {len(PERIODS_TASK)} frequencies. The per-frequency panels assume one "
            "mode per frequency; raise RANK or inspect this run before trusting the report.")

# colourings: elapsed time, and the analytic phase of each target frequency
T_SEG = D[next(iter(RUNS))]["H"].shape[0]
t_axis = np.arange(T_SEG)
COLORINGS = [("time", t_axis, SEQ, "step")]
for P in PERIODS_TASK:
    COLORINGS.append((f"phase T={P:g}", (2 * np.pi * t_axis / P) % (2 * np.pi),
                      CYCLIC, "phase (rad)"))

print("computing UMAP ...")
import umap  # noqa: E402

for n in RUNS:
    D[n]["umap"] = umap.UMAP(n_components=3, n_neighbors=40, min_dist=0.30,
                             random_state=0).fit_transform(D[n]["H"])


def regime_color(name):
    """Palette entry for a run, chosen by its activation rather than its label."""
    return COLOR["tanh" if NONLIN[name] else "linear"]


# --------------------------------------------------------------------------- figures
def embed3d(name, key, title, axes):
    """A 3D embedding with one trace per colouring; buttons switch which is shown."""
    fig = go.Figure()
    P = D[name][key]
    for ci, (label, vals, scale, cbar) in enumerate(COLORINGS):
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
            marker=marker(vals, scale, cbar, 2.4, True),
            visible=(ci == 0), name=label, customdata=t_axis,
            hovertemplate="step %{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter3d(
        x=[P[0, 0]], y=[P[0, 1]], z=[P[0, 2]], mode="markers",
        marker=dict(size=7, color="crimson"), name="start", hoverinfo="name"))
    fig.update_layout(**layout3d(title, axes), showlegend=False,
                      updatemenus=vis_switcher(COLORINGS, 1, n_extra=1))
    return fig


def eigenplane_fig(name):
    """Trajectory in each output-coupled mode's own plane, one trace per colouring."""
    d = D[name]
    fig = go.Figure()
    b = (d["pinv"] @ (d["H"] - d["mu"]).T).T
    for i, j in enumerate(d["osc"]):
        c = b[:, j]
        for ci, (label, vals, scale, cbar) in enumerate(COLORINGS):
            fig.add_trace(go.Scatter(
                x=c.real, y=c.imag, mode="markers",
                xaxis=f"x{i + 1}", yaxis=f"y{i + 1}",
                marker=marker(vals, scale, cbar, 3.4, i == 2),
                visible=(ci == 0), name=label, customdata=t_axis,
                hovertemplate="step %{customdata}<extra></extra>"))
    dom = [(0.0, 0.29), (0.355, 0.645), (0.71, 1.0)]
    ax = {}
    for i, (per, (a, bb)) in enumerate(zip(d["periods"], dom, strict=False)):
        ax[f"xaxis{i + 1}"] = dict(domain=[a, bb], anchor=f"y{i + 1}",
                                   title=f"Re b · mode T≈{per:.1f}",
                                   gridcolor=C_GRID, zeroline=False,
                                   scaleanchor=f"y{i + 1}")
        ax[f"yaxis{i + 1}"] = dict(anchor=f"x{i + 1}", title="Im b" if i == 0 else "",
                                   gridcolor=C_GRID, zeroline=False)
    fig.update_layout(
        **base_layout(f"Per-frequency eigenplanes · {name}", 380,
                      dict(l=60, r=10, t=76, b=50)),
        showlegend=False,
        updatemenus=vis_switcher(COLORINGS, 3), **ax)
    return fig


NEUTRAL = 0.9          # |mu| above this counts as near-neutral
FLOQUET = {}
for n in RUNS:
    _M = np.eye(D[n]["cfg"]["hidden_size"])
    for _t in range(400, 400 + PERIOD):
        _M = jacobian(D[n]["model"], D[n]["ref"][_t], NONLIN[n]) @ _M
    _mult = np.linalg.eigvals(_M)
    FLOQUET[n] = dict(mult=_mult, n_neutral=int((np.abs(_mult) > NEUTRAL).sum()),
                      largest=float(np.abs(_mult).max()),
                      next_largest=float(np.sort(np.abs(_mult))[-2]))
    print(f"  {n}: {FLOQUET[n]['n_neutral']} Floquet multipliers > {NEUTRAL}, "
          f"max |mu| {FLOQUET[n]['largest']:.4f}")


def floquet_fig():
    fig = go.Figure()
    th = np.linspace(0, 2 * np.pi, 400)
    for i, n in enumerate(RUNS):
        mult = FLOQUET[n]["mult"]
        xa, ya = f"x{i + 1}", f"y{i + 1}"
        fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines",
                                 line=dict(color=C_REF, dash="dash", width=1),
                                 xaxis=xa, yaxis=ya, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=mult.real, y=mult.imag, mode="markers", xaxis=xa, yaxis=ya,
            marker=dict(size=6, color=regime_color(n), opacity=0.85),
            hovertemplate="|μ| = %{customdata:.4f}<extra></extra>",
            customdata=np.abs(mult)))
        n_neut = FLOQUET[n]["n_neutral"]
        fig.add_annotation(text=f"<b>{n}</b> — {n_neut} multiplier(s) with |μ| > {NEUTRAL}",
                           xref="paper", yref="paper", showarrow=False,
                           x=0.16 + 0.55 * i, y=1.06, font=dict(size=12))
    fig.update_layout(
        **base_layout("Floquet multipliers of the monodromy matrix (one period)", 430,
                      dict(l=50, r=10, t=80, b=45)),
        showlegend=False,
        xaxis=dict(domain=[0, 0.44], title="Re μ", gridcolor=C_GRID, zeroline=False,
                   scaleanchor="y", range=[-1.35, 1.35]),
        yaxis=dict(title="Im μ", gridcolor=C_GRID, zeroline=False, range=[-1.35, 1.35]),
        xaxis2=dict(domain=[0.56, 1.0], anchor="y2", title="Re μ", gridcolor=C_GRID,
                    zeroline=False, scaleanchor="y2", range=[-1.35, 1.35]),
        yaxis2=dict(anchor="x2", title="", gridcolor=C_GRID, zeroline=False,
                    range=[-1.35, 1.35]))
    return fig


def eigenplane_perturbation(name, j, frac=0.20):
    v = D[name]["Phi"][:, j]
    e1 = np.real(v) / np.linalg.norm(np.real(v))
    e2 = np.imag(v) - (np.imag(v) @ e1) * e1
    e2 /= np.linalg.norm(e2)
    return (e1 + e2) / np.sqrt(2) * np.linalg.norm(D[name]["ref"][T_PERTURB]) * frac


_REC_CACHE = {}


def recovery_ratios(name):
    """Amplitude of each mode after an in-plane kick, divided by the unperturbed run."""
    if name in _REC_CACHE:
        return _REC_CACHE[name]
    d = D[name]
    br = (d["pinv"] @ (d["ref"] - d["mu"]).T).T
    out = {}
    for j, per in zip(d["osc"], d["periods"], strict=False):
        xi = eigenplane_perturbation(name, j)
        P = rollout(d["model"], d["cfg"],
                    perturb=(T_PERTURB, torch.tensor(xi[None], dtype=torch.float32)))
        bp = (d["pinv"] @ (P - d["mu"]).T).T
        out[round(float(per), 1)] = np.abs(bp[:, j]) / np.abs(br[:, j])
    _REC_CACHE[name] = out
    return out


def recovery_fig():
    fig = go.Figure()
    for i, n in enumerate(RUNS):
        for per, ratio in recovery_ratios(n).items():
            x = np.arange(len(ratio)) - T_PERTURB
            m = (x >= -40) & (x <= 560)
            fig.add_trace(go.Scatter(
                x=x[m], y=ratio[m], mode="lines", xaxis=f"x{i + 1}", yaxis=f"y{i + 1}",
                line=dict(color=regime_color(n), width=1.8), name=f"T≈{per:.0f}",
                legendgroup=n, showlegend=False,
                hovertemplate=f"T≈{per:.0f}<br>%{{x}} steps<br>ratio %{{y:.4f}}<extra></extra>"))
        fig.add_shape(type="line", x0=-40, x1=560, y0=1, y1=1, xref=f"x{i + 1}",
                      yref=f"y{i + 1}", line=dict(color=C_REF, dash="dash", width=1))
        fig.add_annotation(text=f"<b>{n}</b>", xref="paper", yref="paper",
                           showarrow=False, x=0.16 + 0.55 * i, y=1.08, font=dict(size=12))
    fig.update_layout(
        **base_layout("Recovery after a 20% perturbation placed inside one mode's "
                      "eigenplane", 400, dict(l=60, r=10, t=80, b=50)),
        xaxis=dict(domain=[0, 0.44], title="steps since perturbation", gridcolor=C_GRID),
        yaxis=dict(title="amplitude ratio<br>perturbed / reference", gridcolor=C_GRID,
                   range=[0.45, 1.45]),
        xaxis2=dict(domain=[0.56, 1.0], anchor="y2", title="steps since perturbation",
                    gridcolor=C_GRID),
        yaxis2=dict(anchor="x2", gridcolor=C_GRID, range=[0.45, 1.45]))
    return fig


def amplitude_fig():
    fig = go.Figure()
    for i, n in enumerate(RUNS):
        d = D[n]
        for s in range(5):
            h0 = torch.tensor(np.random.default_rng(s).normal(
                0, 0.5, (1, d["cfg"]["hidden_size"])), dtype=torch.float32)
            b = (d["pinv"] @ (rollout(d["model"], d["cfg"], T=1500, h0=h0)
                              - d["mu"]).T).T
            for j, per in zip(d["osc"], d["periods"], strict=False):
                fig.add_trace(go.Scatter(
                    y=np.abs(b[:, j]), mode="lines", xaxis=f"x{i + 1}", yaxis=f"y{i + 1}",
                    line=dict(color=regime_color(n), width=1), opacity=0.7, showlegend=False,
                    hovertemplate=f"T≈{per:.0f}<br>step %{{x}}<br>|b| %{{y:.4f}}<extra></extra>"))
        fig.add_annotation(text=f"<b>{n}</b>", xref="paper", yref="paper",
                           showarrow=False, x=0.16 + 0.55 * i, y=1.08, font=dict(size=12))
    fig.update_layout(
        **base_layout("Per-frequency amplitude |b| from 5 random initial states", 400,
                      dict(l=60, r=10, t=80, b=50)),
        xaxis=dict(domain=[0, 0.44], title="step", gridcolor=C_GRID),
        yaxis=dict(title="|b| (log)", type="log", gridcolor=C_GRID),
        xaxis2=dict(domain=[0.56, 1.0], anchor="y2", title="step", gridcolor=C_GRID),
        yaxis2=dict(anchor="x2", type="log", gridcolor=C_GRID))
    return fig


def variance_fig():
    """Singular-value spectrum of the trajectory: how flat the object really is."""
    fig = go.Figure()
    k = np.arange(1, len(D["linear"]["evr"]) + 1)
    for n in RUNS:
        evr = D[n]["evr"]
        fig.add_trace(go.Bar(x=k, y=evr, name=n, marker_color=regime_color(n), opacity=0.85,
                             hovertemplate="PC%{x}<br>%{y:.3%} of variance<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=k, y=np.cumsum(evr), mode="lines+markers", xaxis="x2", yaxis="y2",
            name=n, line=dict(color=regime_color(n), width=1.8), marker=dict(size=5),
            showlegend=False,
            hovertemplate="PC1–%{x}<br>%{y:.3%} cumulative<extra></extra>"))
    fig.add_shape(type="line", x0=6.5, x1=6.5, y0=0, y1=1, xref="x2", yref="y2",
                  line=dict(color=C_REF, dash="dash", width=1))
    fig.update_layout(
        **base_layout("Variance spectrum · per component and cumulative", 340,
                      dict(l=60, r=10, t=60, b=50)),
        barmode="group",
        legend=dict(orientation="h", x=0, y=1.14, font=dict(size=11)),
        xaxis=dict(domain=[0, 0.44], title="component", gridcolor=C_GRID, dtick=1),
        yaxis=dict(title="fraction of variance", gridcolor=C_GRID),
        xaxis2=dict(domain=[0.56, 1.0], anchor="y2", title="components kept",
                    gridcolor=C_GRID, dtick=1),
        yaxis2=dict(anchor="x2", title="cumulative", gridcolor=C_GRID, range=[0, 1.02]))
    return fig


def torus_fig(name):
    d = D[name]
    axes = tuple(f"phase · T≈{p:.0f}" for p in d["periods"])
    fig = embed3d(name, "torus", f"Phase torus · {name}", axes)
    fig.update_layout(scene=dict(
        xaxis=dict(range=[0, 2 * np.pi], tickvals=[0, np.pi, 2 * np.pi],
                   ticktext=["0", "π", "2π"]),
        yaxis=dict(range=[0, 2 * np.pi], tickvals=[0, np.pi, 2 * np.pi],
                   ticktext=["0", "π", "2π"]),
        zaxis=dict(range=[0, 2 * np.pi], tickvals=[0, np.pi, 2 * np.pi],
                   ticktext=["0", "π", "2π"])))
    return fig


def eigenplane_traj_fig(name, frac=0.20):
    """Reference vs perturbed trajectory drawn inside each mode's own plane."""
    d = D[name]
    fig = go.Figure()
    br = (d["pinv"] @ (d["ref"] - d["mu"]).T).T
    seg = slice(T_PERTURB, T_PERTURB + 400)
    for i, (j, per) in enumerate(zip(d["osc"], d["periods"], strict=False)):
        xi = eigenplane_perturbation(name, j, frac)
        P = rollout(d["model"], d["cfg"],
                    perturb=(T_PERTURB, torch.tensor(xi[None], dtype=torch.float32)))
        bp = (d["pinv"] @ (P - d["mu"]).T).T
        xa, ya = f"x{i + 1}", f"y{i + 1}"
        for arr, lab, col, w in ((br, "reference", C_REF, 1.2),
                                 (bp, "perturbed", regime_color(name), 1.6)):
            c = arr[seg, j]
            fig.add_trace(go.Scatter(
                x=c.real, y=c.imag, mode="lines", xaxis=xa, yaxis=ya,
                line=dict(color=col, width=w), name=lab, legendgroup=lab,
                showlegend=(i == 0), opacity=0.9 if lab == "perturbed" else 0.6,
                hovertemplate=f"{lab} · T≈{per:.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[bp[T_PERTURB, j].real], y=[bp[T_PERTURB, j].imag], mode="markers",
            xaxis=xa, yaxis=ya, marker=dict(size=8, color="crimson", symbol="x"),
            name="kick", legendgroup="kick", showlegend=(i == 0),
            hovertemplate="perturbation applied here<extra></extra>"))
    dom = [(0.0, 0.29), (0.355, 0.645), (0.71, 1.0)]
    ax = {}
    for i, (per, (a, bb)) in enumerate(zip(d["periods"], dom, strict=False)):
        ax[f"xaxis{i + 1}"] = dict(domain=[a, bb], anchor=f"y{i + 1}",
                                   title=f"Re b · T≈{per:.0f}", gridcolor=C_GRID,
                                   zeroline=False, scaleanchor=f"y{i + 1}")
        ax[f"yaxis{i + 1}"] = dict(anchor=f"x{i + 1}", title="Im b" if i == 0 else "",
                                   gridcolor=C_GRID, zeroline=False)
    fig.update_layout(
        **base_layout(f"Perturbed vs reference orbit · {name}", 380,
                      dict(l=60, r=10, t=76, b=50)),
        legend=dict(orientation="h", x=0, y=1.16, font=dict(size=11)), **ax)
    return fig


# --------------------------------------------------------------------------- assemble
# Everything the section text quotes is computed here, so the prose cannot drift away from
# whichever runs the report was pointed at.
N_START = 5                                   # random initial states in amplitude_fig
EVR6 = {n: float(D[n]["evr"][:6].sum()) for n in RUNS}
WIND = [max(PERIODS_TASK) / p for p in PERIODS_TASK]
LIN, TAN = list(RUNS)[0], list(RUNS)[1]
REC_TAIL = {n: {per: float(r[-1]) for per, r in recovery_ratios(n).items()} for n in RUNS}
_rt = lambda n: (min(REC_TAIL[n].values()), max(REC_TAIL[n].values()))  # noqa: E731
print("  PC1-6 variance:", {n: round(v, 4) for n, v in EVR6.items()})
print("  recovery ratio at end:", {n: {k: round(v, 4) for k, v in d.items()}
                                   for n, d in REC_TAIL.items()})

SECTIONS = [
    ("Per-frequency eigenplanes",
     "Each output pair reads one mode; this is the trajectory in that mode's own plane. "
     "Use the buttons to recolour by elapsed time or by any target frequency's phase. In "
     "the linear network the phase of a pair's own frequency wraps exactly once around its "
     "circle. In the nonlinear network the same construction on the <em>origin</em> "
     "Jacobian fails — these use the DMD basis instead, which works for both.",
     [eigenplane_fig("linear"), eigenplane_fig("tanh")]),
    ("Phase geometry: the 3-torus",
     f"Taking the phase of each of the {len(PERIODS_TASK)} modes as a coordinate puts the "
     f"trajectory on a {len(PERIODS_TASK)}-torus. The orbit is a single closed curve "
     f"winding around it — "
     + ", ".join(f"{w:.3g} turns of the T={per:g} circle" for w, per in
                 zip(WIND[:-1], PERIODS_TASK[:-1], strict=False))
     + " for every one of the slowest, matching the "
     + " : ".join(f"{p:g}" for p in PERIODS_TASK)
     + " period ratios. This is the structure the PCA view below is a shadow of.",
     [torus_fig("linear"), torus_fig("tanh")]),
    ("Trajectory in PCA space",
     f"The leading three principal components — a 3D shadow of the {2 * len(PERIODS_TASK)}D "
     f"torus above, which is why it looks tangled. The spectrum on the right shows why "
     f"{2 * len(PERIODS_TASK)} is the right number: PC1–6 capture "
     + " and ".join(f"{EVR6[n]:.1%} ({n})" for n in RUNS)
     + " of the variance, two dimensions per frequency, and the remainder is flat.",
     [embed3d("linear", "pca", "3D PCA · linear", ("PC1", "PC2", "PC3")),
      embed3d("tanh", "pca", "3D PCA · nonlinear", ("PC1", "PC2", "PC3")),
      variance_fig()]),
    ("Trajectory in UMAP space",
     "Included as a control rather than a result. Because the object is flat, there is no "
     "curvature for UMAP to unfold, and forcing 6 dimensions into 3 discards structure: "
     "recolour by time and the colours interleave, while the slowest phase stays "
     "monotonic — UMAP has collapsed the revolutions onto a single loop.",
     [embed3d("linear", "umap", "3D UMAP · linear", ("UMAP1", "UMAP2", "UMAP3")),
      embed3d("tanh", "umap", "3D UMAP · nonlinear", ("UMAP1", "UMAP2", "UMAP3"))]),
    ("Stability: Floquet multipliers",
     f"Eigenvalues of the monodromy matrix, the product of instantaneous Jacobians around "
     f"one period of the slowest frequency ({PERIOD} steps). Counting multipliers with "
     f"|μ| > {NEUTRAL}: "
     + ", ".join(f"<b>{n}</b> has {FLOQUET[n]['n_neutral']}" for n in RUNS)
     + ". A single near-neutral multiplier means an attracting limit cycle whose one "
       "neutral direction is the phase; one per task dimension means the whole task "
       "subspace is neutral and nothing attracts.",
     [floquet_fig()]),
    ("Amplitude from random initial states",
     f"Amplitude of each frequency, |b|, from {N_START} random starts. A network with an "
     f"attracting cycle converges to the same amplitude per frequency regardless of where "
     f"it began; one without keeps whatever the initial condition supplied.",
     [amplitude_fig()]),
    ("Recovery from a perturbation inside one eigenplane",
     "Confining the kick to a single mode's own plane removes the geometric confound that "
     f"a random direction in {CFG['hidden_size']} dimensions barely intersects the "
     f"{2 * len(PERIODS_TASK)}-dimensional task subspace. With that removed, the amplitude "
     f"ratio at the end of the window spans "
     + "; ".join(f"<b>{n}</b> {_rt(n)[0]:.3f}–{_rt(n)[1]:.3f}" for n in RUNS)
     + " — a ratio that stays away from 1 means the kick was never corrected, and one that "
       "returns to 1 means it was.",
     [recovery_fig()]),
    ("The same perturbation seen inside the plane",
     "The orbits themselves, reference against perturbed, drawn in each mode's own plane "
     "with the kick marked. The linear orbit is displaced onto a new circle of a different "
     "radius and simply stays there, because every radius is an equally valid orbit. The "
     "nonlinear orbit spirals back onto the one circle the limit cycle allows.",
     [eigenplane_traj_fig("linear"), eigenplane_traj_fig("tanh")]),
]

def _describe(name):
    cfg = D[name]["cfg"]
    parts = [f"{name} = {cfg['activation']}/{cfg['dynamics_type']}"]
    if cfg.get("recurrent_parameterization") == "clark":
        parts += [f"&gamma; = {cfg.get('output_coupling_gamma')}",
                  f"s = {cfg.get('wrec_init_scale')}"]
    parts.append(f"g = {cfg.get('recurrent_gain')}")
    return " · ".join(str(x) for x in parts)


CHIPS = [
    f"{CFG['n_pairs']} cos/sine pairs · T = "
    + ", ".join(f"{p:g}" for p in PERIODS_TASK) + " steps",
    f"N = {CFG['hidden_size']} · dt = {CFG['dt']:g} · seed {CFG.get('seed', 0)}",
    *[_describe(n) for n in RUNS],
    "autonomous (zero input)",
    f"{T_ROLL}-step rollout, trained on {CFG['num_time_steps']}",
]
INTRO = ("How a trained RNN represents three simultaneous frequencies, and how the linear "
         "and nonlinear regimes differ. The companion to the flip-flop report: there the "
         "network has to <em>remember</em> over six timescales, here it has to "
         "<em>generate</em> three. All panels are interactive — rotate the 3D views, and "
         "use the buttons to recolour by elapsed time or by the phase of any target "
         "frequency.")

HTML = page("Sine-wave trajectory geometry", INTRO, CHIPS, SECTIONS,
            "notebooks/tau_init_grid/8_interactive_report.py")

os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
with open(OUT_HTML, "w") as f:
    f.write(HTML)
print(f"wrote {OUT_HTML}  ({os.path.getsize(OUT_HTML) / 1e6:.1f} MB)")
