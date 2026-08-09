"""Build a self-contained interactive HTML report of the n-bit flip-flop analysis.

The counterpart of ``8_interactive_report.py`` for the memory task. The sine task asks the
network to *generate* three frequencies; the flip-flop asks it to *hold* several bits whose
pulses arrive at different rates. The question is the same in both cases — how does the
network allocate its timescales, and what changes between the linear and nonlinear regime —
but the objects are different: fixed points and decay modes rather than limit cycles and
oscillatory modes.

    python 9_flipflop_report.py           # writes figs/flipflop_interactive_report.html

Run from the inner ``timescales`` package directory. Set OUT_HTML to change the target.
"""

import itertools
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
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
    DIVERGING,
    SEQ,
    base_layout,
    layout3d,
    marker,
    page,
    vis_switcher,
)
from train import _create_rnn_model  # noqa: E402

EXP = os.environ.get("SWEEP_ROOT", os.path.join(os.getcwd(), "logs", "experiments"))

# Which runs to analyse. Each entry maps a label to a path relative to SWEEP_ROOT. The two
# labels in MAIN carry the detailed panels; any others appear only in the comparison
# section at the end. Override with e.g.
#   FF_RUNS="linear=<relpath>,tanh=<relpath>"  FF_MAIN="linear,tanh"
DEFAULT_RUNS = {
    "linear g0.5": "tau_init_grid/ff_uniform_full_g0.5/seed_0",
    "linear": "tau_init_grid/ff_uniform_full_g0.9/seed_0",
    "tanh g0.5": "tau_init_grid_tanh_voltage/ff_uniform_full_g0.5/seed_0",
    "tanh": "tau_init_grid_tanh_voltage/ff_uniform_full_g0.9/seed_0",
}


def _parse_runs(spec, default):
    if not spec:
        return dict(default)
    out = {}
    for item in spec.split(","):
        label, _, rel = item.partition("=")
        if not rel:
            raise ValueError(f"FF_RUNS entries must be label=path, got {item!r}")
        out[label.strip()] = rel.strip()
    return out


RUNS = _parse_runs(os.environ.get("FF_RUNS"), DEFAULT_RUNS)
MAIN = tuple(s.strip() for s in os.environ.get("FF_MAIN", "linear,tanh").split(","))
missing = [m for m in MAIN if m not in RUNS]
if missing:
    raise SystemExit(f"FF_MAIN names {missing} are not in FF_RUNS ({list(RUNS)})")
# a label is treated as nonlinear unless it says linear; the config is the real authority
# and is re-checked once loaded
NONLIN = {k: ("linear" not in k.lower()) for k in RUNS}
OUT_HTML = os.environ.get("OUT_HTML", os.path.join(
    os.getcwd(), "notebooks", "tau_init_grid", "figs",
    "flipflop_interactive_report.html"))

T_HOLD = int(os.environ.get("T_HOLD", 600))    # steps of zero input after writing
T_DRIVE = int(os.environ.get("T_DRIVE", 1000))  # length of a normally-driven trial


# --------------------------------------------------------------------------- model
def load(name):
    d = os.path.join(EXP, RUNS[name])
    cfg = yaml.safe_load(open(os.path.join(d, "config_seed0.yaml")))
    model, _ = _create_rnn_model(cfg)
    model.load_state_dict(torch.load(os.path.join(d, "final_model_seed0.pth"),
                                     map_location="cpu", weights_only=False))
    model.eval()
    return model, cfg


def jacobian(model, h, nonlinear):
    st = model.rnn_step
    a = st.current_alphas.detach().numpy()
    W = st.W_rec.weight.detach().numpy() * float(st.recurrent_weight_scale)
    d = (1 - np.tanh(h) ** 2) if nonlinear else np.ones_like(h)
    return np.diag(1 - a) + (a[:, None] * W) * d[None, :]


def readout_features(name, X):
    """The variable the linear readout consumes, mirroring RNN.readout_features.

    Under voltage dynamics the state is a preactivation and the readout sees phi(x); under
    rate dynamics the state is already a rate. phi is the run's own activation — applying
    tanh unconditionally is wrong for an Identity/voltage run, which is exactly what the
    Clark grid's "linear" cell is.
    """
    cfg = D[name]["cfg"] if name in D else CFGS[name][1]
    if cfg["dynamics_type"] != "voltage":
        return X
    return np.tanh(X) if cfg["activation"].lower() == "tanh" else X


def hold_test(model, T=T_HOLD):
    """Write every pattern with one pulse, then hold with zero input."""
    u = np.zeros((len(PATTERNS), T, N_BITS), dtype=np.float32)
    u[:, 0, :] = 2 * PATTERNS - 1          # +1 sets a bit, -1 resets it
    with torch.no_grad():
        hs, out = model(torch.tensor(u))
    return hs.numpy(), out.numpy()


def count_attractors(hT, tol=0.10):
    """Greedy count of end states that are more than ``tol``·‖h‖ apart."""
    scale = np.median(np.linalg.norm(hT, axis=1))
    keep = []
    for i in range(len(hT)):
        if all(np.linalg.norm(hT[i] - hT[j]) > tol * scale for j in keep):
            keep.append(i)
    return keep


# The task shape comes from the runs themselves — n_bits and the per-bit pulse rates differ
# between sweeps (6 bits at 200/143/100/50/20/10 steps in the tau-init grid, 3 bits at
# 200/50/10 in the rich-learning grid), so nothing below may assume a bit count.
print("loading models ...")
CFGS = {}
for name in RUNS:
    CFGS[name] = load(name)
CFG = CFGS[MAIN[0]][1]
for name, (_m, cfg) in CFGS.items():
    if cfg["n_bits"] != CFG["n_bits"] or list(cfg["p_pulse"]) != list(CFG["p_pulse"]):
        raise SystemExit(f"run {name!r} has a different task shape "
                         f"(n_bits={cfg['n_bits']}, p_pulse={cfg['p_pulse']}); "
                         "the panels compare bit-for-bit and would be meaningless")
    NONLIN[name] = cfg["activation"].lower() != "identity"

N_BITS = int(CFG["n_bits"])
DT = float(CFG["dt"])
HOLD_STEPS = [1.0 / p for p in CFG["p_pulse"]]   # mean steps between pulses, per bit
PATTERNS = np.array(list(itertools.product([0, 1], repeat=N_BITS)), dtype=np.float32)
print(f"  {N_BITS} bits, mean hold per bit (steps): "
      f"{[round(h) for h in HOLD_STEPS]}, {len(PATTERNS)} patterns")

print("probing memory ...")
D = {}
for name in RUNS:
    model, cfg = CFGS[name]
    hs, out = hold_test(model)
    hT = hs[:, -1, :]
    with torch.no_grad():
        hnext = model.rnn_step(torch.zeros(len(PATTERNS), N_BITS),
                               torch.tensor(hT)).numpy()
    D[name] = dict(
        model=model, cfg=cfg, hold_h=hs, hold_out=out, hT=hT,
        correct=((out > 0.5) == (PATTERNS[:, None, :] > 0.5)),
        drift=np.linalg.norm(hnext - hT, axis=1)
        / np.maximum(np.linalg.norm(hT, axis=1), 1e-12),
        attractors=count_attractors(hT))
    print(f"  {name}: {len(D[name]['attractors'])}/{len(PATTERNS)} distinct end states, "
          f"median drift {np.median(D[name]['drift']):.1e}")


# --------------------------------------------------------------- driven trajectories
print("simulating driven trials ...")
rng = np.random.default_rng(0)
p_arr = np.asarray(CFG["p_pulse"], dtype=np.float32)
u_drive = np.zeros((1, T_DRIVE, N_BITS), dtype=np.float32)
state = np.zeros(N_BITS, dtype=np.float32)
bit_state = np.zeros((T_DRIVE, N_BITS), dtype=np.float32)
for t in range(T_DRIVE):
    fire = rng.random(N_BITS) < p_arr
    sign = rng.integers(0, 2, N_BITS) * 2 - 1
    u_drive[0, t] = fire * sign
    state[fire & (sign > 0)] = 1.0
    state[fire & (sign < 0)] = 0.0
    bit_state[t] = state

DRIVEN = {}
for name in MAIN:
    d = D[name]
    with torch.no_grad():
        hs, out = d["model"](torch.tensor(u_drive))
    H = hs.numpy()[0]
    mu = H.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(H - mu, full_matrices=False)
    DRIVEN[name] = dict(
        H=H, out=out.numpy()[0], pca=(H - mu) @ Vt[:3].T, pcs=Vt[:3], mu=mu,
        evr=(S ** 2 / (S ** 2).sum())[:10],
        # the readout consumes phi(h) under voltage dynamics and h under rate dynamics;
        # modal decompositions below must be taken in the same variable
        feats=readout_features(name, H),
        W_out=(d["model"].readout_scale * d["model"].W_out.weight).detach().numpy())


# --------------------------------------------------------------------- mode structure
def dmd_basis(X, r):
    """Rank-r exact DMD on the (T, N) sequence X."""
    Y = (X - X.mean(0, keepdims=True)).T
    A, B = Y[:, :-1], Y[:, 1:]
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r].T
    ev, W = np.linalg.eig(Ur.T @ B @ Vr @ np.diag(1 / Sr))
    return ev, B @ Vr @ np.diag(1 / Sr) @ W


def basis_of(name, kind):
    """Eigenbasis to project onto: origin Jacobian, fixed-point Jacobian, or DMD."""
    d = D[name]
    if kind == "dmd":
        return dmd_basis(DRIVEN[name]["feats"], DMD_RANK)
    h = (np.zeros(d["cfg"]["hidden_size"]) if kind == "origin"
         else d["hT"][len(PATTERNS) - 1])
    ev, V = np.linalg.eig(jacobian(d["model"], h, NONLIN[name]))
    order = np.argsort(-np.abs(ev))
    return ev[order], V[:, order]


def mode_table(name, kind="origin"):
    """Decompose each bit's readout over a modal basis.

    ``coupling`` is the fraction of a bit's readout variance carried by each mode's own
    contribution c_bj·b_j(t) over a driven trial. That differs from the plain |W_out·V|
    used elsewhere in the repo: a mode with a large readout weight contributes nothing if
    its coefficient never moves, and weighting by what the mode actually does is what makes
    the assignment below come out one-to-one.
    """
    d = D[name]
    ev, V = basis_of(name, kind)
    feats = DRIVEN[name]["feats"] if name in DRIVEN else None
    if feats is None:                      # off-main gains: fall back to the static weight
        W_out = (d["model"].readout_scale * d["model"].W_out.weight).detach().numpy()
        C = np.abs(W_out @ V)
        C = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-30)
        contrib = None
    else:
        W_out = DRIVEN[name]["W_out"]
        B = (np.linalg.pinv(V) @ (feats - feats.mean(0, keepdims=True)).T).T  # T x M
        Cw = W_out @ V                                                       # n_bits x M
        C = np.zeros((N_BITS, V.shape[1]))
        contrib = np.zeros((N_BITS, len(B)))
        for b in range(N_BITS):
            cb = np.real(B * Cw[b][None, :])          # T x M, this mode's readout share
            v = cb.var(axis=0)
            C[b] = v / max(v.sum(), 1e-30)
            contrib[b] = cb[:, int(np.argmax(v))]
    # |lambda| = 1 is a perfect integrator; clip so tau stays finite and positive
    tau = -1.0 / np.log(np.clip(np.abs(ev), 1e-12, 1 - 1e-9))
    best = [int(np.argmax(C[b])) for b in range(N_BITS)]
    n90 = [int(np.searchsorted(np.cumsum(np.sort(C[b])[::-1]), 0.90) + 1)
           for b in range(N_BITS)]
    return dict(ev=ev, V=V, tau=tau, C=C, best=best, n90=n90, contrib=contrib)


DMD_RANK = 24
print("decomposing the readout over modal bases ...")
MODES = {n: mode_table(n) for n in RUNS}
BASES = {(n, k): mode_table(n, k) for n in MAIN
         for k in ("origin", "fixedpt", "dmd")}
for n in MAIN:
    m = MODES[n]
    print(f"  {n}: dominant-mode τ {[round(m['tau'][j], 1) for j in m['best']]}"
          f"  share {[round(m['C'][b][m['best'][b]], 2) for b in range(N_BITS)]}"
          f"  modes for 90% {m['n90']}")

t_axis = np.arange(T_DRIVE)
COLORINGS = [("time", t_axis, SEQ, "step")]
for b in range(N_BITS):
    COLORINGS.append((f"bit {b}", bit_state[:, b], DIVERGING,
                      f"bit {b} (hold≈{HOLD_STEPS[b]:.0f})"))


# --------------------------------------------------------------------------- figures
TAU_LIM = [2.0, 400.0]


def clamp_tau(taus):
    """Clamp to the plotted range; a mode with |lambda| >= 1 has tau -> infinity and would
    otherwise vanish off the top of the axis without the reader noticing."""
    v = np.asarray(taus, dtype=float)
    return np.clip(v, *TAU_LIM), (v > TAU_LIM[1]) | (v < TAU_LIM[0])


def coupling_fig(name):
    """Which mode each bit reads, and how that mode's timescale relates to its hold time."""
    m = MODES[name]
    fig = go.Figure()
    # Plot coupling against the mode's own timescale rather than its index: the peaks then
    # land directly on the timescale each bit reads, and no mode is cut off by truncating
    # the ordering (bits 4 and 5 read modes several hundred places down).
    bit_cols = sample_colorscale("Turbo", np.linspace(0.08, 0.92, N_BITS))
    tau_best = [m["tau"][j] for j in m["best"]]
    top = max(float(m["C"].max()), 1e-6)
    # The dominant-mode markers sit in their own lane above the data rather than on top of
    # it, so no point is hidden behind one; a stem drops from each to the point it marks.
    lane = top * 1.14
    for b in range(N_BITS):
        fig.add_trace(go.Scatter(
            x=m["tau"], y=m["C"][b], mode="markers", xaxis="x", yaxis="y",
            marker=dict(color=bit_cols[b], size=3.4), opacity=0.55,
            name=f"bit {b}", legendgroup=f"b{b}",
            hovertemplate=f"bit {b}<br>mode τ=%{{x:.1f}}"
                          "<br>share of readout variance %{y:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[tau_best[b], tau_best[b]], y=[m["C"][b][m["best"][b]], lane],
            mode="lines", xaxis="x", yaxis="y",
            line=dict(color=bit_cols[b], width=0.9, dash="dot"),
            legendgroup=f"b{b}", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[tau_best[b]], y=[lane], mode="markers", xaxis="x", yaxis="y",
            marker=dict(size=9, color=bit_cols[b], symbol="triangle-down",
                        line=dict(width=0.6, color="rgba(0,0,0,0.35)")),
            legendgroup=f"b{b}", showlegend=False,
            hovertemplate=f"bit {b} → τ=%{{x:.1f}} steps<extra></extra>"))
    y_clamped, off = clamp_tau(tau_best)
    fig.add_trace(go.Scatter(
        x=HOLD_STEPS, y=y_clamped, mode="markers", xaxis="x2", yaxis="y2",
        marker=dict(size=11, color=bit_cols,
                    symbol=["star" if o else "circle" for o in off],
                    line=dict(width=0.6, color="rgba(0,0,0,.4)")),
        showlegend=False,
        customdata=np.stack([np.arange(N_BITS), np.asarray(tau_best, dtype=float)], -1),
        hovertemplate="bit %{customdata[0]:.0f}<br>hold≈%{x:.0f} steps"
                      "<br>mode τ=%{customdata[1]:.3g} steps<extra></extra>"))
    if off.any():
        fig.add_annotation(text="★ off scale (|λ| ≥ 1, τ → ∞)", xref="x2 domain",
                           yref="y2 domain", x=0.03, y=0.04, showarrow=False,
                           xanchor="left", font=dict(size=10, color=C_REF))
    fig.add_trace(go.Scatter(x=TAU_LIM, y=TAU_LIM, mode="lines", xaxis="x2", yaxis="y2",
                             line=dict(color=C_REF, dash="dash", width=1),
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        **base_layout(f"Bit → mode coupling · {name}", 380,
                      dict(l=66, r=14, t=70, b=52)),
        legend=dict(orientation="h", x=0, y=1.14, font=dict(size=10)),
        xaxis=dict(domain=[0, 0.42], type="log", range=np.log10([1, 400]),
                   title="timescale τ of the mode (steps)", gridcolor=C_GRID),
        yaxis=dict(title="share of readout variance", gridcolor=C_GRID,
                   range=[-0.02 * top, lane * 1.10]),
        xaxis2=dict(domain=[0.60, 1.0], anchor="y2", type="log",
                    range=np.log10(TAU_LIM), title="bit's mean hold time (steps)",
                    gridcolor=C_GRID),
        yaxis2=dict(anchor="x2", type="log", range=np.log10(TAU_LIM),
                    title="τ of mode read (steps)", gridcolor=C_GRID,
                    scaleanchor="x2"))
    return fig


def projection_fig(name, kind="origin", t0=0, t1=520):
    """Project the driven trajectory onto the mode each bit reads, next to that bit.

    The memory-task analogue of the sine report's eigenplanes. There the coupled mode was
    a complex pair and the trajectory drew a circle in its plane; here the coupled modes
    are mostly real decay modes, so the natural picture is the scalar coefficient against
    time, which should look like a staircase that steps at pulses and leaks between them.
    """
    m = BASES[name, kind]
    bit_cols = sample_colorscale("Turbo", np.linspace(0.08, 0.92, N_BITS))
    fig = go.Figure()
    xs, ys = np.arange(t0, t1), None
    ncol = min(N_BITS, 3)
    nrow = int(np.ceil(N_BITS / ncol))
    span = 1.0 / ncol
    dom_x = [(i * span, i * span + span * 0.86) for i in range(ncol)]
    vspan = 1.0 / nrow
    dom_y = [(1.0 - (r + 1) * vspan, 1.0 - r * vspan - (0.12 * vspan if nrow > 1 else 0.0))
             for r in range(nrow)]
    ax = {}
    for b in range(N_BITS):
        k = b + 1
        xa, ya = f"x{k}", f"y{k}"
        c = m["contrib"][b][t0:t1]
        ys = (c - c.min()) / max(c.max() - c.min(), 1e-12)      # rescale to the bit's 0–1
        fig.add_trace(go.Scatter(
            x=xs, y=bit_state[t0:t1, b], mode="lines", xaxis=xa, yaxis=ya,
            line=dict(color=C_REF, width=2.6, shape="hv"), opacity=0.55,
            name="true bit", legendgroup="bit", showlegend=(b == 0),
            hovertemplate="true bit %{y:.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", xaxis=xa, yaxis=ya,
            line=dict(color=bit_cols[b], width=1.5),
            name="dominant mode", legendgroup="mode", showlegend=(b == 0),
            hovertemplate="step %{x}<br>rescaled coefficient %{y:.2f}<extra></extra>"))
        r = abs(np.corrcoef(m["contrib"][b], bit_state[:, b])[0, 1])
        fig.add_annotation(
            text=f"bit {b} · hold≈{HOLD_STEPS[b]:.0f} · τ={m['tau'][m['best'][b]]:.0f}"
                 f" · r={r:.2f}",
            # annotation refs reject "x1"/"y1"; the first axis must be spelled "x"/"y"
            xref=f"{'x' if k == 1 else xa} domain",
            yref=f"{'y' if k == 1 else ya} domain", x=0.02, y=1.16,
            showarrow=False, xanchor="left", font=dict(size=10.5, color=bit_cols[b]))
        col, row = b % ncol, b // ncol
        ax[f"xaxis{k}"] = dict(domain=list(dom_x[col]), anchor=ya, gridcolor=C_GRID,
                               title="step" if row == nrow - 1 else "",
                               showticklabels=(row == nrow - 1))
        ax[f"yaxis{k}"] = dict(domain=list(dom_y[row]), anchor=xa, gridcolor=C_GRID,
                               range=[-0.12, 1.12], showticklabels=(col == 0),
                               title="bit / coefficient" if col == 0 else "")
    fig.update_layout(
        **base_layout(f"Driven trajectory projected onto each bit's dominant mode · "
                      f"{name} ({kind} basis)", 260 * nrow + 60,
                      dict(l=64, r=12, t=92, b=48)),
        legend=dict(orientation="h", x=0, y=1.13, font=dict(size=11)), **ax)
    return fig


def rank_fig():
    """How many modes it actually takes to account for a bit's readout."""
    labels = {"origin": "origin Jacobian", "fixedpt": "fixed-point Jacobian",
              "dmd": f"DMD (rank {DMD_RANK})"}
    syms = {"origin": "circle", "fixedpt": "square", "dmd": "diamond"}
    fig = go.Figure()
    for i, name in enumerate(MAIN):
        for kind, sym in syms.items():
            m = BASES[name, kind]
            fig.add_trace(go.Scatter(
                x=list(range(N_BITS)), y=m["n90"], mode="markers+lines",
                xaxis=f"x{i + 1}", yaxis=f"y{i + 1}",
                marker=dict(size=8, symbol=sym, color=COLOR[name], opacity=0.85),
                line=dict(color=COLOR[name], width=1, dash="dot"),
                name=labels[kind], legendgroup=kind, showlegend=(i == 0),
                hovertemplate="bit %{x}<br>%{y} modes for 90%<extra></extra>"))
        fig.add_annotation(text=f"<b>{name}</b>", xref="paper", yref="paper",
                           showarrow=False, x=0.16 + 0.55 * i, y=1.12, font=dict(size=12))
    fig.update_layout(
        **base_layout("Modes needed for 90% of a bit's readout variance", 330,
                      dict(l=64, r=12, t=92, b=48)),
        legend=dict(orientation="h", x=0, y=1.26, font=dict(size=10.5)),
        xaxis=dict(domain=[0, 0.44], title="bit", dtick=1, gridcolor=C_GRID),
        yaxis=dict(type="log", title="number of modes", gridcolor=C_GRID),
        xaxis2=dict(domain=[0.56, 1.0], anchor="y2", title="bit", dtick=1,
                    gridcolor=C_GRID),
        yaxis2=dict(anchor="x2", type="log", gridcolor=C_GRID))
    return fig


def basis_compare_fig():
    """Does the dominant mode's timescale match the hold time, in any basis?"""
    labels = {"origin": "origin Jacobian", "fixedpt": "fixed-point Jacobian",
              "dmd": f"DMD (rank {DMD_RANK})"}
    fig = go.Figure()
    syms = {"origin": "circle", "fixedpt": "square", "dmd": "diamond"}
    for i, name in enumerate(MAIN):
        for kind, sym in syms.items():
            m = BASES[name, kind]
            raw = [m["tau"][j] for j in m["best"]]
            yv, off = clamp_tau(raw)
            fig.add_trace(go.Scatter(
                x=HOLD_STEPS, y=yv, mode="markers",
                xaxis=f"x{i + 1}", yaxis=f"y{i + 1}",
                marker=dict(size=9, color=COLOR[name], opacity=0.85,
                            symbol=["star" if o else sym for o in off],
                            line=dict(width=0.6, color="rgba(0,0,0,0.35)")),
                name=labels[kind], legendgroup=kind, showlegend=(i == 0),
                customdata=np.stack([np.arange(N_BITS), np.asarray(raw, float)], -1),
                hovertemplate="bit %{customdata[0]:.0f}<br>hold≈%{x:.0f}"
                              "<br>τ=%{customdata[1]:.3g}<extra></extra>"))
        fig.add_annotation(text="★ off scale", xref="paper", yref="paper", showarrow=False,
                           x=0.02 + 0.56 * i, y=0.02, xanchor="left",
                           font=dict(size=10, color=C_REF))
        fig.add_trace(go.Scatter(x=TAU_LIM, y=TAU_LIM, mode="lines", xaxis=f"x{i + 1}",
                                 yaxis=f"y{i + 1}", showlegend=False,
                                 line=dict(color=C_REF, dash="dash", width=1),
                                 hoverinfo="skip"))
        fig.add_annotation(text=f"<b>{name}</b>", xref="paper", yref="paper",
                           showarrow=False, x=0.16 + 0.55 * i, y=1.10, font=dict(size=12))
    lg = np.log10(TAU_LIM)
    fig.update_layout(
        **base_layout("Timescale of the dominant mode versus the bit's hold time", 400,
                      dict(l=64, r=12, t=96, b=52)),
        legend=dict(orientation="h", x=0, y=1.20, font=dict(size=10.5)),
        xaxis=dict(domain=[0, 0.44], type="log", range=lg,
                   title="bit's mean hold time (steps)", gridcolor=C_GRID),
        yaxis=dict(type="log", range=lg, title="τ of dominant mode (steps)",
                   gridcolor=C_GRID, scaleanchor="x"),
        xaxis2=dict(domain=[0.56, 1.0], anchor="y2", type="log", range=lg,
                    title="bit's mean hold time (steps)", gridcolor=C_GRID),
        yaxis2=dict(anchor="x2", type="log", range=lg, gridcolor=C_GRID,
                    scaleanchor="x2"))
    return fig


def retention_fig():
    """Per-bit readout accuracy as the hold gets longer."""
    fig = go.Figure()
    for i, name in enumerate(MAIN):
        d = D[name]
        acc = d["correct"].mean(axis=0)            # T x n_bits
        for b in range(N_BITS):
            fig.add_trace(go.Scatter(
                x=t_axis[:T_HOLD], y=acc[:, b], mode="lines",
                xaxis=f"x{i + 1}", yaxis=f"y{i + 1}",
                line=dict(color=COLOR[name], width=1.6),
                opacity=0.30 + 0.70 * b / (N_BITS - 1), showlegend=False,
                hovertemplate=(f"bit {b} (hold≈{HOLD_STEPS[b]:.0f})"
                               "<br>%{x} steps held<br>%{y:.3f} correct<extra></extra>")))
            fig.add_annotation(x=T_HOLD, y=acc[-1, b], text=f" {b}", xref=f"x{i + 1}",
                               yref=f"y{i + 1}", showarrow=False, xanchor="left",
                               font=dict(size=9, color=COLOR[name]))
        fig.add_shape(type="line", x0=0, x1=T_HOLD, y0=0.5, y1=0.5,
                      xref=f"x{i + 1}", yref=f"y{i + 1}",
                      line=dict(color=C_REF, dash="dash", width=1))
        fig.add_annotation(text=f"<b>{name}</b>", xref="paper", yref="paper",
                           showarrow=False, x=0.16 + 0.55 * i, y=1.08, font=dict(size=12))
    fig.update_layout(
        **base_layout("Fraction of bits still read out correctly after holding", 400,
                      dict(l=60, r=24, t=80, b=50)),
        xaxis=dict(domain=[0, 0.44], title="steps of zero input", gridcolor=C_GRID),
        yaxis=dict(title="fraction correct", gridcolor=C_GRID, range=[0.42, 1.03]),
        xaxis2=dict(domain=[0.56, 1.0], anchor="y2", title="steps of zero input",
                    gridcolor=C_GRID),
        yaxis2=dict(anchor="x2", gridcolor=C_GRID, range=[0.42, 1.03]))
    return fig


def spectrum_fig():
    """Jacobian eigenvalues at the origin and at a written memory state."""
    fig = go.Figure()
    th = np.linspace(0, 2 * np.pi, 400)
    panels = [(MAIN[0], "origin"), (MAIN[1], "origin"), (MAIN[1], "memory state")]
    for i, (name, where) in enumerate(panels):
        d = D[name]
        h = (np.zeros(d["cfg"]["hidden_size"]) if where == "origin"
             else d["hT"][len(PATTERNS) - 1])
        ev = np.linalg.eigvals(jacobian(d["model"], h, NONLIN[name]))
        xa, ya = f"x{i + 1}", f"y{i + 1}"
        fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", xaxis=xa,
                                 yaxis=ya, line=dict(color=C_REF, dash="dash", width=1),
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=ev.real, y=ev.imag, mode="markers", xaxis=xa, yaxis=ya,
            marker=dict(size=5, color=COLOR[name], opacity=0.8),
            customdata=np.abs(ev),
            hovertemplate="|λ| = %{customdata:.4f}<extra></extra>"))
        n_out = int((np.abs(ev) > 1).sum())
        note = f"{n_out} outside" if n_out else "all inside"
        fig.add_annotation(text=f"<b>{name}</b> · {where} — {note}",
                           xref="paper", yref="paper", showarrow=False,
                           x=0.145 + 0.355 * i, y=1.07, font=dict(size=11.5))
    dom = [(0.0, 0.29), (0.355, 0.645), (0.71, 1.0)]
    ax = {}
    for i, (a, b) in enumerate(dom):
        ax[f"xaxis{i + 1}"] = dict(domain=[a, b], anchor=f"y{i + 1}", title="Re λ",
                                   gridcolor=C_GRID, zeroline=False, range=[-1.25, 1.25],
                                   scaleanchor=f"y{i + 1}")
        ax[f"yaxis{i + 1}"] = dict(anchor=f"x{i + 1}", title="Im λ" if i == 0 else "",
                                   gridcolor=C_GRID, zeroline=False, range=[-1.25, 1.25])
    fig.update_layout(**base_layout("Jacobian spectrum, autonomous dynamics", 400,
                                    dict(l=54, r=10, t=80, b=50)),
                      showlegend=False, **ax)
    return fig


def pca2d_fig(name):
    """The three PC planes as ordinary SVG scatters.

    The 3D view below needs WebGL, which is not available in every viewer (sandboxed
    iframes in particular), and a rotatable blob is in any case harder to read than the
    planes. These render anywhere and carry the same colouring switcher.
    """
    P = DRIVEN[name]["pca"]
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig = go.Figure()
    for i, (a, b) in enumerate(pairs):
        for ci, (label, vals, scale, cbar) in enumerate(COLORINGS):
            fig.add_trace(go.Scatter(
                x=P[:, a], y=P[:, b], mode="markers",
                xaxis=f"x{i + 1}", yaxis=f"y{i + 1}",
                marker=marker(vals, scale, cbar, 2.8, i == 2),
                visible=(ci == 0), name=label, customdata=t_axis,
                hovertemplate="step %{customdata}<extra></extra>"))
    dom = [(0.0, 0.28), (0.34, 0.62), (0.68, 0.96)]
    ax = {}
    for i, ((a, b), (lo, hi)) in enumerate(zip(pairs, dom, strict=False)):
        ax[f"xaxis{i + 1}"] = dict(domain=[lo, hi], anchor=f"y{i + 1}",
                                   title=f"PC{a + 1}", gridcolor=C_GRID, zeroline=False)
        ax[f"yaxis{i + 1}"] = dict(anchor=f"x{i + 1}", title=f"PC{b + 1}",
                                   gridcolor=C_GRID, zeroline=False,
                                   scaleanchor=f"x{i + 1}")
    fig.update_layout(
        **base_layout(f"PC planes of a driven trial · {name}", 380,
                      dict(l=54, r=10, t=76, b=50)),
        showlegend=False, updatemenus=vis_switcher(COLORINGS, 3), **ax)
    return fig


def readout_geometry_fig(name, thresh=0.5):
    """PCA cloud with each bit's dominant Jacobian eigendirection and readout boundary.

    Two overlays, both exact only within the displayed slice:

    * the dominant Jacobian eigenvector for each bit, drawn through the mean. Only its
      component inside PC1–3 can be shown, so each is labelled with the fraction of its
      norm that survives the projection — a direction with a small fraction is mostly
      pointing out of the picture and the arrow means little.
    * the readout boundary for each bit. The readout is affine in the feature variable, so
      the set {feats : s·w_b·feats = thresh} is a hyperplane in N dimensions; intersecting
      it with the 3-dimensional PCA slice through the mean gives the plane drawn here.
      Points off the slice carry readout contributions that are not shown, so the plane
      separates the projected cloud only as well as PC1–3 capture the readout — the
      annotation reports how often the in-slice side actually matches the bit.
    """
    if NONLIN[name]:
        raise ValueError(f"{name!r} is nonlinear: the readout consumes phi(h), so its "
                         "decision boundary is curved in the PCA-of-h space this figure "
                         "draws. Build it only for Identity runs.")
    dr = DRIVEN[name]
    P, pcs, mu = dr["pca"], dr["pcs"], dr["mu"]
    W_out = dr["W_out"]                                # (n_bits, N), scale already folded in
    bit_cols = sample_colorscale("Turbo", np.linspace(0.08, 0.92, N_BITS))
    radius = float(np.sqrt((P ** 2).sum(1).mean()))
    fig = go.Figure()

    for ci, (label, vals, scale, cbar) in enumerate(COLORINGS):
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
            marker=marker(vals, scale, cbar, 2.0, True),
            visible=(ci == 0), name=label, customdata=t_axis,
            hovertemplate="step %{customdata}<extra></extra>"))

    m = MODES[name]
    extras = 0
    for b in range(N_BITS):
        v = m["V"][:, m["best"][b]]
        v = np.real(v) if np.abs(np.imag(v)).max() < 1e-9 else np.real(v)
        v = v / max(np.linalg.norm(v), 1e-12)
        pv = pcs @ v                                   # component inside the slice
        frac = float(np.linalg.norm(pv))
        if frac > 1e-9:
            e = pv / frac * radius * 1.6
            fig.add_trace(go.Scatter3d(
                x=[-e[0], e[0]], y=[-e[1], e[1]], z=[-e[2], e[2]], mode="lines",
                line=dict(color=bit_cols[b], width=6),
                name=f"bit {b} eigendirection (τ={m['tau'][m['best'][b]]:.3g}, "
                     f"{frac:.0%} in slice)",
                hovertemplate=f"bit {b} dominant eigendirection<br>"
                              f"{frac:.1%} of its norm lies in PC1–3<extra></extra>"))
            extras += 1

        # readout boundary restricted to the slice: c·a = rhs, with a the PC coordinates
        c = pcs @ W_out[b]
        rhs = thresh - float(W_out[b] @ mu[0])
        nc = float(np.linalg.norm(c))
        if nc < 1e-12:
            continue
        n_hat = c / nc
        a0 = n_hat * (rhs / nc)
        helper = np.eye(3)[int(np.argmin(np.abs(n_hat)))]
        e1 = np.cross(n_hat, helper)
        e1 /= max(np.linalg.norm(e1), 1e-12)
        e2 = np.cross(n_hat, e1)
        L = radius * 1.9
        corners = np.array([a0 + s1 * L * e1 + s2 * L * e2
                            for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        side = (P @ n_hat) > (rhs / nc)
        acc = float(max((side == (bit_state[:, b] > 0.5)).mean(),
                        (side != (bit_state[:, b] > 0.5)).mean()))
        fig.add_trace(go.Mesh3d(
            x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
            i=[0, 0], j=[1, 2], k=[2, 3], color=bit_cols[b], opacity=0.18,
            name=f"bit {b} readout boundary ({acc:.0%} of the slice on the right side)",
            showlegend=True, hoverinfo="name"))
        extras += 1

    fig.update_layout(**layout3d(f"Readout geometry · {name}", ("PC1", "PC2", "PC3"), 620),
                      showlegend=True,
                      legend=dict(orientation="v", x=0, y=-0.02, yanchor="top",
                                  font=dict(size=10)),
                      updatemenus=vis_switcher(COLORINGS, 1, n_extra=extras))
    return fig


def embed3d(name):
    """PCA of a normally driven trial, recolourable by time or by any bit's state."""
    fig = go.Figure()
    P = DRIVEN[name]["pca"]
    for ci, (label, vals, scale, cbar) in enumerate(COLORINGS):
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
            marker=marker(vals, scale, cbar, 2.4, True),
            visible=(ci == 0), name=label, customdata=t_axis,
            hovertemplate="step %{customdata}<extra></extra>"))
    fig.update_layout(**layout3d(f"3D PCA of a driven trial · {name}",
                                 ("PC1", "PC2", "PC3")),
                      showlegend=False,
                      updatemenus=vis_switcher(COLORINGS, 1))
    return fig


def memory_states_fig(bits=None):
    """The written states, projected onto the readout plane of two bits.

    Defaults to the two bits whose end-state readouts separate most in the nonlinear run,
    so the plane shown is one where something is actually retained. With n_bits = 2 there
    is only one choice; the report never assumes a particular bit index.
    """
    if bits is None:
        d0 = D[MAIN[-1]]
        W0 = (d0["model"].readout_scale * d0["model"].W_out.weight).detach().numpy()
        f0 = readout_features(MAIN[-1], d0["hT"])
        pr = f0 @ W0.T
        spread = [abs(pr[PATTERNS[:, b] > 0.5, b].mean() - pr[PATTERNS[:, b] < 0.5, b].mean())
                  for b in range(N_BITS)]
        bits = sorted(np.argsort(spread)[-2:])
    bx, by = int(bits[0]), int(bits[1])
    fig = go.Figure()
    for i, name in enumerate(MAIN):
        d = D[name]
        W_out = (d["model"].readout_scale * d["model"].W_out.weight).detach().numpy()
        feats = readout_features(name, d["hT"])
        proj = feats @ W_out.T                  # n_patterns x n_bits readout values
        keep = d["attractors"]
        fig.add_trace(go.Scatter(
            x=proj[:, bx], y=proj[:, by], mode="markers", xaxis=f"x{i + 1}",
            yaxis=f"y{i + 1}",
            marker=dict(size=9, color=2 * PATTERNS[:, bx] + PATTERNS[:, by],
                        colorscale="Viridis", showscale=False,
                        line=dict(width=0.5, color="rgba(0,0,0,0.35)")),
            customdata=PATTERNS,
            hovertemplate="pattern %{customdata}<br>readout "
                          "(%{x:.2f}, %{y:.2f})<extra></extra>"))
        fig.add_annotation(
            text=f"<b>{name}</b> — {len(keep)} distinct end state"
                 f"{'s' if len(keep) != 1 else ''} of {len(PATTERNS)}",
            xref="paper", yref="paper", showarrow=False,
            x=0.18 + 0.55 * i, y=1.08, font=dict(size=12))
        for ref, axref in ((0.5, f"x{i + 1}"), (0.5, f"y{i + 1}")):
            fig.add_shape(type="line", line=dict(color=C_REF, dash="dot", width=1),
                          xref=f"x{i + 1}", yref=f"y{i + 1}",
                          **(dict(x0=ref, x1=ref, y0=-0.6, y1=1.6) if axref[0] == "x"
                             else dict(y0=ref, y1=ref, x0=-0.6, x1=1.6)))
    fig.update_layout(
        **base_layout(f"Where the {len(PATTERNS)} written patterns end up after "
                      f"{T_HOLD} steps of hold", 400, dict(l=60, r=10, t=80, b=52)),
        showlegend=False,
        xaxis=dict(domain=[0, 0.44], title=f"readout of bit {bx}", gridcolor=C_GRID),
        yaxis=dict(title=f"readout of bit {by}", gridcolor=C_GRID, scaleanchor="x"),
        xaxis2=dict(domain=[0.56, 1.0], anchor="y2", title=f"readout of bit {bx}",
                    gridcolor=C_GRID),
        yaxis2=dict(anchor="x2", gridcolor=C_GRID, scaleanchor="x2"))
    return fig


def perturbation_fig(frac=0.25):
    """Kick a written state in a random direction and watch the readout."""
    fig = go.Figure()
    rng_p = np.random.default_rng(1)
    for i, name in enumerate(MAIN):
        d = D[name]
        model, N = d["model"], d["cfg"]["hidden_size"]
        h0 = d["hold_h"][:, 200, :]                       # written, already settled
        base = np.linalg.norm(h0, axis=1, keepdims=True)
        for _rep in range(3):
            xi = rng_p.normal(0, 1, (len(PATTERNS), N))
            xi = frac * base * xi / np.linalg.norm(xi, axis=1, keepdims=True)
            u = np.zeros((len(PATTERNS), 200, N_BITS), dtype=np.float32)
            with torch.no_grad():
                _, out = model(torch.tensor(u),
                               init_hidden=torch.tensor(h0 + xi, dtype=torch.float32))
            acc = ((out.numpy() > 0.5) == (PATTERNS[:, None, :] > 0.5)).mean(axis=(0, 2))
            fig.add_trace(go.Scatter(
                y=acc, mode="lines", xaxis=f"x{i + 1}", yaxis=f"y{i + 1}",
                line=dict(color=COLOR[name], width=1.5), opacity=0.8, showlegend=False,
                hovertemplate="%{x} steps after kick<br>%{y:.3f} correct<extra></extra>"))
        fig.add_shape(type="line", x0=0, x1=200, y0=0.5, y1=0.5, xref=f"x{i + 1}",
                      yref=f"y{i + 1}", line=dict(color=C_REF, dash="dash", width=1))
        fig.add_annotation(text=f"<b>{name}</b>", xref="paper", yref="paper",
                           showarrow=False, x=0.16 + 0.55 * i, y=1.08, font=dict(size=12))
    fig.update_layout(
        **base_layout(f"Readout after a {frac:.0%} random kick away from a written state",
                      380, dict(l=60, r=10, t=80, b=50)),
        xaxis=dict(domain=[0, 0.44], title="steps since kick", gridcolor=C_GRID),
        yaxis=dict(title="fraction of bits correct", gridcolor=C_GRID, range=[0.42, 1.03]),
        xaxis2=dict(domain=[0.56, 1.0], anchor="y2", title="steps since kick",
                    gridcolor=C_GRID),
        yaxis2=dict(anchor="x2", gridcolor=C_GRID, range=[0.42, 1.03]))
    return fig


def gain_fig():
    """Attractor count and long-hold accuracy across every run supplied."""
    fig = go.Figure()
    labels = list(RUNS)
    counts = [len(D[n]["attractors"]) for n in labels]
    held = [float(D[n]["correct"][:, -1, :].mean()) for n in labels]
    cols = [COLOR["tanh" if NONLIN[n] else "linear"] for n in labels]
    fig.add_trace(go.Bar(x=labels, y=counts, marker_color=cols, opacity=0.85,
                         xaxis="x", yaxis="y",
                         hovertemplate="%{x}<br>%{y} distinct end states<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=held, marker_color=cols, opacity=0.85,
                         xaxis="x2", yaxis="y2",
                         hovertemplate="%{x}<br>%{y:.3f} correct<extra></extra>"))
    fig.add_shape(type="line", x0=-0.5, x1=len(labels) - 0.5, y0=0.5, y1=0.5,
                  xref="x2", yref="y2",
                  line=dict(color=C_REF, dash="dash", width=1))
    fig.update_layout(
        **base_layout(f"Distinct end states, and accuracy after a {T_HOLD}-step hold",
                      340, dict(l=58, r=10, t=56, b=64)),
        showlegend=False,
        xaxis=dict(domain=[0, 0.44], gridcolor=C_GRID),
        yaxis=dict(title=f"distinct end states (of {len(PATTERNS)})", gridcolor=C_GRID),
        xaxis2=dict(domain=[0.56, 1.0], anchor="y2", gridcolor=C_GRID),
        yaxis2=dict(anchor="x2", title="fraction of bits correct", gridcolor=C_GRID,
                    range=[0, 1.05]))
    return fig


# --------------------------------------------------------------------------- assemble
# Everything the section text quotes is computed here. These reports get pointed at
# different sweeps, and a hard-coded number silently becomes a false claim the moment the
# runs change, so the prose reads its figures from the same arrays the panels do.
def cohens_d(name):
    P = DRIVEN[name]["pca"]
    out = []
    for b in range(N_BITS):
        hi, lo = P[bit_state[:, b] > 0.5], P[bit_state[:, b] < 0.5]
        if len(hi) < 10 or len(lo) < 10:
            out.append(float("nan"))
            continue
        pooled = np.sqrt((hi.var(0) + lo.var(0)) / 2)
        out.append(float(np.linalg.norm((hi.mean(0) - lo.mean(0))
                                        / np.maximum(pooled, 1e-12))))
    return out


def outside_unit(name, at):
    d = D[name]
    h = (np.zeros(d["cfg"]["hidden_size"]) if at == "origin"
         else d["hT"][len(PATTERNS) - 1])
    ev = np.linalg.eigvals(jacobian(d["model"], h, NONLIN[name]))
    return int((np.abs(ev) > 1).sum()), float(np.abs(ev).max())


def rng_str(vals, fmt="{:.2f}"):
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}–{fmt.format(hi)}"


STAT = {}
for n in MAIN:
    m = MODES[n]
    tau_b = [float(m["tau"][j]) for j in m["best"]]
    STAT[n] = dict(
        tau=tau_b,
        ratio=[tb / h for tb, h in zip(tau_b, HOLD_STEPS, strict=False)],
        # "ordered" means slower-held bits read slower modes, the pattern the sine report
        # found; holds are given slowest-first so tau should decrease with bit index
        ordered=all(a > b for a, b in zip(tau_b, tau_b[1:], strict=False)),
        share=[float(m["C"][b][m["best"][b]]) for b in range(N_BITS)],
        n90=list(m["n90"]),
        held=[float(D[n]["correct"][:, -1, b].mean()) for b in range(N_BITS)],
        pcvar=float(DRIVEN[n]["evr"][:3].sum()),
        cohen=cohens_d(n),
        n_out_origin=outside_unit(n, "origin"),
        n_out_mem=outside_unit(n, "memory"),
        n_att=len(D[n]["attractors"]),
    )
    s = STAT[n]
    print(f"  {n}: tau {[round(x, 1) for x in s['tau']]} vs holds "
          f"{[round(h) for h in HOLD_STEPS]}  ordered={s['ordered']}  "
          f"share {rng_str(s['share'])}  n90 {min(s['n90'])}-{max(s['n90'])}  "
          f"held {[round(x, 2) for x in s['held']]}  PC1-3 {s['pcvar']:.1%}  "
          f"cohen {rng_str(s['cohen'])}  outside|1| origin {s['n_out_origin'][0]}")

LIN, TAN = MAIN[0], MAIN[1]
n_lin, n_tanh = STAT[LIN]["n_att"], STAT[TAN]["n_att"]
HELD_STR = {n: ", ".join(f"bit {b} {STAT[n]['held'][b]:.2f}" for b in range(N_BITS))
            for n in MAIN}

print("building figures ...")

SECTIONS = [
    ("Each bit reads one slow mode",
     "For each bit, the share of that bit's readout variance carried by each mode's own "
     "contribution, plotted against the mode's decay timescale; the triangle above marks "
     "the dominant mode. This weights a mode by what it actually does, not just by its "
     "readout weight — a mode with a large weight contributes nothing if its coefficient "
     "never moves. Against required hold times of "
     + ", ".join(f"{h:.0f}" for h in HOLD_STEPS)
     + " steps, the dominant modes sit at "
     + "; ".join(f"<b>{n}</b> τ = " + ", ".join(f"{x:.3g}" for x in STAT[n]["tau"])
                 for n in MAIN)
     + ". "
     + ". "
     + (("Both runs place slower modes on the longer-held bits."
         if STAT[MAIN[0]]["ordered"] else
         "Neither run places slower modes on the longer-held bits.")
        if STAT[MAIN[0]]["ordered"] == STAT[MAIN[1]]["ordered"] else
        "; ".join(f"<b>{n}</b> " + ("does" if STAT[n]["ordered"] else "does <em>not</em>")
                  + " place slower modes on the longer-held bits" for n in MAIN) + ".")
     + " Whether the points track the diagonal in the right panel is the whole question: "
       "a run that allocates timescales to match the task lands on it, and one that does "
       "not, does not.",
     [coupling_fig("linear"), coupling_fig("tanh")]),
    ("Projections onto the coupled modes",
     "The memory-task counterpart of the sine report's eigenplanes. There the coupled mode "
     "was a complex pair and the trajectory drew a circle in its plane; here the coupled "
     "modes are real decay modes, so the picture is the scalar coefficient against time. "
     "In the linear network it steps at that bit's pulses and leaks back between them, at "
     "the rate its own eigenvalue sets. The third panel carries the necessary caveat: how "
     "much of the readout the dominant mode actually accounts for, and how many modes it "
     "takes to reach 90%. Here the dominant mode carries "
     + "; ".join(f"<b>{n}</b> {rng_str(STAT[n]['share'])}" for n in MAIN)
     + " of the readout variance, and 90% needs "
     + "; ".join(f"<b>{n}</b> {min(STAT[n]['n90'])}–{max(STAT[n]['n90'])} modes"
                 for n in MAIN)
     + " out of the origin Jacobian's basis. When that count is small the assignment is "
       "close to a genuine low-rank description; when it runs to the hundreds the "
       "dominant mode only <em>sets the timescale</em> and most of the readout lives "
       "elsewhere.",
     [projection_fig("linear"), projection_fig("tanh", "dmd"), rank_fig()]),
    ("Does any basis work for the nonlinear network?",
     "The sine report found that the origin Jacobian was the wrong basis for the tanh "
     "network and that DMD recovered the right one. This panel asks whether that rescue "
     "happens here, by repeating the assignment in three candidate bases: the Jacobian at "
     "the origin, the Jacobian at a settled memory state, and DMD fitted to the driven "
     "trial. Points on the diagonal mean the basis found modes whose timescales match the "
     "task; points marked ★ are off scale because their mode has |λ| ≥ 1 and so an "
     "infinite timescale — which is what a genuine attractor looks like through a linear "
     "lens, and is a reason no mode timescale can express that kind of memory.",
     [basis_compare_fig()]),
    ("How long the memory survives",
     f"All {len(PATTERNS)} patterns are written with a single pulse, then the input is set "
     f"to zero and the readout is scored against the pattern that was written. A bit that "
     f"sits at 1.0 is held indefinitely; one that falls to 0.5 has decayed to chance. "
     f"After {T_HOLD} steps: "
     + "; ".join(f"<b>{n}</b> {HELD_STR[n]}" for n in MAIN)
     + ". Note the networks are only ever trained on holds drawn from the pulse "
       "statistics, so the right-hand end of these curves is beyond what training "
       "required.",
     [retention_fig()]),
    ("Why: one fixed point versus several",
     f"A linear autonomous network has exactly one fixed point, the origin, so no bit can "
     f"be held forever: if every eigenvalue is inside the unit circle then everything "
     f"decays. Counting eigenvalues outside the unit circle — <b>{LIN}</b> at the origin: "
     f"{STAT[LIN]['n_out_origin'][0]} (max |λ| = {STAT[LIN]['n_out_origin'][1]:.4f}); "
     f"<b>{TAN}</b> at the origin: {STAT[TAN]['n_out_origin'][0]} "
     f"(max |λ| = {STAT[TAN]['n_out_origin'][1]:.4f}); <b>{TAN}</b> at a settled memory "
     f"state: {STAT[TAN]['n_out_mem'][0]} "
     f"(max |λ| = {STAT[TAN]['n_out_mem'][1]:.4f}). An unstable origin together with a "
     f"contracting memory state is the signature of genuine attractors holding the bits — "
     f"the memory-task counterpart of the neutral-versus-attracting distinction the sine "
     f"report found with Floquet multipliers.",
     [spectrum_fig()]),
    ("Where the written patterns end up",
     f"Each of the {len(PATTERNS)} written patterns after {T_HOLD} steps of hold, "
     f"projected onto the readout directions of the two bits that separate most. The "
     f"{MAIN[0]} run collapses them onto "
     f"{'a single point' if n_lin == 1 else f'{n_lin} points'}; the {MAIN[1]} run keeps "
     f"{n_tanh} distinct states. A run that retains k bits should show 2^k occupied "
     f"corners, so the count is a direct read-out of how many memories survive.",
     [memory_states_fig()]),
    ("Trajectory of a driven trial",
     "The leading principal components of the hidden state during an ordinary "
     "stimulus-driven trial, recolourable by elapsed time or by the current value of any "
     "one bit. Separation is graded rather than all-or-nothing; measured as Cohen's d "
     "across PC1–3 it is "
     + "; ".join(f"<b>{n}</b> " + ", ".join(f"bit {b} {STAT[n]['cohen'][b]:.1f}"
                                            for b in range(N_BITS)) for n in MAIN)
     + ". Read these planes with the dimensionality in mind — PC1–3 capture only "
     + " and ".join(f"{STAT[n]['pcvar']:.0%} ({n})" for n in MAIN)
     + " of a driven trial's variance, nothing like the flat attractor of the sine task, "
       "so a bit that does not visibly split the cloud here is not thereby absent. The PC "
       "planes come first because they render everywhere; the rotatable 3D views below "
       "need WebGL and are blank in viewers that do not provide it.",
     [pca2d_fig(MAIN[0]), pca2d_fig(MAIN[1]), embed3d(MAIN[0]), embed3d(MAIN[1])]),
    ("Readout geometry: eigendirections and decision boundaries",
     "Two overlays on the same cloud, for the linear run only. The thick bars are the "
     "dominant Jacobian eigendirection for each bit, drawn through the mean; the "
     "translucent sheets are that bit's readout decision boundary. Both are exact only "
     "<em>within</em> the displayed slice, and the legend says how much that costs: each "
     "eigendirection is labelled with the fraction of its norm that survives projection "
     "into PC1–3, and each boundary with how often the side of the plane a point falls on "
     "actually matches the bit. A direction with a small in-slice fraction is mostly "
     "pointing out of the picture, and a boundary well below 100% is being crossed by "
     "readout contributions from components the projection discards. This is drawn only "
     "for the linear run because the readout consumes φ(h): with the identity that is h "
     "itself, so the boundary is a genuine hyperplane in the space plotted, whereas under "
     "tanh it is curved here and a plane would be a fiction.",
     [readout_geometry_fig(MAIN[0])]),
    ("Recovery from a kick",
     "A random perturbation of 25% of the state norm is applied to an already-written "
     "state, then the network runs on with zero input. The nonlinear network's stable bits "
     "are pulled back and the readout recovers; the linear network has nothing to be "
     "pulled back to, and simply continues its decay.",
     [perturbation_fig()]),
    ("Dependence on the gain",
     f"The same two measurements for every run supplied: how many distinct states survive "
     f"a {T_HOLD}-step hold, and how many bits are still read correctly at the end. A "
     f"linear network has one fixed point whatever else changes, as it must; any count "
     f"above one there means the end states are still drifting rather than settled. "
     f"Median relative drift per step was "
     + "; ".join(f"<b>{n}</b> {np.median(D[n]['drift']):.1e}" for n in RUNS)
     + " — below about 1e-6 these are fixed points to machine precision, and well above "
       "it they are better described as a slow manifold.",
     [gain_fig()]),
]

def _describe(name):
    cfg = D[name]["cfg"]
    bits = [f"{name} = {cfg['activation']}/{cfg['dynamics_type']}"]
    if cfg.get("recurrent_parameterization") == "clark":
        bits.append(f"γ = {cfg.get('output_coupling_gamma')}")
        bits.append(f"s = {cfg.get('wrec_init_scale')}")
    bits.append(f"g = {cfg.get('recurrent_gain')}")
    return " · ".join(str(b) for b in bits)


CHIPS = [
    f"{N_BITS}-bit flip-flop · mean holds "
    + ", ".join(f"{h:.0f}" for h in HOLD_STEPS) + " steps",
    f"N = {CFG['hidden_size']} · dt = {DT} · seed {CFG.get('seed', 0)}",
    *[_describe(n) for n in MAIN],
    f"{T_HOLD}-step hold probe · {T_DRIVE}-step driven trial",
]
INTRO = (f"How a trained RNN holds {N_BITS} bits whose pulses arrive at {N_BITS} different "
         "rates, and how the linear and nonlinear regimes differ. The companion to the "
         "sine-wave report: there the network had to <em>generate</em> three timescales, "
         f"here it has to <em>remember</em> over {N_BITS}. All panels are interactive — "
         "rotate the 3D views, and use the buttons to recolour by elapsed time or by the "
         "state of any bit.")

HTML = page("Flip-flop memory geometry", INTRO, CHIPS, SECTIONS,
            "notebooks/tau_init_grid/9_flipflop_report.py")

os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
with open(OUT_HTML, "w") as f:
    f.write(HTML)
print(f"wrote {OUT_HTML}  ({os.path.getsize(OUT_HTML) / 1e6:.1f} MB)")
