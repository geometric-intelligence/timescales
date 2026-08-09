# %% [markdown]
# # Sine-wave hidden-state trajectories: three views of the same orbit
#
# Uniform time-constant initialization. The sine-wave task is autonomous (the input is
# identically zero), so the trajectory is generated entirely by the recurrent dynamics
# from a fixed initial state.
#
# Set `RUN_DIR` to switch between the linear (Identity/rate) and nonlinear (Tanh/voltage)
# network; figures are labelled by regime. The two behave very differently, and the
# reconstruction check in section 5 is what tells you which case you are in.
#
# Three projections of the same trajectory:
#
# 1. **Per-frequency eigenplanes.** For each cos/sine output pair we take the eigenmode
#    that pair couples to most strongly and plot the trajectory in that mode's own 2D
#    plane — the coefficient $c_j(t) = (V^{-1}h(t))_j$, drawn as $(\mathrm{Re}\,c_j,
#    \mathrm{Im}\,c_j)$. Note the inverse: $V$ is not orthogonal, so $V^\top$ would not
#    give the coefficient of $h$ in the eigenbasis.
# 2. **3D PCA**, the variance-ordered view.
# 3. **3D UMAP**, a nonlinear embedding. Section 3 shows the trajectory is confined to a
#    6-dimensional *linear* subspace, so UMAP has no curvature to unfold here and is
#    reported with that caveat rather than as the primary view.
#
# Each is shown with two colourings: elapsed **time**, and the **phase of the target
# pair**. Because `random_phase=False` and `dt=1`, the target phase is known exactly,
# $\varphi_k(t) = 2\pi t / T_k$, so phase is computed rather than estimated. Phase is
# cyclic, so it uses a cyclic colormap — a sequential one would put a false seam at
# $2\pi$.

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# `train` is a top-level module inside the inner package directory (it does
# `from callbacks import ...`), so that directory has to be on the path, not just the
# repo root. Run this notebook from the inner `timescales` directory.
for _cand in (os.getcwd(), os.path.abspath(os.path.join(os.getcwd(), ".."))):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

RUN = os.environ.get(
    "RUN_DIR",
    os.path.join(os.getcwd(), "logs", "experiments", "tau_init_grid",
                 "sine_uniform_full_g0.9", "seed_0"))
FIGS_DIR = os.environ.get(
    "FIGS_DIR", os.path.join(os.getcwd(), "notebooks", "tau_init_grid", "figs"))
os.makedirs(FIGS_DIR, exist_ok=True)
# Label figures by regime so running both does not overwrite one with the other.
LABEL = "tanh" if "tanh_voltage" in RUN else "linear"

# The network was trained on 100 steps. Rolling out longer gives UMAP enough points and
# shows more revolutions, at the cost of slow amplitude drift where |lambda| != 1 exactly
# (over 600 steps the |lambda| = 0.9995 mode decays to ~74% of its amplitude).
T_ROLL = 600
T_TRAIN = 100
CMAP_TIME, CMAP_PHASE = "viridis", "twilight"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})

# %% [markdown]
# ## Roll out the trained network and build the three coordinate systems

# %%
with open(os.path.join(RUN, "config_seed0.yaml")) as f:
    config = yaml.safe_load(f)
PERIODS = [float(p) for p in config["periods"]]
DT = float(config["dt"])

from train import _create_rnn_model  # noqa: E402

model, _ = _create_rnn_model(config)
model.load_state_dict(
    torch.load(os.path.join(RUN, "final_model_seed0.pth"), map_location="cpu",
               weights_only=False))
model.eval()

# Autonomous rollout: zero input, the initial state the task specifies.
inputs = torch.zeros(1, T_ROLL, config["input_size"])
h0 = torch.full((1, config["hidden_size"]), float(config["init_hidden_value"]))
with torch.no_grad():
    H, Y = model(inputs=inputs, init_hidden=h0)
H = H[0].numpy()          # (T, N)
Y = Y[0].numpy()          # (T, n_out)

blob = torch.load(os.path.join(RUN, "spectral_final.pt"), map_location="cpu",
                  weights_only=False)
V = np.asarray(blob["V"], dtype=complex)
lam = np.asarray(blob["eigvals_eig"], dtype=complex)
W_out = np.asarray(blob["W_out"], dtype=float)

# Coefficients in the eigenbasis, and the mode each output couples to most strongly.
C = np.linalg.solve(V, H.T.astype(complex)).T        # (T, N) complex
coup = np.abs(W_out @ V)
top = np.argmax(coup, axis=1)

# Analytic target phase per pair; cyclic in [0, 2pi).
t = np.arange(T_ROLL) * DT
phase = {k: (2 * np.pi * t / P) % (2 * np.pi) for k, P in enumerate(PERIODS)}

# PCA of the hidden states.
mu = H.mean(0, keepdims=True)
_, S, Vt = np.linalg.svd(H - mu, full_matrices=False)
evr = S**2 / (S**2).sum()
P_pca = (H - mu) @ Vt[:3].T

print(f"rollout {T_ROLL} steps (trained on {T_TRAIN})   hidden {H.shape}")
print(f"PCA: PC1-3 = {100 * evr[:3].sum():.1f}% of variance")
print("\noutput pair -> most strongly coupled mode:")
for k, P in enumerate(PERIODS):
    j = top[2 * k]
    per = 2 * np.pi / abs(np.angle(lam[j])) if abs(np.angle(lam[j])) > 1e-9 else np.inf
    same = "same mode" if top[2 * k] == top[2 * k + 1] else "DIFFERENT modes"
    print(f"  pair {k} (target T={P:g}): outputs {2 * k},{2 * k + 1} -> mode {j}  "
          f"|λ|={abs(lam[j]):.4f}  period={per:.2f}  ({same})")

# %% [markdown]
# ## 1 — Per-frequency eigenplanes
#
# Each column is one frequency pair, in the plane of the eigenmode that pair reads from.
# Top row is coloured by time, bottom row by that pair's target phase.

# %%
fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.0))
for k, P in enumerate(PERIODS):
    j = top[2 * k]
    c = C[:, j]
    for row, (col_by, cmap, lbl) in enumerate((
            (t, CMAP_TIME, "time (steps)"),
            (phase[k], CMAP_PHASE, f"target phase, T={P:g}"))):
        ax = axes[row][k]
        sc = ax.scatter(c.real, c.imag, c=col_by, cmap=cmap, s=4, alpha=0.9)
        ax.scatter(c.real[0], c.imag[0], color="crimson", s=40, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(r"Re $c_j$", fontsize=8)
        ax.set_ylabel(r"Im $c_j$", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(sc, ax=ax, fraction=0.046, label=lbl)
        if row == 0:
            per = 2 * np.pi / abs(np.angle(lam[j]))
            ax.set_title(f"pair {k}: target T={P:g}\nmode {j}, period {per:.1f}, "
                         f"|λ|={abs(lam[j]):.4f}", fontsize=9)
fig.suptitle("1 — Trajectory in the eigenplane of each output-coupled mode "
             f"({LABEL}, uniform init, g=0.9; {T_ROLL}-step rollout, red = start)",
             fontsize=10.5, y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, f"sine1_eigenplanes_{LABEL}.png"))
plt.show()

# %% [markdown]
# **Reading it.** The projection is only meaningful if these modes actually carry the
# output, which section 5 tests. Two very different pictures are possible.
#
# *Linear network.* Each panel is a circle traversed at that pair's own frequency, and the
# phase colouring completes exactly one cycle around the ring — the colour wheel maps onto
# the circle once, which is the signature of a one-to-one assignment. Two features there
# are real rather than artifacts: the T=20 panel lands on ~20 discrete angular positions
# because the mode's period is exactly 20.00 steps, so with unit time steps the trajectory
# revisits the same 20 phases every revolution; and the T=100 panel spirals slowly inward
# because that mode sits just inside the unit circle (|λ| = 0.9995), losing ~26% of its
# amplitude over 600 steps — invisible within the 100-step training window.
#
# *Nonlinear network.* The panels collapse into blobs with no circular structure, the two
# channels of a pair select *different* modes, and the mode periods bear no relation to
# the targets. This is the expected outcome, not a bug: the eigendecomposition here is of
# the Jacobian **linearized at the origin**, while a trained Tanh network operates on a
# limit cycle far from the origin, where that linearization does not describe the flow.
# The origin eigenmodes are simply the wrong coordinate system for it.

# %% [markdown]
# ## 2 — 3D PCA

# %%
def line3d(ax, P, color_by, cmap, lw=1.1):
    """Path coloured by an arbitrary quantity, so direction and structure both show."""
    pts = P[:, :3].reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, cmap=cmap, linewidths=lw)
    lc.set_array(color_by[:-1])
    ax.add_collection3d(lc)
    for dim, setlim in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        lo, hi = P[:, dim].min(), P[:, dim].max()
        pad = 0.08 * max(hi - lo, 1e-9)
        setlim(lo - pad, hi + pad)
    return lc


SLOW = len(PERIODS) - 1   # colour the global views by the slowest pair's phase

fig = plt.figure(figsize=(11.0, 4.6))
for i, (col_by, cmap, lbl) in enumerate((
        (t, CMAP_TIME, "time (steps)"),
        (phase[SLOW], CMAP_PHASE, f"target phase, T={PERIODS[SLOW]:g}"))):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    lc = line3d(ax, P_pca, col_by, cmap)
    ax.scatter(*P_pca[0, :3], color="crimson", s=40, depthshade=False, zorder=5)
    for setter, name, k in ((ax.set_xlabel, "PC1", 0), (ax.set_ylabel, "PC2", 1),
                            (ax.set_zlabel, "PC3", 2)):
        setter(f"{name} ({100 * evr[k]:.0f}%)", fontsize=8)
    ax.tick_params(labelsize=6)
    fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.10, label=lbl)
fig.suptitle(f"2 — 3D PCA of the hidden state, {LABEL}  "
             f"(PC1-3 = {100 * evr[:3].sum():.1f}% of variance)", fontsize=10.5, y=1.00)
fig.subplots_adjust(left=0.02, right=0.94, top=0.88, bottom=0.05, wspace=0.22)
fig.savefig(os.path.join(FIGS_DIR, f"sine2_pca_3d_{LABEL}.png"))
plt.show()

# %% [markdown]
# ## 3 — How many dimensions does the trajectory actually occupy?
#
# Worth establishing before reaching for a nonlinear embedding. Three complex-conjugate
# mode pairs span a **6-dimensional real subspace**, and if the output is carried entirely
# by those three modes the hidden trajectory should be confined to it.

# %%
print("Cumulative explained variance of the hidden trajectory:")
for k in range(1, 8):
    print(f"   PC1-{k}: {100 * evr[:k].sum():11.5f}%")
print(f"\n   variance beyond PC6: {100 * evr[6:].sum():.3f}%")
print("   (residual is the transient from the other 509 modes, which decays)")

# %% [markdown]
# The trajectory is confined to a 6-dimensional **linear** subspace to within ~1%. That
# is worth keeping in mind for what follows: the object is a 3-torus — three
# incommensurate rotations — sitting inside a flat 6D subspace, not a curved manifold.
#
# ## 4 — 3D UMAP
#
# UMAP is fit on the raw 512-dimensional states. Given the result above, it is being asked
# to compress a 3-torus from 6 dimensions into 3, which cannot be done faithfully: some
# structure has to be discarded. It is included because it was asked for and because the
# failure mode is informative, but PCA is the more honest view of this particular object.
#
# Read the two colourings against each other. If the embedding preserved the trajectory,
# **time** would vary smoothly along the curve. Instead time comes out interleaved while
# the slow **phase** is monotonic — UMAP has collapsed the six revolutions onto a single
# loop parameterized by the slowest frequency, discarding the two faster ones. The
# embedding is also parameter-sensitive; `n_neighbors` between 10 and 100 changes the
# shape without changing that conclusion.

# %%
import umap  # noqa: E402

reducer = umap.UMAP(n_components=3, n_neighbors=40, min_dist=0.30,
                    metric="euclidean", random_state=0)
E = reducer.fit_transform(H)
print(f"UMAP embedding: {E.shape}")

fig = plt.figure(figsize=(11.0, 4.6))
for i, (col_by, cmap, lbl) in enumerate((
        (t, CMAP_TIME, "time (steps)"),
        (phase[SLOW], CMAP_PHASE, f"target phase, T={PERIODS[SLOW]:g}"))):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    sc = ax.scatter(E[:, 0], E[:, 1], E[:, 2], c=col_by, cmap=cmap, s=5,
                    depthshade=False)
    ax.scatter(*E[0], color="crimson", s=40, depthshade=False, zorder=5)
    for setter, name in ((ax.set_xlabel, "UMAP1"), (ax.set_ylabel, "UMAP2"),
                         (ax.set_zlabel, "UMAP3")):
        setter(name, fontsize=8)
    ax.tick_params(labelsize=6)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.10, label=lbl)
fig.suptitle(f"4 — 3D UMAP of the hidden state, {LABEL} (fit on all 512 dimensions)",
             fontsize=10.5, y=1.00)
fig.subplots_adjust(left=0.02, right=0.94, top=0.88, bottom=0.05, wspace=0.22)
fig.savefig(os.path.join(FIGS_DIR, f"sine3_umap_3d_{LABEL}.png"))
plt.show()

# %% [markdown]
# ## 5 — Reconstruction check
#
# The projections above are only meaningful if the coupled modes really do carry the
# output. Reconstructing each channel from its own mode alone and comparing against the
# network's actual output tests that directly, and is what separates the two regimes:
# R² near 1 means the single mode accounts for the channel, while R² near 0 (or negative,
# meaning worse than predicting the mean) means the eigenplane view above should not be
# read as the network's working coordinate system.

# %%
print("Per-channel reconstruction from the single coupled mode (R^2 over the rollout):\n")
for k, P in enumerate(PERIODS):
    for ch in (2 * k, 2 * k + 1):
        j = top[ch]
        # contribution of mode j (and its conjugate) to output channel ch
        rec = 2.0 * np.real((W_out[ch] @ V[:, j]) * C[:, j])
        ss = 1.0 - np.var(Y[:, ch] - rec) / max(np.var(Y[:, ch]), 1e-12)
        print(f"  pair {k} (T={P:g}) channel {ch}: mode {j}  R² = {ss:+.3f}")

# %% [markdown]
# ## 6 — Coloured by each target phase in turn
#
# The task has three frequencies, and each one induces its own phase. Colouring the same
# embedding by each phase separately shows which of them the geometry is organized by:
# a phase that varies smoothly around the structure is one the embedding resolves, while
# a phase that looks scrambled is one it has folded away.

# %%
COLORINGS = [("time (steps)", t, CMAP_TIME)] + [
    (f"target phase, T={P:g}", phase[k], CMAP_PHASE) for k, P in enumerate(PERIODS)]

for name, E3, evr3 in (("pca", P_pca, evr), ("umap", E, None)):
    fig = plt.figure(figsize=(4.0 * len(COLORINGS), 4.2))
    for i, (lbl, vals, cmap) in enumerate(COLORINGS):
        ax = fig.add_subplot(1, len(COLORINGS), i + 1, projection="3d")
        sc = ax.scatter(E3[:, 0], E3[:, 1], E3[:, 2], c=vals, cmap=cmap, s=4,
                        depthshade=False)
        ax.scatter(*E3[0, :3], color="crimson", s=34, depthshade=False, zorder=5)
        ax.set_title(lbl, fontsize=9)
        base = "PC" if name == "pca" else "UMAP"
        for d, setter in enumerate((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel)):
            extra = f" ({100 * evr3[d]:.0f}%)" if evr3 is not None else ""
            setter(f"{base}{d + 1}{extra}", fontsize=7)
        ax.tick_params(labelsize=6)
        fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.16, shrink=0.72)
    fig.suptitle(f"6 — 3D {base} of the hidden state, {LABEL}, coloured four ways",
                 fontsize=11, y=1.00)
    fig.subplots_adjust(left=0.02, right=0.96, top=0.86, bottom=0.04, wspace=0.28)
    fig.savefig(os.path.join(FIGS_DIR, f"sine6_{name}_phases_{LABEL}.png"))
    plt.show()

# %% [markdown]
# ## 7 — Interactive versions, written to HTML
#
# The 3D structure is much easier to judge when it can be rotated, so the same two
# embeddings are written out as standalone interactive pages. Each has a dropdown that
# switches the colouring between elapsed time and the three target phases, so the
# comparison in section 6 can be made by eye on the same rotated view.
#
# Plotly is embedded in the file rather than loaded from a CDN, so the pages work offline.

# %%
import plotly.graph_objects as go  # noqa: E402
from plotly.colors import cyclical  # noqa: E402

PLOTLY_CMAP = {CMAP_TIME: "Viridis", CMAP_PHASE: cyclical.Twilight}


def interactive_3d(E3, axis_base, evr3=None):
    """Scatter3d with a dropdown that swaps which quantity colours the points."""
    def axis_label(d):
        extra = f" ({100 * evr3[d]:.0f}%)" if evr3 is not None else ""
        return f"{axis_base}{d + 1}{extra}"

    lbl0, vals0, cmap0 = COLORINGS[0]
    fig = go.Figure(go.Scatter3d(
        x=E3[:, 0], y=E3[:, 1], z=E3[:, 2], mode="markers",
        marker=dict(size=2.6, color=vals0, colorscale=PLOTLY_CMAP[cmap0],
                    colorbar=dict(title=lbl0), opacity=0.9),
        hovertemplate="step %{customdata}<extra></extra>",
        customdata=np.arange(len(E3)),
        name="trajectory"))
    # mark the start so the direction of travel is unambiguous; no legend entry,
    # which would otherwise sit on top of the colorbar title
    fig.add_trace(go.Scatter3d(
        x=E3[:1, 0], y=E3[:1, 1], z=E3[:1, 2], mode="markers",
        marker=dict(size=7, color="crimson"), name="start", showlegend=False))

    buttons = [dict(label=lbl, method="restyle",
                    args=[{"marker.color": [vals],
                           "marker.colorscale": [PLOTLY_CMAP[cmap]],
                           "marker.colorbar.title": lbl}, [0]])
               for lbl, vals, cmap in COLORINGS]
    # Titles are rendered as HTML headings instead of plotly titles, so the dropdown
    # has clear space above the scene rather than overlapping the title text.
    fig.update_layout(
        height=640, showlegend=False,
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                          x=0.0, xanchor="left", y=1.0, yanchor="bottom",
                          pad=dict(b=6))],
        scene=dict(xaxis_title=axis_label(0), yaxis_title=axis_label(1),
                   zaxis_title=axis_label(2)),
        margin=dict(l=0, r=0, t=54, b=0))
    return fig


figs = [
    (f"3D PCA (PC1-3 = {100 * evr[:3].sum():.1f}% of variance)",
     interactive_3d(P_pca, "PC", evr)),
    ("3D UMAP (fit on all 512 dimensions)", interactive_3d(E, "UMAP")),
]
html_path = os.path.join(FIGS_DIR, f"sine_interactive_{LABEL}.html")
with open(html_path, "w") as f:
    f.write(f"<html><head><meta charset='utf-8'><title>sine-wave trajectories "
            f"({LABEL})</title></head><body style='font-family:system-ui;margin:24px'>")
    f.write(f"<h2>Sine-wave hidden-state trajectories — {LABEL} network</h2>"
            f"<p>Uniform time-constant initialization, g = 0.9, seed 0. "
            f"{T_ROLL}-step autonomous rollout (trained on {T_TRAIN}). "
            f"Use the dropdown on each plot to switch the colouring; drag to rotate.</p>")
    for i, (heading, fg) in enumerate(figs):
        f.write(f"<h3 style='margin:28px 0 0'>{heading}</h3>")
        f.write(fg.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)))
    f.write("</body></html>")
print(f"wrote {html_path}")
