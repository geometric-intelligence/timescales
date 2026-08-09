# Trajectory geometry of trained sine-wave RNNs — running notes

Working notes on how trained RNNs represent the heterogeneous sine-wave task. Meant to be
edited and extended as we go, not a finished document. Findings are numbered so they can
be referred to; each records what was asked, how it was measured, and what came out,
including the ones that turned out to be wrong.

**Setup for everything below unless stated otherwise.** Sine-wave task, 3 cos/sine pairs
at periods T = 20, 50, 100 steps, dt = 1, zero input (autonomous generation), fixed
initial state h₀ = 1. Uniform time-constant initialization, g = 0.9, seed 0, N = 512, all
parameters trainable. Two regimes, identical apart from two settings:

| | linear | nonlinear |
|---|---|---|
| activation | Identity | Tanh |
| dynamics | rate | voltage |
| update | h⁺ = (1−α)h + α·gWh | h⁺ = (1−α)h + α·gW·tanh(h) |

Networks are rolled out 600–800 steps although trained on 100, which is fine for the
autonomous task but means slow drift accumulates — see finding 8.

**Reproduce:** `notebooks/tau_init_grid/5_sine_projections.py` (projections, per-phase
colourings, interactive HTML), `6_effective_modes_and_geometry.py` (DMD, torus),
`7_perturbations_and_fixed_points.py` (findings 5–7).

**Interactive report:** `notebooks/tau_init_grid/8_interactive_report.py` writes
`figs/sine_interactive_report.html`, a single self-contained page (Plotly's JS is embedded,
so it works offline and can be moved anywhere) collecting the panels behind findings 1–3,
5, 7 and 9–11: eigenplanes, the phase torus, PCA with the variance spectrum, UMAP, Floquet
multipliers, amplitude from random starts, and the eigenplane-restricted perturbation shown
both as an amplitude ratio and as the orbit itself. Every scatter panel has buttons to
recolour by elapsed time or by the phase of any one of the three target frequencies.

**Two items flagged for review** (also in `todos.md`): the Floquet/monodromy machinery
behind findings 3, 7 and 10, and the claim in findings 7 and 11 that phase is a neutral
direction. Findings that depend on them are marked.

---

## 0. Definitions used throughout

Stated explicitly because several findings are quoted as ratios of these quantities.

**Rank-r DMD.** Stack the centred states of the reference trajectory as snapshot matrices
X = [h₀ … h_{T−2}]ᵀ and X′ = [h₁ … h_{T−1}]ᵀ, each (N × T−1), after subtracting the time
mean μ of the reference. Take the SVD X = UΣVᵀ and keep the leading r = 6 components
(Uᵣ, Σᵣ, Vᵣ). The reduced operator is

    Ã = Uᵣᵀ X′ Vᵣ Σᵣ⁻¹        (a 6×6 matrix)

with eigendecomposition Ã w = λ w. The DMD modes are Φ = X′ Vᵣ Σᵣ⁻¹ W, an (N × 6) matrix
whose columns Φ_k are directions in state space, each with its own eigenvalue λ_k.

**Why r = 6.** The normalized singular values of the snapshot matrix are

    1.000, 0.991, 0.918, 0.856, 0.786, 0.767, | 0.034, 0.033, 0.032, 0.031, …

a 22× drop between the 6th and 7th. Six is where the data stops supporting more, matching
the 6-dimensional trajectory of finding 1 (three complex-conjugate pairs).

**The coefficient b_k(t).** Projecting a state into that basis,

    b(t) = Φ⁺ (h(t) − μ)          (Φ⁺ = Moore–Penrose pseudoinverse, b ∈ ℂ⁶)

so that h(t) ≈ μ + Σ_k Φ_k b_k(t). Each b_k is complex; **|b_k(t)| is its modulus**, the
instantaneous amplitude of mode k. Under the ideal linear evolution b_k(t) = b_k(0)·λ_kᵗ,
so |b_k(t)| = |b_k(0)|·|λ_k|ᵗ — constant when |λ_k| = 1. That is what makes it the right
"amplitude" observable, unlike ‖h‖, which oscillates as the state moves around the orbit.

Modes come in conjugate pairs and we index one per pair; the physical contribution of a
pair is 2·Re(Φ_k b_k), so |b_k| differs from the physical amplitude by a fixed factor. It
cancels in every ratio below.

**The amplitude ratio.** For a perturbed rollout h^p and the unperturbed reference h^r,
both projected with the **same** Φ and μ (fitted on the reference),

    ratio_k(t) = |b_k^p(t)| / |b_k^r(t)| = |[Φ⁺(h^p(t) − μ)]_k| / |[Φ⁺(h^r(t) − μ)]_k|

Comparing at equal t means any global drift |λ_k|ᵗ affects numerator and denominator alike
and cancels, so the ratio isolates the perturbation's effect. It is dimensionless and
invariant to the scaling of Φ. A value of 1 means the perturbed mode has recovered the
reference amplitude.

**Phase offset** is the argument of the same quantity, arg(b_k^p(t) / b_k^r(t)).

---

## 1. The hidden trajectory is 6-dimensional and essentially flat — in *both* regimes

Cumulative PCA variance of the hidden states:

| regime | PC1–3 | PC1–6 | beyond PC6 |
|---|---|---|---|
| linear | 68.1% | **98.71%** | 1.29% |
| nonlinear | 58.9% | **99.31%** | 0.69% |

Three complex-conjugate mode pairs span a 6-dimensional real subspace, and that is where
the trajectory lives. The residual is the transient from the other ~509 modes, which
decays.

This was a surprise for the nonlinear network: the dimensionality is the same as the
linear one. Whatever the nonlinearity is doing, it is not adding dimensions.

Consequence: there is no curved manifold for a nonlinear embedding to unfold, which is why
UMAP does not help here (finding 9).

## 2. In the linear network, each output pair reads exactly one Jacobian eigenmode

Taking, for each output channel, the eigenmode it couples to most strongly
(argmaxⱼ |W_out V|ₖⱼ) and reading off that mode's period:

| output pair | target period | mode | measured period | \|λ\| |
|---|---|---|---|---|
| 0 (ch 0,1) | 20 | 0 | 20.00 | 1.0001 |
| 1 (ch 2,3) | 50 | 2 | 49.97 | 1.0001 |
| 2 (ch 4,5) | 100 | 4 | 99.74 | 0.9995 |

Both channels of a pair select the *same* mode — three modes for six channels. All sit
essentially on the unit circle, which is what sustains the oscillation.

**Validation:** reconstructing each output channel from that single mode alone gives
**R² = 1.000** on all six channels. So this is not just "a mode with the right frequency
exists" — that one mode fully accounts for the channel.

In the mode's own plane the trajectory is a circle, and colouring by the corresponding
target phase wraps the colour wheel exactly once around it.

Two features that look like artifacts but are not: the T = 20 panel lands on ~20 discrete
angles because its period is exactly 20.00 steps and the sampling interval is 1; the
T = 100 panel spirals slowly inward because |λ| = 0.9995 costs ~26% amplitude over 600
steps (invisible inside the 100-step training window).

## 3. In the nonlinear network the origin Jacobian is the wrong basis — and this is why

Same construction on the Tanh network gives nothing: the two channels of a pair select
*different* modes, the periods (7.2, 2.0, 9.3) are unrelated to the targets, and
single-mode reconstruction gives R² between −0.30 and 0.00.

The reason is mechanical. For voltage dynamics the Jacobian is

    J(h) = diag(1−α) + diag(α)·gW·diag(1 − tanh²(h))

At the origin tanh′ = 1, and this is exactly the Jacobian stored by
`SpectralSnapshotCallback` (verified: reconstruction matches the stored J to 1.3e-7). But
a trained Tanh network never goes near the origin — it runs on a limit cycle where
tanh′(h) ≈ 0.87, which *reduces* the effective recurrent gain.

| | max\|λ\| at origin | max\|λ\| along trajectory | mean tanh′(h) |
|---|---|---|---|
| linear | 1.0001 | 1.0001 (identical — J is constant) | — |
| nonlinear | **1.1109** | **0.9686 ± 0.0110** | 0.869 |

**Precisely what the middle column is.** At each of 10 sampled steps t ∈ {400, 440, …,
760} we form the full 512×512 instantaneous Jacobian J(h(t)), take all 512 eigenvalues,
and keep the largest modulus. The table reports mean ± s.d. of those 10 numbers, so the
spread is variation *along the orbit*, not a statistical error.

**Caveat — this quantity does not establish contraction.** The J(h(t)) do not commute, and
a product of matrices each with spectral radius < 1 can still grow. Instantaneous
eigenvalues are suggestive only; the object that actually governs stability around a cycle
is the monodromy matrix (finding 10). An earlier phrasing of this note said the nonlinear
network is "contracting on the attractor" on the strength of 0.9686 — that inference was
not licensed, though finding 10 happens to support the conclusion.

So the origin is *unstable* in the nonlinear network, and saturation pulls the effective
gain down. Evaluating J on the trajectory instead of at the origin already recovers periods
of 20.7 / 56.7 / 98.1 — close to the targets.

**This supersedes an earlier statement** that the nonlinear network's structure "is not
describable by the eigenmodes". It is not describable by the *origin* eigenmodes.

## 4. DMD supplies the right basis, and recovers the same one-to-one structure

Fitting the best linear operator to h(t+1) ≈ A h(t) along the trajectory (rank-6 exact
DMD, mean removed) and using *its* eigenvectors:

| regime | basis | periods found | one mode per pair? | single-mode R² |
|---|---|---|---|---|
| linear | origin Jacobian | 20.00 / 49.97 / 99.74 | yes | 1.000 |
| linear | **DMD** | 19.98 / 50.12 / 99.37 | yes | 0.997–1.000 |
| nonlinear | origin Jacobian | 7.18 / 2.00 / 9.27 | **no** | −0.30 … 0.00 |
| nonlinear | **DMD** | **19.89 / 49.56 / 99.31** | **yes** | **0.93–0.98** |

In the DMD planes the Tanh trajectory traces clean circles cleanly parameterized by target
phase — the same picture as the linear network.

The linear rows are the control: DMD agrees with the Jacobian there, as it must for
genuinely linear dynamics, so the method is not manufacturing structure.

**Caveat on interpretation.** The DMD operator is *not* the Jacobian at any single point.
It is closer to a time-averaged (monodromy-like) linearization around the limit cycle:
the instantaneous J(h(t)) varies along the orbit, and DMD averages that. The instantaneous
Jacobian gets the periods approximately right (20.7 / 56.7 / 98.1); DMD gets them closer
(19.89 / 49.56 / 99.31). Not yet checked rigorously against Floquet theory.

## 5. The global geometry: a closed (5, 2, 1) curve on a 3-torus

Three circles, one per frequency, so the orbit lies on a 3-torus inside the 6-dimensional
subspace. Which curve depends on the frequency ratios: periods 20, 50, 100 have least
common multiple 100, so an exact solution closes after 100 steps having wound 5, 2 and 1
times around the three circles — a closed curve, not a space-filling one.

| regime | winding turns / 100 steps | relative ‖h(t+100) − h(t)‖ |
|---|---|---|
| linear | 5.01, 2.00, 1.01 | 0.023 |
| nonlinear | 5.03, 2.02, 1.01 | 0.126 |

Unrolling the torus (plotting two mode phases against each other, coloured by the third)
gives the expected family of parallel lines rather than a filled square.

**Both regimes have the same geometry.** Only the basis needed to see it differs.

## 6. Fixed-point structure differs sharply

Iterating the autonomous map from 8 random initial states for 4000 steps:

| regime | final ‖h‖ across 8 starts | interpretation |
|---|---|---|
| linear | 0.749 … 1.871 (spread 1.12) | amplitude set by initial condition — no attraction |
| nonlinear | 9.380 … 9.792 (spread 0.41) | **an attractor**; all starts converge to one cycle |

The linear network has a single fixed point at the origin, marginally stable
(|λ| ≈ 1.0001), so orbits are neutral: a continuum of them, each amplitude preserved
forever. The nonlinear network has an *unstable* origin (|λ| = 1.111) plus a stable limit
cycle, and the saturation in finding 3 is the mechanism that sets its amplitude.

**‖h‖ is the wrong observable here** — it oscillates as the state moves around the orbit,
so the curves look sinusoidal about a mean and the comparison is muddied. The sharp
version is the amplitude of each frequency separately, |b_k| in the DMD basis, which is
constant on the orbit. From 3 random initial states, measured at t ≈ 2500:

| regime | start | T≈20 | T≈50 | T≈100 |
|---|---|---|---|---|
| linear | 0 | 0.113 | 0.472 | 0.242 |
| linear | 1 | 0.182 | 0.323 | 0.092 |
| linear | 2 | 0.485 | 0.838 | 0.339 |
| nonlinear | 0 | 4.323 | 3.595 | 3.664 |
| nonlinear | 1 | 4.322 | 3.594 | 3.663 |
| nonlinear | 2 | 4.323 | 3.594 | 3.663 |

The nonlinear network sets **each frequency's amplitude independently, reproducible to
four significant figures** regardless of where it started. The linear network's amplitudes
are whatever the initial condition happened to supply, varying by 4–5× across starts.

## 7. Response to perturbations: the difference is in the amplitude, not the total distance

Perturbing the hidden state at t = 300 by a random direction, averaged over 5 directions,
and tracking relative distance to the unperturbed trajectory:

| regime | size | t+1 | t+25 | t+100 | t+490 | residual / initial |
|---|---|---|---|---|---|---|
| linear | 20% | 0.1011 | 0.0200 | 0.0174 | 0.0176 | 0.17× |
| nonlinear | 20% | 0.0850 | 0.0178 | 0.0123 | 0.0087 | 0.10× |

The ratio is identical across 5%/20%/50% in both regimes, so both are in a
linear-response regime around their orbit.

The naive expectation — nonlinear recovers, linear is permanently displaced — is right
about amplitude but the raw distance is a poor way to see it. **Both** damp most of a kick
quickly, and the reason is geometric rather than dynamical: a random direction in 512-D
barely intersects the 6-D task subspace. Measured over 200 random directions, the fraction
of a unit perturbation lying in that subspace is **0.1079 ± 0.0287** (linear) and
0.1069 ± 0.0316 (nonlinear), against the value expected for a random direction,
√(6/512) = **0.1083**. So ~89% of any random kick sits in fast-decaying directions and
disappears within ~25 steps in *either* network. The linear network is not correcting
anything; it is discarding the part of the perturbation that was never in the persistent
subspace to begin with.

**The decisive test is to remove that confound** by confining the perturbation to a single
mode's own 2D eigenplane, so 100% of it lands in the persistent subspace. Amplitude ratio
(perturbed/unperturbed) of the perturbed mode, at +1, +50, +200 and +580 steps:

| regime | mode | +1 | +50 | +200 | +580 |
|---|---|---|---|---|---|
| linear | T=20 | 1.2043 | 1.2044 | 1.2044 | **1.2044** |
| linear | T=50 | 1.2665 | 1.2665 | 1.2665 | **1.2661** |
| linear | T=100 | 0.5858 | 0.5917 | 0.5855 | **0.5896** |
| nonlinear | T≈20 | 1.0370 | 1.0039 | 1.0084 | **0.9993** |
| nonlinear | T≈50 | 0.9686 | 0.9578 | 1.0013 | **1.0010** |
| nonlinear | T≈100 | 1.2545 | 1.0232 | 0.9998 | **1.0006** |

The linear network shows *literally zero* correction — a 20% amplitude error is still a
20% error 580 steps later, to four decimal places. The nonlinear network returns every
mode to within 0.1%, including a 25% error on the slowest mode. This is the cleanest form
of the result, and it is the version worth showing.

The real difference shows in the *mode amplitudes*. Decomposing the residual in the DMD
basis (amplitude ratio perturbed/reference, and phase offset):

| regime | mode | amplitude just after → late | phase offset just after → late |
|---|---|---|---|
| linear | T=20 | 1.0212 → **1.0195** | +0.25° → +0.23° |
| linear | T=50 | 1.0144 → **1.0138** | +0.52° → +0.53° |
| linear | T=100 | 0.9912 → **0.9938** | +0.99° → +0.88° |
| nonlinear | T=20 | 1.0052 → **1.0000** | +1.12° → +0.90° |
| nonlinear | T=50 | 0.9862 → **0.9998** | +0.89° → +0.38° |
| nonlinear | T=100 | 1.0079 → **1.0003** | −1.48° → +0.15° |

The nonlinear network restores every mode amplitude to within 0.03% of the unperturbed
value — that is the attractor doing its work. The linear network does not: its amplitude
errors (2.0%, 1.4%, 0.6%) barely move over 480 steps, because a marginally stable linear
system has nothing that restores amplitude.

What survives in the nonlinear case is a small **phase** offset, which is expected: phase
along a limit cycle is a neutral direction (Floquet multiplier 1), so a phase shift is
never corrected. That, not incomplete contraction, is why the total residual plateaus
around 1% rather than going to zero.

**Output accuracy is a bad metric for this.** Over a 600-step rollout both networks'
output R² is dominated by their own frequency drift (e.g. period 19.89 vs 20 accumulates
~1 radian of phase error over 600 steps), which swamps the perturbation. Perturbed vs
unperturbed late R² differ by <0.01 in both regimes.

## 8. Rollout length is a confound to watch

Networks were trained on 100 steps and are rolled out 600–800. Over that horizon:
|λ| = 0.9995 costs ~26% amplitude (linear T = 100 mode); period mismatches of 0.1–0.4
steps accumulate into radians of phase error. Anything measured late in a rollout needs
to separate this from the effect being studied.

## 9. UMAP does not help here, and the failure is diagnostic

3D UMAP on the raw 512-dimensional states produces a figure-eight in both regimes. Reading
the colourings against each other shows what it did: the slow phase is monotonic around
the curve while **time is interleaved**, meaning it collapsed the six revolutions onto a
single loop and discarded the two faster frequencies. Changing `n_neighbors` from 10 to
100 changes the shape but not that conclusion.

This is consistent with finding 1: the object is a 3-torus inside a *flat* 6-dimensional
subspace. There is no curvature to unfold, and compressing 6 dimensions into 3 has to
discard something. PCA is the honest 3-dimensional view; UMAP is best kept as a negative
control.

## 10. Floquet multipliers: one neutral direction in the nonlinear network, six in the linear one

The correct stability object around a cycle is the **monodromy matrix**, the product of the
instantaneous Jacobians around one period, M = ∏ₜ J(h(t)). Its eigenvalues are the Floquet
multipliers: a multiplier of modulus 1 is a neutral direction, below 1 contracting, above 1
expanding. Computed over t = 400…499 (one approximate period of 100 steps):

| regime | top \|multipliers\| |
|---|---|
| linear | 1.0056, 1.0056, 1.0056, 1.0056, 0.951, 0.951, 0, … |
| nonlinear | **1.0245**, 0.482, 0.052, 0.039, 0.010, 0, … |

This is the crisp statement of the difference:

* The **nonlinear** network has **one** near-neutral multiplier and everything else strongly
  contracting (≤ 0.48). One neutral direction plus transverse contraction is the definition
  of an attracting limit cycle, and the neutral direction is the phase.
* The **linear** network has **six** near-neutral multipliers — the whole task subspace is
  neutral. Nothing is attracting, which is why finding 7's eigenplane perturbation is never
  corrected.

Consistency check: for the linear network J is constant, so M = J¹⁰⁰ and the multipliers
must be λ¹⁰⁰. Per-step, 1.0056^(1/100) = 1.00006, matching the max |λ| = 1.0001 quoted
(rounded) in finding 3.

Neither network's neutral multiplier is exactly 1 (deviations 5.6e-3 and 2.5e-2) because
the orbit is not exactly 100-periodic — see finding 8.

## 11. Why the phase offset is never corrected

A perturbation along the cycle is a shift in *when* the network is on its orbit, not *where*
the orbit is. The dynamics are autonomous — the update contains no reference to absolute
time — so if h*(t) is a solution then h*(t + s) is the same solution shifted, and the
system has no way to prefer one over the other. Formally this is the neutral Floquet
multiplier of finding 10: its eigenvector is the tangent to the cycle, and perturbations
along it are neither amplified nor damped.

So the nonlinear network restores the *shape* of its output (amplitudes to within 0.1%) but
cannot re-lock its *timing*. Recovering phase would need an external reference — an input
the network could synchronize to — which this task, being autonomous with zero input, never
provides. It is a property of the task, not a shortcoming of the network.

This also means "robustness to perturbation" needs care as a claim: the nonlinear network is
robust in amplitude and permanently sensitive in phase.

## 12. Is the DMD operator "the Koopman operator"? Not quite

Worth being careful about, since the terms get used loosely.

The **Koopman operator** is exact, linear and global, but acts on an infinite-dimensional
space of observables rather than on the state. What DMD computes is a finite-rank
approximation to it, restricted to the observables you supply — here the identity, i.e.
linear functions of the state — and fitted to data from one region of state space.

Two consequences for what we have:

* It is **not state-dependent**, but it is **not globally valid** either — and these are
  different things. Not state-dependent: it is one fixed matrix, applied identically
  wherever you are, unlike the true Jacobian J(h) which differs at every state. Not
  globally valid: it was *fitted* to data from one region, and nothing constrains it
  elsewhere. Measured directly, as relative one-step prediction error
  ‖predicted − true‖/‖true‖ for the nonlinear network:

  | evaluated at | error |
  |---|---|
  | states on the attractor | **0.047** |
  | random states at attractor scale | 0.974 |
  | states near the origin | **1.025** |

  Off the attractor the model is worse than useless (an error above 1 is worse than
  predicting zero). Two reasons: the fitted eigenvalues carry |λ| ≈ 1 whereas the true
  Jacobian at the origin has max |λ| = 1.111, and Φ spans only 6 of 512 dimensions, so any
  component outside that subspace is projected away and cannot be represented at all. The
  operator is linear *in form* but local *in validity*.
* For the **linear** network the distinction collapses: the Koopman operator restricted to
  linear observables *is* the system matrix, so DMD is exact and global there. That is why
  the linear rows of finding 4 agree with the Jacobian to 3 decimal places.

The defensible phrasing is "a rank-6 DMD approximation of the Koopman operator on linear
observables, valid on the attractor" — not "the Koopman operator". Whether it coincides with
the monodromy matrix of finding 10 is still open; they are different constructions that
happen to describe the same cycle.

---

## 13. Flip-flop: the same question, with fixed points instead of cycles

**Setup.** 6-bit flip-flop, N = 512, dt = 0.1, uniform τ init, g = 0.9, seed 0, both
regimes, `ff_uniform_full_g0.9`. Pulses arrive at a different rate per bit,
`p_pulse = [0.005, 0.007, 0.01, 0.02, 0.05, 0.1]`, so the mean interval a bit must be held
is 200 / 143 / 100 / 50 / 20 / 10 steps. Trained bit accuracy is 0.936 (linear) and 0.858
(nonlinear); neither run reached the convergence threshold, so these are partial solutions
and the analysis below describes what they actually learned, not what the task permits.

**13a. Each bit reads one slow mode, and the timescales are ordered.** Decomposing each
bit's readout over the eigenvectors of the autonomous Jacobian at the origin gives a clean
one-to-one assignment in the **linear** network:

| bit | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| mean hold (steps) | 200 | 143 | 100 | 50 | 20 | 10 |
| τ of the mode it reads | 145 | 110 | 71 | 41 | 17 | 11.7 |
| ratio | 0.73 | 0.77 | 0.71 | 0.83 | 0.84 | 1.17 |
| share of readout variance | 0.49 | 0.52 | 0.43 | 0.67 | 0.04 | 0.10 |

Six bits, six different modes, in the same order as the hold times, at 0.7–1.2×. This is the
memory-task counterpart of finding 2 — there each output pair read one oscillatory mode at
its target period, here each bit reads one decay mode at roughly its required hold time.

*Definition, and why it matters.* "Coupling" here is the fraction of a bit's readout
**variance** carried by that mode's own contribution c_bj·b_j(t) over a driven trial, not
the plain |W_out·V| used by `mode_analysis.pair_couplings`. The distinction is not cosmetic:
under the plain weight, bits 4 and 5 pick modes at τ = 16.8 and **3.2** against holds of 20
and 10, and the last one looks like a failure. Weighting by what the mode actually does —
a mode with a large readout weight contributes nothing if its coefficient never moves —
moves bit 5 to τ = 11.7 and makes all six line up. Both numbers are in the git history of
this file; the variance-weighted one is the defensible one.

Worth keeping separate from the older per-task mode table quoted under Open questions
(137/93/63/37/17/2.5): that came from an earlier pilot-grid sweep directory and used
the nearest-neighbour matching metric, so those numbers are not directly comparable to
either definition above, even though they show the same ordering.

**13a′. But the assignment is not a low-rank basis, unlike the sine case.** The dominant
mode carries only 43–67% of the readout variance for the four slow bits (4% and 10% for the
two fast ones), and **81–217 Jacobian modes** are needed to reach 90% of a bit's readout
variance. So the assignment identifies the mode that *sets the timescale*; it does not say
the readout lives in a 6-dimensional modal subspace the way rank-6 DMD did for the sine
task. Rank-24 DMD fitted to the driven trial *is* low-rank by construction (4–16 modes for
90%), but its timescales do not match the hold times in either regime.

**13a″. No basis rescues the nonlinear network.** In the sine case the origin Jacobian was
the wrong basis and DMD recovered the right one. That does not happen here. Across the
Jacobian at the origin, the Jacobian at a settled memory state, and rank-24 DMD, the
nonlinear network's dominant-mode timescales are unordered and off-diagonal
(origin: 3.7 / 4.6 / ∞ / ∞ / 30.7 / 6.7 against holds 200 / 143 / 100 / 50 / 20 / 10; the
two infinities are the |λ| = 1.0716 unstable pair). This is consistent with 13d: the
nonlinear network's memory is held by *which attractor it is in*, and no linear mode
timescale can express that, because an attractor's timescale is infinite.

**13b. Per-neuron time constants do not explain it.** The learned single-unit τ span only
2.9–15.4 steps, while the modes above run out to 145. The long timescales are a property of
the recurrent connectivity, not of the units.

**13c. The linear network cannot hold a bit, and does not.** An autonomous linear network
has exactly one fixed point (h = 0, since (I − gW) is non-singular here), so every mode
decays: max |λ| = 0.9931. Writing all 64 patterns with one pulse and then holding with zero
input, all 64 end states collapse onto a single point, and the readout is at chance on
*every* bit by 200 steps — fastest bit first. It scores 0.94 on the task only because
pulses keep arriving.

**13d. The nonlinear network has genuine attractors, but only a few.** At g = 0.9 the origin
is unstable (a complex pair at |λ| = 1.0716), and the 64 written patterns settle onto **4
distinct fixed points** — machine-precision fixed points, relative drift 2×10⁻⁸, with every
Jacobian eigenvalue inside the unit circle (max 0.967). Those 4 = 2² states are exactly the
combinations of the two bits it holds perfectly and indefinitely (bits 2 and 3, accuracy
1.000 at 600 steps); the other four bits decay to chance. This is the same
neutral-versus-attracting split as findings 10 and 7, in the memory setting.

**13e. Attractor count is a property of the solution, not of the nonlinearity.** At g = 0.5
the nonlinear network reaches 16 distinct end states and holds more bits, and it is the
better network (0.971 vs 0.858 bit accuracy). But its end states drift at 3×10⁻⁴ with
max |λ| = 0.9971, so they are better described as a slow manifold than as sharp attractors.
The linear network has one fixed point at either gain, as it must.

**13f. The driven trajectory is not low-dimensional, unlike the sine attractor.** PC1–3 of a
1000-step driven trial capture 62% (linear) and 70% (nonlinear) of the variance, against
the ~99% that PC1–6 captured on the sine attractor (finding 1). Colouring the PC cloud by
each bit separates it, but in a graded way: Cohen's d across PC1–3 is 0.5–2.2 for all six
bits in the linear network with none standing out, and in the nonlinear network the bit it
holds perfectly (bit 2) reaches d = 4.9 while the bits it has lost still sit at 1.2–2.1. A
bit that does not visibly split the cloud in these planes is therefore not thereby absent —
most of the variance is elsewhere.

**Reproduce:** `notebooks/tau_init_grid/9_flipflop_report.py`, which writes
`figs/flipflop_interactive_report.html`. Note the 3D panels need WebGL; the PC-plane panels
above them are plain SVG and render in any viewer.

---

## 14. Replication on the Adam / Clark discovery grid

Both reports were re-run on `rich_learning_adam_discovery_4k`, a sweep that differs from the
tau-init grid in almost every way except the tasks: Clark parameterization
(h⁺ uses (g/√N)·J·φ(h), readout V·φ(h)/(N·γ)), Adam with no scheduler and no weight decay,
**fixed** time constants rather than learned, g = 0.5 throughout, and 4000 steps. Cells
chosen by the researcher: sine at s = 0.64, γ = 1.0; flip-flop at s = 0.04, γ = 0.03. Note
the flip-flop there is **3 bits** (holds 200 / 50 / 10), not 6, and "linear" is
Identity/**voltage** — algebraically the same update as Identity/rate, so still comparable.

**14a. The sine findings replicate essentially unchanged.** Rank-6 DMD recovers periods
20.00 / 50.02 / 100.0 (linear) and 19.97 / 49.9 / 99.57 (tanh) against targets 20 / 50 / 100.
Floquet multipliers over one slow period: 6 with |μ| > 0.9 in the linear network, 1 in the
nonlinear one — the same neutral-subspace versus attracting-limit-cycle split as finding 10,
now on a different optimizer, parameterization and gain. PC1–6 capture 100.0% and 99.86% of
the variance. After an eigenplane-restricted kick the linear amplitude ratios end at
1.12 / 0.87 / 0.91 (never corrected) while the nonlinear ones end at 1.02 / 1.00 / 1.00.

**14b. The flip-flop fixed-point finding replicates; the timescale-allocation finding does
not.** The nonlinear run has 2 eigenvalues outside the unit circle at the origin and settles
into **4 = 2² distinct end states**, exactly matching the two bits it holds perfectly
(bits 0 and 1, at 1.00 after a 600-step hold; the fastest bit is at chance). The linear run
has nothing outside the unit circle and **every** bit decays to chance. That is finding 13c/13d
intact.

But the ordered bit → timescale assignment of finding 13a does not survive: dominant-mode
τ = 29.6 / 43.4 / 12.6 against holds of 200 / 50 / 10, so the longest-held bit does *not*
read the slowest mode. Two candidate reasons, neither tested: time constants are fixed here
rather than learned, and γ = 0.03 concentrates the readout enormously — 90% of a bit's
readout variance needs only 2–6 modes here versus 81–217 in the tau-init grid. The low rank
makes the assignment cleaner to state and simultaneously shows it is not aligned to the task.

**Consequence for the reports.** Every number quoted in the section text of both reports is
now computed from the loaded runs rather than written in. This was not cosmetic: pointing
the old flip-flop report at these runs would have left it asserting an ordered assignment
that is absent, "43–67%" shares that are actually 27–89%, and "roughly a hundred modes" that
are actually 2–6. One pre-existing inconsistency surfaced in the process — the sine report's
"PC1–6 capture 98.7% / 99.3%" was taken from finding 1's window, while the figure beside it
plots a different segment where the true values are 100.0% / 99.76%.

**14c. The readout directions are well aligned with the leading PCs.** Drawing each bit's
decision boundary in the PC1–3 slice of the linear run: the readout is affine in the feature
variable, so `{f : s·w_b·f = 0.5}` is a hyperplane, and intersecting it with the slice
through the mean gives a plane. Restricted to that slice the planes still put **100% / 96% /
96%** of the trajectory on the correct side for bits 0 / 1 / 2, even though PC1–3 capture
only 87.4% of the trial's variance. So almost all of the *readout-relevant* variance lives
in the top three components — a contrast with the sine case, where the trajectory subspace
sat at principal angles of 68–83° to the readout row space (see Open questions).

The dominant Jacobian eigendirections are less well contained: the fraction of each one's
norm surviving projection into PC1–3 is 67% (bit 0), 85% (bit 1), 45% (bit 2). The bit-2 bar
in particular is mostly pointing out of the picture and should not be read as a direction in
the plotted space.

**14d. Correction: readout features were computed wrongly for Identity/voltage runs.** The
flip-flop report hardcoded `phi = tanh` whenever `dynamics_type == "voltage"`, but
`RNN.readout_features` applies the run's *own* activation. Every run analysed before this
sweep was Identity/**rate** or Tanh/voltage, where the two agree, so nothing earlier is
affected. The Clark grid's linear cell is Identity/**voltage**, where they do not: |h| there
reaches 1.19 and `max |tanh(h) − h| = 0.36`, a 30% relative error. Now fixed to mirror the
model. The effect on the published numbers turned out to be negligible — dominant-mode
shares move from 0.6619/0.7612/0.2695 to 0.6584/0.7602/0.2674 and the mode counts do not
move at all, because variance *fractions* are robust to tanh's compression at these
amplitudes — but the fix matters for any run with larger states.

**Reproduce:**

```
SWEEP_ROOT=<...>/logs/experiments \
SINE_RUNS="linear=rich_learning_adam_discovery_4k/sine_linear_adam_s0.64_gamma1/seed_0,\
tanh=rich_learning_adam_discovery_4k/sine_tanh_adam_s0.64_gamma1/seed_0" \
OUT_HTML=.../figs/sine_rich_s0.64_gamma1.html \
python notebooks/tau_init_grid/8_interactive_report.py
```

and the same shape with `FF_RUNS` for `9_flipflop_report.py`.

---

## Open questions

- **DMD versus monodromy.** Finding 10 computed the Floquet multipliers and finding 4 the
  DMD eigenvalues, but they have not been compared directly. They describe the same cycle by
  different routes and need not agree; whether they do is worth checking.
- **Off-attractor validity.** Finding 12 argues the DMD operator is local to the attractor.
  Testing it directly — predict h(t+1) from a state far from the cycle and measure the error
  — would turn that argument into a measurement.
- **Other fixed points.** Only the origin has been identified in the nonlinear network.
  Solving h = gW·tanh(h) from many starts would say whether others exist off the cycle.
- **Readout alignment.** The 6-D trajectory subspace sits at principal angles of 68–83° to
  the readout row space, i.e. most state variance is in directions the readout barely
  sees. Not chased down; may or may not be meaningful.
- **Power-law initialization.** Everything here is uniform init. The extraction scripts
  take `SCHEME=powerlaw`, and the power-law sine networks fail on the fastest frequency,
  so their mode structure should differ in an interpretable way.
- **Flip-flop: how the attractors are arranged.** Finding 13d shows 4 fixed points at
  g = 0.9 and 13e shows 16 at g = 0.5, but not whether they sit at the corners of a
  hypercube in the readout coordinates, nor what separates them — the saddles between
  adjacent memory states have not been located.
- **Flip-flop: why only some bits get attractors.** Bits 2 and 3 are held perfectly and the
  rest are not, and it is not the slowest or fastest bits that survive. Whether this is
  seed-specific or reflects something about which hold times are reachable at a given gain
  is unresolved.
- **Flip-flop: what carries the other half of the readout?** Finding 13a′ shows the
  dominant mode accounts for at most two thirds of a bit's readout variance and that ~100
  modes are needed for 90%. Whether the remainder is a structured second population or an
  unstructured spread over fast modes has not been looked at.
- **Coupling metric.** The variance-weighted definition in 13a disagrees with
  `mode_analysis.pair_couplings` on the fast bits. If the variance-weighted version is the
  right one, the repo metric and anything computed from it should be revisited.
- **Why the flip-flop timescale allocation replicates on one grid and not the other.**
  Finding 14b shows the ordered bit → mode assignment is absent under Clark/Adam with fixed
  time constants. Learned-versus-fixed τ and the readout concentration set by γ are the two
  obvious suspects and are separable: re-run the same cell with `learn_time_constants:
  true`, and sweep γ at fixed s. Until that is done, finding 13a should be read as a
  property of the tau-init grid rather than of the task.
- **Generality.** All of the above is seed 0, g = 0.9. Worth confirming the qualitative
  claims hold across seeds and gains before leaning on them.
