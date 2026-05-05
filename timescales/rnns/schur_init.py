"""
Schur-diagonal spectral surgery initialization for the recurrent weight matrix.

Procedure (round-trip W -> J -> H̃ -> J̃ -> W̃):
    1. Sample a base W (e.g. N(0, 1/sqrt(N)))
    2. Build the Jacobian under the configured per-unit alphas:
           J = (I - A) + g A W,    A = diag(alpha_i)
    3. Real Schur factor:  J = Q H Q^T  (Q orthogonal, H upper quasi-triangular)
    4. For each diagonal block of H, replace its eigenvalue magnitudes by
       exp(-1/tau_i) for tau_i drawn from p(tau) ∝ tau^{-beta} on
       [tau_min, tau_max], preserving phases for 2x2 (complex-conjugate)
       blocks and the sign for 1x1 (real) blocks.
    5. Reconstruct J̃ = Q H̃ Q^T.
    6. Solve back for the new recurrent weight matrix:
           W̃ = (1/g) A^{-1} (J̃ - (I - A))
       (A is diagonal, so A^{-1} is element-wise 1/alpha_i.)

The result is a real W̃ such that the discrete-time linearisation of the RNN
has the prescribed eigenvalue magnitudes (and hence tau_eff distribution)
while keeping the *eigenbasis* identical to the base J.

This file lifts the helpers from
`timescales/notebooks/theory/5_discrete_linear_rnn_spectra.py` (Section 2b)
into a reusable module with a few small hardenings.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import schur as _schur


# ---------------------------------------------------------------------------
# Power-law tau sampler
# ---------------------------------------------------------------------------

def sample_powerlaw_taus(
    N: int,
    beta: float,
    tau_min: float,
    tau_max: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    CDF inversion for p(tau) ∝ tau^{-beta} on [tau_min, tau_max].

    beta=1 corresponds to log-uniform (equal mass per decade).
    beta=0 corresponds to uniform on [tau_min, tau_max].

    Returns a 1-D ndarray of length N.
    """
    if tau_min <= 0.0 or tau_max <= tau_min:
        raise ValueError(
            f"Need 0 < tau_min < tau_max, got tau_min={tau_min}, tau_max={tau_max}"
        )
    if rng is None:
        rng = np.random.default_rng()

    u = rng.uniform(0.0, 1.0, N)
    if abs(beta - 1.0) < 1e-6:
        return tau_min * (tau_max / tau_min) ** u
    e = 1.0 - beta
    return (tau_min**e + u * (tau_max**e - tau_min**e)) ** (1.0 / e)


# ---------------------------------------------------------------------------
# Schur surgery on a single matrix
# ---------------------------------------------------------------------------

def detect_schur_block_sizes(H: np.ndarray, block_eps: float = 1e-12) -> list[int]:
    """Walk a (quasi-)triangular real Schur form and return the size of each
    diagonal block (1 for real, 2 for conjugate-pair). The list sums to N.
    """
    N = H.shape[0]
    sizes: list[int] = []
    i = 0
    while i < N:
        if i < N - 1 and abs(H[i + 1, i]) > block_eps:
            sizes.append(2)
            i += 2
        else:
            sizes.append(1)
            i += 1
    return sizes


def schur_spectral_surgery(
    J: np.ndarray,
    taus_new: np.ndarray,
    block_eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Replace the eigenvalue magnitudes of `J`'s real Schur diagonal blocks with
    exp(-1/tau) for each tau in `taus_new`.

    In the real Schur form, a complex conjugate pair sits in a single 2x2
    diagonal block and shares one magnitude; a real eigenvalue sits in a 1x1
    block. So `taus_new` is interpreted *per block*: its length must match the
    number of diagonal blocks K (with N/2 ≤ K ≤ N).

    Phase (for 2x2 blocks) and sign (for 1x1 blocks) are preserved; the
    eigenbasis Q and the strict upper-triangular part of H above the diagonal
    blocks are unchanged.

    Args:
        J:         real square matrix (N, N).
        taus_new:  target tau_eff values, length K = number of Schur blocks.
                   Sorted descending internally so that the slowest target maps
                   to the largest |λ| block (scipy's default Schur ordering).
        block_eps: tolerance for detecting 2x2 vs 1x1 Schur blocks.

    Returns:
        J_tilde:    Q H_tilde Q^T   (real, shape (N, N))
        H_tilde:    modified Schur form
        Q:          orthogonal Schur basis (unchanged)
        eigs_old:   eigenvalues read off the original H, in block order
        eigs_new:   eigenvalues read off H_tilde, in block order
    """
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"J must be square, got shape {J.shape}")
    N = J.shape[0]
    if not np.all(taus_new > 0):
        raise ValueError("All target taus must be positive.")

    H, Q = _schur(J, output="real")
    block_sizes = detect_schur_block_sizes(H, block_eps=block_eps)
    K = len(block_sizes)

    if taus_new.shape != (K,):
        raise ValueError(
            f"taus_new must have shape ({K},) (one per Schur block), "
            f"got {taus_new.shape}."
        )

    mags_new = np.exp(-1.0 / np.sort(taus_new)[::-1])

    H_tilde = H.copy()
    eigs_old: list[complex] = []
    eigs_new: list[complex] = []

    i = 0
    for k, bsize in enumerate(block_sizes):
        mag_target = mags_new[k]
        if bsize == 2:
            a = H[i, i]
            b = H[i, i + 1]
            c = H[i + 1, i]
            d = H[i + 1, i + 1]
            tr = a + d
            det = a * d - b * c
            disc = (tr * tr) / 4.0 - det
            if disc >= 0:
                im_old = 0.0
            else:
                im_old = np.sqrt(-disc)
            eigs_old.append(complex(tr / 2.0, im_old))
            eigs_old.append(complex(tr / 2.0, -im_old))

            mag_old = np.sqrt(max(det, 0.0))
            s = mag_target / mag_old if mag_old > block_eps else 0.0
            H_tilde[i, i] = s * a
            H_tilde[i, i + 1] = s * b
            H_tilde[i + 1, i] = s * c
            H_tilde[i + 1, i + 1] = s * d

            tr_new = s * tr
            det_new = s * s * det
            disc_new = (tr_new * tr_new) / 4.0 - det_new
            im_new = np.sqrt(-disc_new) if disc_new < 0 else 0.0
            eigs_new.append(complex(tr_new / 2.0, im_new))
            eigs_new.append(complex(tr_new / 2.0, -im_new))
            i += 2
        else:
            r = H[i, i]
            eigs_old.append(complex(r, 0.0))
            r_new = np.sign(r) * mag_target if abs(r) > block_eps else mag_target
            H_tilde[i, i] = r_new
            eigs_new.append(complex(r_new, 0.0))
            i += 1

    J_tilde = Q @ H_tilde @ Q.T
    return J_tilde, H_tilde, Q, np.array(eigs_old), np.array(eigs_new)


# ---------------------------------------------------------------------------
# Round-trip: W -> J -> surgery -> W̃
# ---------------------------------------------------------------------------

def compute_W_tilde(
    W: np.ndarray,
    alphas: np.ndarray,
    g: float,
    beta_H: float,
    tau_min: float,
    tau_max: float,
    rng: np.random.Generator | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | dict:
    """
    Apply Schur spectral surgery to the Jacobian induced by (W, A, g) and
    return the recurrent weight matrix W_tilde that realises it.

    Algorithm:
        J     = (I - A) + g A W,          A = diag(alphas)
        J_t   = Q H_tilde Q^T   (Schur surgery with power-law taus)
        W_t   = (1/g) A^{-1} (J_t - (I - A))

    Args:
        W:              base recurrent weight matrix, shape (N, N).
        alphas:         per-unit alphas, shape (N,). Must be > 0 (no diagonal
                        is forced trivial -- A^{-1} is element-wise 1/alphas).
        g:              recurrent gain (must be != 0).
        beta_H:         power-law exponent for the inserted tau distribution.
        tau_min, tau_max: support of the power-law sampler.
        rng:            optional NumPy Generator (for reproducibility).
        return_diagnostics:
                        if True, return a dict with the modified Jacobian,
                        its Schur factors, sampled taus, and recovered
                        eigenvalues alongside W_tilde -- useful for tests
                        and notebooks.

    Returns:
        W_tilde (np.ndarray) by default, or a dict with diagnostics if
        return_diagnostics=True.
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square, got shape {W.shape}")
    N = W.shape[0]
    if alphas.shape != (N,):
        raise ValueError(f"alphas must have shape ({N},), got {alphas.shape}")
    if not np.all(alphas > 0):
        raise ValueError("All alphas must be > 0 for A to be invertible.")
    if abs(g) < 1e-12:
        raise ValueError("recurrent_gain g must be non-zero for surgery.")

    A_diag = alphas.astype(np.float64)
    W64 = W.astype(np.float64)

    J = np.diag(1.0 - A_diag) + (A_diag[:, None] * W64) * g

    # Real Schur first to count blocks K, then sample exactly K target taus.
    H_init, Q_init = _schur(J, output="real")
    K = len(detect_schur_block_sizes(H_init))
    taus_new = sample_powerlaw_taus(K, beta_H, tau_min, tau_max, rng=rng)
    J_tilde, H_tilde, Q, eigs_old, eigs_new = schur_spectral_surgery(J, taus_new)

    inv_A_diag = 1.0 / A_diag
    W_tilde = (inv_A_diag[:, None] / g) * (J_tilde - np.diag(1.0 - A_diag))

    if not np.all(np.isfinite(W_tilde)):
        raise RuntimeError(
            "W_tilde contains non-finite entries; check alphas / g / tau range."
        )

    if return_diagnostics:
        return {
            "W_tilde": W_tilde.astype(W.dtype),
            "J": J,
            "J_tilde": J_tilde,
            "H_tilde": H_tilde,
            "Q": Q,
            "taus_new": taus_new,
            "eigs_old": eigs_old,
            "eigs_new": eigs_new,
        }
    return W_tilde.astype(W.dtype)
