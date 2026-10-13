"""
Shared utilities for the ring-attractor weight-inference experiments.

Provides:
  - build_network     : sparse Mexican-hat connectivity on a ring.
  - simulate          : rate-model ODE with smooth OU input + global inhibition.
  - add_obs_noise     : Gaussian observation noise on u, r.
  - prepare_regression: compute target = u + tau*du/dt - I ; warmup + stride.
  - fit_*             : inference methods (constrained / weighted ridge /
                        blind OLS / blind Lasso / pairwise / row-drop).
  - compute_metrics   : rel, corr, sign, spurious RMS.
  - masking / imputation helpers for the missing-data experiment.
  - plot_weight_matrix / plot_true_vs_inferred for compact figure code.

Network model (Wilson-Cowan rate equations):
  tau du_i/dt = -u_i + sum_j W_ij r_j + I_i(t) - gamma * sum_j r_j
  r_i = phi(u_i) = tanh(max(u_i, 0))
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. Network + simulation
# ----------------------------------------------------------------------
def phi(x):
    return np.tanh(np.maximum(x, 0.0))


def build_network(N=60, r_exc=0.35, r_inh=1.20, j_exc=0.80, j_inh=-0.50,
                   heterogeneity=0.30, rng=None):
    """Sparse ring Mexican-hat connectivity. Returns W, connectome, theta, dtheta."""
    if rng is None:
        rng = np.random.default_rng()
    theta = 2 * np.pi * np.arange(N) / N
    dtheta = np.abs(theta[:, None] - theta[None, :])
    dtheta = np.minimum(dtheta, 2 * np.pi - dtheta)
    W = np.zeros((N, N))
    W[dtheta < r_exc] = j_exc
    W[(dtheta >= r_exc) & (dtheta < r_inh)] = j_inh
    W *= 1.0 + heterogeneity * rng.standard_normal((N, N))
    W[dtheta >= r_inh] = 0.0
    np.fill_diagonal(W, 0.0)
    connectome = (W != 0).astype(int)
    return W, connectome, theta, dtheta


def simulate(W, theta, *, tau=0.020, dt=0.001, T=8.0, gamma=0.15,
              amp=3.0, sigma_input=0.25, omega_tau=0.8, omega_sig=2.0,
              rng=None):
    """
    Integrate the rate ODE with a smoothly wandering Gaussian input bump.
    Returns U, R, I_ext, center_hist, each shape (S, N) except center_hist (S,).
    """
    if rng is None:
        rng = np.random.default_rng()
    N = W.shape[0]
    S = int(T / dt)
    a_ou = np.exp(-dt / omega_tau)
    ou_scale = omega_sig * np.sqrt(1.0 - a_ou ** 2)

    u = np.zeros(N)
    U = np.zeros((S, N))
    R = np.zeros((S, N))
    I_ext = np.zeros((S, N))
    center_hist = np.zeros(S)
    center = rng.uniform(0, 2 * np.pi)
    omega = 0.0
    for t in range(S):
        omega = a_ou * omega + ou_scale * rng.standard_normal()
        center = (center + omega * dt) % (2 * np.pi)
        center_hist[t] = center
        ang = np.abs(theta - center)
        ang = np.minimum(ang, 2 * np.pi - ang)
        I = amp * np.exp(-0.5 * (ang / sigma_input) ** 2)
        I_ext[t] = I
        r = phi(u)
        du = (-u + W @ r + I - gamma * r.sum()) / tau
        u = u + dt * du
        U[t], R[t] = u, r
    return U, R, I_ext, center_hist


def add_obs_noise(U, R, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    U_n = U + sigma * rng.standard_normal(U.shape)
    R_n = np.clip(R + sigma * rng.standard_normal(R.shape), 0, None)
    return U_n, R_n


def prepare_regression(U, R, I_ext=None, *, tau=0.020, dt=0.001,
                        warmup=500, stride=3):
    """
    target(t,i) = u_i + tau*du_i/dt - I_i(t)   if I_ext is known
                = u_i + tau*du_i/dt            if I_ext is None (input unknown;
                                                handle as latent in the fit).
    """
    dU = np.gradient(U, dt, axis=0)
    target = U + tau * dU
    if I_ext is not None:
        target = target - I_ext
    idx = slice(warmup, None, stride)
    Rs = R[idx]
    tg = target[idx]
    sum_r = Rs.sum(axis=1, keepdims=True)
    return Rs, sum_r, tg


# ----------------------------------------------------------------------
# 2. Inference methods
# ----------------------------------------------------------------------
def fit_constrained(Rs, sum_r, target, support):
    """OLS with per-row hard support constraint (known binary connectome)."""
    N = Rs.shape[1]
    W_hat = np.zeros((N, N))
    gammas = np.zeros(N)
    for i in range(N):
        j_idx = np.flatnonzero(support[i])
        X = np.hstack([Rs[:, j_idx], -sum_r])
        coef, *_ = np.linalg.lstsq(X, target[:, i], rcond=None)
        W_hat[i, j_idx] = coef[:-1]
        gammas[i] = coef[-1]
    return W_hat, gammas


def prepare_regression_unknown_input(U, R, *, tau=0.020, dt=0.001,
                                       hp_cutoff_hz=15.0,
                                       warmup=500, stride=3,
                                       edge=500):
    """
    Regression data when the external input I(t) is UNKNOWN but SLOW
    compared to the membrane time constant tau.

    The dynamics
        u + tau*du/dt = W r + I(t) - gamma * sum(r)
    become linear in (W, gamma) once I(t) is removed. If I has no power
    above hp_cutoff_hz (here 15 Hz: bump drifts on ~1 s scale vs 20 ms
    membrane time constant), a zero-phase high-pass filter on Y, R, and
    sum(r) kills I while preserving the fast recurrent+membrane signal.

    Returns (Rs, sum_r, target) ready to plug into any of the fit_* methods.
    """
    from scipy.signal import butter, sosfiltfilt
    dU = np.gradient(U, dt, axis=0)
    Y = U + tau * dU
    s = R.sum(axis=1, keepdims=True)
    sos = butter(4, hp_cutoff_hz, btype="highpass", fs=1.0 / dt, output="sos")
    Y_hp = sosfiltfilt(sos, Y, axis=0)
    R_hp = sosfiltfilt(sos, R, axis=0)
    s_hp = sosfiltfilt(sos, s, axis=0)
    idx = slice(max(warmup, edge), -edge if edge > 0 else None, stride)
    return R_hp[idx], s_hp[idx], Y_hp[idx]


def fit_weighted_ridge(Rs, sum_r, target, p_matrix,
                        sigma_max=1.5, noise_var=4e-4, eps=1e-3):
    """
    MAP Bayesian regression with per-edge Gaussian prior
    w_ij ~ N(0, p_ij * sigma_max^2). gamma gets a weak prior.
    """
    N = Rs.shape[1]
    X = np.hstack([Rs, -sum_r])
    W_hat = np.zeros((N, N))
    gammas = np.zeros(N)
    for i in range(N):
        lam = noise_var / (sigma_max ** 2 * (p_matrix[i] + eps))
        Lambda = np.diag(np.concatenate([lam, [1e-6]]))
        A = X.T @ X + Lambda
        b = X.T @ target[:, i]
        coef = np.linalg.solve(A, b)
        W_hat[i] = coef[:-1]
        gammas[i] = coef[-1]
    return W_hat, gammas


def fit_ols(Rs, sum_r, target):
    """Blind OLS, no prior at all. Uses SVD-based pseudoinverse (lstsq)."""
    X = np.hstack([Rs, -sum_r])
    coef_all, *_ = np.linalg.lstsq(X, target, rcond=None)  # (N+1, N)
    return coef_all[:-1, :].T, coef_all[-1, :]


def fit_ridge(Rs, sum_r, target, lam=1e-3):
    """Uniform ridge (not weighted). gamma unpenalised."""
    N = Rs.shape[1]
    X = np.hstack([Rs, -sum_r])
    Lambda = lam * np.eye(N + 1); Lambda[-1, -1] = 1e-6
    coef = np.linalg.solve(X.T @ X + Lambda, X.T @ target)
    return coef[:-1, :].T, coef[-1, :]


def fit_lasso(Rs, sum_r, target, alpha=1e-3, max_iter=5000):
    """
    Blind Lasso with unpenalised gamma (via per-row partialling-out).
    Requires sklearn.
    """
    from sklearn.linear_model import Lasso
    N = Rs.shape[1]
    W_hat = np.zeros((N, N))
    gammas = np.zeros(N)
    s = -sum_r
    ssT = float((s.T @ s)[0, 0])
    for i in range(N):
        y = target[:, i]
        sRs = ((s.T @ Rs) / ssT).ravel()
        sy = float(((s.T @ y) / ssT).ravel()[0])
        X_res = Rs - s * sRs
        y_res = y - s.ravel() * sy
        m = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter, tol=1e-5)
        m.fit(X_res, y_res)
        W_hat[i] = m.coef_
        gammas[i] = sy - float(sRs @ m.coef_)
    return W_hat, gammas


# ----------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------
def compute_metrics(W_true, W_hat, connectome):
    """Return rel, corr, sign, spurious; all computed on true edges only."""
    N = W_true.shape[0]
    mask = connectome.astype(bool)
    non_edges = ~mask & ~np.eye(N, dtype=bool)
    if np.any(np.isnan(W_hat[mask])):
        return dict(rel=np.nan, corr=np.nan, sign=np.nan,
                    spurious=np.nan, failed=True)
    mae = np.mean(np.abs(W_hat[mask] - W_true[mask]))
    rel = mae / np.mean(np.abs(W_true[mask]))
    corr = np.corrcoef(W_true[mask], W_hat[mask])[0, 1]
    sign = np.mean(np.sign(W_true[mask]) == np.sign(W_hat[mask]))
    spur = np.sqrt(np.mean(W_hat[non_edges] ** 2)) if non_edges.any() else 0.0
    return dict(rel=rel, corr=corr, sign=sign, spurious=spur, failed=False)


# ----------------------------------------------------------------------
# 4. Missing-data helpers
# ----------------------------------------------------------------------
def random_mask(shape, p, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.random(shape) < p


def burst_mask(shape, n_bursts, len_range_ms, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    mask = np.zeros(shape, dtype=bool)
    for _ in range(n_bursts):
        length = int(rng.integers(len_range_ms[0], len_range_ms[1] + 1))
        start = int(rng.integers(0, shape[0] - length))
        mask[start:start + length, :] = True
    return mask


def interp_columns(X):
    """Linear interpolation along time (axis 0) per column."""
    Y = X.copy().astype(float)
    t = np.arange(Y.shape[0])
    for j in range(Y.shape[1]):
        col = Y[:, j]
        nan = np.isnan(col)
        if nan.any() and (~nan).any():
            Y[nan, j] = np.interp(t[nan], t[~nan], col[~nan])
    return Y


# ----------------------------------------------------------------------
# 5. Plot helpers
# ----------------------------------------------------------------------
def plot_weight_matrix(ax, W, title="", vmax=None, cmap="RdBu_r"):
    if vmax is None:
        vmax = np.nanmax(np.abs(W))
    ax.imshow(W, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def plot_true_vs_inferred(ax, W_true, W_hat, connectome, label="",
                            vmax=None, alpha=0.6, s=8):
    mask = connectome.astype(bool)
    ax.scatter(W_true[mask], W_hat[mask], s=s, alpha=alpha, label=label)
    if vmax is None:
        vmax = np.max(np.abs(W_true))
    ax.plot([-1.1*vmax, 1.1*vmax], [-1.1*vmax, 1.1*vmax], "k--", lw=0.6)
    ax.set_xlim(-1.1*vmax, 1.1*vmax); ax.set_ylim(-1.1*vmax, 1.1*vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("true w"); ax.set_ylabel("inferred w")
