"""
Experiment 5: weight inference with MISSING VALUES.

Two missingness patterns:
  A. RANDOM  - each (t, neuron) independently NaN with prob p_miss (dropout).
  B. BURST   - random contiguous time windows where ALL neurons are NaN
               (movement artefacts / stimulation blanking).

Four handling strategies, all constrained to the true connectome:
  INTERP      - linear interpolation along time, then fit normally.
  ROW_DROP    - per-row listwise deletion; drop samples where the row's
                target or any of its presynaptic regressors is missing.
  PAIRWISE    - available-case normal equations (blind, no prior). Each
                (j, k) of X^T X is estimated from samples where both
                column j AND column k are observed.
  PAIRWISE+CON- pairwise but restricted to the connectome support.

Also a sweep over random missingness rate with INTERP (reference curve).

Outputs:
  figures/05_missing_values.png   - 2 rows (patterns) x 5 cols (truth + 4 methods)
  figures/05_missing_sweep.png    - error + gamma vs % values missing
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import ring_attractor as ra

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

rng = np.random.default_rng(5)

# 1. Simulate
W_true, connectome, theta, _ = ra.build_network(rng=rng)
N = W_true.shape[0]
tau, dt, T = 0.020, 0.001, 12.0
GAMMA = 0.15

U, R, I_ext, _ = ra.simulate(W_true, theta, tau=tau, dt=dt, T=T,
                              gamma=GAMMA, rng=rng)
U_clean, R_clean = ra.add_obs_noise(U, R, 0.02, rng=rng)
S = U.shape[0]

# 2. Missingness masks
P_MISS = 0.10
mask_random = ra.random_mask((S, N), P_MISS, rng=rng)
mask_burst  = ra.burst_mask((S, N), n_bursts=30, len_range_ms=(30, 120),
                             rng=rng)
print(f"Random mask: {mask_random.mean()*100:.1f}% NaN")
print(f"Burst mask : {mask_burst.mean()*100:.1f}% NaN\n")


# 3. Four handling strategies
def fit_interp_oracle(U_c, R_c, mask, support):
    U_fill = ra.interp_columns(np.where(mask, np.nan, U_c))
    R_fill = ra.interp_columns(np.where(mask, np.nan, R_c))
    Rs, sum_r, target = ra.prepare_regression(U_fill, R_fill, I_ext,
                                               tau=tau, dt=dt,
                                               warmup=500, stride=3)
    return ra.fit_constrained(Rs, sum_r, target, support)


def _build_nan(U_c, R_c, mask):
    U_nan = np.where(mask, np.nan, U_c)
    R_nan = np.where(mask, np.nan, R_c)
    dU_nan = np.gradient(U_nan, dt, axis=0)
    target_nan = U_nan + tau * dU_nan - I_ext
    return R_nan, target_nan, ~np.isnan(R_nan), ~np.isnan(target_nan)


def fit_row_drop(U_c, R_c, mask, support):
    R_nan, target_nan, obs_r, obs_t = _build_nan(U_c, R_c, mask)
    sum_r = np.nansum(R_nan, axis=1, keepdims=True)
    idx = slice(500, None, 3)
    R_use, target_use = R_nan[idx], target_nan[idx]
    obs_r_use, obs_t_use = obs_r[idx], obs_t[idx]
    sum_r_use = sum_r[idx]

    W_hat = np.zeros((N, N)); gammas = np.zeros(N); n_valid = np.zeros(N, int)
    for i in range(N):
        j_idx = np.flatnonzero(support[i])
        col_obs = obs_r_use[:, j_idx].all(axis=1)
        row_ok = obs_t_use[:, i] & col_obs
        if row_ok.sum() < len(j_idx) + 2:
            W_hat[i, j_idx] = np.nan; gammas[i] = np.nan; continue
        X = np.hstack([R_use[row_ok][:, j_idx], -sum_r_use[row_ok]])
        y = target_use[row_ok, i]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        W_hat[i, j_idx] = coef[:-1]
        gammas[i] = coef[-1]
        n_valid[i] = row_ok.sum()
    return W_hat, gammas, n_valid


def fit_pairwise(U_c, R_c, mask, support=None, reg=1e-2):
    """Available-case normal equations. (X^T X)_{jk} uses samples where both
    j AND k are observed. `support` optionally restricts regressors per row."""
    R_nan, target_nan, obs_r, obs_t = _build_nan(U_c, R_c, mask)
    sum_r = np.nansum(R_nan, axis=1, keepdims=True)
    idx = slice(500, None, 3)
    R_use, target_use = R_nan[idx], target_nan[idx]
    obs_r_use, obs_t_use = obs_r[idx], obs_t[idx]
    sum_r_use = sum_r[idx]
    R_z = np.where(obs_r_use, R_use, 0.0)
    Or = obs_r_use.astype(float)
    Tsamp = R_use.shape[0]

    W_hat = np.zeros((N, N)); gammas = np.zeros(N)
    for i in range(N):
        j_idx = np.arange(N) if support is None else np.flatnonzero(support[i])
        Rz_sub = R_z[:, j_idx]; Or_sub = Or[:, j_idx]
        gam_col = -sum_r_use
        obs_gam = np.ones(Tsamp)

        sum_XX_r = Rz_sub.T @ Rz_sub
        n_XX_r = Or_sub.T @ Or_sub
        sum_Xg_r = (Rz_sub * Or_sub).T @ gam_col
        n_Xg_r = Or_sub.T @ obs_gam[:, None]
        sum_gg = float((gam_col * gam_col).sum())
        n_gg = float(obs_gam.sum())

        K = len(j_idx)
        XTX = np.zeros((K + 1, K + 1))
        XTX[:K, :K] = sum_XX_r / np.maximum(n_XX_r, 1)
        XTX[:K, K] = (sum_Xg_r / np.maximum(n_Xg_r, 1)).ravel()
        XTX[K, :K] = XTX[:K, K]
        XTX[K, K] = sum_gg / max(n_gg, 1)

        y = target_use[:, i]
        y_z = np.where(obs_t_use[:, i], y, 0.0)
        ot = obs_t_use[:, i].astype(float)
        sum_Xy_r = Rz_sub.T @ y_z
        n_Xy_r = Or_sub.T @ ot
        sum_gy = float((gam_col.ravel() * y_z).sum())
        n_gy = float(ot.sum())
        XTy = np.zeros(K + 1)
        XTy[:K] = sum_Xy_r / np.maximum(n_Xy_r, 1)
        XTy[K]  = sum_gy / max(n_gy, 1)

        try:
            coef = np.linalg.solve(XTX + reg * np.eye(K + 1), XTy)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(XTX + reg * np.eye(K + 1), XTy,
                                    rcond=None)[0]
        W_hat[i, j_idx] = coef[:-1]
        gammas[i] = coef[-1]
    return W_hat, gammas


def fmt(m, gam=None):
    if m.get("failed", False):
        return "FAILED"
    tail = f"  gamma={gam:.3f}" if gam is not None else ""
    return (f"rel={m['rel']:.3f}  r={m['corr']:.3f}  "
            f"sign={m['sign']*100:5.1f}%  spur={m['spurious']:.3f}{tail}")


# 4. Evaluate on both masks
all_results = {}
for miss_name, mask in [("RANDOM 10%", mask_random), ("BURST", mask_burst)]:
    print(f"=== {miss_name} ===")
    W1, g1     = fit_interp_oracle(U_clean, R_clean, mask, connectome)
    W2, g2, nv = fit_row_drop     (U_clean, R_clean, mask, connectome)
    W3, g3     = fit_pairwise     (U_clean, R_clean, mask, support=None)
    W4, g4     = fit_pairwise     (U_clean, R_clean, mask, support=connectome)
    m1 = ra.compute_metrics(W_true, W1, connectome)
    m2 = ra.compute_metrics(W_true, W2, connectome)
    m3 = ra.compute_metrics(W_true, W3, connectome)
    m4 = ra.compute_metrics(W_true, W4, connectome)
    print(f"  INTERP       (con): {fmt(m1, g1.mean())}")
    print(f"  ROW_DROP     (con): {fmt(m2, g2.mean())}  "
          f"[avg {nv.mean():.0f} valid samples/row]")
    print(f"  PAIRWISE  (blind) : {fmt(m3, g3.mean())}")
    print(f"  PAIRWISE+CON      : {fmt(m4, g4.mean())}")
    all_results[miss_name] = [
        (W1, m1, "INTERP"),
        (W2, m2, "ROW_DROP"),
        (W3, m3, "PAIRWISE (blind)"),
        (W4, m4, "PAIRWISE+CON"),
    ]
    print()

# 5. Main figure: 2 rows x 5 cols (truth + 4 methods)
vmax = np.max(np.abs(W_true))
fig, ax = plt.subplots(2, 5, figsize=(17, 7.5))
for ri, (miss_name, rows) in enumerate(all_results.items()):
    ra.plot_weight_matrix(ax[ri, 0], W_true,
                           f"True W\n({miss_name})", vmax=vmax)
    for ci, (W, m, name) in enumerate(rows, start=1):
        if m.get("failed", False):
            ax[ri, ci].text(0.5, 0.5, f"{name}\nFAILED",
                             ha="center", va="center",
                             transform=ax[ri, ci].transAxes)
            ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
        else:
            ra.plot_weight_matrix(
                ax[ri, ci], W,
                f"{name}\nrel={m['rel']:.2f}  r={m['corr']:.2f}",
                vmax=vmax)

plt.tight_layout()
out = os.path.join(FIGDIR, "05_missing_values.png")
plt.savefig(out, dpi=120)
print(f"Saved {out}")

# 6. Sweep over random missingness rate (INTERP + connectome)
fracs = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
rels, corrs, gams = [], [], []
for p in fracs:
    mk = ra.random_mask((S, N), p, rng=rng) if p > 0 \
        else np.zeros((S, N), dtype=bool)
    W_b, g_b = fit_interp_oracle(U_clean, R_clean, mk, connectome)
    m = ra.compute_metrics(W_true, W_b, connectome)
    rels.append(m["rel"]); corrs.append(m["corr"]); gams.append(g_b.mean())
    print(f"  interp, random {p*100:.0f}% missing -> "
          f"rel={m['rel']:.3f}  r={m['corr']:.3f}  gamma={g_b.mean():.3f}")

fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
ax2[0].plot(np.array(fracs) * 100, rels, "o-", label="rel err")
ax2[0].plot(np.array(fracs) * 100, 1 - np.array(corrs), "s-",
             label="1 - corr")
ax2[0].set_xlabel("% values missing (random)")
ax2[0].set_ylabel("error")
ax2[0].set_title("INTERP + connectome-oracle")
ax2[0].grid(alpha=0.3); ax2[0].legend()

ax2[1].plot(np.array(fracs) * 100, gams, "o-", color="C2")
ax2[1].axhline(GAMMA, color="k", ls="--", lw=0.8, label=f"true gamma={GAMMA}")
ax2[1].set_xlabel("% values missing")
ax2[1].set_ylabel("inferred gamma")
ax2[1].set_title("gamma vs missingness rate")
ax2[1].grid(alpha=0.3); ax2[1].legend()

plt.tight_layout()
out2 = os.path.join(FIGDIR, "05_missing_sweep.png")
plt.savefig(out2, dpi=120)
print(f"Saved {out2}")
