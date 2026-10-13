"""
Experiment 3: weight inference with NO connectome prior.

The experimenter knows the dynamics and records (u, r), but has no idea which
pairs are connected. Four estimators:

  OLS     - plain least squares, all N+1 unknowns per row.
  RIDGE   - uniform L2 penalty.
  LASSO   - uniform L1 penalty (sparsity -> rediscovers a support).
  ORACLE  - constrained to the true connectome (upper-bound reference).

Questions:
  - Can we recover weights (magnitude + sign) with no structural prior?
  - Does LASSO rediscover the Mexican-hat connectome from activity alone?

Outputs:
  figures/03_no_connectome.png         - matrices + scatter vs truth
  figures/03_support_recovery.png      - lasso support + alpha sweep
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import ring_attractor as ra

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

rng = np.random.default_rng(2)

# 1. Simulate (longer trial -> more samples)
W_true, connectome, theta, _ = ra.build_network(rng=rng)
U, R, I_ext, _ = ra.simulate(W_true, theta, T=12.0, rng=rng)
N = W_true.shape[0]

NOISE = 0.02
U_obs, R_obs = ra.add_obs_noise(U, R, NOISE, rng=rng)
Rs, sum_r, target = ra.prepare_regression(U_obs, R_obs, I_ext,
                                           warmup=500, stride=3)

print(f"Dynamics: N={N}, T_samples={Rs.shape[0]}, "
      f"unknowns per row = {N + 1}")
X_full = np.hstack([Rs, -sum_r])
print(f"Design condition number: {np.linalg.cond(X_full):.2e}")

# 2. Four estimators
W_ols,    g_ols    = ra.fit_ols(Rs, sum_r, target)
W_ridge,  g_ridge  = ra.fit_ridge(Rs, sum_r, target, lam=1e-3)
W_lasso,  g_lasso  = ra.fit_lasso(Rs, sum_r, target, alpha=1e-3)
W_oracle, g_oracle = ra.fit_constrained(Rs, sum_r, target, connectome)

def report(name, W_hat, gammas):
    m = ra.compute_metrics(W_true, W_hat, connectome)
    print(f"{name:8s}  rel={m['rel']:.3f}  r={m['corr']:.3f}  "
          f"sign={m['sign']*100:4.1f}%  spur={m['spurious']:.3f}  "
          f"gamma={gammas.mean():.3f}")
    return m

print(f"\nMetrics (true edges, n={int(connectome.sum())})")
m_ols    = report("OLS",    W_ols,    g_ols)
m_ridge  = report("RIDGE",  W_ridge,  g_ridge)
m_lasso  = report("LASSO",  W_lasso,  g_lasso)
m_oracle = report("ORACLE", W_oracle, g_oracle)

# 3. Lasso support recovery
lasso_nz = (np.abs(W_lasso) > 1e-6).astype(int)
tp = ((lasso_nz == 1) & (connectome == 1)).sum()
fp = ((lasso_nz == 1) & (connectome == 0)).sum()
fn = ((lasso_nz == 0) & (connectome == 1)).sum()
print(f"\nLASSO support: TP={tp} FP={fp} FN={fn}  "
      f"precision={tp/max(tp+fp,1):.2f}  recall={tp/max(tp+fn,1):.2f}")

# 4. Main figure
fig, ax = plt.subplots(2, 3, figsize=(13, 8))
vmax = np.max(np.abs(W_true))

ra.plot_weight_matrix(ax[0, 0], W_true,  "True W",                                 vmax=vmax)
ra.plot_weight_matrix(ax[0, 1], W_ols,   f"OLS  rel={m_ols['rel']:.2f}  r={m_ols['corr']:.2f}",    vmax=vmax)
ra.plot_weight_matrix(ax[0, 2], W_ridge, f"Ridge  rel={m_ridge['rel']:.2f}  r={m_ridge['corr']:.2f}", vmax=vmax)
ra.plot_weight_matrix(ax[1, 0], W_lasso, f"Lasso  rel={m_lasso['rel']:.2f}  r={m_lasso['corr']:.2f}", vmax=vmax)
ra.plot_weight_matrix(ax[1, 1], W_oracle, f"Oracle  rel={m_oracle['rel']:.2f}  r={m_oracle['corr']:.2f}", vmax=vmax)

mask = connectome.astype(bool)
ax[1, 2].scatter(W_true[mask], W_ols[mask],    s=6, alpha=0.3, label="OLS")
ax[1, 2].scatter(W_true[mask], W_ridge[mask],  s=6, alpha=0.5, label="Ridge")
ax[1, 2].scatter(W_true[mask], W_lasso[mask],  s=6, alpha=0.5, label="Lasso")
ax[1, 2].scatter(W_true[mask], W_oracle[mask], s=6, alpha=0.5, label="Oracle")
lim = 1.1 * vmax
ax[1, 2].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
ax[1, 2].set_xlim(-lim, lim); ax[1, 2].set_ylim(-lim, lim)
ax[1, 2].set_aspect("equal")
ax[1, 2].set_xlabel("true w"); ax[1, 2].set_ylabel("inferred w")
ax[1, 2].legend(fontsize=8); ax[1, 2].set_title("Inferred vs true, edges only")

plt.tight_layout()
out = os.path.join(FIGDIR, "03_no_connectome.png")
plt.savefig(out, dpi=120)
print(f"Saved {out}")

# 5. Lasso support recovery figure + alpha sweep
fig2, ax2 = plt.subplots(1, 3, figsize=(13, 4))
ax2[0].imshow(lasso_nz, cmap="gray_r", vmin=0, vmax=1)
ax2[0].set_title(f"LASSO support\n"
                 f"P={tp/max(tp+fp,1):.2f}  R={tp/max(tp+fn,1):.2f}")
ax2[0].set_xticks([]); ax2[0].set_yticks([])
ax2[1].imshow(connectome, cmap="gray_r", vmin=0, vmax=1)
ax2[1].set_title("True binary connectome")
ax2[1].set_xticks([]); ax2[1].set_yticks([])

alphas = np.logspace(-4, -1, 15)
f1s, precs, recs = [], [], []
for alpha in alphas:
    W_a, _ = ra.fit_lasso(Rs, sum_r, target, alpha=alpha)
    nz = (np.abs(W_a) > 1e-6).astype(int)
    t1 = ((nz == 1) & (connectome == 1)).sum()
    fp_ = ((nz == 1) & (connectome == 0)).sum()
    fn_ = ((nz == 0) & (connectome == 1)).sum()
    p = t1 / max(t1 + fp_, 1); r = t1 / max(t1 + fn_, 1)
    f1s.append(2 * p * r / max(p + r, 1e-9))
    precs.append(p); recs.append(r)

ax2[2].semilogx(alphas, f1s, "o-", label="F1")
ax2[2].semilogx(alphas, precs, "s--", label="precision")
ax2[2].semilogx(alphas, recs, "^--", label="recall")
ax2[2].set_xlabel("Lasso alpha")
ax2[2].set_ylabel("score")
ax2[2].set_title("Lasso sparsity sweep")
ax2[2].legend()

plt.tight_layout()
out2 = os.path.join(FIGDIR, "03_support_recovery.png")
plt.savefig(out2, dpi=120)
print(f"Saved {out2}")
