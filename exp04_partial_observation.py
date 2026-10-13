"""
Experiment 4: weight inference with PARTIAL observation.

Same ring-attractor simulation, but now the experimenter records only a
subset of neurons. The unobserved neurons become structured confounders
because their rates are correlated with the observed rates through the
shared bump dynamics.

We probe two sampling patterns:
  - RANDOM       - half the neurons chosen uniformly at random
  - CONTIGUOUS   - one half of the ring (an arc)

and two blind inference methods (no connectome prior):
  - OLS
  - LASSO

Metrics are computed only on the observed-to-observed submatrix of W.

Outputs:
  figures/04_partial_observation.png   - two rows (RANDOM/CONTIG) x methods
  figures/04_fraction_sweep.png        - Lasso error/gamma vs % observed
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import ring_attractor as ra

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

rng = np.random.default_rng(3)

# 1. Simulate full network
W_true, connectome, theta, _ = ra.build_network(rng=rng)
N = W_true.shape[0]
GAMMA = 0.15

U, R, I_ext, _ = ra.simulate(W_true, theta, T=12.0, gamma=GAMMA, rng=rng)
U_obs, R_obs = ra.add_obs_noise(U, R, 0.02, rng=rng)
Rs_full, sum_r_full, target_full = ra.prepare_regression(
    U_obs, R_obs, I_ext, warmup=500, stride=3)

# 2. Sampling patterns
N_OBS = N // 2
obs_random = np.sort(rng.choice(N, size=N_OBS, replace=False))
obs_contig = np.arange(N_OBS)


def fit_partial(obs_idx, method="ols", alpha=1e-3):
    """Fit W submatrix on observed neurons, treating unobserved rates as
    hidden confounders. gamma enters as an extra (unpenalised) regressor."""
    Rs = Rs_full[:, obs_idx]
    sum_r = Rs.sum(axis=1, keepdims=True)
    target = target_full[:, obs_idx]
    if method == "ols":
        return ra.fit_ols(Rs, sum_r, target)
    if method == "lasso":
        return ra.fit_lasso(Rs, sum_r, target, alpha=alpha)
    raise ValueError(method)


def evaluate(name, obs_idx, W_sub, gammas):
    W_sub_true = W_true[np.ix_(obs_idx, obs_idx)]
    sub_connect = (W_sub_true != 0).astype(int)
    np.fill_diagonal(sub_connect, 0)
    m = ra.compute_metrics(W_sub_true, W_sub, sub_connect)
    print(f"  {name:8s}  rel={m['rel']:.3f}  r={m['corr']:.3f}  "
          f"sign={m['sign']*100:4.1f}%  spur={m['spurious']:.3f}  "
          f"gamma={gammas.mean():.3f}")
    m["gamma"] = gammas.mean()
    return m


print(f"N={N}, observing {N_OBS}/{N} neurons "
      f"({100*N_OBS/N:.0f}%), true gamma={GAMMA}\n")

results = {}
for label, obs_idx in [("RANDOM half", obs_random),
                        ("CONTIGUOUS half", obs_contig)]:
    print(f"Sampling pattern: {label}")
    W_ols, g_ols = fit_partial(obs_idx, "ols")
    W_las, g_las = fit_partial(obs_idx, "lasso", alpha=1e-3)
    r_ols = evaluate("OLS",   obs_idx, W_ols, g_ols)
    r_las = evaluate("LASSO", obs_idx, W_las, g_las)
    results[label] = (obs_idx, W_ols, W_las, r_ols, r_las)
    print()

# 3. Observation-fraction sweep (random, Lasso only)
fracs = np.linspace(0.2, 1.0, 9)
sweep_rel, sweep_corr, sweep_gamma = [], [], []
for f in fracs:
    n_obs = max(5, int(round(f * N)))
    idx = np.sort(rng.choice(N, size=n_obs, replace=False))
    W_l, g_l = fit_partial(idx, "lasso", alpha=1e-3)
    r = evaluate(f"frac={f:.2f}", idx, W_l, g_l)
    sweep_rel.append(r["rel"])
    sweep_corr.append(r["corr"])
    sweep_gamma.append(r["gamma"])

# 4. Main figure
vmax = np.max(np.abs(W_true))
fig, ax = plt.subplots(2, 4, figsize=(15, 8))

for row, (label, _) in enumerate([("RANDOM half", obs_random),
                                   ("CONTIGUOUS half", obs_contig)]):
    obs_idx, W_ols, W_las, r_ols, r_las = results[label]
    W_sub_true = W_true[np.ix_(obs_idx, obs_idx)]
    sub_connect = (W_sub_true != 0).astype(int)
    np.fill_diagonal(sub_connect, 0)

    ra.plot_weight_matrix(ax[row, 0], W_sub_true,
                           f"True W[obs,obs]\n({label})", vmax=vmax)
    ra.plot_weight_matrix(ax[row, 1], W_ols,
                           f"OLS  rel={r_ols['rel']:.2f}  r={r_ols['corr']:.2f}",
                           vmax=vmax)
    ra.plot_weight_matrix(ax[row, 2], W_las,
                           f"Lasso  rel={r_las['rel']:.2f}  r={r_las['corr']:.2f}",
                           vmax=vmax)

    a = ax[row, 3]
    mask = sub_connect.astype(bool)
    a.scatter(W_sub_true[mask], W_ols[mask], s=8, alpha=0.5, label="OLS")
    a.scatter(W_sub_true[mask], W_las[mask], s=8, alpha=0.5, label="Lasso")
    lim = 1.1 * vmax
    a.plot([-lim, lim], [-lim, lim], "k--", lw=0.6)
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_aspect("equal")
    a.set_xlabel("true w"); a.set_ylabel("inferred w")
    a.legend(fontsize=8)
    a.set_title(f"{label}\n(true edges only)")

plt.tight_layout()
out = os.path.join(FIGDIR, "04_partial_observation.png")
plt.savefig(out, dpi=120)
print(f"Saved {out}")

# 5. Fraction-sweep figure
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
ax2[0].plot(fracs * 100, sweep_rel, "o-", label="relative err")
ax2[0].plot(fracs * 100, 1 - np.array(sweep_corr), "s-",
             label="1 - correlation")
ax2[0].set_xlabel("% neurons observed"); ax2[0].set_ylabel("error metric")
ax2[0].set_title("Lasso recovery vs observation fraction")
ax2[0].grid(alpha=0.3); ax2[0].legend()

ax2[1].plot(fracs * 100, sweep_gamma, "o-", color="C2")
ax2[1].axhline(GAMMA, color="k", ls="--", lw=0.8, label=f"true gamma={GAMMA}")
ax2[1].set_xlabel("% neurons observed")
ax2[1].set_ylabel("inferred gamma")
ax2[1].set_title("gamma bias vs observation fraction")
ax2[1].grid(alpha=0.3); ax2[1].legend()

plt.tight_layout()
out2 = os.path.join(FIGDIR, "04_fraction_sweep.png")
plt.savefig(out2, dpi=120)
print(f"Saved {out2}")
