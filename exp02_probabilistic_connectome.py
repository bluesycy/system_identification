"""
Experiment 2: weight inference with a PROBABILISTIC connectome.

Each potential (i, j) has an estimated connection probability p_ij -- e.g.
from axon/dendrite proximity ("Peters's rule"), light microscopy, or a noisy
EM reconstruction. We compare:

  A. ORACLE          - fit using the true binary connectome (upper bound).
  B. THRESHOLD       - binarise p at 0.5 and fit within that support.
  C. WEIGHTED RIDGE  - don't binarise. Place a Gaussian prior
                       w_ij ~ N(0, p_ij * sigma_max^2) on every potential
                       edge and do MAP regression (ridge with per-entry
                       penalty lambda_ij propto 1/p_ij).

Outputs:
  figures/02_probabilistic_connectome.png   - matrices + scatter
  figures/02_separability.png               - |W|-histograms + F1 vs threshold
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import ring_attractor as ra

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

rng = np.random.default_rng(1)

# 1. Ground truth
W_true, true_bin, theta, dtheta = ra.build_network(rng=rng)
N = W_true.shape[0]

# 2. Peters'-rule-like probabilistic connectome.
# True edges: Beta(5, 3) (mean 0.625, sometimes missed).
# Non-edges: proximity kernel * Beta(5, 3) -- nearby non-edges look plausible.
proximity = np.exp(-0.5 * (dtheta / 0.8) ** 2)
p_conn = np.where(
    true_bin == 1,
    rng.beta(5, 3, size=(N, N)),
    proximity * rng.beta(5, 3, size=(N, N)),
)
np.fill_diagonal(p_conn, 0.0)
p_conn = np.clip(p_conn, 0.0, 1.0)

# 3. Simulate
U, R, I_ext, _ = ra.simulate(W_true, theta, rng=rng)
NOISE = 0.02
U_obs, R_obs = ra.add_obs_noise(U, R, NOISE, rng=rng)
Rs, sum_r, target = ra.prepare_regression(U_obs, R_obs, I_ext,
                                           warmup=500, stride=5)

# 4. Three strategies
W_oracle, g_oracle = ra.fit_constrained(Rs, sum_r, target, true_bin)
W_thresh, g_thresh = ra.fit_constrained(Rs, sum_r, target,
                                         (p_conn > 0.5).astype(int))
W_ridge, g_ridge = ra.fit_weighted_ridge(Rs, sum_r, target, p_conn,
                                          sigma_max=1.5,
                                          noise_var=NOISE ** 2)

def report(name, W_hat, gammas):
    m = ra.compute_metrics(W_true, W_hat, true_bin)
    print(f"{name:18s}  rel={m['rel']:.3f}  r={m['corr']:.3f}  "
          f"sign={m['sign']*100:4.1f}%  spur={m['spurious']:.3f}  "
          f"gamma={gammas.mean():.3f}")
    return m

print(f"\nTrue edges: {int(true_bin.sum())}.  Noise sd={NOISE}.\n")
m_oracle = report("A oracle",         W_oracle, g_oracle)
m_thresh = report("B threshold p>0.5", W_thresh, g_thresh)
m_ridge  = report("C weighted ridge",  W_ridge,  g_ridge)

# 5. Anatomy-alone threshold quality + F1 sweep on |W_ridge|
tp = ((p_conn > 0.5) & (true_bin == 1)).sum()
fp = ((p_conn > 0.5) & (true_bin == 0)).sum()
fn = ((p_conn <= 0.5) & (true_bin == 1)).sum()
print(f"\nSupport from anatomy alone (p>0.5):  "
      f"TP={tp} FP={fp} FN={fn}  "
      f"precision={tp/(tp+fp):.2f}  recall={tp/(tp+fn):.2f}")

mask_off = ~np.eye(N, dtype=bool)
absW = np.abs(W_ridge)[mask_off]
truth = true_bin[mask_off].astype(bool)
thresholds = np.linspace(0.0, absW.max(), 200)
f1s = []
for th in thresholds:
    pred = absW > th
    tp_ = (pred & truth).sum()
    fp_ = (pred & ~truth).sum()
    fn_ = (~pred & truth).sum()
    prec = tp_ / max(tp_ + fp_, 1)
    rec  = tp_ / max(tp_ + fn_, 1)
    f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
f1s = np.array(f1s)
best = int(np.argmax(f1s))
th_best = thresholds[best]
print(f"Best |W_ridge| threshold {th_best:.3f}  -> F1={f1s[best]:.3f}")

# 6. Main figure
fig, ax = plt.subplots(2, 4, figsize=(17, 8.5))
vmax = np.max(np.abs(W_true))

ra.plot_weight_matrix(ax[0, 0], W_true, "True W", vmax=vmax)
ax[0, 1].imshow(p_conn, cmap="viridis", vmin=0, vmax=1)
ax[0, 1].set_title("p_ij (probabilistic connectome)")
ax[0, 2].imshow((p_conn > 0.5).astype(int), cmap="gray", vmin=0, vmax=1)
ax[0, 2].set_title("binarised p>0.5")
ax[0, 3].imshow(true_bin, cmap="gray", vmin=0, vmax=1)
ax[0, 3].set_title("true binary connectome")
for a in ax[0]:
    a.set_xticks([]); a.set_yticks([])

ra.plot_weight_matrix(ax[1, 0], W_oracle,
                       f"A oracle\nrel={m_oracle['rel']:.3f}", vmax=vmax)
ra.plot_weight_matrix(ax[1, 1], W_thresh,
                       f"B threshold p>0.5\nrel={m_thresh['rel']:.3f}",
                       vmax=vmax)
ra.plot_weight_matrix(ax[1, 2], W_ridge,
                       f"C weighted ridge\nrel={m_ridge['rel']:.3f}",
                       vmax=vmax)

mask_t = true_bin.astype(bool)
ax[1, 3].scatter(W_true[mask_t], W_oracle[mask_t], s=8, alpha=0.5,
                  label=f"A oracle r={m_oracle['corr']:.3f}")
ax[1, 3].scatter(W_true[mask_t], W_thresh[mask_t], s=8, alpha=0.5,
                  label=f"B threshold r={m_thresh['corr']:.3f}")
ax[1, 3].scatter(W_true[mask_t], W_ridge[mask_t], s=8, alpha=0.5,
                  label=f"C w-ridge r={m_ridge['corr']:.3f}")
lim = 1.1 * vmax
ax[1, 3].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
ax[1, 3].set_xlim(-lim, lim); ax[1, 3].set_ylim(-lim, lim)
ax[1, 3].set_aspect("equal")
ax[1, 3].set_xlabel("true w"); ax[1, 3].set_ylabel("inferred w")
ax[1, 3].legend(loc="upper left", fontsize=8)
ax[1, 3].set_title("Inferred vs true (true edges only)")

plt.tight_layout()
out = os.path.join(FIGDIR, "02_probabilistic_connectome.png")
plt.savefig(out, dpi=120)
print(f"Saved {out}")

# 7. Separability diagnostic
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
bins = np.linspace(0, np.max(np.abs(W_ridge)), 60)
ax2[0].hist(np.abs(W_ridge[true_bin == 1]), bins=bins, alpha=0.7,
             label=f"true edges (n={int(true_bin.sum())})", color="C1")
ax2[0].hist(np.abs(W_ridge[(true_bin == 0) & mask_off]), bins=bins,
             alpha=0.7, label="non-edges", color="C0")
ax2[0].axvline(th_best, color="k", ls="--", lw=1,
                label=f"F1 cut = {th_best:.2f}")
ax2[0].set_yscale("log")
ax2[0].set_xlabel("|inferred weight|")
ax2[0].set_ylabel("count"); ax2[0].legend()
ax2[0].set_title("Weighted-ridge separates true vs false edges")

ax2[1].plot(thresholds, f1s, lw=1.5)
ax2[1].axvline(th_best, color="k", ls="--", lw=1)
ax2[1].set_xlabel("threshold on |inferred W|")
ax2[1].set_ylabel("F1 (binary connectome recovery)")
ax2[1].set_title("F1 vs threshold")

plt.tight_layout()
out2 = os.path.join(FIGDIR, "02_separability.png")
plt.savefig(out2, dpi=120)
print(f"Saved {out2}")
