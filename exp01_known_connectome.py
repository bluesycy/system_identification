"""
Experiment 1: weight inference with a KNOWN connectome but UNKNOWN input.

Scenario: we record firing rates r(t) and the membrane-potential-like
variable u(t). We know:
  - the binary connectome (which (i, j) pairs are connected)
  - the membrane time constant tau

We do NOT know:
  - the sign or magnitude of each W_ij
  - the global inhibition gamma
  - the external input I_i(t)

Why it's still solvable: I(t) here is a bump that drifts on ~1 s, while
the membrane time constant tau = 20 ms -- so the recurrent+membrane signal
has power out to ~50 Hz while I has essentially no power above ~2 Hz. A
zero-phase high-pass filter on all regressors and the target kills I and
leaves the W*r + gamma*sum(r) contribution intact.

Concretely, from
    u + tau*du/dt = W r + I(t) - gamma * sum(r),
applying a high-pass filter HP[*] to both sides and noting HP[I] ~ 0:
    HP[u + tau*du/dt] = W HP[r] - gamma * HP[sum(r)].
Then plain constrained LS on HP-filtered data recovers (W, gamma).

Outputs:
  figures/01_known_connectome.png   - true vs inferred W, power spectra
  figures/01_ring_activity.gif      - neurons on the ring coloured by rate
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import ring_attractor as ra

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

rng = np.random.default_rng(0)

# 1. Build + simulate
W_true, connectome, theta, _ = ra.build_network(rng=rng)
N = W_true.shape[0]
print(f"Network: N={N}, {connectome.sum()} edges "
      f"({connectome.mean()*100:.1f}% density)")

U, R, I_ext, center_hist = ra.simulate(W_true, theta, rng=rng)
S = U.shape[0]
dt = 0.001

# 2. Unknown-input inference via high-pass filter (no I_ext used).
HP_CUT = 15.0
Rs, sum_r, target = ra.prepare_regression_unknown_input(
    U, R, hp_cutoff_hz=HP_CUT, warmup=500, stride=3, edge=500)
W_hat, gammas = ra.fit_constrained(Rs, sum_r, target, connectome)

m = ra.compute_metrics(W_true, W_hat, connectome)
print(f"[HP {HP_CUT} Hz]  rel={m['rel']:.3f}  r={m['corr']:.4f}  "
      f"sign={m['sign']*100:.1f}%  spur={m['spurious']:.3f}  "
      f"gamma={gammas.mean():.3f}  (true=0.15)")

# Reference: oracle (I known) on the same data, for calibration.
Rs2, sum_r2, tg2 = ra.prepare_regression(U, R, I_ext, warmup=500, stride=5)
W_ref, g_ref = ra.fit_constrained(Rs2, sum_r2, tg2, connectome)
m_ref = ra.compute_metrics(W_true, W_ref, connectome)
print(f"[oracle I ]  rel={m_ref['rel']:.3f}  r={m_ref['corr']:.4f}  "
      f"gamma={g_ref.mean():.3f}")

# 3. Main figure: matrices + true-vs-inferred + power spectra
fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
vmax = np.max(np.abs(W_true))

ra.plot_weight_matrix(ax[0, 0], W_true, "True W", vmax=vmax)
ra.plot_weight_matrix(ax[0, 1], W_hat,
                       f"Inferred W (I unknown, HP {HP_CUT:.0f} Hz)",
                       vmax=vmax)
ra.plot_weight_matrix(ax[0, 2], W_hat - W_true,
                       "Residual (inferred - true)", vmax=vmax)

ax[1, 0].imshow(R.T, aspect="auto", cmap="viridis",
                 extent=[0, S * dt, 0, N], origin="lower")
ax[1, 0].set_title("Activity R(t)")
ax[1, 0].set_xlabel("time (s)"); ax[1, 0].set_ylabel("neuron")

ra.plot_true_vs_inferred(ax[1, 1], W_true, W_hat, connectome, vmax=vmax)
ax[1, 1].set_title(f"r = {m['corr']:.4f}")

# Power spectra: I vs u+tau*du/dt for one neuron, to justify the cutoff
dU = np.gradient(U, dt, axis=0)
Y = U + 0.020 * dU
j = 0
f = np.fft.rfftfreq(S, d=dt)
P_I = np.abs(np.fft.rfft(I_ext[:, j])) ** 2
P_Y = np.abs(np.fft.rfft(Y[:, j])) ** 2
ax[1, 2].loglog(f[1:], P_I[1:], label="I(t)    (true, unknown)")
ax[1, 2].loglog(f[1:], P_Y[1:], label="u+tau du/dt")
ax[1, 2].axvline(HP_CUT, color="k", ls="--", lw=0.8,
                  label=f"HP cutoff {HP_CUT:.0f} Hz")
ax[1, 2].set_xlabel("frequency (Hz)"); ax[1, 2].set_ylabel("power")
ax[1, 2].set_title(f"neuron {j}: input is slow, signal is fast")
ax[1, 2].legend(fontsize=8)

plt.tight_layout()
out = os.path.join(FIGDIR, "01_known_connectome.png")
plt.savefig(out, dpi=120)
print(f"Saved {out}")

# 4. Ring-activity movie
SUBSAMPLE = 40
frame_idx = np.arange(0, S, SUBSAMPLE)
x, y = np.cos(theta), np.sin(theta)

fig_m, axm = plt.subplots(figsize=(6, 6))
axm.set_aspect("equal"); axm.axis("off")
axm.set_xlim(-1.5, 1.5); axm.set_ylim(-1.5, 1.5)
axm.plot(x, y, color="lightgray", lw=1, zorder=0)
scat = axm.scatter(x, y, c=R[0], cmap="magma", vmin=0, vmax=1,
                    s=160, edgecolor="k", linewidth=0.5, zorder=2)
in_dot, = axm.plot([], [], "o", color="cyan", ms=14,
                    markeredgecolor="k", zorder=3)
title = axm.set_title("t = 0.000 s")
cbar = plt.colorbar(scat, ax=axm, shrink=0.75)
cbar.set_label("firing rate r")

def update(t):
    scat.set_array(R[t])
    c = center_hist[t]
    in_dot.set_data([1.3 * np.cos(c)], [1.3 * np.sin(c)])
    title.set_text(f"t = {t*dt:.3f} s")
    return scat, in_dot, title

anim = FuncAnimation(fig_m, update, frames=frame_idx, interval=50, blit=False)
mov_out = os.path.join(FIGDIR, "01_ring_activity.gif")
anim.save(mov_out, writer=PillowWriter(fps=20))
print(f"Saved {mov_out}")
