# One-column loss curves for the report (21 Nov 2026): the transformer-only run that trained longest (TYW, round TY1,
# 60 epochs; cross-validation 1981-2013, 5 folds x 3 seeds, mean per epoch). Same numbers as loss_curves_best_transformer.py,
# panels stacked for one column.
#   BSISO head: training loss as logged (normalised MSE of PC1/PC2) and the same loss on the held-out summers.
#   Rain-map head: held-out mean squared error per cell of the weekly rain-anomaly map (z units, 283 cells).
import os, glob
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); K = RUNS
OUT = os.path.join(ROOT, "figures", "loss_curves_report.png")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix"})
INK, INK2, GRID, BLUE, ORANGE = "#1f1f1d", "#55544f", "#e4e3de", "#2a78d6", "#d9622b"
CM = 1 / 2.54

R = [pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(K, "TY1", "out", "out", "cv_TYW_f*_s*.pkl")))]
sd = np.concatenate([r["AUX"] for r in {r["fold"]: r for r in R}.values()]).reshape(-1, 2).std(0)
eps = np.array(sorted(R[0]["snaps"]))
bs_va, map_va = [], []
for e in eps:
    b, m = [], []
    for r in R:
        W = r["W"]; S = r["snaps"][e][..., 2:4]; D = r["w_snaps"][e]
        b.append(np.nanmean(((S - r["AUX"]) / sd) ** 2))
        Z = W["LOCW"].astype("float32") + np.einsum("nwp,pc->nwc", W["GW"] + D[..., 2:], W["V"])
        m.append(((Z - W["ZW"].astype("float32")) ** 2)[W["VW"]].mean())
    bs_va.append(np.mean(b)); map_va.append(np.mean(m))
bs_va, map_va = np.array(bs_va), np.array(map_va)
bs_tr = np.mean([r["hist"][:, 2] for r in R], 0)

fig, axs = plt.subplots(2, 1, figsize=(8.6 * CM, 7.4 * CM))
plt.subplots_adjust(hspace=0.7, left=0.15, right=0.97, top=0.92, bottom=0.12)
ax = axs[0]
ax.plot(np.arange(1, len(bs_tr) + 1), bs_tr, color=BLUE, lw=1.4, ls="--", label="training")
ax.plot(eps, bs_va, color=ORANGE, lw=1.6, label="held-out summers")
eb = eps[bs_va.argmin()]
ax.plot(eb, bs_va.min(), "o", ms=5, mfc="none", mec=INK, mew=1.1)
ax.annotate(f"lowest held-out: epoch {eb}", (eb, bs_va.min()), xytext=(eb + 12, bs_va.min() - 0.1), fontsize=7.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_title("BSISO head (daily PC1, PC2)", loc="left", fontsize=9, color=INK)
ax.set_ylabel("normalised MSE", fontsize=8, color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
ax = axs[1]
ax.plot(eps, map_va, color=ORANGE, lw=1.6, label="held-out summers")
em = eps[map_va.argmin()]
ax.plot(em, map_va.min(), "o", ms=5, mfc="none", mec=INK, mew=1.1)
ax.annotate(f"lowest held-out: epoch {em}", (em, map_va.min()), xytext=(em + 7, map_va.min() + 0.001), fontsize=7.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_title("Weekly rain-anomaly map head", loc="left", fontsize=9, color=INK)
ax.set_ylabel("MSE per cell (z units)", fontsize=8, color=INK2)
ax.legend(frameon=False, fontsize=7.5, loc="upper center")
for ax in axs:
    ax.set_xlabel("epoch", fontsize=8, color=INK2); ax.set_xticks(range(0, 61, 10)); ax.set_xlim(0, 61)
    ax.tick_params(labelsize=7, colors=INK2)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.savefig(OUT, dpi=300, facecolor="white")
print("saved", OUT, "| BSISO min", round(bs_va.min(), 4), "ep", eb, "| map min", round(map_va.min(), 4), "ep", em)
