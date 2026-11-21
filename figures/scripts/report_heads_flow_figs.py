# Report figures for page 5 (21 Nov 2026), transformer only, black and white, block-flow style (grey 3-D boxes, bold
# underlined titles, sizes written on the arrows, stacked cards for the forecast days).
#  head_bsiso_flow.png : encoder output -> keep rows 1-14 -> BSISO head (Linear 32->2) -> 14 daily forecasts (PC1, PC2)
#  head_rain_flow.png  : encoder output -> keep rows 1-14 -> weekly mean -> rain head (Linear 32->30) -> x 30 EOF patterns
#                        -> 2 weekly maps on the 283 IMD cells
#  bsiso_cor_transformer_bw.png : BSISO correlation by lead, test 2014-2024: transformer only, ECMWF, persistence (no colour)
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
import common as nc

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans"})
CM = 1 / 2.54
K = "#111111"
SIDE, BOTTOM = "#9a9a9a", "#c9c9c9"
DEP = 0.13
TITLES = []


def box3d(ax, x, y, w, h, title, sub=None, fs=6.9):
    ax.add_patch(Polygon([(x + w, y), (x + w + DEP, y - DEP), (x + w + DEP, y + h - DEP), (x + w, y + h)], closed=True,
                         fc=SIDE, ec=K, lw=0.7, zorder=2))
    ax.add_patch(Polygon([(x, y), (x + DEP, y - DEP), (x + w + DEP, y - DEP), (x + w, y)], closed=True, fc=BOTTOM, ec=K,
                         lw=0.7, zorder=2))
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=K, lw=0.9, zorder=3))
    yt = y + h / 2 + (0.13 if sub else 0)
    t = ax.text(x + w / 2, yt, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=K, zorder=4)
    TITLES.append((ax, t))
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=fs - 0.6, color=K, zorder=4)


def underline_titles(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax, t in TITLES:
        bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
        ax.plot([bb.x0, bb.x1], [bb.y0 + 0.02, bb.y0 + 0.02], color=K, lw=0.7, zorder=5)
    TITLES.clear()


def arrow(ax, pts, label=None, lpos=None, ha="center", va="bottom"):
    for p0, p1 in zip(pts[:-2], pts[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=K, lw=0.9, zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=9, color=K, lw=0.9, shrinkA=0, shrinkB=0, zorder=1))
    if label:
        ax.text(*lpos, label, ha=ha, va=va, fontsize=7.0, fontweight="bold", color=K)


def cards(ax, x, y, w, h, n, dx, dy, texts, fs=7.2):
    """n stacked cards, the front one at (x, y); texts = (front text, label on the last card)."""
    for k in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + k * dx, y + k * dy), w, h, fc="white", ec=K, lw=0.8, zorder=3 + (n - k)))
    ax.text(x + w / 2, y + h / 2, texts[0], ha="center", va="center", fontsize=fs, color=K, zorder=20, linespacing=1.15)


# ======================================================================= 1) BSISO head
W, H = 8.6, 5.0
fig = plt.figure(figsize=(W * CM, H * CM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, W), ylim=(0, H)); ax.axis("off")
bw, bh, y1 = 1.76, 0.95, 3.55
xa, xb, xc = 0.06, 3.08, 6.08
box3d(ax, xa, y1, bw, bh, "Encoder", "output X")
box3d(ax, xb, y1, bw, bh, "Keep rows", "1–14")
box3d(ax, xc, y1, bw, bh, "BSISO head", r"Linear 32$\rightarrow$2")
arrow(ax, [(xa + bw + DEP, y1 + bh / 2), (xb, y1 + bh / 2)], r"80 $\times$ 32", ((xa + bw + DEP + xb) / 2, y1 + bh / 2 + 0.07))
arrow(ax, [(xb + bw + DEP, y1 + bh / 2), (xc, y1 + bh / 2)], r"14 $\times$ 32", ((xb + bw + DEP + xc) / 2, y1 + bh / 2 + 0.07))
# dropped rows (like a purge stream)
arrow(ax, [(xb + bw / 2, y1 - DEP), (xb + bw / 2, 2.35)])
ax.text(xb + bw / 2, 2.28, "Data rows\ndropped\n" r"66 $\times$ 32", ha="center", va="top", fontsize=7.2, fontweight="bold", color=K,
        linespacing=1.1)
# 14 daily forecasts as stacked cards
cw, ch = 2.0, 0.95
cx, cy = 5.55, 0.3
arrow(ax, [(xc + bw / 2, y1 - DEP), (xc + bw / 2, cy + ch + 3 * 0.13 + 0.4)], r"14 $\times$ 2", (xc + bw / 2 + 0.1, 2.85), ha="left",
      va="center")
cards(ax, cx, cy, cw, ch, 4, 0.13, 0.13, ("BSISO forecast\nPC1, PC2\nday T+1",))
ax.text(cx + 3 * 0.13 + cw + 0.05, cy + ch + 3 * 0.13 + 0.02, "…T+14", ha="right", va="bottom", fontsize=6.8, color=K)
ax.text(cx - 0.1, cy + ch / 2, "one card per\nlead day", ha="right", va="center", fontsize=7.0, color=K, linespacing=1.1)
underline_titles(fig)
fig.savefig(os.path.join(OUTD, "head_bsiso_flow.png"), dpi=300, facecolor="white")
plt.close(fig)
print("BSISO flow saved")

# ======================================================================= 2) rain-map head
ds = xr.open_dataset(os.path.join(nc.DATA, "imd_india_025_jjas_1979_2024.nc"))
tp = ds["tp"].isel(time=0).values; la25, lo25 = ds["lat"].values, ds["lon"].values; ds.close()
mask = np.load(os.path.join(nc.DATA, "kaggle_bsiso_extras", "india_mask_025.npy")) & np.isfinite(tp)
ny, nx = (len(la25) - 1) // 4, (len(lo25) - 1) // 4
M1 = mask[:ny * 4, :nx * 4].reshape(ny, 4, nx, 4).mean((1, 3)) >= 0.5
if la25[0] > la25[-1]:
    M1 = M1[::-1]
print("IMD cells", int(M1.sum()))


def india_outline(ax, x, y, w, z):
    """outline of the 283-cell IMD grid, fitted in a w-wide box at (x, y)."""
    ny_, nx_ = M1.shape
    h = w * ny_ / nx_
    xs = x + (np.arange(nx_) + 0.5) / nx_ * w
    ys = y + (np.arange(ny_) + 0.5) / ny_ * h
    P = np.pad(M1.astype(float), 1)
    xs_p = np.r_[xs[0] - w / nx_, xs, xs[-1] + w / nx_]; ys_p = np.r_[ys[0] - h / ny_, ys, ys[-1] + h / ny_]
    ax.contour(xs_p, ys_p, P, levels=[0.5], colors=K, linewidths=0.7, zorder=z)
    return h


W, H = 8.6, 6.3
fig = plt.figure(figsize=(W * CM, H * CM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, W), ylim=(0, H)); ax.axis("off")
y1 = 4.15
box3d(ax, xa, y1, bw, bh, "Encoder", "output X")
box3d(ax, xb, y1, bw, bh, "Keep rows", "1–14")
box3d(ax, xc, y1, bw, bh, "Week means", "days 1–7, 8–14")
arrow(ax, [(xa + bw + DEP, y1 + bh / 2), (xb, y1 + bh / 2)], r"80 $\times$ 32", ((xa + bw + DEP + xb) / 2, y1 + bh / 2 + 0.07))
arrow(ax, [(xb + bw + DEP, y1 + bh / 2), (xc, y1 + bh / 2)], r"14 $\times$ 32", ((xb + bw + DEP + xc) / 2, y1 + bh / 2 + 0.07))
arrow(ax, [(xb + bw / 2, y1 + bh), (xb + bw / 2, 5.6)])
ax.text(xb + bw / 2 + 0.1, 5.75, r"Data rows dropped  66 $\times$ 32", ha="center", va="bottom", fontsize=7.2, fontweight="bold",
        color=K)
y2 = 1.55
xd, xe = xc, xb
box3d(ax, xd, y2, bw, bh, "Rain head", r"Linear 32$\rightarrow$30")
box3d(ax, xe, y2, bw, bh, "Projection", r"$\times$ 30 EOFs")
arrow(ax, [(xc + bw / 2, y1 - DEP), (xd + bw / 2, y2 + bh)], r"2 $\times$ 32", (xc + bw / 2 + 0.1, (y1 + y2 + bh) / 2), ha="left",
      va="center")
arrow(ax, [(xd, y2 + bh / 2), (xe + bw + DEP, y2 + bh / 2)], r"2 $\times$ 30", ((xd + xe + bw + DEP) / 2, y2 + bh / 2 + 0.07))
ax.text(xe + bw / 2, y2 - DEP - 0.08, r"patterns: 30 $\times$ 283", ha="center", va="top", fontsize=7.0, color=K)
# two weekly maps as stacked cards
mw_, mh_ = 1.5, 2.2
mx, my = 0.25, 0.6
arrow(ax, [(xe, y2 + bh / 2), (mx + mw_ + 0.13 + 0.02, y2 + bh / 2)], r"2 $\times$ 283", ((xe + mx + mw_ + 0.15) / 2, y2 + bh / 2 + 0.07))
for k, nm in ((1, "week 2"), (0, "week 1")):
    x0, y0 = mx + 0.13 * k, my + 0.13 * k
    ax.add_patch(Rectangle((x0, y0), mw_, mh_, fc="white", ec=K, lw=0.8, zorder=4 + 3 * (1 - k)))
    if k == 0:
        india_outline(ax, x0 + 0.2, y0 + 0.1, mw_ - 0.4, 9)
        ax.text(x0 + mw_ / 2, y0 + mh_ - 0.08, "Rain map\nweek 1", ha="center", va="top", fontsize=7.0, color=K,
                zorder=10, linespacing=1.1)
ax.text(mx + mw_ + 0.13 + 0.05, my + mh_ + 0.13, "week 2", ha="left", va="center", fontsize=7.0, color=K)
ax.text(mx + mw_ / 2, my - 0.08, "283 IMD cells", ha="center", va="top", fontsize=7.0, color=K)
underline_titles(fig)
fig.savefig(os.path.join(OUTD, "head_rain_flow.png"), dpi=300, facecolor="white")
plt.close(fig)
print("rain flow saved")

# ======================================================================= 3) BSISO correlation, transformer only (black and white)
exec(open(os.path.join(HERE, "poster_transformer_figs.py"), encoding="utf-8").read().split("# ================================================================ 1.")[0])
KL = K; K = "#111111"                                           # the poster header sets K to the run folder
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix"})
dcols = [f"d{k}" for k in range(1, 15)]
B = pd.read_csv(os.path.join(PD, "test_W3BF_bsiso.csv"), index_col=0)
Tr = pd.read_pickle(os.path.join(KL, "TXMF", "out", "out", "final_TXM.pkl"))
tx = np.r_[1, bcor(np.mean([s[20][..., 2:4] for s in Tr["deep_snaps"]], 0), Tr["AUX"])]
ec, pe = np.r_[1, B.loc["ECMWF", dcols].values], np.r_[1, B.loc["persistence", dcols].values]
print("persistence", np.round(pe, 3)); print("transformer", np.round(tx, 3)); print("ECMWF", np.round(ec, 3))
L = np.arange(0, 15)
fig, ax = plt.subplots(figsize=(8.6 * CM, 5.6 * CM))
plt.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.17)
ax.axhline(0.5, color="#888888", lw=0.8, ls=(0, (4, 3)))
ax.plot(L, ec, color="#555555", lw=1.3, ls="--", marker="s", ms=3.2, mfc="white", label="ECMWF")
ax.plot(L, tx, color=K, lw=1.6, marker="o", ms=3.2, label="Transformer")
ax.plot(L, pe, color="#777777", lw=1.2, ls=":", marker="^", ms=3.2, mfc="white", label="Persistence")
ax.set(xlim=(-0.3, 14.3), ylim=(-0.1, 1.04), xticks=range(0, 15, 2))
ax.set_xlabel("lead time (days)", fontsize=8.5); ax.set_ylabel("BSISO correlation", fontsize=8.5)
ax.tick_params(labelsize=7.5)
ax.text(14.2, 0.52, "COR = 0.5", ha="right", va="bottom", fontsize=7.5, color="#555555")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7.8, loc="lower left")
fig.savefig(os.path.join(OUTD, "bsiso_cor_transformer_bw.png"), dpi=300, facecolor="white")
plt.close(fig)
print("BSISO graph saved")
