# Report figures (21 Nov 2026), block-flow style, black and white.
#  logistic_plus_transformer.png : inputs -> frozen linear/logistic models (fitted first) + transformer correction -> forecast
#  ere_head_flow.png             : week state -> ERE head (Linear 32->2) -> P(any ERE), P(large | any) -> three week classes
#  ere_loss_terms.png            : held-out weighted loss terms per epoch for the transformer-only 60-epoch run (TYW):
#                                  1.0 L_ERE, 0.5 L_map, 0.5 L_BSISO (as in profA.loss_fn)
import os, glob
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__)); KL = RUNS
OUTD = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans"})
CM = 1 / 2.54
K = "#111111"; SIDE, BOTTOM = "#9a9a9a", "#c9c9c9"; DEP = 0.13
TITLES = []


def box3d(ax, x, y, w, h, title, sub=None, fs=6.9):
    ax.add_patch(Polygon([(x + w, y), (x + w + DEP, y - DEP), (x + w + DEP, y + h - DEP), (x + w, y + h)], closed=True, fc=SIDE, ec=K, lw=0.7, zorder=2))
    ax.add_patch(Polygon([(x, y), (x + DEP, y - DEP), (x + w + DEP, y - DEP), (x + w, y)], closed=True, fc=BOTTOM, ec=K, lw=0.7, zorder=2))
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=K, lw=0.9, zorder=3))
    nsub = 0 if not sub else sub.count("\n") + 1
    yt = y + h - 0.27 if nsub > 1 else y + h / 2 + 0.17      # title near the top, sub text below its underline
    t = ax.text(x + w / 2, yt, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=K, zorder=4)
    TITLES.append((ax, t))
    if sub:
        ax.text(x + w / 2, yt - 0.24, sub, ha="center", va="top", fontsize=fs - 0.6, color=K, zorder=4, linespacing=1.15)


def underline_titles(fig):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for ax, t in TITLES:
        bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
        ax.plot([bb.x0, bb.x1], [bb.y0 + 0.02, bb.y0 + 0.02], color=K, lw=0.7, zorder=5)
    TITLES.clear()


def arrow(ax, pts, label=None, lpos=None, ha="center", va="bottom", fs=6.8, bold=True):
    for p0, p1 in zip(pts[:-2], pts[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=K, lw=0.9, zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=9, color=K, lw=0.9, shrinkA=0, shrinkB=0, zorder=1))
    if label:
        ax.text(*lpos, label, ha=ha, va=va, fontsize=fs, fontweight="bold" if bold else "normal", color=K, linespacing=1.1)


# ======================================================================= 1) logistic + transformer
W, H = 8.6, 5.2
fig = plt.figure(figsize=(W * CM, H * CM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, W), ylim=(0, H)); ax.axis("off")
bh = 1.15
yI, yL, yT = 2.0, 3.85, 0.3                       # inputs / linear / transformer rows
box3d(ax, 0.08, yI, 1.45, bh, "Inputs", "last 30 days\nup to day T")
box3d(ax, 2.7, yL, 2.3, bh, "Linear models", "ridge and logistic,\nfitted first, frozen")
box3d(ax, 2.7, yT, 2.3, bh, "Transformer", "learns a\ncorrection")
xp, yp = 5.95, yI + bh / 2
ax.add_patch(Circle((xp, yp), 0.2, fc="white", ec=K, lw=0.9, zorder=4))
ax.plot([xp - 0.12, xp + 0.12], [yp, yp], color=K, lw=0.9, zorder=5); ax.plot([xp, xp], [yp - 0.12, yp + 0.12], color=K, lw=0.9, zorder=5)
box3d(ax, 6.55, yI, 1.85, bh, "Forecast", "BSISO, maps")
xv = 2.25
arrow(ax, [(1.53 + DEP, yI + bh * 0.75), (xv, yI + bh * 0.75), (xv, yL + bh / 2), (2.7, yL + bh / 2)],
      "compact\nfeatures", (xv - 0.1, 3.7), ha="right", va="center")
arrow(ax, [(1.53 + DEP, yI + bh * 0.25), (xv, yI + bh * 0.25), (xv, yT + bh / 2), (2.7, yT + bh / 2)],
      "80\ntokens", (xv - 0.1, 1.45), ha="right", va="center")
arrow(ax, [(5.0 + DEP, yL + bh / 2), (xp, yL + bh / 2), (xp, yp + 0.2)], "linear\nforecast", (xp + 0.12, 3.75), ha="left", va="center")
arrow(ax, [(5.0 + DEP, yT + bh / 2), (xp, yT + bh / 2), (xp, yp - 0.2)], "correction\n" r"$\Delta$", (xp + 0.12, 1.45), ha="left", va="center")
arrow(ax, [(xp + 0.2, yp), (6.55, yp)])
underline_titles(fig)
fig.savefig(os.path.join(OUTD, "logistic_plus_transformer.png"), dpi=300, facecolor="white")
plt.close(fig)
print("logistic + transformer saved")

# ======================================================================= 2) ERE head
W, H = 8.6, 4.5
fig = plt.figure(figsize=(W * CM, H * CM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, W), ylim=(0, H)); ax.axis("off")
y1, bw = 3.2, 1.9
xa, xb, xc = 0.06, 3.3, 6.5
box3d(ax, xa, y1, bw, bh, "Week state", r"$\bar{\mathbf{h}}_w$, mean of" "\n" r"7 rows of $\mathbf{H}$")
box3d(ax, xb, y1, bw, bh, "ERE head", r"Linear 32$\rightarrow$2")
box3d(ax, xc, y1, 1.95, bh, "Sigmoid", "on each logit")
arrow(ax, [(xa + bw + DEP, y1 + bh / 2), (xb, y1 + bh / 2)], "w = 1, 2", ((xa + bw + DEP + xb) / 2, y1 + bh / 2 + 0.07), fs=6.4)
arrow(ax, [(xb + bw + DEP, y1 + bh / 2), (xc, y1 + bh / 2)], "2 logits", ((xb + bw + DEP + xc) / 2, y1 + bh / 2 + 0.07), fs=6.4)
# three classes
bx, by, bW, bH = 0.3, 0.2, 4.5, 1.75
ax.add_patch(Polygon([(bx + bW, by), (bx + bW + DEP, by - DEP), (bx + bW + DEP, by + bH - DEP), (bx + bW, by + bH)], closed=True, fc=SIDE, ec=K, lw=0.7, zorder=2))
ax.add_patch(Polygon([(bx, by), (bx + DEP, by - DEP), (bx + bW + DEP, by - DEP), (bx + bW, by)], closed=True, fc=BOTTOM, ec=K, lw=0.7, zorder=2))
ax.add_patch(Rectangle((bx, by), bW, bH, fc="white", ec=K, lw=0.9, zorder=3))
t = ax.text(bx + bW / 2, by + bH - 0.25, "Three classes for the week", ha="center", va="center", fontsize=6.9, fontweight="bold", zorder=4)
TITLES.append((ax, t))
for k, (nm, fx) in enumerate((("no ERE", r"$1-p_a$"), ("small or medium only", r"$p_a\,(1-p_l)$"), ("at least one large day", r"$p_a\,p_l$"))):
    yy = by + bH - 0.68 - 0.36 * k
    ax.text(bx + 0.2, yy, nm, ha="left", va="center", fontsize=6.6, zorder=4)
    ax.text(bx + bW - 0.2, yy, fx, ha="right", va="center", fontsize=7.2, zorder=4)
xm = xc + 1.95 / 2; ym = by + bH / 2
arrow(ax, [(xm, y1 - DEP), (xm, ym), (bx + bW + DEP, ym)])
ax.text(xm - 0.12, 2.35, r"$p_a=P(\mathrm{any\ ERE})$" "\n" r"$p_l=P(\mathrm{large}\mid\mathrm{any})$", ha="right", va="center",
        fontsize=6.4, color=K, linespacing=1.5)
underline_titles(fig)
fig.savefig(os.path.join(OUTD, "ere_head_flow.png"), dpi=300, facecolor="white")
plt.close(fig)
print("ERE head saved")

# ======================================================================= 3) held-out loss terms, transformer only (TYW)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix"})
bce = lambda z, y: np.logaddexp(0, z) - y * z
R = [pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(KL, "TY1", "out", "out", "cv_TYW_f*_s*.pkl")))]
sd = np.concatenate([r["AUX"] for r in {r["fold"]: r for r in R}.values()]).reshape(-1, 2).std(0)
eps = np.array(sorted(R[0]["snaps"])); rows = []
for e in eps:
    acc = []
    for r in R:
        Wd = r["W"]; S = r["snaps"][e]; D = r["w_snaps"][e]
        b = np.nanmean(((S[..., 2:4] - r["AUX"]) / sd) ** 2)
        m = np.isfinite(Wd["YWa"]); l1 = bce(Wd["LW"][..., 0] + D[..., 0], np.nan_to_num(Wd["YWa"]))[m].mean()
        m2 = np.isfinite(Wd["YWl"]); l2 = bce(Wd["LW"][..., 1] + D[..., 1], np.nan_to_num(Wd["YWl"]))[m2].mean()
        mp = ((Wd["GW"] + D[..., 2:] - Wd["RW"]) ** 2)[Wd["VW"]].mean() / Wd["wsc"]
        acc.append((0.5 * b, l1 + l2, 0.5 * mp))
    rows.append(np.mean(acc, 0))
a = np.array(rows)
print("ERE term min %.3f at epoch %d, last %.3f; BSISO %.3f -> %.3f; map %.3f -> %.3f" % (a[:, 1].min(), eps[a[:, 1].argmin()], a[-1, 1],
      a[0, 0], a[-1, 0], a[0, 2], a[-1, 2]))
fig, ax = plt.subplots(figsize=(8.6 * CM, 4.8 * CM))
plt.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.2)
C_ERE, C_MAP, C_BS = "#d9622b", "#2a78d6", "#1a9e77"          # coloured (user, 21 Nov); line styles kept for grey print
ax.grid(color="#e4e3de", lw=0.6); ax.set_axisbelow(True)
ax.plot(eps, a[:, 1], color=C_ERE, lw=1.7, label=r"ERE: $1.0\,L_{\mathrm{ERE}}$")
ax.plot(eps, a[:, 2], color=C_MAP, lw=1.4, ls="--", label=r"map: $0.5\,L_{\mathrm{map}}$")
ax.plot(eps, a[:, 0], color=C_BS, lw=1.5, ls=":", label=r"BSISO: $0.5\,L_{\mathrm{BSISO}}$")
ie = int(np.argmin(a[:, 1]))
ax.plot(eps[ie], a[ie, 1], "o", ms=4.5, mfc="none", mec=K, mew=1.0)
ax.annotate(f"ERE lowest at epoch {eps[ie]}", (eps[ie], a[ie, 1]), xytext=(eps[ie] + 5, 0.78), fontsize=7.5, color=K, va="center",
            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6, shrinkB=3))
ax.set(xlim=(0, 61), xticks=range(0, 61, 10))
ax.set_xlabel("epoch", fontsize=8); ax.set_ylabel("held-out loss term", fontsize=8)
ax.tick_params(labelsize=7)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7.3, loc="lower right", bbox_to_anchor=(1.0, 0.24))
fig.savefig(os.path.join(OUTD, "ere_loss_terms.png"), dpi=300, facecolor="white")
plt.close(fig)
print("loss terms saved")
