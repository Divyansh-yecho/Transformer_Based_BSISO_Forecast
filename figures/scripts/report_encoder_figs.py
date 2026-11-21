# Report figures (21 Nov 2026), section "Attention is all you need", after the token matrix.
#  enc_layer_flow.png : one encoder layer in the poster's style (LN, multi-head self-attention, add, LN, FFN with GELU, add),
#                       with the size of the data at every step. Sizes are the model's (d = 32, 4 heads of 8, FFN 64).
#  attn_steps.png     : attention for one query, step by step (scores -> softmax weights -> weighted sum of values).
#                       The numbers are a worked example, not trained values.
#  gelu_curve.png     : GELU against ReLU.
# The attention and GELU figures use a light hand-drawn line effect (matplotlib path.sketch).
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np
from math import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(ROOT, "figures")
CM = 1 / 2.54
K, K2 = "#222222", "#555555"
LN_C, ATT_C, FFN_C, BG = "#fbdcaa", "#fff5a3", "#dfdcf6", "#f2f3f5"


def rbox(ax, x, y, w, h, fc, text, fs=8, ec=K, lw=0.9, style="round,pad=0,rounding_size=0.12", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=K, zorder=4, linespacing=1.15)


def arrow(ax, p0, p1, lw=0.9, col=K):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=col, lw=lw, shrinkA=0, shrinkB=0, zorder=2))


def plus(ax, x, y, r=0.16):
    ax.add_patch(Circle((x, y), r, fc="white", ec=K, lw=0.9, zorder=4))
    ax.plot([x - 0.6 * r, x + 0.6 * r], [y, y], color=K, lw=0.9, zorder=5)
    ax.plot([x, x], [y - 0.6 * r, y + 0.6 * r], color=K, lw=0.9, zorder=5)


# ======================================================================= 1) encoder layer, poster style, with sizes
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans"})
W, H = 8.6, 9.5
fig = plt.figure(figsize=(W * CM, H * CM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, W), ylim=(0, H)); ax.axis("off")
cx, bw = 3.05, 3.1                                              # centre and width of the main boxes
bx = cx - bw / 2
rbox(ax, bx - 0.35, 0.25, bw + 0.7, 0.62, "white", r"Token matrix $X$", fs=8)
# encoder box
ex0, ey0, ex1, ey1 = 0.55, 1.35, 5.55, 7.8
ax.add_patch(FancyBboxPatch((ex0, ey0), ex1 - ex0, ey1 - ey0, boxstyle="round,pad=0,rounding_size=0.08", fc=BG, ec=K2,
                            lw=0.9, ls=(0, (4, 2.5)), zorder=1))
ax.text(ex0, ey1 + 0.1, "Encoder layer", ha="left", va="bottom", fontsize=8.5, color=K)
ax.text(ex0 - 0.22, (ey0 + ey1) / 2, r"$N = 3$", rotation=90, ha="center", va="center", fontsize=8.5, color=K)
y_ln1, y_att, y_p1, y_ln2, y_ffn, y_p2 = 1.95, 2.85, 4.3, 4.9, 5.8, 7.3
rbox(ax, bx, y_ln1, bw, 0.5, LN_C, "Layer Norm")
rbox(ax, bx, y_att, bw, 0.95, ATT_C, "Multi-head\nself-attention", fs=8.5)
plus(ax, cx, y_p1)
rbox(ax, bx, y_ln2, bw, 0.5, LN_C, "Layer Norm")
rbox(ax, bx, y_ffn, bw, 0.95, FFN_C, "Feed-forward\n(64, GELU)", fs=8.5)
plus(ax, cx, y_p2)
arrow(ax, (cx, 0.87), (cx, y_ln1))
arrow(ax, (cx, y_ln1 + 0.5), (cx, y_att))
arrow(ax, (cx, y_att + 0.95), (cx, y_p1 - 0.16))
arrow(ax, (cx, y_p1 + 0.16), (cx, y_ln2))
arrow(ax, (cx, y_ln2 + 0.5), (cx, y_ffn))
arrow(ax, (cx, y_ffn + 0.95), (cx, y_p2 - 0.16))
# residual connections (left for attention, right for the feed-forward block)
xl, xr = ex0 + 0.35, ex1 - 0.35
yb1 = (0.87 + y_ln1) / 2 + 0.12
ax.plot([cx, xl, xl], [yb1, yb1, y_p1], color=K, lw=0.9, zorder=2)
arrow(ax, (xl, y_p1), (cx - 0.16, y_p1))
ax.text(xl - 0.12, (yb1 + y_p1) / 2, "residual", rotation=90, ha="center", va="center", fontsize=7, color=K2)
yb2 = (y_p1 + y_ln2) / 2 + 0.03
ax.plot([cx, xr, xr], [yb2, yb2, y_p2], color=K, lw=0.9, zorder=2)
arrow(ax, (xr, y_p2), (cx + 0.16, y_p2))
ax.text(xr + 0.12, (yb2 + y_p2) / 2, "residual", rotation=90, ha="center", va="center", fontsize=7, color=K2)
# output
arrow(ax, (cx, y_p2 + 0.16), (cx, 8.3))
rbox(ax, bx - 0.35, 8.3, bw + 0.7, 0.62, "white", r"New $X$ (richer rows)", fs=8)
# sizes on the right
sx = 5.75
sz = dict(ha="left", va="center", fontsize=7.2, color=K)
ax.text(sx, 0.56, r"$80\times32$", **sz)
ax.text(sx, y_ln1 + 0.25, r"$80\times32$", **sz)
ax.text(sx, y_att + 0.48, "per head:\n" r"$Q,K,V$: $80\times8$" "\n" r"weights: $80\times80$" "\n" r"4 heads $\rightarrow 80\times32$",
        ha="left", va="center", fontsize=7.0, color=K, linespacing=1.2)
ax.text(sx, y_p1, r"$80\times32$", **sz)
ax.text(sx, y_ln2 + 0.25, r"$80\times32$", **sz)
ax.text(sx, y_ffn + 0.48, r"$80\times32\rightarrow80\times64$" "\n" r"GELU" "\n" r"$80\times64\rightarrow80\times32$",
        ha="left", va="center", fontsize=7.0, color=K, linespacing=1.2)
ax.text(sx, y_p2, r"$80\times32$", **sz)
ax.text(sx, 8.61, r"$80\times32$", **sz)
fig.savefig(os.path.join(OUTD, "enc_layer_flow.png"), dpi=300, facecolor="white")
plt.close(fig)
print("encoder flow saved")

# ======================================================================= 2) attention for one query, step by step
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "path.sketch": (0.5, 70, 1.0)})
s = np.array([2.1, 0.3, -0.5, 1.4, 0.0])
a = np.exp(s) / np.exp(s).sum()
W, H = 8.6, 6.8
fig = plt.figure(figsize=(W * CM, H * CM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, W), ylim=(0, H)); ax.axis("off")
xs = 2.35 + np.arange(5) * 1.25
bwid = 0.8
lab = dict(ha="right", va="center", fontsize=8.5, color=K)
# row 1: the query and the keys
rbox(ax, 0.25, 5.45, 1.2, 0.5, "white", r"query $\mathbf{q}_i$", fs=8.5, style="round,pad=0,rounding_size=0.08")
for j, x in enumerate(xs):
    rbox(ax, x - bwid / 2, 5.45, bwid, 0.5, "white", rf"$\mathbf{{k}}_{j + 1}$", fs=9, style="round,pad=0,rounding_size=0.08")
ax.text(xs[0] - 0.6, 5.7, "keys", **lab) if False else None
# row 2: scores
ax.text(1.75, 4.62, r"score $s_j$", **lab)
ax.text(xs[2], 6.45, "compare the query with every key", ha="center", va="center", fontsize=8.5, color=K2)
for j, x in enumerate(xs):
    arrow(ax, (x, 5.43), (x, 4.85), lw=0.7, col=K2)
    ax.text(x, 4.62, f"{s[j]:.1f}", ha="center", va="center", fontsize=9, color=K)
# row 3: softmax weights as bars
ax.text(1.75, 3.35, r"softmax $a_j$", **lab)
ax.text(1.75, 3.0, "(sum = 1)", ha="right", va="center", fontsize=7.5, color=K2)
for j, x in enumerate(xs):
    arrow(ax, (x, 4.38), (x, 4.0), lw=0.7, col=K2)
    hb = 1.0 * a[j] / a.max()
    ax.add_patch(Rectangle((x - 0.25, 2.55), 0.5, hb, fc="#f2c9a3", ec=K, lw=0.8, zorder=3))
    ax.text(x, 2.55 + hb + 0.08, f"{a[j]:.2f}", ha="center", va="bottom", fontsize=8, color=K)
ax.plot([xs[0] - 0.5, xs[-1] + 0.5], [2.55, 2.55], color=K, lw=0.7)
# row 4: values, weighted
ax.text(1.75, 1.85, r"values $\mathbf{v}_j$", **lab)
for j, x in enumerate(xs):
    rbox(ax, x - bwid / 2, 1.6, bwid, 0.5, "#f2c9a3" if a[j] > 0.2 else "white", rf"$\mathbf{{v}}_{j + 1}$", fs=9,
         style="round,pad=0,rounding_size=0.08")
# row 5: weighted sum
ox = (xs[0] + xs[-1]) / 2
rbox(ax, ox - 1.55, 0.2, 3.1, 0.55, "white", r"$\mathbf{z}_i=a_1\mathbf{v}_1+\cdots+a_5\mathbf{v}_5$", fs=9.5,
     style="round,pad=0,rounding_size=0.08")
for j, x in enumerate(xs):
    arrow(ax, (x, 1.58), (ox + (x - ox) * 0.45, 0.77), lw=0.4 + 2.2 * a[j], col=K2)
ax.text(1.75, 0.47, "output", **lab)
fig.savefig(os.path.join(OUTD, "attn_steps.png"), dpi=300, facecolor="white")
plt.close(fig)
print("attention steps saved")

# ======================================================================= 3) GELU against ReLU
W, H = 8.6, 4.1
fig = plt.figure(figsize=(W * CM, H * CM))
ax = fig.add_axes([0.1, 0.13, 0.86, 0.82])
x = np.linspace(-3, 3, 400)
gelu = np.array([v * 0.5 * (1 + erf(v / np.sqrt(2))) for v in x])
relu = np.maximum(x, 0)
ax.plot(x, relu, color="#888888", lw=1.2, ls=(0, (4, 3)), label="ReLU")
ax.plot(x, gelu, color="#d9622b", lw=1.8, label=r"GELU $=x\,\Phi(x)$")
ax.axhline(0, color=K, lw=0.6); ax.axvline(0, color=K, lw=0.6)
ax.set(xlim=(-3, 3), ylim=(-0.6, 3.05))
ax.set_xlabel("input $x$", fontsize=8.5); ax.set_ylabel("output", fontsize=8.5)
ax.tick_params(labelsize=7.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
im = int(np.argmin(gelu))
ax.plot(x[im], gelu[im], "o", ms=3.5, color="#d9622b")
ax.annotate("small negative inputs\nleak through a little\n(lowest point " + f"{gelu[im]:.2f})", (x[im], gelu[im]),
            xytext=(-2.95, 0.75), fontsize=7.5, color=K, arrowprops=dict(arrowstyle="-", color=K2, lw=0.6))
ax.text(0.9, 2.55, "large inputs pass\nalmost unchanged", fontsize=7.5, color=K)
ax.text(-2.95, 0.12, "large negative inputs " r"$\rightarrow 0$", fontsize=7.5, color=K)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.savefig(os.path.join(OUTD, "gelu_curve.png"), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print("gelu saved")
