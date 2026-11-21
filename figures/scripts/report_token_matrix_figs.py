# Report figures (21 Nov 2026), section "Attention is all you need", subsection on the token matrix.
# Plain black-and-white style (white boxes, thin black lines, one light grey for the learned rows).
#  tm_token_matrix.png      : the 80 x 32 token matrix X; rows 1-14 learned lead queries, rows 15-80 data tokens.
#  tm_position_embedding.png: data token = content + position vector; query token = learned vector.
# Group sizes and order are the model's (profA.py, config W3N).
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix"})
CM = 1 / 2.54
K = "black"
GREY = "#e6e6e6"
LW = 0.7

# ======================================================================= figure 1: the token matrix
W, H = 8.4, 7.5
fig = plt.figure(figsize=(W * CM, H * CM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, W), ylim=(0, H)); ax.axis("off")
mx, my, mw, mh = 2.6, 0.7, 2.6, 6.0
rh = mh / 80
rows = [("lead queries", 14), ("BSISO", 16), ("rain band", 6), ("maps", 44)]
y = my + mh
for k, (nm, n) in enumerate(rows):
    yb = y - n * rh
    ax.add_patch(Rectangle((mx, yb), mw, n * rh, fc=GREY if k == 0 else "white", ec=K, lw=LW))
    y = yb
# the six maps inside the maps block (thin dashed lines)
y = my + mh - 36 * rh
for nm, n in [("u850", 6), ("MSLP", 10), ("IMD rain", 10), ("TCWV", 6), ("SST", 6), ("Z850", 6)]:
    ax.text(mx + mw / 2, y - n * rh / 2, f"{nm} ({n})", ha="center", va="center", fontsize=7)
    y -= n * rh
    if y > my + 0.01:
        ax.plot([mx, mx + mw], [y, y], color=K, lw=0.4, ls=(0, (2, 2)))
# labels inside the first three blocks
y = my + mh
for nm, n in rows[:3]:
    ax.text(mx + mw / 2, y - n * rh / 2, f"{nm} ({n})", ha="center", va="center", fontsize=8)
    y -= n * rh
ax.text(mx + mw / 2, my + mh + 0.12, r"$X\in\mathbb{R}^{80\times32}$", ha="center", va="bottom", fontsize=10)
ax.annotate("", (mx, my - 0.2), (mx + mw, my - 0.2), arrowprops=dict(arrowstyle="<->", color=K, lw=0.6))
ax.text(mx + mw / 2, my - 0.28, "32 numbers per token", ha="center", va="top", fontsize=8)


def brace(x, ytop, ybot, side):
    s = 0.12 if side == "left" else -0.12
    ax.plot([x + s, x, x, x + s], [ytop - 0.02, ytop - 0.02, ybot + 0.02, ybot + 0.02], color=K, lw=LW)


yq = my + mh - 14 * rh
brace(mx - 0.3, my + mh, yq, "left")
brace(mx - 0.3, yq, my, "left")
ax.text(mx - 0.42, (my + mh + yq) / 2, "14 learned\ntokens", ha="right", va="center", fontsize=8.5, linespacing=1.15)
ax.text(mx - 0.42, (yq + my) / 2, "66 data\ntokens", ha="right", va="center", fontsize=8.5, linespacing=1.15)
# row numbers on the right
y = my + mh
r0 = 1
for nm, n in rows:
    ax.text(mx + mw + 0.12, y - n * rh / 2, f"rows {r0}–{r0 + n - 1}", ha="left", va="center", fontsize=7.5)
    y -= n * rh; r0 += n
fig.savefig(os.path.join(OUTD, "tm_token_matrix.png"), dpi=300, facecolor="white")
plt.close(fig)
print("token matrix saved")

# ======================================================================= figure 2: one row = content + position
W, H = 8.6, 3.0
fig = plt.figure(figsize=(W * CM, H * CM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, W), ylim=(0, H)); ax.axis("off")
sw, sh = 1.75, 0.32


def row_box(x, y, fill="white"):
    ax.add_patch(Rectangle((x, y), sw, sh, fc=fill, ec=K, lw=LW))
    for k in range(1, 8):                                       # a few ticks: a row of numbers
        ax.plot([x + k * sw / 8] * 2, [y, y + sh], color=K, lw=0.3)


xs = [1.2, 3.6, 6.0]
ya = 2.05
ax.text(0.1, ya + sh / 2, "data\ntoken", ha="left", va="center", fontsize=8, linespacing=1.1)
row_box(xs[0], ya); row_box(xs[1], ya); row_box(xs[2], ya)
ax.text((xs[0] + sw + xs[1]) / 2, ya + sh / 2, "+", ha="center", va="center", fontsize=12)
ax.text((xs[1] + sw + xs[2]) / 2, ya + sh / 2, "=", ha="center", va="center", fontsize=12)
for x, sym, txt in ((xs[0], r"$\mathbf{e}_i$", "content (inputs)"), (xs[1], r"$\mathbf{p}_i$", "position (learned)"),
                    (xs[2], r"$\mathbf{x}_i$", r"row $i$ of $X$")):
    ax.text(x + sw / 2, ya + sh + 0.07, sym, ha="center", va="bottom", fontsize=10)
    ax.text(x + sw / 2, ya - 0.07, txt, ha="center", va="top", fontsize=7.5)
yb = 0.5
ax.text(0.1, yb + sh / 2, "query\ntoken", ha="left", va="center", fontsize=8, linespacing=1.1)
row_box(xs[1], yb, GREY); row_box(xs[2], yb, GREY)
ax.text((xs[1] + sw + xs[2]) / 2, yb + sh / 2, "=", ha="center", va="center", fontsize=12)
ax.text(xs[1] + sw / 2, yb + sh + 0.07, r"$\mathbf{q}_k$", ha="center", va="bottom", fontsize=10)
ax.text(xs[2] + sw / 2, yb + sh + 0.07, r"$\mathbf{x}_k$", ha="center", va="bottom", fontsize=10)
ax.text(xs[1] + sw / 2, yb - 0.07, "learned, no input", ha="center", va="top", fontsize=7.5)
ax.text(xs[2] + sw / 2, yb - 0.07, r"row $k$ of $X$", ha="center", va="top", fontsize=7.5)
fig.savefig(os.path.join(OUTD, "tm_position_embedding.png"), dpi=300, facecolor="white")
plt.close(fig)
print("position figure saved")
