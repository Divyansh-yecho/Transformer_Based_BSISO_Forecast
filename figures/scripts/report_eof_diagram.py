# Report figure (20 Nov 2026): schematic of EOF compression (not real data).
# The last 30 days of a field (stacked cards); each day = c1 x EOF1 + c2 x EOF2 + ... + c20 x EOF20;
# the 20 coefficients of each of the 30 days go to the model (a 30 x 20 input).
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figures", "eof_diagram.png")
INK, INK2, BLUE = "#1f1f1d", "#52514e", "#2a78d6"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix"})

NY, NX = 7, 9                                                   # a small grid that still looks like a grid
y, x = np.meshgrid(np.linspace(-1, 1, NY), np.linspace(-1, 1, NX), indexing="ij")
E = [np.exp(-((x - 0.1) ** 2 + (y + 0.1) ** 2) / 1.2),           # EOF 1: one large dome
     np.tanh(1.6 * y),                                           # EOF 2: north-south seesaw
     np.tanh(1.6 * x) * (0.6 + 0.4 * np.cos(1.2 * y)),           # EOF 3: west-east seesaw
     np.sin(3.3 * x + 0.6) * np.cos(3.0 * y)]                    # EOF 20: small-scale pattern
E = [e / np.abs(e).max() for e in E]
DAY = -1.0 * E[0] + 0.35 * E[1] - 0.2 * E[2] + 0.05 * E[3]
LIM = np.abs(DAY).max()
CMAP = plt.get_cmap("RdBu_r")

XW, YH = 30.4, 7.8
fig = plt.figure(figsize=(10, 10 * YH / XW))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, XW), ylim=(0, YH), aspect="equal")
ax.axis("off")
Z = [10]                                                        # draw order: later tiles on top


def tile(x0, yc, f, w, lim=1.0, fade=0.0):
    h = w * NY / NX
    ext = (x0, x0 + w, yc - h / 2, yc + h / 2)
    z = Z[0]; Z[0] += 3
    rgba = CMAP((f + lim) / (2 * lim))
    rgba[..., :3] = rgba[..., :3] * (1 - fade) + fade             # fade towards white for the cards behind
    ax.imshow(rgba, origin="lower", interpolation="nearest", extent=ext, zorder=z)
    for i in range(1, NX):
        ax.plot([x0 + i * w / NX] * 2, ext[2:], color="white", lw=0.8, zorder=z + 1)
    for j in range(1, NY):
        ax.plot(ext[:2], [ext[2] + j * h / NY] * 2, color="white", lw=0.8, zorder=z + 1)
    ax.add_patch(Rectangle((x0, ext[2]), w, h, fill=False, ec=INK2, lw=0.9, zorder=z + 2))
    return x0 + w / 2, ext[2]


def t(x0, y0, s, size=13, **k):
    k.setdefault("color", INK)
    ax.text(x0, y0, s, ha="center", va="center", fontsize=size, zorder=100, **k)


# the last 30 days, stacked: back cards first
YC, DW = 4.35, 3.2
for k, off in enumerate((0.7, 0.35)):
    tile(1.0 - off, YC - off, np.roll(DAY, 2 - k, axis=1), DW, lim=LIM, fade=0.45 - 0.2 * k)
cx, bot = tile(1.0, YC, DAY, DW, lim=LIM)
t(cx - 0.35, 6.95, "Last 30 days of MSLP anomaly")
t(cx - 0.35, bot - 1.1, r"each day: $14\times17=238$ numbers", 12)

W = 2.5
starts = [7.9, 13.9, 19.9, 27.7]
names = ["EOF 1", "EOF 2", "EOF 3", "EOF 20"]
ops = [r"$=\ \ c_1\,\times$", r"$+\ \ c_2\,\times$", r"$+\ \ c_3\,\times$", r"$+\ \cdots\ +\ \ c_{20}\,\times$"]
prev_end = 1.0 + DW
for x0, f, nm, op in zip(starts, E, names, ops):
    t((prev_end + x0 - 0.5) / 2, YC - 0.25, op, 15)
    for k, (off, amp) in enumerate(((0.5, 0.55), (0.25, 0.8))):   # earlier days: same pattern, another coefficient
        tile(x0 - off, YC - off, amp * f, W, fade=0.45 - 0.2 * k)
    cx, bot = tile(x0, YC, f, W)
    t(cx - 0.25, bot - 0.95, nm, 12)
    prev_end = x0 + W
t(18.3, 6.95, "fixed patterns, found once from the training years", 13, color=INK2)

# the coefficients of each day, stacked the same way: a 30 x 20 input
labels = [r"$c_1$", r"$c_2$", r"$c_3$", r"$\cdots$", r"$c_{20}$"]
bx, by, bw, bh = 8.8, 0.55, 1.25, 0.95
for off, fc in ((0.3, "#f7f9fc"), (0.15, "#f2f6fb")):
    for i in range(5):
        ax.add_patch(FancyBboxPatch((bx + i * bw - off, by - off), bw - 0.15, bh, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc=fc, ec="#8db5e8", lw=0.8, zorder=50))
for i, l in enumerate(labels):
    ax.add_patch(FancyBboxPatch((bx + i * bw, by), bw - 0.15, bh, boxstyle="round,pad=0,rounding_size=0.15",
                                fc="#eef3fa", ec=BLUE, lw=1.1, zorder=60))
    t(bx + i * bw + (bw - 0.15) / 2, by + bh / 2, l, 14)
ax.text(bx + 5 * bw + 0.35, by + bh / 2, r"30 days $\times$ 20 numbers go to the transformer", fontsize=13, color=INK2, ha="left", va="center", zorder=100)
ax.add_patch(FancyArrowPatch((2.3, 1.6), (bx - 0.55, by + bh / 2 - 0.1), connectionstyle="arc3,rad=0.3",
                             arrowstyle="-|>", mutation_scale=16, color=INK2, lw=1.2, zorder=70))
fig.savefig(OUT, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.05)
print("saved", OUT)
