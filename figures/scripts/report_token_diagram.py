# Report figure (20 Nov 2026): schematic of how one fast field (MSLP or the IMD rain map) becomes 10 tokens.
# 30 daily maps (T-29..T), each already 12 numbers (20 EOF coefficients -> Linear(20->12)):
#   days T-29..T-5: 5 blocks of 5 days, 5 x 12 = 60 numbers -> Linear(60->32) -> 5 tokens
#   days T-4..T:    one day each, 12 numbers -> Linear(12->32) -> 5 tokens
# Clean version: large text, labels outside the arrows.
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figures", "token_diagram.png")
INK, INK2 = "#1f1f1d", "#52514e"
OLD, NEW = "#1f1f1d", "#1f1f1d"                                # black and white (21 Nov): older days white, last 5 days light grey
OLD_L, NEW_L = "#ffffff", "#d9d9d9"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix"})

XW, YH = 22.6, 8.6
fig = plt.figure(figsize=(6.2, 6.2 * YH / XW))
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, XW), ylim=(0, YH), aspect="equal"); ax.axis("off")


def t(x, y, s, size=13, **k):
    k.setdefault("color", INK); k.setdefault("ha", "center"); k.setdefault("va", "center")
    ax.text(x, y, s, fontsize=size, **k)


def arrow(x0, y0, x1, y1, col):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=col, lw=1.2,
                                 shrinkA=0, shrinkB=0))


# day boxes: 25 older days in 5 blocks, then the newest 5 days spread out
x0, dw, sq, yd, hd = 1.75, 0.5, 0.44, 6.0, 0.6
xs = [x0 + d * dw + 0.35 * (d // 5) for d in range(25)]
xs += [xs[-1] + 1.25 + d * 1.15 for d in range(5)]
for d, x in enumerate(xs):
    col, fc = (NEW, NEW_L) if d >= 25 else (OLD, OLD_L)
    ax.add_patch(Rectangle((x, yd), sq, hd, fc=fc, ec=col, lw=1.0))
t(xs[0] - 0.15, yd + hd / 2, r"$T-29$", 11, color=INK2, ha="right")
t(xs[-1] + sq + 0.15, yd + hd / 2, r"$T$", 11, color=INK2, ha="left")
blue_mid = (xs[0] + xs[24] + sq) / 2
red_mid = (xs[25] + xs[29] + sq) / 2
t(blue_mid, 8.3, "older days", 14, color=OLD)
t(blue_mid, 7.45, "one token per 5 days", 12, color=OLD)
t(red_mid, 8.3, "last 5 days", 14, color=NEW)
t(red_mid, 7.45, "one token per day", 12, color=NEW)

# tokens (6 cells stand for 32 numbers)
ty, th, tw = 1.75, 2.4, 0.5
cent = [((xs[5 * b] + xs[5 * b + 4] + sq) / 2, OLD, OLD_L) for b in range(5)]
cent += [(xs[d] + sq / 2, NEW, NEW_L) for d in range(25, 30)]
for b in range(5):                                           # bracket under each 5-day block
    a, z = xs[5 * b], xs[5 * b + 4] + sq
    ax.plot([a, a, z, z], [yd - 0.08, yd - 0.22, yd - 0.22, yd - 0.08], color=OLD, lw=1.0)
for c, col, fc in cent:
    arrow(c, yd - (0.22 if col == OLD else 0.0) - 0.02, c, ty + th + 0.05, col)
    for k in range(6):
        ax.add_patch(Rectangle((c - tw / 2, ty + k * th / 6), tw, th / 6, fc=fc, ec=col, lw=0.8))
t(blue_mid, 1.0, r"each: $5\times12=60$ numbers $\rightarrow$ Linear $\rightarrow$ 32", 12, color=OLD)
t(red_mid, 1.0, r"each: $12\rightarrow$ Linear $\rightarrow32$", 12, color=NEW)
fig.savefig(OUT, dpi=260, facecolor="white", bbox_inches="tight", pad_inches=0.04)
print("saved", OUT)
