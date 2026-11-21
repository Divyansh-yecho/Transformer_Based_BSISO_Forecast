# Poster schematic of Model P's transformer (user, 15 Nov 2026), drawn in the style of the reference figure the user gave
# (dashed grey layer box, orange norm boxes, yellow attention, lavender feed-forward, pink linear projection, blue dense,
# green outputs, positional-encoding circles). Every number is read from profA.py config W3N (Model P):
#   d = 32, 3 encoder layers, 4 heads, feed-forward 64 (GELU), pre-norm (norm_first), final LayerNorm.
#   tokens: 14 learned lead queries + BSISO 16 (11 five-day + 5 daily) + rain band 6 (1 five-day + 5 daily)
#           + maps (20 EOFs/day -> 12 -> patch tokens): u850, TCWV, SST, Z850 6 each (five-day), MSLP and IMD rain 10 each
#           (5 five-day + 5 daily) = 80 tokens; learned position embedding per group; FiLM from Nino3.4, DMI, season.
#   heads: per lead Dense(32 -> BSISO PC1/PC2 correction); per week mean of its 7 lead states -> Dense(32 -> 2 ERE logits
#           + 30 rain-map EOF coefficients). Each output is added to its frozen linear part (not drawn: "just the transformer").
# Mini maps and curves are real inputs (1 Aug 2019 issue day) for illustration.
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
HERE = os.path.dirname(os.path.abspath(__file__)); D = KDATA
FIG = os.path.join(ROOT, "figures")
NORM, ATT, FFN, PROJ, DENSE, OUT, LAYER, PE = "#fbd9a6", "#fff59d", "#dcdcf8", "#f7a8c8", "#cfe3f6", "#cdeecb", "#f1f2f3", "#e3f1fb"
FILM = "#e6f4f1"; INK = "#111111"; INK2 = "#4a4a4a"
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(14, 10)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 140); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, fc, text, fs=11.5, bold=False, round_=0.6, ec=INK, lw=1.1, ls="-", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={round_}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=color, zorder=4,
                linespacing=1.15)


def arrow(x0, y0, x1, y1, lw=1.1, head=True):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>" if head else "-", lw=lw, color=INK, mutation_scale=11,
                                                       shrinkA=0, shrinkB=0), zorder=2)


def line(xs, ys, lw=1.1):
    ax.plot(xs, ys, color=INK, lw=lw, zorder=2, solid_capstyle="butt")


def pe(x, y, r=2.1):                                                       # positional-encoding circle with a sine wave
    ax.add_patch(Circle((x, y), r, fc=PE, ec=INK, lw=1.1, zorder=3))
    t = np.linspace(-1, 1, 60); ax.plot(x + 0.75 * r * t, y + 0.42 * r * np.sin(np.pi * t), color=INK, lw=1.0, zorder=4)


def plus(x, y, r=1.15):
    ax.add_patch(Circle((x, y), r, fc="white", ec=INK, lw=1.1, zorder=3)); ax.text(x, y - 0.05, "+", ha="center", va="center", fontsize=12, zorder=4)


def img(a, x, y, w, h, cmap, vlim=None, z=5, mask=None):
    a = np.array(a, float)
    if mask is not None:
        a = np.where(mask, a, np.nan)
    v = vlim or np.nanpercentile(np.abs(a), 98)
    ax.imshow(a, extent=(x, x + w, y, y + h), origin="lower", cmap=cmap, vmin=-v, vmax=v, aspect="auto", zorder=z, interpolation="bilinear")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=INK2, lw=0.7, zorder=z + 0.1))


def stack(arrs, x, y, w, h, cmap, dx=0.9, dy=0.9, mask=None):              # a small stack of map tiles (like the reference figure)
    for k, a in enumerate(arrs):
        img(a, x + k * dx, y + k * dy, w, h, cmap, z=5 + k, mask=mask)


# ---------------------------------------------------------------- real inputs for the icons (issue day 1 Aug 2019)
mp, mn, bi = np.load(os.path.join(D, "maps.npz")), np.load(os.path.join(D, "maps_new.npz")), np.load(os.path.join(D, "base_inputs.npz"))
dates = pd.DatetimeIndex(mp["dates"]); T = int(np.where(dates == pd.Timestamp("2019-08-01"))[0][0]); doy = dates.dayofyear.values
anom = lambda A, i: A[i] - A[np.abs(doy - doy[i]) <= 10].mean(0)
u850, mslp, tcwv = anom(mp["u850"], T), anom(mp["mslp"], T), anom(mn["tcwv"], T)
sst, z850 = anom(mn["sst"], T), anom(mn["z850"], T)
lr = np.log1p(bi["R1"]); rain = anom(lr, T); M1 = bi["M1"]
bsd = pd.DatetimeIndex(bi["bs_dates"]); bsv = pd.DataFrame(bi["bs"], index=bsd).loc[dates[T] - pd.Timedelta(days=59):dates[T]].values
band = bi["band"][T - 9:T + 1]; band = band - bi["band"][np.abs(doy - doy[T]) <= 10].mean(0)
wk = np.mean([anom(lr, T + k) for k in range(1, 8)], 0)

# ---------------------------------------------------------------- inputs (bottom)
ax.text(70, 1.2, "Inputs known on issue day T", ha="center", va="bottom", fontsize=12, color=INK2, style="italic")
# BSISO
gx = [4, 33, 58]
ax.add_patch(Rectangle((gx[0], 8), 22, 9, fc="white", ec=INK2, lw=0.7, zorder=4))
tt = np.arange(60)
for j, c in zip(range(2), ("#1f5fa8", "#c2452d")):
    ax.plot(gx[0] + 0.8 + 20.4 * tt / 59, 12.5 + 1.6 * bsv[:, j], color=c, lw=1.2, zorder=5)
ax.text(gx[0] + 11, 6.6, "BSISO index\nPC1–4 + amplitudes, days T−59…T", ha="center", va="top", fontsize=10.5)
# rain band
img(band.T, gx[1] + 1, 8, 16, 9, "BrBG", z=5)
ax.text(gx[1] + 9, 6.6, "IMD rain band (25 latitudes)\n+ central-India rain, days T−9…T", ha="center", va="top", fontsize=10.5)
# field maps (EOF coefficients per day)
tiles = [(u850, "RdBu_r", None, "u850"), (mslp, "RdBu_r", None, "MSLP"), (rain, "BrBG", M1, "IMD rain"),
         (tcwv, "BrBG", None, "TCWV*"), (sst, "RdBu_r", None, "SST*"), (z850, "RdBu_r", None, "Z850*")]
for k, (a, cm, msk, nm) in enumerate(tiles):
    x0 = gx[2] + k * 9.4
    stack([a, a, a], x0, 9.2, 6.6, 5.6, cm, dx=0.55, dy=0.55, mask=msk)
    ax.text(x0 + 3.9, 17.9, nm, ha="center", va="bottom", fontsize=10)
ax.text(gx[2] + 27.5, 6.6, "Daily maps → 20 EOF coefficients each, days T−29…T\n(* slow background removed)", ha="center", va="top", fontsize=10.5)
# lead queries
box(119, 9, 17, 8, "white", "14 lead queries\nT+1 … T+14\n(learned)", fs=10.5, ls="--", ec=INK2)

# token patches + linear projection
tok = [(gx[0] + 11, "16 tokens\n11 five-day + 5 daily"), (gx[1] + 9, "6 tokens\n1 five-day + 5 daily"),
       (gx[2] + 27.5, "44 tokens: 6 five-day per map\nMSLP, IMD rain: 5 five-day + 5 daily")]
for cx, t in tok:
    arrow(cx, 20.6 if cx > 60 else 18.2, cx, 22.6)
    w = 30 if cx > 60 else 20
    box(cx - w / 2, 22.6, w, 5.6, PROJ, "Linear Projection", fs=11.5, round_=0.9)
    ax.text(cx + w / 2 + 0.6, 25.4, "", fontsize=9)
    ax.text(cx, 31.0, t, ha="center", va="center", fontsize=9.4, color=INK2, zorder=6, bbox=dict(fc="white", ec="none", pad=0.6))
# positional embeddings
for cx in [tok[0][0], tok[1][0], tok[2][0], 127.5]:
    y0 = 37.4
    if cx == 127.5:                                                    # the lead queries are learned vectors: no position embedding
        line([cx, cx], [17, 39.5]); continue
    line([cx, cx], [28.2, y0 - 2.1])
    pe(cx, y0)
ax.text(tok[0][0] - 3.2, 37.4, "Position\nembedding", ha="right", va="center", fontsize=10)
# merge into one sequence
yS = 43.6
for cx in [tok[0][0], tok[1][0], tok[2][0], 127.5]:
    line([cx, cx], [39.5, yS])
line([tok[0][0], 127.5], [yS, yS])
box(40, yS + 1.6, 46, 4.4, "white", "Token sequence: 80 tokens × 32 numbers", fs=11, ec=INK2)
arrow(63, yS, 63, yS + 1.6)
# FiLM
box(89, yS + 1.4, 20, 4.8, FILM, "FiLM scale & shift\nNiño3.4, DMI, season", fs=10.3)
arrow(89, yS + 3.8, 86, yS + 3.8)

# ---------------------------------------------------------------- encoder layer
ex, ey, ew, eh = 38, 54, 50, 37
ax.add_patch(FancyBboxPatch((ex, ey), ew, eh, boxstyle="round,pad=0,rounding_size=0.4", fc=LAYER, ec=INK, lw=1.2, ls=(0, (5, 3)), zorder=1))
ax.text(ex + 1.2, ey + eh + 0.8, "Encoder Layer", ha="left", va="bottom", fontsize=12.5)
ax.text(ex - 1.6, ey + eh - 5, "N = 3", ha="center", va="center", rotation=90, fontsize=12)
ax.text(ex + ew - 1.2, ey + 1.2, "4 heads, width 32", ha="right", va="bottom", fontsize=9.6, color=INK2)
cx = ex + ew / 2; arrow(63, yS + 6, cx, ey + 3.4)
box(cx - 10, ey + 3.4, 20, 3.6, NORM, "Layer Norm", fs=11)
arrow(cx, ey + 7.0, cx, ey + 8.6)
box(cx - 11, ey + 8.6, 22, 7.0, ATT, "Multi-head\nself-attention", fs=11.5)
arrow(cx, ey + 15.6, cx, ey + 17.3); plus(cx, ey + 18.45)
line([cx, cx - 16, cx - 16], [ey + 1.9, ey + 1.9, ey + 18.45]); arrow(cx - 16, ey + 18.45, cx - 1.15, ey + 18.45)
line([cx, cx], [ey + 0.0, ey + 3.4])
arrow(cx, ey + 19.6, cx, ey + 21.4)
box(cx - 10, ey + 21.4, 20, 3.6, NORM, "Layer Norm", fs=11)
arrow(cx, ey + 25.0, cx, ey + 26.6)
box(cx - 11, ey + 26.6, 22, 6.2, FFN, "Feed forward\nNetwork (64, GELU)", fs=11)
arrow(cx, ey + 32.8, cx, ey + 34.0); plus(cx, ey + 35.15)
line([cx, cx + 16, cx + 16], [ey + 20.3, ey + 20.3, ey + 35.15]); arrow(cx + 16, ey + 35.15, cx + 1.15, ey + 35.15)
ax.text(cx - 17.2, ey + 10, "Residual connection", rotation=90, ha="center", va="center", fontsize=9.4)
# final norm and lead states
line([cx, cx], [ey + 36.3, ey + eh + 3.0]); arrow(cx, ey + eh + 3.0, cx, ey + eh + 3.6)
box(cx - 10, ey + eh + 3.6, 20, 3.6, NORM, "Layer Norm", fs=11)
line([cx, cx], [ey + eh + 7.2, 98.0]); line([cx, 98], [98.0, 98.0])
ax.text(cx + 11.5, 98.6, "14 lead states (outputs at the 14 query positions)", ha="left", va="bottom", fontsize=10.3)

# ---------------------------------------------------------------- heads and outputs (right)
line([98, 98], [98.0, 64.0])
# BSISO head
yb = 84; arrow(98, yb, 103, yb)
box(103, yb - 6, 3.6, 12, DENSE, "", round_=0.4); ax.text(104.8, yb, "Dense", rotation=90, ha="center", va="center", fontsize=10.5, zorder=5)
arrow(106.6, yb, 110, yb); box(110, yb - 4.2, 28, 8.4, OUT, "BSISO PC1, PC2\ndays 1–14", fs=11.5, round_=0.9)
ax.text(96.8, yb, "each\nlead", ha="right", va="center", fontsize=9, color=INK2)
# weekly heads
yw = 64; arrow(98, yw, 103, yw)
ax.text(96.8, yw, "mean over\ndays 1–7\nand 8–14", ha="right", va="center", fontsize=9, color=INK2)
box(103, yw - 9, 3.6, 18, DENSE, "", round_=0.4); ax.text(104.8, yw, "Dense", rotation=90, ha="center", va="center", fontsize=10.5, zorder=5)
arrow(106.6, yw + 5, 110, yw + 5); box(110, yw + 1.2, 28, 7.6, OUT, "Weekly ERE class\nweeks 1 and 2", fs=11.5, round_=0.9)
arrow(106.6, yw - 5, 110, yw - 5); box(110, yw - 9.6, 28, 9.2, OUT, "", round_=0.9)
ax.text(111.5, yw - 5, "Weekly rain-\nanomaly map\n(30 EOFs →\n283 cells)", ha="left", va="center", fontsize=10.2, zorder=6, linespacing=1.1)
stack([wk, wk], 125.6, yw - 8.6, 9.0, 6.6, "BrBG", dx=0.6, dy=0.6, mask=M1)
ax.text(124, yw - 10.6, "Each output is the transformer's correction,\nadded to that head's frozen linear part.", ha="center", va="top",
        fontsize=9.6, color=INK2, style="italic")
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(FIG, f"7_transformer_schematic.{ext}"), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15)
print("saved 7_transformer_schematic.png/pdf/svg")
