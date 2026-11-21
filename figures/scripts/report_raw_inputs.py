# Report figure (20 Nov 2026): what the raw input maps look like for one forecast day.
# Top: the MSLP grid of one day (14 x 17, ERA5), with a zoom on 3 x 3 cells showing the actual numbers.
# Bottom: the other five daily maps on their own grids. Each panel is drawn as a stack: the model reads the last 30 days.
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import common as nc

HERE = os.path.dirname(os.path.abspath(__file__))
PIN = os.path.join(WORK, "profA_ext")
OUT = os.path.join(ROOT, "figures", "raw_inputs.png")
INK, INK2 = "#1f1f1d", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix"})
PROJ = ccrs.PlateCarree()
DAY = pd.Timestamp("1990-07-15")

z = np.load(os.path.join(PIN, "maps.npz")); n = np.load(os.path.join(PIN, "maps_new.npz"))
i = int(np.where(pd.DatetimeIndex(z["dates"]) == DAY)[0][0])


def north_up(F, la):
    return (F[::-1], la[::-1]) if la[0] > la[-1] else (F, la)


mslp, mla = north_up(z["mslp"][i] / 100.0, z["mslp_lat"]); mlo = z["mslp_lon"]
u850, ula = north_up(z["u850"][i], z["u_lat"]); ulo = z["u_lon"]
z850, zla = north_up(n["z850"][i] / 9.80665, n["z_lat"]); zlo = n["z_lon"]
tcwv, tla = north_up(n["tcwv"][i], n["tcwv_lat"]); tlo = n["tcwv_lon"]
sst, _ = north_up(np.where(n["sst"][i] > 100, n["sst"][i] - 273.15, np.nan), n["tcwv_lat"])
ds = xr.open_dataset(os.path.join(nc.DATA, "imd_india_025_jjas_1979_2024.nc"))
tp = ds["tp"].sel(time=DAY).values.astype("float32"); la25, lo25 = ds["lat"].values, ds["lon"].values; ds.close()
mask = np.load(os.path.join(nc.DATA, "kaggle_bsiso_extras", "india_mask_025.npy")) & np.isfinite(tp)
ny, nx = (len(la25) - 1) // 4, (len(lo25) - 1) // 4
blk = mask[:ny * 4, :nx * 4].reshape(ny, 4, nx, 4)
M1 = blk.mean((1, 3)) >= 0.5
R1 = np.nanmean(np.where(blk, tp[:ny * 4, :nx * 4].reshape(ny, 4, nx, 4), np.nan), axis=(1, 3))
R1 = np.where(M1, R1, np.nan)
rla, rlo = la25[:ny * 4].reshape(ny, 4).mean(1), lo25[:nx * 4].reshape(nx, 4).mean(1)
R1, rla = north_up(R1, rla)
print("IMD cells:", int(np.isfinite(R1).sum()))


def ext_of(la, lo):
    dla, dlo = abs(np.diff(la).mean()), abs(np.diff(lo).mean())
    return (lo[0] - dlo / 2, min(lo[-1] + dlo / 2, 179.99), la[0] - dla / 2, la[-1] + dla / 2)


STACK = []


def stack_behind(fig, a, n=2, d=0.008):
    STACK.append((a, n, d))


def draw_stacks(fig):
    fig.canvas.draw()                                            # cartopy fixes the final box sizes here
    for a, n, d in STACK:
        p = a.get_position()
        a.set_zorder(5)
        for k in range(n, 0, -1):
            b = fig.add_axes([p.x0 - k * d, p.y0 - k * d * 1.4, p.width, p.height])
            b.set_zorder(5 - k); b.set_xticks([]); b.set_yticks([])
            b.patch.set_facecolor("white")
            for sp in b.spines.values():
                sp.set_edgecolor("#9b9a96"); sp.set_linewidth(0.7)


def mapax(fig, rect, F, la, lo, cmap, grid=False, vmin=None, vmax=None, coast_lw=0.5):
    a = fig.add_axes(rect, projection=PROJ)
    e = ext_of(la, lo)
    a.set_extent(e, crs=PROJ)
    im = a.imshow(F, origin="lower", extent=e, transform=PROJ, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    a.add_feature(cfeature.COASTLINE.with_scale("110m" if e[1] - e[0] > 60 else "50m"), lw=coast_lw, edgecolor=INK)
    if grid:
        dla, dlo = abs(np.diff(la).mean()), abs(np.diff(lo).mean())
        for xv in np.r_[lo - dlo / 2, lo[-1] + dlo / 2]:
            a.plot([xv, xv], e[2:], color="white", lw=0.5, alpha=0.8, transform=PROJ)
        for yv in np.r_[la - dla / 2, la[-1] + dla / 2]:
            a.plot(e[:2], [yv, yv], color="white", lw=0.5, alpha=0.8, transform=PROJ)
    for s in a.spines.values():
        s.set_edgecolor(INK2); s.set_linewidth(0.8)
    return a, im


fig = plt.figure(figsize=(8.0, 7.0))
# ---- top: MSLP grid + zoom on 3 x 3 cells
a, im = mapax(fig, [0.04, 0.50, 0.50, 0.44], mslp, mla, mlo, "Blues_r", grid=True, coast_lw=0.8)
stack_behind(fig, a)
a.set_title(f"MSLP on {DAY:%d %b %Y}: $14\\times17$ grid = 238 numbers", fontsize=11, loc="left")
iy0, ix0 = int(np.argmin(np.abs(mla - 21.0))), int(np.argmin(np.abs(mlo - 79.5)))   # 3 x 3 cells over central India
d = 1.5
x0, y0 = mlo[ix0] - d / 2, mla[iy0] - d / 2
a.add_patch(Rectangle((x0, y0), 3 * d, 3 * d, fill=False, ec="#e8590c", lw=1.8, transform=PROJ, zorder=6))
zx = fig.add_axes([0.64, 0.56, 0.31, 0.31])
cm = plt.get_cmap("Blues_r"); nrm = im.norm
for r in range(3):
    for c in range(3):
        v = mslp[iy0 + r, ix0 + c]
        zx.add_patch(Rectangle((c, r), 1, 1, fc=cm(nrm(v)), ec="white", lw=2))
        rr, gg, bb, _ = cm(nrm(v))
        zx.text(c + 0.5, r + 0.5, f"{v:.1f}", ha="center", va="center", fontsize=12,
                color="white" if 0.299 * rr + 0.587 * gg + 0.114 * bb < 0.55 else INK)
zx.set(xlim=(0, 3), ylim=(0, 3), aspect="equal"); zx.set_xticks([]); zx.set_yticks([])
for s in zx.spines.values():
    s.set_edgecolor("#e8590c"); s.set_linewidth(1.8)
zx.set_title("zoom: one number per cell (hPa)", fontsize=10.5)
for (xa, ya), (xb, yb) in (((x0 + 3 * d, y0 + 3 * d), (0, 3)), ((x0 + 3 * d, y0), (0, 0))):
    fig.add_artist(ConnectionPatch((xa, ya), (xb, yb), coordsA=a.transData, coordsB=zx.transData, color="#e8590c", lw=0.9))

# ---- bottom: the other five fields, each on its own grid
panels = [(u850, ula, ulo, "RdBu_r", r"850 hPa zonal wind, $7\times15$", True, (-18, 18)),
          (z850, zla, zlo, "viridis", r"850 hPa height, $11\times28$", True, None),
          (tcwv, tla, tlo, "YlGnBu", r"water vapour, $40\times100$", False, None),
          (sst, tla, tlo, "inferno", r"SST, $40\times100$", False, (20, 31)),
          (R1, rla, rlo, "Blues", "IMD rain, 283 cells", True, (0, 40))]
rects = [[0.04, 0.27, 0.27, 0.15], [0.37, 0.27, 0.27, 0.15], [0.70, 0.27, 0.27, 0.15],
         [0.04, 0.04, 0.27, 0.15], [0.37, 0.04, 0.27, 0.15]]
for (F, la, lo, cmap, ttl, grid, lim), rect in zip(panels, rects):
    vmin, vmax = (lim if lim else (np.nanpercentile(F, 2), np.nanpercentile(F, 98)))
    if F is R1:
        rect = [rect[0] + 0.06, rect[1], 0.15, rect[3]]
    a, _ = mapax(fig, rect, F, la, lo, cmap, grid=grid, vmin=vmin, vmax=vmax)
    stack_behind(fig, a)
    a.set_title(ttl, fontsize=9.5, loc="left")
fig.text(0.70, 0.115, "Each panel is one day.\nThe model reads the last\n30 days of every field.", fontsize=10.5,
         color=INK2, va="center", ha="left")
draw_stacks(fig)
fig.savefig(OUT, dpi=230, facecolor="white", bbox_inches="tight", pad_inches=0.05)
print("saved", OUT)
