# Night run 13-14 Nov 2026 (file 38), processing step P1: IMD 0.25 deg rain of the early years -> the same model inputs as
# modelA_prep_base.py (1-deg 4x4 block means on the 283 Indian cells, central-India rain, rain-band profile 6-31N over 70-95E),
# April-September, with the date shifted back one day exactly as before. The land mask, valid-cell mask and block mask are taken from
# the 1979 file, exactly as modelA_prep_base.py did, so the early years use identical cells.
# Check: reprocessing 1979 with this code must reproduce base_inputs.npz (R1, cir, band) exactly.
# Output: other/intermediate/ext/imd_early.npz (dates, R1, cir, band) ; usage: python ext_prep_imd.py 1951 1978
import os, sys
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, xarray as xr
import common as nc

Y0, Y1 = int(sys.argv[1]), int(sys.argv[2])
OUT = os.path.join(nc.WORK, "ext"); os.makedirs(OUT, exist_ok=True)
SRC_OLD, SRC_NEW = os.path.join(nc.DATA, "imd_rain_025"), os.path.join(nc.DATA, "imd_rain_025_early")
ref = xr.open_dataset(os.path.join(nc.DATA, "imerg_india_025_jjas_2000_2024.nc")); LAT, LON = ref["lat"].values, ref["lon"].values; ref.close()
lat_i = 6.5 + 0.25 * np.arange(129); lon_i = 66.5 + 0.25 * np.arange(135)
ia = np.array([np.argmin(np.abs(LAT - v)) for v in lat_i]); io = np.array([np.argmin(np.abs(LON - v)) for v in lon_i])
mask = np.load(os.path.join(nc.DATA, "kaggle_bsiso_extras", "india_mask_025.npy"))
ny, nx = (len(LAT) - 1) // 4, (len(LON) - 1) // 4; LA, LO = np.meshgrid(LAT, LON, indexing="ij")


def load(y, src):
    nd = 366 if pd.Timestamp(y, 12, 31).dayofyear == 366 else 365
    a = np.fromfile(os.path.join(src, f"ind{y}_rfp25.grd"), "<f4").reshape(nd, 129, 135)
    lab = pd.date_range(f"{y}-01-01", periods=nd) - pd.Timedelta(days=1); s = (lab.month >= 4) & (lab.month <= 9)
    g = np.full((s.sum(), len(LAT), len(LON)), np.nan, "float32"); g[:, ia[:, None], io[None, :]] = np.where(a[s] < -900, np.nan, a[s])
    return lab[s], g


_, g79 = load(1979, SRC_OLD)                                            # masks exactly as modelA_prep_base.py (first year = 1979)
valid = np.isfinite(g79).mean(0) > 0.99; m = mask & valid
blk = m[:ny * 4, :nx * 4].reshape(ny, 4, nx, 4); M1 = blk.mean((1, 3)) >= 0.5; ci = m & nc.ci_box(LA, LO)
lat1 = LAT[:ny * 4].reshape(ny, 4).mean(1); lon1 = LON[:nx * 4].reshape(nx, 4).mean(1); rows = (lat1 >= 6) & (lat1 <= 31); cols = (lon1 >= 70) & (lon1 <= 95)


def proc(lab, g):
    g = np.where(m[None], g, np.nan)
    R1 = np.nanmean(np.where(blk[None], g[:, :ny * 4, :nx * 4].reshape(len(g), ny, 4, nx, 4), np.nan), axis=(2, 4))
    R1 = np.where(M1[None], R1, np.nan).astype("float32"); cir = np.nanmean(g[:, ci], 1).astype("float32")
    band = np.nanmean(R1[:, rows][:, :, cols], axis=2); band = np.where(np.isfinite(band), band, 0).astype("float32")
    return R1, cir, band


B = np.load(os.path.join(nc.WORK, "modelA", "base_inputs.npz")); bd = pd.DatetimeIndex(B["dates"])
lab, g = load(1979, SRC_OLD); R1, cir, band = proc(lab, g); sel = (bd.year == 1979) & bd.isin(lab)
ok = (np.array_equal(np.nan_to_num(R1, nan=-9), np.nan_to_num(B["R1"][sel], nan=-9)) and np.allclose(cir, B["cir"][sel], equal_nan=True)
      and np.allclose(band, B["band"][sel])); assert (B["M1"] == M1).all()
print("CHECK reprocessing 1979 reproduces base_inputs.npz exactly:", ok, flush=True); assert ok
D, RR, CC, BB, rep = [], [], [], [], []
for y in range(Y0, Y1 + 1):
    lab, g = load(y, SRC_NEW); R1, cir, band = proc(lab, g)
    nanf = float(np.isnan(R1[:, M1]).mean()); D.append(lab); RR.append(R1); CC.append(cir); BB.append(band)
    jj = (lab.month >= 6) & (lab.month <= 9)
    rep.append(dict(year=y, days=len(lab), nan_cells=nanf, jjas_mean_mm=float(np.nanmean(R1[jj][:, M1])), ci_jjas_mm=float(np.nanmean(cir[jj])),
                    max_mm=float(np.nanmax(R1))))
R = pd.DataFrame(rep); print(R.round(3).to_string(index=False))
ref_ = [float(np.nanmean(B["R1"][(bd.year == y) & (bd.month >= 6) & (bd.month <= 9)][:, B["M1"]])) for y in range(1979, 2014)]
print(f"reference 1979-2013 JJAS mean rain {np.mean(ref_):.2f} mm/d (sd over years {np.std(ref_):.2f}); early years {R.jjas_mean_mm.mean():.2f} (sd {R.jjas_mean_mm.std():.2f})")
dates = pd.DatetimeIndex(np.concatenate(D))
np.savez_compressed(os.path.join(OUT, "imd_early.npz"), dates=dates.values.astype("datetime64[D]").astype(str), R1=np.concatenate(RR), cir=np.concatenate(CC),
                    band=np.concatenate(BB), M1=M1, lat1=lat1, lon1=lon1)
R.to_csv(os.path.join(OUT, "imd_early_check.csv"), index=False); print("saved", len(dates), "days")
