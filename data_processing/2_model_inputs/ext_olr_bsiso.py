# Night run 13-14 Nov 2026 (file 38), processing steps P3 + P4, from NCEP/NCAR Reanalysis 1 (data/ncep_r1, 1959-2022; light enough
# for the night's slow link and one homogeneous source over the whole period).
# P3 OLR proxy: NCEP model OLR (T62 grid) bilinear to the 2.5 deg NOAA OLR grid (10S-40N 40-160E); per grid point OLR = a + b * x,
#    fitted on 1981-2013 (all days), checked on 2014-2022 (bias, RMSE, correlation of daily anomalies).
#    -> data/olr_merged/olr_daily_1959_2024_proxy_merged.nc (1959-1978 calibrated proxy; 1979-2024 the NOAA file unchanged).
# P4 BSISO index rebuild (Lee et al. 2013 style): daily OLR and u850 over 10S-40N 40-160E; anomaly from a 3-harmonic calendar
#    climatology (1981-2010) minus the mean of the previous 120 days; each field scaled by its sd; multivariate EOFs of May-Oct
#    1981-2010; ridge from the leading 20 PCs to the official PC1-4 (fitted 1981-2013, all days), variance matched on 1981-2013;
#    checked on 2014-2022 (per-PC and bivariate correlation, amplitude correlation, phase agreement).
#    Version A: NOAA OLR (proxy before 1979) + NCEP u850.  Version B: NCEP OLR + NCEP u850 (exists the same way in every year).
#    -> data/bsiso_extended/BSISO_extended_1959_2026.csv (official where it exists, i.e. from 1981; version B before; column source).
import os, sys, glob
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd, xarray as xr
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as nc
D = DATA; E = os.path.join(WORK, "ext")
os.makedirs(E, exist_ok=True); REP = []; say = lambda m: (print(m, flush=True), REP.append(m))
full = pd.date_range("1959-01-01", "2022-12-31")


def load(name):
    A, dd = [], []
    for y in range(1959, 2023):
        z = np.load(os.path.join(D, "ncep_r1", f"{name}_{y}.npz")); A.append(z["data"]); dd.append(z["dates"]); la, lo = z["lat"], z["lon"]
    dd = pd.DatetimeIndex(np.concatenate(dd)); A = np.concatenate(A)
    assert len(dd) == len(full) and (dd == full).all(), f"{name}: calendar not complete"
    s = np.argsort(la); return A[:, s], la[s], lo
NO, nla, nlo = load("olr"); NU, ula, ulo = load("u850")
say(f"NCEP OLR {NO.shape} mean {NO.mean():.1f} W m-2; NCEP u850 {NU.shape} mean {NU.mean():.2f} m/s; complete daily calendars 1959-2022; NaN {np.isnan(NO).mean() + np.isnan(NU).mean():.4f}")
O = xr.open_dataset(os.path.join(D, "olr_merged", "olr_daily_1979_2024_merged.nc")).olr.sortby("lat").sel(lat=slice(-10, 40), lon=slice(40, 160)).load()
gl, gn = O.lat.values, O.lon.values; G = np.stack(np.meshgrid(gl, gn, indexing="ij"), -1).reshape(-1, 2)
regrid = lambda A, la, lo: np.stack([RegularGridInterpolator((la, lo), A[i])(G).reshape(len(gl), len(gn)) for i in range(len(A))]).astype("float32")
X = regrid(NO, nla, nlo); Ub = regrid(NU, ula, ulo)                     # both on the 2.5 deg 10S-40N 40-160E grid

# ---------------- P3 OLR proxy
od = pd.DatetimeIndex(O.time.values); com = full[full >= od[0]]; io_ = od.get_indexer(com); ix = full.get_indexer(com)
Yo = O.values[io_]; Xc = X[ix]; tr = (com.year >= 1981) & (com.year <= 2013); va = (com.year >= 2014) & (com.year <= 2022)
xm, ym = Xc[tr].mean(0), Yo[tr].mean(0); b = ((Xc[tr] - xm) * (Yo[tr] - ym)).sum(0) / ((Xc[tr] - xm) ** 2).sum(0); a = ym - b * xm
Ph = a + b * X
def anom(A, d, ref):
    F = A.reshape(len(A), -1); cl = pd.DataFrame(F[ref]).groupby(d[ref].dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction="both").values
    cl = pd.DataFrame(np.vstack([cl[-30:], cl, cl[:30]])).rolling(31, center=True).mean().values[30:-30]
    return F - cl[d.dayofyear - 1]
Ya = anom(Yo, com, tr); Pa = anom(Ph[ix], com, tr); cv = np.array([np.corrcoef(Ya[va][:, j], Pa[va][:, j])[0, 1] for j in range(Ya.shape[1])])
say(f"OLR proxy, check 2014-2022 (not used for fitting): bias {np.mean(Ph[ix][va] - Yo[va]):+.2f} W m-2, RMSE {np.sqrt(np.mean((Ph[ix][va] - Yo[va]) ** 2)):.1f} "
    f"(NOAA daily sd {Yo[va].std():.1f}); correlation of daily anomalies median {np.median(cv):.3f}, 10th pct {np.percentile(cv, 10):.3f}, min {cv.min():.3f}")
early = full.year <= 1978
xr.Dataset({"olr": (("time", "lat", "lon"), np.concatenate([Ph[early], O.values]).astype("float32"))},
           coords={"time": np.concatenate([full[early].values, od.values]), "lat": gl, "lon": gn},
           attrs={"proxy_until": "1978-12-31", "source": "1959-1978: NCEP/NCAR R1 model OLR (ulwrf.ntat) calibrated per grid point to NOAA OLR on "
                  "1981-2013; 1979-2024: olr_daily_1979_2024_merged.nc unchanged", "made_by": "ext_olr_bsiso.py (file 38)"}
           ).to_netcdf(os.path.join(D, "olr_merged", "olr_daily_1959_2024_proxy_merged.nc"))
say("saved data/olr_merged/olr_daily_1959_2024_proxy_merged.nc (lat 10S-40N, lon 40-160E)")

# ---------------- P4 BSISO rebuild
B = nc.load_bsiso()[["p1", "p2", "q1", "q2"]].reindex(full)
def prep(F):
    F = F.reshape(len(F), -1).astype("float64"); w = 2 * np.pi * full.dayofyear.values / 365.25
    H = np.column_stack([np.ones(len(F))] + [f(k * w) for k in (1, 2, 3) for f in (np.sin, np.cos)]); ref = (full.year >= 1981) & (full.year <= 2010)
    A = F - H @ np.linalg.lstsq(H[ref], F[ref], rcond=None)[0]
    c = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(A, 0)]); prev = np.full_like(A, np.nan); k = np.arange(120, len(A))
    prev[k] = (c[k] - c[k - 120]) / 120; A = A - prev
    return A / np.nanstd(A[ref])
biv = lambda a_, b_: (a_ * b_).sum() / np.sqrt((a_ ** 2).sum() * (b_ ** 2).sum())
def rebuild(olr, tag):
    Z = np.concatenate([prep(olr), prep(Ub)], 1); ok = np.isfinite(Z).all(1)
    ref = ok & (full.year >= 1981) & (full.year <= 2010) & full.month.isin([5, 6, 7, 8, 9, 10])
    mu = Z[ref].mean(0); _, _, Vt = np.linalg.svd(Z[ref] - mu, full_matrices=False); P = (Z - mu) @ Vt[:20].T; P = P / P[ref].std(0)
    trn = ok & (full.year >= 1981) & (full.year <= 2013) & B.notna().all(1).values
    X1 = np.column_stack([np.ones(trn.sum()), P[trn]]); W = np.linalg.solve(X1.T @ X1 + np.diag(np.r_[0, np.ones(20)]), X1.T @ B.values[trn])
    Rb = np.column_stack([np.ones(len(P)), P]) @ W; Rb = Rb / Rb[trn].std(0) * B.values[trn].std(0); Rb[~ok] = np.nan
    for nm, m_ in (("all days", np.ones(len(full), bool)), ("JJAS", full.month.isin([6, 7, 8, 9]))):
        vv = m_ & ok & (full.year >= 2014) & (full.year <= 2022) & B.notna().all(1).values; o = B.values[vv]; r = Rb[vv]
        pc = [np.corrcoef(o[:, j], r[:, j])[0, 1] for j in range(4)]; amp = np.corrcoef(np.hypot(o[:, 0], o[:, 1]), np.hypot(r[:, 0], r[:, 1]))[0, 1]
        st = np.hypot(o[:, 0], o[:, 1]) > 1; ph = np.mean(nc.phase8(o[st, 0], o[st, 1]) == nc.phase8(r[st, 0], r[st, 1]))
        pm = np.mean(np.abs(((nc.phase8(o[st, 0], o[st, 1]) - nc.phase8(r[st, 0], r[st, 1]) + 4) % 8) - 4) <= 1)
        say(f"BSISO rebuild {tag}, check 2014-2022 {nm}: PC1 {pc[0]:.3f} PC2 {pc[1]:.3f} PC3 {pc[2]:.3f} PC4 {pc[3]:.3f} | bivariate BSISO1 "
            f"{biv(o[:, :2], r[:, :2]):.3f} BSISO2 {biv(o[:, 2:], r[:, 2:]):.3f} | amplitude corr {amp:.3f} | same phase {100 * ph:.0f}% (within 1 phase {100 * pm:.0f}%)")
    return Rb
olrA = np.full_like(X, np.nan); olrA[ix] = Yo
RA = rebuild(np.where(np.isnan(olrA), Ph, olrA), "A (NOAA OLR + NCEP u850)")
RB = rebuild(X, "B (NCEP OLR + NCEP u850; used before 1981)")
outd = os.path.join(D, "bsiso_extended"); os.makedirs(outd, exist_ok=True)
off = B.notna().all(1).values; use = ~off & np.isfinite(RB).all(1)
ext = pd.DataFrame(np.where(off[:, None], B.values, RB), index=full, columns=["p1", "p2", "q1", "q2"]); ext["source"] = np.where(off, "official", np.where(use, "rebuilt_NCEP", ""))
later = nc.load_bsiso()[["p1", "p2", "q1", "q2"]]; later = later[later.index > full[-1]]; later["source"] = "official"
ext = pd.concat([ext[ext.source != ""], later]); ext.index.name = "date"; ext.to_csv(os.path.join(outd, "BSISO_extended_1959_2026.csv"))
pd.DataFrame(RB, index=full, columns=["p1", "p2", "q1", "q2"]).to_csv(os.path.join(outd, "BSISO_rebuilt_versionB_1959_2022.csv"))
say(f"saved data/bsiso_extended/BSISO_extended_1959_2026.csv: {ext.index[0].date()} .. {ext.index[-1].date()}, {ext.source.value_counts().to_dict()}")
open(os.path.join(E, "olr_bsiso_report.txt"), "w").write("\n".join(REP))
