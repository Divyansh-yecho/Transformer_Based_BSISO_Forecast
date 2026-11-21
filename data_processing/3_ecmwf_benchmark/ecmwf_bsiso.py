# ECMWF's own BSISO forecast skill (file 41), for the poster's BSISO benchmark.
# ECMWF does not issue a BSISO index, but the index is computed from OLR and u850 fields, and ECMWF forecasts both. Standard verification
# (as for the MJO / BSISO in S2S studies): project ECMWF's forecast OLR and u850 onto the BSISO patterns.
#   Mapping ("version A", file 38): daily NOAA OLR + NCEP u850 on the 2.5 deg grid 10S-40N 40-160E; anomaly from a 3-harmonic calendar
#   climatology (1981-2010); minus the mean of the previous 120 days; each field / its sd; 20 multivariate EOFs (May-Oct 1981-2010);
#   ridge to the official PC1-4 (fitted 1981-2013); variance matched on 1981-2013. On 2014-2022 it reproduces the official BSISO1 at 0.988.
#   ECMWF forecast for day D = S + k (start S, 00 UTC): u850 = mean of steps k and k+1; OLR = -(ttr[k+1] - ttr[k]) / 86400; ensemble
#   mean of 11 members (control + 10). Anomaly = forecast minus ECMWF's own climatology for that lead and calendar day (+-15 d,
#   leave-one-year-out over 2005-2024 hindcasts). The previous-120-day mean uses observed anomalies before S and forecast anomalies
#   from S on. Then the same scaling, EOFs, ridge and variance factors as the observed index.
# Scored exactly like our models: test issue days 2014-2024 (start on T, else T-1), bivariate correlation of PC1/PC2 with the official
# index per lead 1-14, horizon (correlation >= 0.5). Output: intermediate/maphead/ecmwf_bsiso.csv and ecmwf_bsiso_forecasts.npz
import os, sys, glob
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd, xarray as xr
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as nc
D_ = DATA; OUT = os.path.join(WORK, "maphead")
KW = dict(engine="cfgrib", backend_kwargs={"indexpath": ""})
full = pd.date_range("1979-01-01", "2024-12-31")
# ---------------- observed fields on the 2.5 deg grid
O = xr.open_dataset(os.path.join(D_, "olr_merged", "olr_daily_1979_2024_merged.nc")).olr.sortby("lat").sel(lat=slice(-10, 40), lon=slice(40, 160)).load()
gl, gn = O.lat.values, O.lon.values; G = np.stack(np.meshgrid(gl, gn, indexing="ij"), -1).reshape(-1, 2)
olr_o = O.reindex(time=full).values.astype("float64")
def dates_of(z):
    d = z["dates"]
    try:
        return pd.DatetimeIndex(d)
    except Exception:
        return pd.Timestamp("1800-01-01") + pd.to_timedelta(d.astype(float), unit="h")
U, ud = [], []
for y in range(1979, 2025):
    z = np.load(os.path.join(D_, "ncep_r1", f"u850_{y}.npz")); la, lo = z["lat"], z["lon"]; s = np.argsort(la)
    A = z["data"][:, s]; ip = RegularGridInterpolator((la[s], lo), np.moveaxis(A, 0, -1))
    U.append(np.moveaxis(ip(G), -1, 0).reshape(len(A), len(gl), len(gn))); ud.append(dates_of(z))
ud = pd.DatetimeIndex(np.concatenate(ud)); assert (ud == full).all(); u_o = np.concatenate(U).astype("float64")
assert np.isfinite(olr_o).all() and np.isfinite(u_o).all(), "observed fields incomplete"
# ---------------- version-A mapping (fitted on observations only)
w = 2 * np.pi * full.dayofyear.values / 365.25
H = np.column_stack([np.ones(len(full))] + [f(k * w) for k in (1, 2, 3) for f in (np.sin, np.cos)]); ref = (full.year >= 1981) & (full.year <= 2010)
def harm(F):
    F = F.reshape(len(F), -1); return F, np.linalg.lstsq(H[ref], F[ref], rcond=None)[0]
def prev120(A):
    c = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(A, 0)]); p = np.full_like(A, np.nan); k = np.arange(120, len(A)); p[k] = (c[k] - c[k - 120]) / 120; return p
Fo, Co = harm(olr_o); Fu, Cu = harm(u_o)
Ao, Au = Fo - H @ Co, Fu - H @ Cu                                       # harmonic anomalies (observed)
Zo, Zu = Ao - prev120(Ao), Au - prev120(Au); so, su = np.nanstd(Zo[ref]), np.nanstd(Zu[ref])
Z = np.concatenate([Zo / so, Zu / su], 1); ok = np.isfinite(Z).all(1)
B = nc.load_bsiso()[["p1", "p2", "q1", "q2"]].reindex(full)
r2 = ok & ref & full.month.isin([5, 6, 7, 8, 9, 10]); mu = Z[r2].mean(0); _, _, Vt = np.linalg.svd(Z[r2] - mu, full_matrices=False); Vt = Vt[:20]
P = (Z - mu) @ Vt.T; ps = P[r2].std(0); P = P / ps
trn = ok & (full.year >= 1981) & (full.year <= 2013) & B.notna().all(1).values
X1 = np.column_stack([np.ones(trn.sum()), P[trn]]); W = np.linalg.solve(X1.T @ X1 + np.diag(np.r_[0, np.ones(20)]), X1.T @ B.values[trn])
Rb = np.column_stack([np.ones(len(P)), P]) @ W; vf = B.values[trn].std(0) / Rb[trn].std(0)
to_pcs = lambda z: (np.column_stack([np.ones(len(z)), ((z - mu) @ Vt.T) / ps]) @ W) * vf
chk = (full.year >= 2014) & (full.year <= 2022) & ok & B.notna().all(1).values
bv = lambda a, b: (a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum())
print(f"mapping check on observations 2014-2022: bivariate BSISO1 {bv(to_pcs(Z[chk])[:, :2], B.values[chk][:, :2]):.3f}", flush=True)
# ---------------- ECMWF forecasts: daily-mean OLR and u850 per start and lead day 0..20 (ensemble mean)
def ens(kind, md):
    out = []
    for ft in ("cf", "pf"):
        ds = xr.open_dataset(os.path.join(D_, "s2s_ecmwf", f"{kind}_{ft}_{md}.grib"), **KW); da = ds[list(ds.data_vars)[0]]
        if "isobaricInhPa" in da.dims:
            da = da.sel(isobaricInhPa=850)
        out.append(da.sortby("latitude").mean("number") * 10 if "number" in da.dims else da.sortby("latitude"))
    return (out[0] + out[1]) / 11.0
mds = sorted({os.path.basename(f).split("_")[2].split(".")[0] for f in glob.glob(os.path.join(D_, "s2s_ecmwf", "ttr_pf_*.grib"))})
ST, FU, FO = [], [], []
for md in mds:
    u = ens("pl", md); t = ens("ttr", md); assert np.allclose(u.latitude.values, gl) and np.allclose(u.longitude.values, gn)
    uu = u.values; tt = t.values; ST.append(pd.DatetimeIndex(u.time.values).normalize())
    FU.append(0.5 * (uu[:, :21] + uu[:, 1:22])); FO.append(-np.diff(tt, axis=1)[:, :21] / 86400.0)
    print(md, end=" ", flush=True)
ST = pd.DatetimeIndex(np.concatenate(ST)); FU = np.concatenate(FU).reshape(len(ST), 21, -1); FO = np.concatenate(FO).reshape(len(ST), 21, -1)
o_ = np.argsort(ST.values); ST, FU, FO = ST[o_], FU[o_], FO[o_]
print(f"\nECMWF starts: {len(ST)} ({ST.min().date()} .. {ST.max().date()}); OLR mean {FO.mean():.1f} W m-2, u850 mean {FU.mean():.2f} m/s", flush=True)
# model climatology per lead and calendar day, leave one year out
def model_anom(F):
    A = np.full_like(F, np.nan); yrs = ST.year.values
    for k in range(F.shape[1]):
        dk = (ST + pd.Timedelta(days=k)).dayofyear.values
        for y in np.unique(yrs):
            o = yrs != y
            for i in np.where(yrs == y)[0]:
                m = o & (np.minimum(np.abs(dk - dk[i]), 366 - np.abs(dk - dk[i])) <= 15); A[i, k] = F[i, k] - F[m, k].mean(0)
    return A
AO, AU = model_anom(FO), model_anom(FU)
# forecast PCs: previous-120-day mean from observed anomalies before S and forecast anomalies from S on
pos = full.get_indexer(ST); co = np.vstack([np.zeros((1, Ao.shape[1])), np.cumsum(Ao, 0)]); cu = np.vstack([np.zeros((1, Au.shape[1])), np.cumsum(Au, 0)])
PC = np.full((len(ST), 21, 4), np.nan)
for i, p0 in enumerate(pos):
    zz = []
    for k in range(21):
        nobs = 120 - k                                                   # days D-120 .. S-1 observed, S .. D-1 forecast
        so_ = (co[p0] - co[p0 - nobs]) + AO[i, :k].sum(0); su_ = (cu[p0] - cu[p0 - nobs]) + AU[i, :k].sum(0)
        zz.append(np.concatenate([(AO[i, k] - so_ / 120) / so, (AU[i, k] - su_ / 120) / su]))
    PC[i] = to_pcs(np.stack(zz))
# ---------------- score on our test issue days (start on T, else T-1), leads 1-14
iss = np.load(os.path.join(OUT, "cache_H7W.npz")); ISSd = pd.DatetimeIndex(iss["DATES"])[iss["ISS"]]; IY = iss["IY"]; AUX = iss["AUX"]
te = np.where(IY >= 2014)[0]; spos = pd.Series(np.arange(len(ST)), index=ST)
F = np.full((len(te), 14, 2), np.nan)
for n, r in enumerate(te):
    T = ISSd[r]
    for o in (1, 0):                                                    # T wins over T-1
        s = spos.get(T - pd.Timedelta(days=o))
        if s is not None:
            F[n] = PC[s, o + 1:o + 15, :2]
okr = np.isfinite(F).all((1, 2)); print(f"test issue days with an ECMWF start on T or T-1: {okr.sum()} of {len(te)}", flush=True)
O_ = AUX[te][okr]; Fv = F[okr]; okb = np.isfinite(O_).all(2)
cor = np.array([bv(Fv[okb[:, k], k], O_[okb[:, k], k]) for k in range(14)])
b_ = np.where(cor < 0.5)[0]; hz = 14.0 if not len(b_) else b_[0] + (cor[b_[0] - 1] - 0.5) / (cor[b_[0] - 1] - cor[b_[0]])
print("ECMWF BSISO1 bivariate correlation, test 2014-2024, leads 1-14:", np.round(cor, 3).tolist())
print(f"days 8-14 mean {cor[7:].mean():.3f}; horizon (>= 0.5) {hz:.1f} d" if len(b_) else f"days 8-14 mean {cor[7:].mean():.3f}; correlation >= 0.5 through day 14 (day 14: {cor[13]:.3f})")
pd.DataFrame({"lead": range(1, 15), "ecmwf_cor": cor}).to_csv(os.path.join(OUT, "ecmwf_bsiso.csv"), index=False)
np.savez_compressed(os.path.join(OUT, "ecmwf_bsiso_forecasts.npz"), F=F, rows=te, starts=ST.values.astype("datetime64[D]").astype(str), PC=PC.astype("float32"))
