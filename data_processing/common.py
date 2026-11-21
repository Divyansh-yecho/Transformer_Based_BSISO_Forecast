# Shared helpers for the data-processing scripts (first written for the Nov 5-6 overnight tests): data, event catalogues at 0.25 and 1 degree, the rain-step
# logistic, the normal-chance logit, and RPSS / bootstrap helpers. Builds a cache once (other/intermediate/night/).
import os
import numpy as np, pandas as pd, xarray as xr
from scipy import ndimage

import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import DATA, WORK  # repo-wide paths, see paths.py
NIGHT = os.path.join(WORK, "night")
FIGN = os.path.join(WORK, "figs_night")
os.makedirs(NIGHT, exist_ok=True); os.makedirs(FIGN, exist_ok=True)
CACHE = os.path.join(NIGHT, "events_cache.npz")


def ci_box(la, lo):
    return (la >= 17) & (la < 26.5) & (lo >= 74) & (lo <= 88.5) & ~((lo <= 77.5) & (la <= 21))


def load_bsiso():
    b = pd.read_csv(os.path.join(DATA, "BSISO.INDEX.NORM.LY.data"), sep=r"\s+", skiprows=1, header=None,
                    names=["year", "doy", "p1", "p2", "q1", "q2", "a1", "a2"]).replace(-999.9, np.nan)
    b.index = pd.to_datetime(b.year * 1000 + b.doy, format="%Y%j")
    b = b.asfreq("D")
    b[["p1", "p2", "q1", "q2"]] = b[["p1", "p2", "q1", "q2"]].interpolate(limit=2, limit_area="inside")
    return b


def phase8(p1, p2):
    """BSISO1 phase convention of the project: floor(deg(atan2(PC1, -PC2)) mod 360 / 45) + 1."""
    return (np.floor((np.degrees(np.arctan2(p1, -p2)) % 360) / 45) + 1).astype(int)


def clusters(ev, mask, LA, LO):
    """largest 8-connected cluster size (cells) with centroid in central India, per day."""
    out = np.zeros(len(ev))
    for t in range(len(ev)):
        L, n = ndimage.label(mask & ev[t], structure=np.ones((3, 3)))
        if n:
            idx = np.arange(1, n + 1)
            sz = np.asarray(ndimage.sum(np.ones(mask.shape), L, idx))
            la, lo = np.asarray(ndimage.mean(LA, L, idx)), np.asarray(ndimage.mean(LO, L, idx))
            ok = ci_box(la, lo)
            if ok.any():
                out[t] = sz[ok].max()
    return out


def build_cache():
    ds = xr.open_dataset(os.path.join(DATA, "imerg_india_025_jjas_2000_2024.nc"))
    tp = ds["tp"].values.astype("float32")
    lat, lon = ds["lat"].values, ds["lon"].values
    times = pd.DatetimeIndex(ds["time"].values).normalize()
    ds.close()
    mask = np.load(os.path.join(DATA, "kaggle_bsiso_extras", "india_mask_025.npy"))
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    # ---- 0.25 deg ERE (identical to the final package)
    THR = np.maximum(np.nanpercentile(np.where(tp > 1, tp, np.nan), 99, axis=0), 50).astype("float32")
    ev = np.zeros(tp.shape, bool)
    for t in range(len(times)):
        ev[t] = tp[t] > THR
    MX25 = np.zeros(len(times))
    for t in range(len(times)):                                    # >= 16 cells only, as the package
        L, n = ndimage.label(mask & ev[t], structure=np.ones((3, 3)))
        if n:
            idx = np.arange(1, n + 1)
            sz = np.asarray(ndimage.sum(np.ones(mask.shape), L, idx))
            la, lo = np.asarray(ndimage.mean(LA, L, idx)), np.asarray(ndimage.mean(LO, L, idx))
            ok = ci_box(la, lo) & (sz >= 16)
            if ok.any():
                MX25[t] = sz[ok].max()
    del ev
    ci25 = mask & ci_box(LA, LO)
    CIR = np.log1p(tp[:, ci25].mean(1))
    # ---- 1 deg grid: 4 x 4 block means (lat 5-39, lon 60-99 edges), India if >= 50% of the block is India
    ny, nx = (len(lat) - 1) // 4, (len(lon) - 1) // 4
    tpc = tp[:, :ny * 4, :nx * 4].reshape(len(times), ny, 4, nx, 4)
    mk = mask[:ny * 4, :nx * 4].reshape(ny, 4, nx, 4).mean((1, 3))
    R1 = np.nanmean(np.where(mask[:ny * 4, :nx * 4].reshape(ny, 4, nx, 4)[None], tpc, np.nan), axis=(2, 4)).astype("float32")
    lat1 = lat[:ny * 4].reshape(ny, 4).mean(1); lon1 = lon[:nx * 4].reshape(nx, 4).mean(1)
    M1 = mk >= 0.5
    R1 = np.where(M1[None], R1, np.nan)
    LA1, LO1 = np.meshgrid(lat1, lon1, indexing="ij")
    del tp, tpc
    out = {}
    for q in (99, 95):
        T1 = np.nanpercentile(np.where(R1 > 1, R1, np.nan), q, axis=0)
        out[f"MX1_p{q}"] = clusters(R1 > T1[None], M1, LA1, LO1)
        out[f"THR1_p{q}"] = T1
    np.savez_compressed(CACHE, times=times.values, MX25=MX25, CIR=CIR, R1=R1, M1=M1, lat1=lat1, lon1=lon1, **out)
    print("cache built:", CACHE)


def load():
    if not os.path.exists(CACHE):
        build_cache()
    z = np.load(CACHE, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["times"] = pd.DatetimeIndex(d["times"])
    return d


# ---------------------------------------------------------------- rain-step helpers (as the final package)
def fit_predict(Xtr, ytr, Xte, l2=1e-3):
    Xtr = np.column_stack([np.ones(len(Xtr)), Xtr]); Xte = np.column_stack([np.ones(len(Xte)), Xte])
    w = np.zeros(Xtr.shape[1]); R_ = l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
    for _ in range(30):
        p = 1 / (1 + np.exp(-np.clip(Xtr @ w, -30, 30)))
        step = np.linalg.solve((Xtr * (p * (1 - p))[:, None]).T @ Xtr + R_, Xtr.T @ (p - ytr) + R_ @ w)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return 1 / (1 + np.exp(-np.clip(Xte @ w, -30, 30)))


def clim_logit(y, yrs, pos, yr, npos, smooth=15):
    c = np.array([np.nanmean(np.where((yrs != yr) & (pos == k), y, np.nan)) for k in range(npos)])
    c = np.clip(pd.Series(c).rolling(smooth, center=True, min_periods=1).mean().values, 0.002, 0.998)
    return np.log(c / (1 - c))[pos][:, None]


def cum_eval(YC, Xf, years, kidx, ND, G=None):
    """leave-one-year-out cumulative logistic (normal-chance logit + Xf) for every threshold in YC.
    Returns model and normal-chance cumulative probabilities (min-accumulated) and the scored days."""
    good = np.isfinite(Xf).all(1) if Xf is not None else np.ones(len(years), bool)
    if G is not None:
        good &= G
    PM, PR = np.full((len(YC), len(years)), np.nan), np.full((len(YC), len(years)), np.nan)
    for yr in np.unique(years):
        te, tr = good & (years == yr), good & (years != yr)
        if not te.any():
            continue
        for c, y in enumerate(YC):
            cl = clim_logit(y, years, kidx, yr, ND)
            PR[c, te] = fit_predict(cl[tr], y[tr], cl[te])
            X = cl if Xf is None else np.column_stack([cl, Xf])
            PM[c, te] = fit_predict(X[tr], y[tr], X[te])
    return np.minimum.accumulate(PM, 0), np.minimum.accumulate(PR, 0), good


def rps_by_year(P, YC, G, years, UY):
    Yy = np.array(YC)
    r = np.where(G, ((np.nan_to_num(P) - Yy) ** 2).sum(0), 0)
    return np.array([r[years == y].sum() for y in UY])


BI = np.random.default_rng(0).integers(0, 25, (2000, 25))


def rpss(rm, rr):
    return 1 - rm.sum() / rr.sum()


def gain(rm_a, rm_b, rr, UY):
    """RPSS(b) - RPSS(a) with 90% range over year resamples and both halves."""
    sk = lambda rm: 1 - rm[BI].sum(1) / rr[BI].sum(1)
    d = sk(rm_b) - sk(rm_a)
    h1, h2 = UY <= 2011, UY >= 2012
    hs = lambda rm, m: 1 - rm[m].sum() / rr[m].sum()
    return {"gain": d.mean(), "lo": np.percentile(d, 5), "hi": np.percentile(d, 95),
            "h1": hs(rm_b, h1) - hs(rm_a, h1), "h2": hs(rm_b, h2) - hs(rm_a, h2)}


def passes(g):
    return g["lo"] > 0 and g["h1"] > 0 and g["h2"] > 0
