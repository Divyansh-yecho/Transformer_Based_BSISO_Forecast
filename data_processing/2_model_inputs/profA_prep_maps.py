# Prof model A input maps on the base_inputs calendar (April-September 1979-2024, 8418 days):
#   olr  : merged NOAA OLR, 2.5 deg, 10S-30N 40-160E                 (data/olr_merged)
#   u850, u200 : ERA5 (WeatherBench2 5.625 deg; 2023-24 CDS 1.5 deg interpolated to the same grid), 10S-30N 40-120E
#   mslp : ERA5 1.5 deg, 10-30N 70-95E (WB2; 2023-24 CDS on the same lattice)
#   vo850, vo500 : ERA5 CDS 1.5 deg, 15-28N 72-92E (00/12 UTC daily mean), only if all data/era5_prof_extra/vo_*.nc exist
# Raw physical values are saved (float32); anomalies, scaling and EOFs are computed later from training years only.
# Output: other/intermediate/profA/maps.npz  (and maps_vo.npz for vorticity)
import os, sys, glob
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd, xarray as xr

END = int(os.environ.get("END_YEAR", "2024"))                          # 2025 test (file 32): END_YEAR=2025 OUT_DIR=.../profA_2025
SRC = os.path.join(DATA, "era5_wb2"); OUT = os.environ.get("OUT_DIR", os.path.join(WORK, "profA"))
os.makedirs(OUT, exist_ok=True)
DATES = pd.DatetimeIndex(np.load(os.environ.get("BASE_FILE", os.path.join(WORK, "modelA", "base_inputs.npz")))["dates"])
STAGE = sys.argv[1] if len(sys.argv) > 1 else "main"


def check(name, A):
    print(f"{name}: shape {A.shape}, NaN {np.isnan(A).mean():.4f}, JJAS mean {np.nanmean(A[(DATES.month >= 6) & (DATES.month <= 9)]):.3g}",
          flush=True)


if STAGE == "main":
    out = {"dates": DATES.strftime("%Y-%m-%d").values.astype("U10")}
    # OLR
    o = xr.open_dataset(os.path.join(DATA, "olr_merged", f"olr_daily_1979_{END}_merged.nc")).olr.sortby("lat")
    o = o.sel(lat=slice(-10, 30), lon=slice(40, 160)).sel(time=DATES.values)
    out["olr"] = o.values.astype("float32"); out["olr_lat"] = o.lat.values; out["olr_lon"] = o.lon.values
    check("olr", out["olr"])
    # ERA5 from WB2 (time, lon, lat) + CDS 2023/24
    z0 = np.load(os.path.join(SRC, "wb2_1979.npz"))
    lo_lat, lo_lon, hi_lat, hi_lon = z0["lo_lat"], z0["lo_lon"], z0["hi_lat"], z0["hi_lon"]
    mla = (lo_lat >= -10.5) & (lo_lat <= 30.5); mlo = (lo_lon >= 39) & (lo_lon <= 121)
    hla = (hi_lat >= 10) & (hi_lat <= 30); hlo = (hi_lon >= 70) & (hi_lon <= 95)
    U8, U2, MS, dd = [], [], [], []
    for yr in range(1979, 2023):
        z = np.load(os.path.join(SRC, f"wb2_{yr}.npz"))
        t = lambda a: np.asarray(a, "float32").transpose(0, 2, 1)       # -> (time, lat, lon)
        U8.append(t(z["lo_u_component_of_wind_850"])[:, mla][:, :, mlo]); U2.append(t(z["lo_u_component_of_wind_200"])[:, mla][:, :, mlo])
        MS.append(t(z["hi_mean_sea_level_pressure"])[:, hla][:, :, hlo]); dd.append(z["dates"])
    def cds_fields(yr):
        """CDS 00/12 UTC daily means of u850, u200 (on the 5.625 grid) and MSLP (1.5 grid), plus month of each day."""
        op = lambda f: xr.open_dataset(os.path.join(SRC, f))
        dm = lambda da: da.resample(valid_time="1D").mean().rename(valid_time="time").sortby("latitude")
        uv = xr.concat([op(f"cds{yr}_uva.nc"), op(f"cds{yr}_uvb.nc")], "valid_time"); sl = op(f"cds{yr}_sl.nc")
        def boxmean(da):                                                # WB2's 5.625 deg grid is a conservative (area) mean, so
            out = np.zeros((da.sizes["time"], mla.sum(), mlo.sum()), "float32")   # average the 1.5 deg CDS points in each 5.625 cell
            la, lo = da.latitude.values, da.longitude.values
            for a, c in enumerate(lo_lat[mla]):
                for b, d in enumerate(lo_lon[mlo]):
                    m = da.sel(latitude=la[np.abs(la - c) <= 2.8125], longitude=lo[np.abs(lo - d) <= 2.8125])
                    out[:, a, b] = m.mean(("latitude", "longitude")).values
            return out
        u8, u2 = (boxmean(dm(uv["u"].sel(pressure_level=lev))) for lev in (850, 200))
        ms = dm(sl["msl"]).reindex(latitude=hi_lat[hla], longitude=hi_lon[hlo], method="nearest", tolerance=0.01).values.astype("float32")
        tt = pd.DatetimeIndex(dm(sl["msl"])["time"].values)
        return u8, u2, ms, tt

    # 2023-24 exist only as CDS 00/12 UTC samples; WB2 daily means use 4 samples. The semi-diurnal tide biases a 00/12 mean, so a
    # per-grid-point, per-month bias (CDS 00/12 minus WB2) is measured on the 2022 overlap and removed from 2023-24.
    assert os.path.exists(os.path.join(SRC, "cds2022_sl.nc")), "need CDS 2022 (era5_cds_2024.py 2022) for the sampling-bias correction"
    c8, c2, cm, ct = cds_fields(2022)
    w = pd.DatetimeIndex(dd[-1]); pos = pd.Series(np.arange(len(w)), index=w).reindex(ct).values.astype(int)   # WB2 2022 = last WB2 year
    BIAS = {}
    for k, cv, wv in (("u850", c8, U8[-1]), ("u200", c2, U2[-1]), ("mslp", cm, MS[-1])):
        BIAS[k] = {m: (cv[ct.month == m] - wv[pos][ct.month == m]).mean(0) for m in range(4, 10)}
        print(f"2022 sampling bias {k}: mean {np.mean([b.mean() for b in BIAS[k].values()]):+.3f}, max |bias| {max(np.abs(b).max() for b in BIAS[k].values()):.3f}")
    for yr in range(2023, END + 1):
        u8, u2, ms, tt = cds_fields(yr)
        for k, a, L in (("u850", u8, U8), ("u200", u2, U2), ("mslp", ms, MS)):
            a = a.copy()
            for m in range(4, 10):
                a[tt.month == m] -= BIAS[k][m]
            L.append(a)
        dd.append(tt.strftime("%Y-%m-%d").values)
    ed = pd.DatetimeIndex(np.concatenate(dd))
    idx = pd.Series(np.arange(len(ed)), index=ed).reindex(DATES).values
    assert not np.isnan(idx.astype(float)).any(), "ERA5 dates missing"
    idx = idx.astype(int)
    for k, L in (("u850", U8), ("u200", U2), ("mslp", MS)):
        out[k] = np.concatenate(L, 0)[idx]; check(k, out[k])
    out["u_lat"], out["u_lon"] = lo_lat[mla], lo_lon[mlo]; out["mslp_lat"], out["mslp_lon"] = hi_lat[hla], hi_lon[hlo]
    np.savez_compressed(os.path.join(OUT, "maps.npz"), **out)
    print("saved maps.npz", {k: v.shape for k, v in out.items() if k in ("olr", "u850", "u200", "mslp")})

if STAGE == "vo":
    # 1981-2022: WeatherBench2 (1.5 deg conservative, daily mean of 4 six-hourly fields). 2023-24: CDS (00/12 UTC, 1.5 deg on a
    # lattice offset by 0.5 deg) interpolated to the WB2 points and matched to WB2 per grid point and month on the 2022 overlap
    # (CDS 2022 vs WB2 2022): x' = (x - mean_cds) * sd_wb2 / sd_cds + mean_wb2. Without CDS 2022 only the 00/12 sampling bias
    # measured inside WB2 2022 is removed (flagged). 1979-80 are NaN (never used: issue days start in 1981).
    VD = os.path.join(DATA, "era5_wb2_vo"); CD = os.path.join(DATA, "era5_prof_extra")
    wb, wd = {850: [], 500: []}, []
    for yr in range(1981, 2023):
        z = np.load(os.path.join(VD, f"vo_{yr}.npz"))
        for L in (850, 500):
            wb[L].append(z[f"vo{L}"].transpose(0, 2, 1))                   # (time, lon, lat) -> (time, lat, lon)
        wd.append(z["dates"])
    lat, lon = z["lat"], z["lon"]
    wd = pd.DatetimeIndex(np.concatenate(wd)); W = {L: np.concatenate(wb[L], 0) * 1e5 for L in wb}

    def cds_year(yr):
        d = xr.open_dataset(os.path.join(CD, f"vo_{yr}.nc")); tv = "valid_time" if "valid_time" in d.dims else "time"
        lv = "pressure_level" if "pressure_level" in d.dims else "level"
        da = d["vo"].resample({tv: "1D"}).mean().sortby("latitude").interp(latitude=lat, longitude=lon)
        return {L: da.sel({lv: L}).values * 1e5 for L in (850, 500)}, pd.DatetimeIndex(da[tv].values)

    i22 = wd.year == 2022; mon22 = wd[i22].month
    have22 = os.path.exists(os.path.join(CD, "vo_2022.nc"))
    if have22:
        c22, t22 = cds_year(2022); assert (t22 == wd[i22]).all()
        corr = {}
        for L in (850, 500):
            a, b = c22[L], W[L][i22]
            corr[L] = {m: (a[mon22 == m].mean(0), a[mon22 == m].std(0), b[mon22 == m].mean(0), b[mon22 == m].std(0)) for m in range(4, 10)}
            aa, bb = a - a.mean(0), b - b.mean(0); r = (aa * bb).sum(0) / np.sqrt((aa ** 2).sum(0) * (bb ** 2).sum(0))
            print(f"vo{L} 2022 CDS vs WB2: day-to-day corr median {np.median(r):.3f} min {r.min():.3f}; sd ratio median {np.median(a.std(0) / b.std(0)):.2f}")
    else:
        q = np.load(os.path.join(VD, "vo_2022_0012.npz"))
        corr = {L: {m: (q[f"vo{L}"].transpose(0, 2, 1)[mon22 == m].mean(0) * 1e5, 1.0, W[L][i22][mon22 == m].mean(0), 1.0) for m in range(4, 10)}
                for L in (850, 500)}
        print("WARNING: CDS 2022 vorticity missing -> only the 00/12 sampling bias (from WB2 2022) is removed for 2023-24")
    parts, pd_ = {850: [W[850]], 500: [W[500]]}, [wd]
    def box025(yr):
        """native 0.25 deg CDS, 00/06/12/18 UTC daily mean, averaged over each WB2 1.5 deg cell exactly like WB2's conservative
        regridding (7 x 7 points, edge points half weight, cos-lat weights). 2022 check: corr 0.9998 with WB2, sd ratio 1.00."""
        d = xr.open_dataset(os.path.join(CD, f"vo025_{yr}.nc")); v = d["vo"].resample(valid_time="1D").mean().sortby("latitude")
        la, lo = v.latitude.values, v.longitude.values; A = v.values; lv = list(v.pressure_level.values)
        out = np.zeros((A.shape[0], 2, len(lat), len(lon)), "float32")
        for a, c in enumerate(lat):
            wl = np.where(np.abs(la - c) <= 0.75 + 1e-6, np.where(np.isclose(np.abs(la - c), 0.75), 0.5, 1.0), 0.0) * np.cos(np.deg2rad(la))
            for b_, e in enumerate(lon):
                wo = np.where(np.abs(lo - e) <= 0.75 + 1e-6, np.where(np.isclose(np.abs(lo - e), 0.75), 0.5, 1.0), 0.0)
                Wt = wl[:, None] * wo[None]; out[:, :, a, b_] = (A * Wt).sum((-2, -1)) / Wt.sum()
        return {L: out[:, lv.index(L)] * 1e5 for L in (850, 500)}, pd.DatetimeIndex(v.valid_time.values)

    if all(os.path.exists(os.path.join(CD, f"vo025_{y}.nc")) for y in (2022, 2023, 2024)):
        c22, t22 = box025(2022)
        for L in (850, 500):
            a, b = c22[L], W[L][i22]; aa, bb = a - a.mean(0), b - b.mean(0); r = (aa * bb).sum(0) / np.sqrt((aa ** 2).sum(0) * (bb ** 2).sum(0))
            print(f"vo{L} 2022 check, box-averaged 0.25 CDS vs WB2: corr median {np.median(r):.4f} min {r.min():.4f}, sd ratio {np.median(a.std(0) / b.std(0)):.3f}")
            assert np.median(r) > 0.995, "0.25 deg CDS does not reproduce WB2"
        corr = {L: {m: (0.0, 1.0, 0.0, 1.0) for m in range(4, 10)} for L in (850, 500)}   # identical processing -> no adjustment
        cds_year = box025
        print("using native 0.25 deg CDS, box-averaged, for 2023-24 (no adjustment needed)")
    for yr in range(2023, END + 1):
        if not os.path.exists(os.path.join(CD, f"vo_{yr}.nc")) and not (cds_year is box025 and os.path.exists(os.path.join(CD, f"vo025_{yr}.nc"))):
            print(f"WARNING: CDS vorticity {yr} missing -> NaN (needed only for the final test)"); continue
        c, t = cds_year(yr)
        for L in (850, 500):
            x = c[L].copy()
            for m in range(4, 10):
                mc, sc, mw, sw = corr[L][m]; k = t.month == m; x[k] = (x[k] - mc) * (sw / sc) + mw
            parts[L].append(x.astype("float32"))
        pd_.append(t)
    tt = pd.DatetimeIndex(np.concatenate([np.asarray(p) for p in pd_]))
    pos = pd.Series(np.arange(len(tt)), index=tt).reindex(DATES).values
    out = {"vo_lat": lat, "vo_lon": lon}
    for L in (850, 500):
        A = np.concatenate(parts[L], 0); B = np.full((len(DATES),) + A.shape[1:], np.nan, "float32"); ok = np.isfinite(pos)
        B[ok] = A[pos[ok].astype(int)]; out[f"vo{L}"] = B
        check(f"vo{L} (1e-5 s-1)", B)
        j = (DATES.month >= 6) & (DATES.month <= 9)
        print(f"  JJAS mean |vo{L}| by year:", {y: round(float(np.nanmean(np.abs(B[j & (DATES.year == y)]))), 3) for y in range(2020, END + 1)})
    np.savez_compressed(os.path.join(OUT, "maps_vo.npz"), **out)
    print("saved maps_vo.npz", out["vo850"].shape)
