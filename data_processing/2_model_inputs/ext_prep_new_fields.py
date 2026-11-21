# File 42: Maeda's three extra fields as model inputs on the extended calendar (April-September 1959-2024, 12,078 days):
#   tcwv (precipitable water, kg m-2) and sst (K) on 1.5 deg, 30S-30N 30-180E; z850 (geopotential at 850 hPa, m2 s-2) on 5.625 deg, ~31S-31N 30-180E.
# Sources: 1959-1978 data/era5_wb2_ext/new_early; 1979-2022 data/era5_wb2 (tcwv, sst) + data/era5_wb2_ext/z850; 2023-2024 CDS (00/12 UTC,
# 1.5 deg; z850 averaged into the 5.625 deg boxes) matched to WeatherBench2 per grid point on the 2022 overlap (mean and sd of the
# daily values): x' = (x - mean_cds) / sd_cds * sd_wb2 + mean_wb2.  SST over land is NaN in every source and is set to 0 (a constant, so
# its anomaly is 0). Output: other/intermediate/profA_ext/maps_new.npz. Checks printed: calendar, NaN pattern, per-period statistics,
# year-to-year continuity at the source joins (1978/1979, 2022/2023).
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd, xarray as xr
D = DATA; I = WORK
MP = np.load(os.path.join(I, "profA_ext", "maps.npz")); DATES = pd.DatetimeIndex(MP["dates"])
z0 = np.load(os.path.join(D, "era5_wb2", "wb2_1979.npz")); hl, hn, ll, ln = z0["hi_lat"], z0["hi_lon"], z0["lo_lat"], z0["lo_lon"]
hla = (hl >= -30) & (hl <= 30); hlo = (hn >= 30) & (hn <= 180); lla = np.abs(ll) <= 31; llo = (ln >= 28) & (ln <= 181)
T = lambda a: np.asarray(a, "float32").transpose(0, 2, 1)               # (time, lon, lat) -> (time, lat, lon)
def wb2(y):
    if y <= 1978:
        z = np.load(os.path.join(D, "era5_wb2_ext", "new_early", f"wb2_{y}.npz")); zz = z
    else:
        z = np.load(os.path.join(D, "era5_wb2", f"wb2_{y}.npz")); zz = np.load(os.path.join(D, "era5_wb2_ext", "z850", f"wb2_{y}.npz"))
    assert np.allclose(z["hi_lat"], hl) and np.allclose(zz["lo_lat"], ll)
    return (T(z["hi_total_column_water_vapour"])[:, hla][:, :, hlo], T(z["hi_sea_surface_temperature"])[:, hla][:, :, hlo],
            T(zz["lo_geopotential_850"])[:, lla][:, :, llo], pd.DatetimeIndex(z["dates"]))
def cds(y):
    s = xr.open_dataset(os.path.join(D, "era5_wb2", f"cds{y}_sl.nc")); g = xr.open_dataset(os.path.join(D, "era5_wb2", f"cds{y}_z850.nc"))
    dm = lambda da: da.resample(valid_time="1D").mean().sortby("latitude")
    tc = dm(s["tcwv"]).sel(latitude=hl[hla], longitude=hn[hlo], method="nearest"); ss = dm(s["sst"]).sel(latitude=hl[hla], longitude=hn[hlo], method="nearest")
    assert np.allclose(tc.latitude.values, hl[hla]) and np.allclose(tc.longitude.values, hn[hlo])
    zg = dm(g["z"].isel(pressure_level=0)); la, lo = zg.latitude.values, zg.longitude.values; zv = zg.values
    Z = np.zeros((zv.shape[0], lla.sum(), llo.sum()), "float32")
    for a, c in enumerate(ll[lla]):
        for b, d in enumerate(ln[llo]):
            Z[:, a, b] = zv[:, np.abs(la - c) <= 2.8125][:, :, np.abs(lo - d) <= 2.8125].mean((1, 2))
    return tc.values.astype("float32"), ss.values.astype("float32"), Z, pd.DatetimeIndex(tc.valid_time.values)
out = {k: [] for k in ("tcwv", "sst", "z850")}; dd = []
for y in range(1959, 2023):
    a, b, c, d = wb2(y); out["tcwv"].append(a); out["sst"].append(b); out["z850"].append(c); dd.append(d)
# CDS 2022 overlap -> matching factors per grid point
w22, c22 = wb2(2022), cds(2022); assert (w22[3] == c22[3]).all()
match = {}
for j, k in enumerate(("tcwv", "sst", "z850")):
    mw, sw, mc, sc = np.nanmean(w22[j], 0), np.nanstd(w22[j], 0), np.nanmean(c22[j], 0), np.nanstd(c22[j], 0) + 1e-6
    match[k] = (mw, sw, mc, sc); r = np.corrcoef(np.nan_to_num(w22[j]).ravel(), np.nan_to_num(c22[j]).ravel())[0, 1]
    print(f"2022 overlap {k}: WB2 mean {np.nanmean(mw):.4g} vs CDS {np.nanmean(mc):.4g}; sd ratio {np.nanmean(sw / sc):.3f}; correlation of daily fields {r:.4f}")
for y in (2023, 2024):
    t = cds(y)
    for j, k in enumerate(("tcwv", "sst", "z850")):
        mw, sw, mc, sc = match[k]; out[k].append(((t[j] - mc) / sc * sw + mw).astype("float32"))
    dd.append(t[3])
dates = pd.DatetimeIndex(np.concatenate(dd)); assert len(dates) == len(DATES) and (dates == DATES).all(), "calendar differs from maps.npz"
A = {k: np.concatenate(v) for k, v in out.items()}
land = np.isnan(A["sst"]).all(0); print(f"\nSST land points (NaN in every day): {land.sum()} of {land.size}; other NaN: sst {np.isnan(A['sst'][:, ~land]).sum()}, "
                                         f"tcwv {np.isnan(A['tcwv']).sum()}, z850 {np.isnan(A['z850']).sum()}")
A["sst"] = np.where(land[None], 0.0, A["sst"]).astype("float32")
print("\nper-period April-September statistics (consistency check):")
for k in ("tcwv", "sst", "z850"):
    x = A[k] if k != "sst" else A[k][:, ~land]
    row = []
    for p0, p1 in ((1959, 1968), (1969, 1978), (1979, 1988), (1989, 1998), (1999, 2008), (2009, 2013), (2014, 2022), (2023, 2024)):
        m = (dates.year >= p0) & (dates.year <= p1); row.append(f"{p0}-{str(p1)[2:]} {np.nanmean(x[m]):.5g}")
    print(f"  {k:5s} " + " | ".join(row))
    ym = np.array([np.nanmean(x[dates.year == y]) for y in range(1959, 2025)]); dy = np.abs(np.diff(ym))
    print(f"        largest year-to-year jumps in the annual (Apr-Sep) mean: {', '.join(f'{1960 + i}: {dy[i]:.3g}' for i in np.argsort(dy)[::-1][:3])} | typical {np.median(dy):.3g}")
np.savez_compressed(os.path.join(I, "profA_ext", "maps_new.npz"), dates=dates.strftime("%Y-%m-%d").values.astype("U10"), tcwv=A["tcwv"], sst=A["sst"], z850=A["z850"],
                    tcwv_lat=hl[hla], tcwv_lon=hn[hlo], z_lat=ll[lla], z_lon=ln[llo])
print("\nsaved profA_ext/maps_new.npz", {k: v.shape for k, v in A.items()})
