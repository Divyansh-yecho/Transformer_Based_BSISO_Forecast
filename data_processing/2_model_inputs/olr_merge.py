# Merged daily OLR for the Prof model, 1979-2024, 2.5 deg, 20S-40N 30-180E.
#   1979-2022: NOAA interpolated OLR (Liebmann & Smith; ends 31 Dec 2022)  -> data/olr_noaa_interp/olr.day.mean.nc
#   2023-2024: CPC Blended OLR (CBO), corrected to the interpolated product -> data/olr.cbo-2.5deg.day.mean.nc
# Correction, fitted on the 1991-2022 overlap, per grid point:
#   corrected = (CBO - clim_CBO[doy]) * sd_interp / sd_CBO + clim_interp[doy]
#   clim = day-of-year mean smoothed with a 31-day running window; sd = std of daily anomalies.
# Missing CBO days are filled by linear interpolation in time (at most a few days).
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd, xarray as xr

D = DATA
OUT = os.path.join(D, "olr_merged"); os.makedirs(OUT, exist_ok=True)
box = lambda x: x.sortby("lat").sel(lat=slice(-20, 40), lon=slice(30, 180))
I = box(xr.open_dataset(os.path.join(D, "olr_noaa_interp", "olr.day.mean.nc")).olr).sel(time=slice("1979", "2022")).load()
END = os.environ.get("END_YEAR", "2024")                                # 2025 test (file 32)
C = box(xr.open_dataset(os.environ.get("CBO_FILE", os.path.join(D, "olr.cbo-2.5deg.day.mean.nc"))).olr).sel(time=slice("1991", END)).load()
C = C.interpolate_na("time", limit=5)
# NOAA filled some satellite gaps by copying the previous day (e.g. 21-25 Sep 1983): treat exact repeats as missing and
# interpolate linearly in time between the neighbouring real days (same for CBO)
for nm in ("I", "C"):
    X = globals()[nm]
    rep = np.r_[False, (X.diff("time") == 0).all(("lat", "lon")).values]
    print(f"{nm}: {int(rep.sum())} days identical to the previous day -> interpolated", list(pd.DatetimeIndex(X.time.values[rep]).strftime("%Y-%m-%d"))[:25])
    globals()[nm] = X.where(~xr.DataArray(rep, coords={"time": X.time}, dims="time")).interpolate_na("time")
assert np.allclose(I.lat, C.lat) and np.allclose(I.lon, C.lon), "grids differ"


def clim(x):
    c = x.groupby("time.dayofyear").mean()
    v = np.concatenate([c.values[-15:], c.values, c.values[:15]])
    v = np.stack([v[k:k + 31].mean(0) for k in range(len(c))])
    return xr.DataArray(v, coords=c.coords, dims=c.dims)


Io, Co = xr.align(I.sel(time=slice("1991", "2022")), C.sel(time=slice("1991", "2022")), join="inner")
ci, cc = clim(Io), clim(Co)
ai = Io.groupby("time.dayofyear") - ci; ac = Co.groupby("time.dayofyear") - cc
ratio = ai.std("time") / ac.std("time")
C2 = C.sel(time=slice("2023", END))
fix = ((C2.groupby("time.dayofyear") - cc) * ratio).groupby("time.dayofyear") + ci
fix = fix.drop_vars("dayofyear", errors="ignore").transpose("time", "lat", "lon")
M = xr.concat([I, fix], "time").astype("float32")
M.name = "olr"
M.attrs = {"units": "W m-2", "source": "1979-2022 NOAA interpolated OLR; 2023-2024 CPC Blended OLR corrected to it (see olr_merge.py)"}

# checks: overlap agreement before/after correction, and continuity at the 2022/2023 join
fo = ((Co.groupby("time.dayofyear") - cc) * ratio).groupby("time.dayofyear") + ci
rms_raw, rms_fix = float(np.sqrt(((Co - Io) ** 2).mean())), float(np.sqrt(((fo - Io) ** 2).mean()))
jja = lambda x: x.sel(time=x.time.dt.month.isin([6, 7, 8])).mean().item()
print(f"overlap 1991-2022: rms diff raw {rms_raw:.2f} -> corrected {rms_fix:.2f} W/m2; mean bias raw {float((Co - Io).mean()):.2f} -> {float((fo - Io).mean()):.2f}")
print("JJA mean OLR by year (W/m2):", {y: round(jja(M.sel(time=str(y))), 1) for y in (2019, 2020, 2021, 2022, 2023, 2024)})
print("missing values:", int(M.isnull().sum()), "| days", M.sizes["time"], str(M.time.values[0])[:10], "-", str(M.time.values[-1])[:10])
FN = os.path.join(OUT, f"olr_daily_1979_{END}_merged.nc")
M.to_dataset().to_netcdf(FN, encoding={"olr": {"zlib": True, "complevel": 4}})
print("saved", FN)
