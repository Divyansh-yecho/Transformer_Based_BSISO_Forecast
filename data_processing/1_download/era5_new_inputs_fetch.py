# File 42: the new (Maeda-style) transformer inputs from the public WeatherBench2 ERA5 copy, April-September, fetched month by month
# with retries (same region and daily-mean method as era5_wb2_fetch.py: 30S-60N 0-180E, daily means of the 6-hourly fields).
#   new_early 1959 1978 : precipitable water (total column water vapour), SST, soil water layer 1 (1.5 deg) and specific humidity 500 /
#                         700 hPa + geopotential 850 hPa (5.625 deg) -- the years before the existing era5_wb2 files
#   z850      1979 2022 : geopotential 850 hPa only (not in the existing files for any year)
# Output: D:\...\data\era5_wb2_ext\<tag>\wb2_<year>.npz (keys as era5_wb2: hi_<var>, lo_<var>_<level>, dates, hi/lo lat lon); log fetch.log
import os, sys, time, calendar
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, xarray as xr, dask

TAG, Y0, Y1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
OUT = os.path.join(os.path.join(DATA, "era5_wb2_ext"), TAG); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "fetch.log"); log = lambda m: open(LOG, "a").write(time.strftime("%Y-%m-%d %H:%M:%S ") + m + "\n")
dask.config.set(scheduler="threads", num_workers=6)
SO = {"token": "anon", "requests_timeout": 180}
HI = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr", storage_options=SO, consolidated=True)
LO = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr", storage_options=SO, consolidated=True)
SING = ["total_column_water_vapour", "sea_surface_temperature", "volumetric_soil_water_layer_1"] if TAG == "new_early" else []
PL = ([("specific_humidity", [500, 700])] if TAG == "new_early" else []) + [("geopotential", [850])]
region = lambda ds: ds.sel(latitude=slice(-30, 60), longitude=slice(0, 180))


def piece(y, m):
    sl = slice(f"{y}-{m:02d}-01", f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]}T18"); n = calendar.monthrange(y, m)[1]
    for a in range(10):
        try:
            out = {}
            for v in SING:
                x = region(HI[v].sel(time=sl)).resample(time="1D").mean().load(); assert x.sizes["time"] == n
                out["hi_" + v] = x.values.astype("float32")             # SST is NaN over land by design
            for v, levs in PL:
                x = region(LO[v].sel(time=sl, level=levs)).resample(time="1D").mean().load(); assert x.sizes["time"] == n and np.isfinite(x.values).all()
                for j, lv in enumerate(levs):
                    out[f"lo_{v}_{lv}"] = x.values[:, j].astype("float32")
            out["dates"] = np.array([str(t)[:10] for t in x.time.values]); return out
        except Exception as e:
            log(f"FAIL {y}-{m:02d} try {a + 1}: {type(e).__name__} {repr(e)[:160]}"); time.sleep(min(600, 30 * (a + 1)))
    raise RuntimeError(f"{y}-{m:02d} failed 10 times")


log(f"start (monthly) {TAG} {Y0}-{Y1}")
for y in range(Y0, Y1 + 1):
    fn = os.path.join(OUT, f"wb2_{y}.npz")
    if os.path.exists(fn):
        continue
    t0 = time.time()
    try:
        P = [piece(y, m) for m in range(4, 10)]
    except Exception as e:
        log(f"SKIP {y}: {e}"); continue
    out = {k: np.concatenate([p[k] for p in P]) for k in P[0]}
    out.update(hi_lat=region(HI).latitude.values, hi_lon=region(HI).longitude.values, lo_lat=region(LO).latitude.values, lo_lon=region(LO).longitude.values)
    assert len(out["dates"]) == 183
    np.savez_compressed(fn + ".tmp.npz", **out); os.replace(fn + ".tmp.npz", fn); log(f"OK {y} {len(out['dates'])} days in {(time.time() - t0) / 60:.1f} min")
log("all done")
os._exit(0)
