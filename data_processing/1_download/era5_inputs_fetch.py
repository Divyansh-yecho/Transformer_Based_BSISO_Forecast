# Night run 13-14 Nov 2026 (file 38): the model-input ERA5 fields (MSLP 1.5 deg; u200 + u850 5.625 deg) from the public WeatherBench2
# copy, fetched MONTH BY MONTH with retries per month (the night's link is slow and stalls; a stall now costs one month, not a year).
# Same region, daily means and output format as era5_wb2_fetch_ext.py (data/era5_wb2_ext/<tag>/wb2_<year>.npz, same keys).
#   usage: python era5_inputs_fetch.py early_inputs 1959 1978 | python era5_inputs_fetch.py offseason_inputs 1979 2022
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
MONTHS = {"early_inputs": [4, 5, 6, 7, 8, 9], "offseason_inputs": [1, 2, 3, 10, 11, 12]}[TAG]
region = lambda ds: ds.sel(latitude=slice(-30, 60), longitude=slice(0, 180))


def piece(y, m):
    sl = slice(f"{y}-{m:02d}-01", f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]}T18")
    for a in range(10):
        try:
            ms = region(HI["mean_sea_level_pressure"].sel(time=sl)).resample(time="1D").mean().load()
            uu = region(LO["u_component_of_wind"].sel(time=sl, level=[200, 850])).resample(time="1D").mean().load()
            assert ms.sizes["time"] == calendar.monthrange(y, m)[1] == uu.sizes["time"] and np.isfinite(ms.values).all() and np.isfinite(uu.values).all()
            return ms.values.astype("float32"), uu.values.astype("float32"), np.array([str(t)[:10] for t in ms.time.values])
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
        P = [piece(y, m) for m in MONTHS]
    except Exception as e:
        log(f"SKIP {y}: {e}"); continue
    uu = np.concatenate([p[1] for p in P])                                   # (time, level, lon, lat)
    out = {"hi_mean_sea_level_pressure": np.concatenate([p[0] for p in P]), "lo_u_component_of_wind_200": uu[:, 0], "lo_u_component_of_wind_850": uu[:, 1],
           "dates": np.concatenate([p[2] for p in P]), "hi_lat": region(HI).latitude.values, "hi_lon": region(HI).longitude.values,
           "lo_lat": region(LO).latitude.values, "lo_lon": region(LO).longitude.values}
    np.savez_compressed(fn + ".tmp.npz", **out); os.replace(fn + ".tmp.npz", fn); log(f"OK {y} {len(out['dates'])} days in {(time.time() - t0) / 60:.1f} min")
log("all done")
os._exit(0)
