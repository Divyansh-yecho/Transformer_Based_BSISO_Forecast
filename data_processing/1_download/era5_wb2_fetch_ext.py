# Night run 13-14 Nov 2026 (file 38): ERA5 from the public WeatherBench2 copy (anonymous Google Cloud, no account, no queue) for
# the data extension. Same fields, grids and region as era5_wb2_fetch.py (1.5 deg single levels, 5.625 deg pressure levels,
# 30S-60N 0-180E, daily means of the 6-hourly fields), plus ERA5 top-of-atmosphere net longwave (an OLR equivalent, for rebuilding
# the BSISO index before 1979).
#   usage: python era5_wb2_fetch_ext.py offseason 1979 2022   -> Jan-Mar + Oct-Dec of each year (the months not yet on disk)
#          python era5_wb2_fetch_ext.py all 1959 1978         -> all months (years before the current record)
#          python era5_wb2_fetch_ext.py ttr 1979 2022         -> top net longwave only, Apr-Sep (overlap with NOAA OLR / BSISO)
# Output: D:\...\data\era5_wb2_ext\<tag>\wb2_<year>.npz (keys as era5_wb2 + hi_mean_top_net_long_wave_radiation_flux); log fetch.log.
import os, sys, time
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, xarray as xr, dask

TAG, Y0, Y1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
OUT = os.path.join(os.path.join(DATA, "era5_wb2_ext"), TAG); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "fetch.log")
log = lambda m: open(LOG, "a").write(time.strftime("%Y-%m-%d %H:%M:%S ") + m + "\n")
dask.config.set(scheduler="threads", num_workers=12)
SO = {"token": "anon", "requests_timeout": 600}                    # a dead connection fails and is retried instead of hanging
HI = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr", storage_options=SO, consolidated=True)
LO = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr", storage_options=SO, consolidated=True)
TTR = "mean_top_net_long_wave_radiation_flux"
SING = ["total_column_water_vapour", "sea_surface_temperature", "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3", "mean_sea_level_pressure", TTR]
PL = [("u_component_of_wind", 200), ("u_component_of_wind", 850), ("v_component_of_wind", 200), ("specific_humidity", 500),
      ("specific_humidity", 700)]
if TAG == "ttr":
    SING, PL = [TTR], []
if TAG in ("early_inputs", "offseason_inputs"):                      # slow link (13 Nov night): only the fields the model uses
    SING, PL = ["mean_sea_level_pressure"], []                          # u200 + u850 fetched together below (one pass over the chunks)
RANGES = {"offseason": [("01-01", "03-31T18"), ("10-01", "12-31T18")], "all": [("01-01", "12-31T18")], "ttr": [("04-01", "09-30T18")],
          "early_inputs": [("04-01", "09-30T18")], "offseason_inputs": [("01-01", "03-31T18"), ("10-01", "12-31T18")]}[TAG]
region = lambda ds: ds.sel(latitude=slice(-30, 60), longitude=slice(0, 180))
log(f"start {TAG} {Y0}-{Y1}")
for y in range(Y0, Y1 + 1):
    fn = os.path.join(OUT, f"wb2_{y}.npz")
    if os.path.exists(fn):
        continue
    for attempt in range(5):
        try:
            t0 = time.time(); out = {}
            for v in SING:
                out["hi_" + v] = np.concatenate([region(HI[v].sel(time=slice(f"{y}-{a}", f"{y}-{b}"))).resample(time="1D").mean().load().values
                                                 for a, b in RANGES]).astype("float32")
            for v, lev in PL:
                out[f"lo_{v}_{lev}"] = np.concatenate([region(LO[v].sel(time=slice(f"{y}-{a}", f"{y}-{b}"), level=lev)).resample(time="1D").mean().load().values
                                                      for a, b in RANGES]).astype("float32")
            if TAG in ("early_inputs", "offseason_inputs"):
                uu = np.concatenate([region(LO["u_component_of_wind"].sel(time=slice(f"{y}-{a}", f"{y}-{b}"), level=[200, 850])).resample(time="1D").mean().load().values
                                     for a, b in RANGES]).astype("float32")                     # (time, level, lon, lat)
                out["lo_u_component_of_wind_200"], out["lo_u_component_of_wind_850"] = uu[:, 0], uu[:, 1]
            dts = np.concatenate([region(HI[TTR].sel(time=slice(f"{y}-{a}", f"{y}-{b}"))).resample(time="1D").mean()["time"].values for a, b in RANGES])
            out["dates"] = np.array([str(t)[:10] for t in dts])
            assert all(len(v_) == len(dts) for k, v_ in out.items() if k != "dates"), "length mismatch"
            out["hi_lat"], out["hi_lon"] = region(HI).latitude.values, region(HI).longitude.values
            out["lo_lat"], out["lo_lon"] = region(LO).latitude.values, region(LO).longitude.values
            np.savez_compressed(fn + ".tmp.npz", **out); os.replace(fn + ".tmp.npz", fn)
            log(f"OK {y} {len(dts)} days in {(time.time() - t0) / 60:.1f} min"); break
        except Exception as e:
            log(f"FAIL {y} try {attempt + 1}: {type(e).__name__} {repr(e)[:200]}"); time.sleep(60 * (attempt + 1))
log("all done")
os._exit(0)
