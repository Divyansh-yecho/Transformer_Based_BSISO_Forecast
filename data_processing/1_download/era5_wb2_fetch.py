# ERA5 from the public WeatherBench2 copy on Google Cloud (anonymous, no queue), 1979-2023, April-September, daily means of
# the 6-hourly fields, region 30S-60N, 0-180E:
#   1.5 deg  (240x121 conservative): total column water vapour, SST, soil water layers 1-3, mean sea-level pressure
#   5.625 deg (64x32 conservative):   u200, u850, v200, q500, q700  (pressure levels: the 1.5 deg store keeps all 13 levels in
#                                     each chunk, ~100 GB for this pull, so the coarse store is used; fine for planetary waves,
#                                     jets and large-scale mid-level moisture)
# One file per year: data/era5_wb2/wb2_<year>.npz ; log data/era5_wb2/fetch.log. 2024 (not in WB2) comes from the CDS.
import os, sys, time
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, xarray as xr, dask

OUT = os.path.join(DATA, "era5_wb2")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "fetch.log")
log = lambda m: open(LOG, "a").write(time.strftime("%H:%M:%S ") + m + "\n")
dask.config.set(scheduler="threads", num_workers=16)
SO = {"token": "anon"}
HI = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
                  storage_options=SO, consolidated=True)
LO = xr.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
                  storage_options=SO, consolidated=True)
SING = ["total_column_water_vapour", "sea_surface_temperature", "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3", "mean_sea_level_pressure"]
PL = [("u_component_of_wind", 200), ("u_component_of_wind", 850), ("v_component_of_wind", 200), ("specific_humidity", 500),
      ("specific_humidity", 700)]
Y0, Y1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (1979, 2023)
log(f"start {Y0}-{Y1}")


def region(ds):
    return ds.sel(latitude=slice(-30, 60), longitude=slice(0, 180))


for y in range(Y0, Y1 + 1):
    fn = os.path.join(OUT, f"wb2_{y}.npz")
    if os.path.exists(fn):
        continue
    t0 = time.time()
    out = {}
    sl = slice(f"{y}-04-01", f"{y}-09-30T18")
    for v in SING:
        a = region(HI[v].sel(time=sl)).resample(time="1D").mean().load()
        out["hi_" + v] = a.values.astype("float32")
    for v, lev in PL:
        a = region(LO[v].sel(time=sl, level=lev)).resample(time="1D").mean().load()
        out[f"lo_{v}_{lev}"] = a.values.astype("float32")
    out["dates"] = np.array([str(t)[:10] for t in a["time"].values])
    out["hi_lat"], out["hi_lon"] = region(HI).latitude.values, region(HI).longitude.values
    out["lo_lat"], out["lo_lon"] = region(LO).latitude.values, region(LO).longitude.values
    np.savez_compressed(fn, **out)
    log(f"OK {y} in {(time.time() - t0) / 60:.1f} min")
log("all done")
os._exit(0)
