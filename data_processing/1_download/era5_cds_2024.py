# ERA5 for 2024 (not in the WeatherBench2 copy) from the CDS, at 1.5 deg, April-September, 00/12 UTC, 30S-60N 0-180E.
# Same variables as era5_wb2_fetch.py; the pressure-level fields are later interpolated to the WB2 5.625 deg grid.
import os, sys, time, threading
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
YR = sys.argv[1] if len(sys.argv) > 1 else "2024"
import cdsapi

OUT = os.path.join(DATA, "era5_wb2")
LOG = os.path.join(OUT, f"cds{YR}.log")
log = lambda m: open(LOG, "a").write(time.strftime("%H:%M:%S ") + m + "\n")
C = {"product_type": ["reanalysis"], "year": [YR], "day": [f"{d:02d}" for d in range(1, 32)], "time": ["00:00", "12:00"],
     "data_format": "netcdf", "download_format": "unarchived", "area": [60, 0, -30, 180], "grid": [1.5, 1.5]}
JOBS = [("reanalysis-era5-single-levels", dict(C, month=["04", "05", "06", "07", "08", "09"], variable=[
            "total_column_water_vapour", "sea_surface_temperature", "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
            "volumetric_soil_water_layer_3", "mean_sea_level_pressure"]), f"cds{YR}_sl.nc")]
for half, mons in (("a", ["04", "05", "06"]), ("b", ["07", "08", "09"])):
    JOBS.append(("reanalysis-era5-pressure-levels", dict(C, month=mons, variable=["u_component_of_wind", "v_component_of_wind"],
                 pressure_level=["200", "850"]), f"cds{YR}_uv{half}.nc"))
    if os.environ.get("NOQ") != "1":                                     # humidity is not used by Prof A (skip for 2025-26)
        JOBS.append(("reanalysis-era5-pressure-levels", dict(C, month=mons, variable=["specific_humidity"], pressure_level=["500", "700"]),
                     f"cds{YR}_q{half}.nc"))


def run(ds, req, fn):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        return
    c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", quiet=True, progress=False)
    for a in range(3):
        try:
            t0 = time.time(); c.retrieve(ds, req, p + ".part"); os.replace(p + ".part", p)
            log(f"OK {fn} {(time.time() - t0) / 60:.1f} min"); return
        except Exception as e:
            log(f"FAIL {fn}: " + " | ".join(l for l in str(e).splitlines()[:3] if "key" not in l.lower())); time.sleep(60)


th = [threading.Thread(target=run, args=j) for j in JOBS]
[t.start() for t in th]; [t.join() for t in th]
log("all done")
