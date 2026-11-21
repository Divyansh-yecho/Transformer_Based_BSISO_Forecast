# File 42: ERA5 geopotential at 850 hPa from the CDS for 2022-2024 (2023-24 are not in the WeatherBench2 copy; 2022 is the overlap used
# to match the CDS 00/12 UTC sampling to the WB2 daily means), April-September, 00/12 UTC, 1.5 deg, 30S-60N 0-180E (as era5_cds_2024.py).
# Output: D:\...\data\era5_wb2\cds<year>_z850.nc ; log cds_z850.log.   usage: python era5_cds_z850.py 2022 2023 2024
import os, sys, time, threading
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import cdsapi

OUT = os.path.join(DATA, "era5_wb2")
LOG = os.path.join(OUT, "cds_z850.log"); log = lambda m: open(LOG, "a").write(time.strftime("%Y-%m-%d %H:%M:%S ") + m + "\n")


def run(yr):
    p = os.path.join(OUT, f"cds{yr}_z850.nc")
    if os.path.exists(p):
        return
    req = {"product_type": ["reanalysis"], "year": [str(yr)], "month": ["04", "05", "06", "07", "08", "09"], "day": [f"{d:02d}" for d in range(1, 32)],
           "time": ["00:00", "12:00"], "variable": ["geopotential"], "pressure_level": ["850"], "data_format": "netcdf", "download_format": "unarchived",
           "area": [60, 0, -30, 180], "grid": [1.5, 1.5]}
    c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", quiet=True, progress=False)
    for a in range(5):
        try:
            t0 = time.time(); c.retrieve("reanalysis-era5-pressure-levels", req, p + ".part"); os.replace(p + ".part", p)
            log(f"OK cds{yr}_z850.nc {(time.time() - t0) / 60:.1f} min"); return
        except Exception as e:
            log(f"FAIL cds{yr}_z850 try {a + 1}: " + " | ".join(l for l in str(e).splitlines()[:3] if "key" not in l.lower())); time.sleep(120 * (a + 1))


log("start " + " ".join(sys.argv[1:]))
th = [threading.Thread(target=run, args=(int(y),)) for y in sys.argv[1:]]
[t.start() for t in th]; [t.join() for t in th]
log("all done")
