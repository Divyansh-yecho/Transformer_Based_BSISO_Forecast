# Night run 13-14 Nov 2026 (file 38): NCEP/NCAR Reanalysis 1 daily means from NOAA PSL (public OPeNDAP, server-side subsetting,
# so only the needed level and region travel over the slow link): model OLR at the top of the atmosphere (ulwrf.ntat, T62 Gaussian
# grid) and u850 (2.5 deg), 15S-45N 30-180E, 1959-2022. Used to rebuild the BSISO index before 1981 from one homogeneous source
# (calibrated to the official index on 1981-2013, checked on 2014-2022) and as the OLR proxy for 1959-1978 (calibrated to NOAA OLR).
# The PSL server rate-limits (HTTP 429), so requests are sequential with pauses and backoff.
# Output: D:\...\data\ncep_r1\<var>_<year>.npz ; log fetch.log.   usage: python ncep_fetch_bsiso.py 1959 2022
import os, sys, time
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, xarray as xr

Y0, Y1 = int(sys.argv[1]), int(sys.argv[2])
OUT = os.path.join(DATA, "ncep_r1"); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "fetch.log"); log = lambda m: open(LOG, "a").write(time.strftime("%Y-%m-%d %H:%M:%S ") + m + "\n")
BASE = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Dailies"
VARS = {"olr": (f"{BASE}/other_gauss/ulwrf.ntat.gauss.{{y}}.nc", "ulwrf", {}), "u850": (f"{BASE}/pressure/uwnd.{{y}}.nc", "uwnd", {"level": 850})}


def get(name, y):
    fn = os.path.join(OUT, f"{name}_{y}.npz")
    if os.path.exists(fn):
        return True
    url, v, sel = VARS[name]
    for a in range(10):
        try:
            d = xr.open_dataset(url.format(y=y)); x = d[v].sel(**sel) if sel else d[v]
            x = x.sel(lat=slice(45, -15), lon=slice(30, 180)).load(); d.close()
            n = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
            assert x.sizes["time"] == n and np.isfinite(x.values).all(), f"{x.sizes['time']} days / NaN"
            np.savez_compressed(fn + ".tmp.npz", data=x.values.astype("float32"), lat=x.lat.values, lon=x.lon.values,
                                dates=np.array([str(t)[:10] for t in x.time.values])); os.replace(fn + ".tmp.npz", fn)
            log(f"OK {name} {y} {x.shape}"); time.sleep(4); return True
        except Exception as e:
            log(f"FAIL {name} {y} try {a + 1}: {str(e).splitlines()[0][:160]}"); time.sleep(min(900, 60 * (a + 1)))
    return False


log(f"start {Y0}-{Y1}")
for rnd in range(3):
    miss = [(n, y) for y in range(Y0, Y1 + 1) for n in VARS if not get(n, y)]
    log(f"pass {rnd + 1}: missing {miss}")
    if not miss:
        break
    time.sleep(1800)
log("all done")
