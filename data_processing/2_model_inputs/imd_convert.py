# Convert the IMD 0.25-deg daily gridded rainfall (Pai et al. 2014; binary ind<YYYY>_rfp25.grd from imdpune.gov.in)
# to one NetCDF of JJAS 1979-2024 on the SAME lattice as our IMERG file (lat 40 -> 5 descending, lon 60 -> 100), so the
# ERE code (mask, CI box, 1-deg 4x4 blocks) runs unchanged. Binary layout: per day 129 lat (6.5 -> 38.5) x 135 lon
# (66.5 -> 100), float32 little-endian, -999 = missing (outside India). Each IMD date is the 24 h ending 08:30 IST on
# that date, so it is shifted back one day to match IMERG's calendar day.
import os, sys
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd, xarray as xr

DATA = DATA
SRC = os.path.join(DATA, "imd_rain_025")
Y0, Y1 = 1979, 2024
lat_i = 6.5 + 0.25 * np.arange(129); lon_i = 66.5 + 0.25 * np.arange(135)
ref = xr.open_dataset(os.path.join(DATA, "imerg_india_025_jjas_2000_2024.nc"))
LAT, LON = ref["lat"].values, ref["lon"].values; ref.close()
ia = np.array([np.argmin(np.abs(LAT - v)) for v in lat_i]); io = np.array([np.argmin(np.abs(LON - v)) for v in lon_i])
assert np.allclose(LAT[ia], lat_i) and np.allclose(LON[io], lon_i)
out, tim = [], []
for y in range(Y0, Y1 + 1):
    nd = 366 if pd.Timestamp(y, 12, 31).dayofyear == 366 else 365
    a = np.fromfile(os.path.join(SRC, f"ind{y}_rfp25.grd"), "<f4")
    assert a.size == nd * 129 * 135, (y, a.size)
    a = a.reshape(nd, 129, 135)
    d = pd.date_range(f"{y}-01-01", periods=nd)
    lab = d - pd.Timedelta(days=1)                                   # IMD date D = 24 h ending 08:30 IST on D -> rain of D-1
    s = (lab.month >= 6) & (lab.month <= 9)                         # (checked: IMD(D) vs IMERG(D-1) r = 0.94, vs IMERG(D) 0.68)
    g = np.full((s.sum(), len(LAT), len(LON)), np.nan, "float32")
    v = np.where(a[s] < -900, np.nan, a[s])
    g[:, ia[:, None], io[None, :]] = v
    out.append(g); tim.append(lab[s])
    print(y, f"mean JJAS {np.nanmean(v):.2f} mm/d, valid cells {np.isfinite(v[0]).sum()}", flush=True)
tp = np.concatenate(out)
ds = xr.Dataset({"tp": (("time", "lat", "lon"), tp)}, coords={"time": np.concatenate(tim), "lat": LAT, "lon": LON},
                attrs={"source": "IMD 0.25 deg daily gridded rainfall (Pai et al. 2014, Mausam), imdpune.gov.in",
                       "units": "mm/day", "note": "JJAS only, placed on the IMERG 0.25 lattice; NaN outside the IMD grid"})
f = os.path.join(DATA, f"imd_india_025_jjas_{Y0}_{Y1}.nc")
ds.to_netcdf(f, encoding={"tp": {"zlib": True, "complevel": 4}})
print("saved", f, tp.shape)
