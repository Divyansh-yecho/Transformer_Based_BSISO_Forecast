# File 40: year-round ERA5 model inputs (u850, u200 5.625 deg; MSLP 1.5 deg, Pa) on a daily calendar 1959-01-01 .. 2024-12-31 for the
# off-season BSISO samples of H7W+ (profA.py option era_y). April-September from profA_ext/maps.npz, October-March from
# ext/era5_offseason_inputs.npz (1959-2022). Oct-Dec 2023-24 do not exist (WeatherBench2 ends Jan 2023) and stay NaN (never used:
# off-season samples come only from training years). Output: other/intermediate/profA_ext/maps_yr.npz
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd
I = WORK
MP = np.load(os.path.join(I, "profA_ext", "maps.npz")); OF = np.load(os.path.join(I, "ext", "era5_offseason_inputs.npz"))
D = pd.date_range("1959-01-01", "2024-12-31"); md, od = pd.DatetimeIndex(MP["dates"]), pd.DatetimeIndex(OF["dates"])
assert not set(md) & set(od), "overlap"
out = {"dates": D.strftime("%Y-%m-%d").values.astype("U10")}
for k in ("u850", "u200", "mslp"):
    A = np.full((len(D),) + MP[k].shape[1:], np.nan, "float32"); A[D.get_indexer(md)] = MP[k]; A[D.get_indexer(od)] = OF[k]; out[k] = A
    miss = pd.DatetimeIndex(D[np.isnan(A).any((1, 2))])
    print(f"{k}: {np.isfinite(A).all((1, 2)).sum()} of {len(D)} days present; missing: {miss.min().date() if len(miss) else '-'} .. {miss.max().date() if len(miss) else '-'} ({len(miss)} days, years {sorted(set(miss.year))})")
for k in ("u_lat", "u_lon", "mslp_lat", "mslp_lon"):
    out[k] = MP[k]
np.savez_compressed(os.path.join(I, "profA_ext", "maps_yr.npz"), **out); print("saved profA_ext/maps_yr.npz")
