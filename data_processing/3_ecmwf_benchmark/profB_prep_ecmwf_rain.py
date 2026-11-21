# Prof model B inputs: ECMWF S2S reforecast RAIN only (model year 2025, hindcast years 2005-2024, starts every 2nd day
# 25 May - 21 Sep, 1 control + 10 perturbed members, 2.5 deg, 5-35N 65-100E).
# Per start and lead index j = 0..21 (j = rain on calendar day start + j - 1, 00-24 UTC; j = 0 is set to 0):
#   mean  (N, 22, 13, 15) ensemble-mean daily rain map, mm/day
#   ci_mem (N, 11, 22)    central-India box (17-26.5N, 74-88.5E) daily rain of every member, mm/day
#   cimax (N, 22)         largest ensemble-mean cell rain inside the box
# No observations are used. Output: other/intermediate/profB/ecmwf_rain.npz
import os, glob
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd, xarray as xr

SRC = os.path.join(DATA, "s2s_ecmwf"); OUT = os.path.join(WORK, "profB")
os.makedirs(OUT, exist_ok=True)
KW = dict(engine="cfgrib", backend_kwargs={"indexpath": ""})


def daily(a):
    """accumulated (..., step, lat, lon) -> daily amounts, lead 0 := 0"""
    d = np.clip(np.diff(a, axis=-3), 0, None)
    return np.concatenate([np.zeros_like(d[..., :1, :, :]), d], -3)


recs = []
for md in sorted({os.path.basename(f).split("_")[2].split(".")[0] for f in glob.glob(os.path.join(SRC, "tp_pf_*.grib"))}):
    fc, fp = os.path.join(SRC, f"tp_cf_{md}.grib"), os.path.join(SRC, f"tp_pf_{md}.grib")
    if not (os.path.exists(fc) and os.path.exists(fp)):
        continue
    c = xr.open_dataset(fc, **KW); c = c[list(c.data_vars)[0]]
    p = xr.open_dataset(fp, **KW); p = p[list(p.data_vars)[0]]
    assert c.dims == ("time", "step", "latitude", "longitude") and p.dims[0] == "number" and c.shape[1] == 22
    assert (c.step.values.astype("timedelta64[h]").astype(int) == np.arange(22) * 24).all()
    mem = np.concatenate([daily(c.values)[None], daily(p.values)], 0)          # (11, time, 22, lat, lon), kg m-2 = mm
    la, lo = c.latitude.values, c.longitude.values
    LA, LO = np.meshgrid(la, lo, indexing="ij"); ci = (LA >= 17) & (LA <= 26.5) & (LO >= 74) & (LO <= 88.5)
    starts = pd.DatetimeIndex(c.time.values).normalize()
    mean = mem.mean(0)
    for i, s in enumerate(starts):
        recs.append({"start": s, "mean": mean[i], "ci_mem": mem[:, i][:, :, ci].mean(-1), "cimax": mean[i][:, ci].max(-1)})
    print(md, len(starts), end=" | ", flush=True)
recs = sorted(recs, key=lambda r: r["start"])
st = pd.DatetimeIndex([r["start"] for r in recs])
assert st.is_unique
np.savez_compressed(os.path.join(OUT, "ecmwf_rain.npz"), starts=st.values.astype("datetime64[D]").astype(str), lat=la, lon=lo,
                    mean=np.stack([r["mean"] for r in recs]).astype("float32"), ci_mem=np.stack([r["ci_mem"] for r in recs]).astype("float32"),
                    cimax=np.stack([r["cimax"] for r in recs]).astype("float32"))
m = np.stack([r["ci_mem"] for r in recs])
print(f"\nsaved {len(st)} starts {st.min().date()} - {st.max().date()} | CI box rain lead 1-14 mean {m[:, :, 1:15].mean():.1f} mm/d, "
      f"member spread {m[:, :, 1:15].std(1).mean():.1f}")
