# Model A, step 1: inputs that are already on disk -> one compact file for Kaggle (other/intermediate/modelA/base_inputs.npz)
#  - IMD 0.25 deg daily rain, April-September 1979-2024 (date shifted back 1 day, as imd_convert.py), on the IMERG lattice;
#    1-deg 4x4 block means exactly as night_common / imd_long_record (targets are built from these in the notebook, with
#    thresholds from training years only); rain-band profile (1-deg rows 6-31N averaged over 70-95E) and central-India rain.
#  - BSISO index (daily, year-round), Nino3.4 anomaly and DMI (monthly), past-only BSISO forecasts C1/C2 scheme B (baseline).
import os, sys
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, xarray as xr
import common as nc

END = int(os.environ.get("END_YEAR", "2024"))                          # 2025 test (file 32): END_YEAR=2025, OUT_DIR=.../modelA_2025
OUT = os.environ.get("OUT_DIR", os.path.join(nc.WORK, "modelA")); os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(nc.DATA, "imd_rain_025")
ref = xr.open_dataset(os.path.join(nc.DATA, "imerg_india_025_jjas_2000_2024.nc"))
LAT, LON = ref["lat"].values, ref["lon"].values; ref.close()
lat_i = 6.5 + 0.25 * np.arange(129); lon_i = 66.5 + 0.25 * np.arange(135)
ia = np.array([np.argmin(np.abs(LAT - v)) for v in lat_i]); io = np.array([np.argmin(np.abs(LON - v)) for v in lon_i])
mask = np.load(os.path.join(nc.DATA, "kaggle_bsiso_extras", "india_mask_025.npy"))
ny, nx = (len(LAT) - 1) // 4, (len(LON) - 1) // 4
LA, LO = np.meshgrid(LAT, LON, indexing="ij")
dates, R1s, cir = [], [], []
valid = None
for y in range(1979, END + 1):
    nd = 366 if pd.Timestamp(y, 12, 31).dayofyear == 366 else 365
    a = np.fromfile(os.path.join(SRC, f"ind{y}_rfp25.grd"), "<f4").reshape(nd, 129, 135)
    lab = pd.date_range(f"{y}-01-01", periods=nd) - pd.Timedelta(days=1)
    s = (lab.month >= 4) & (lab.month <= 9)
    g = np.full((s.sum(), len(LAT), len(LON)), np.nan, "float32")
    g[:, ia[:, None], io[None, :]] = np.where(a[s] < -900, np.nan, a[s])
    if valid is None:
        valid = np.isfinite(g).mean(0) > 0.99
        m = mask & valid
        blk = m[:ny * 4, :nx * 4].reshape(ny, 4, nx, 4); M1 = blk.mean((1, 3)) >= 0.5
        ci = m & nc.ci_box(LA, LO)
    g = np.where(m[None], g, np.nan)
    R1 = np.nanmean(np.where(blk[None], g[:, :ny * 4, :nx * 4].reshape(len(g), ny, 4, nx, 4), np.nan), axis=(2, 4))
    R1s.append(np.where(M1[None], R1, np.nan).astype("float32")); cir.append(np.nanmean(g[:, ci], 1)); dates.append(lab[s])
    print(y, end=" ", flush=True)
dates = pd.DatetimeIndex(np.concatenate(dates)); R1 = np.concatenate(R1s); cir = np.concatenate(cir)
lat1 = LAT[:ny * 4].reshape(ny, 4).mean(1); lon1 = LON[:nx * 4].reshape(nx, 4).mean(1)
rows = (lat1 >= 6) & (lat1 <= 31); cols = (lon1 >= 70) & (lon1 <= 95)
band = np.nanmean(R1[:, rows][:, :, cols], axis=2)                   # rain profile by latitude, 70-95E
band = np.where(np.isfinite(band), band, 0).astype("float32")
# BSISO
b = nc.load_bsiso()
bs = b.loc["1979-01-01":f"{END}-12-31", ["p1", "p2", "q1", "q2"]]
# ENSO / IOD monthly anomalies
e = pd.read_csv(os.path.join(nc.DATA, "official_indices", "enso_cpc", "ersst5.nino.mth.91-20.ascii"), sep=r"\s+")
nino = pd.Series(e.iloc[:, 9].values, index=pd.to_datetime(dict(year=e.YR, month=e.MON, day=1)))
d = np.loadtxt(os.path.join(nc.DATA, "official_indices", "iod_psl", "dmi.had.long.data"), skiprows=1, max_rows=2026 - 1870 + 1)
dmi = pd.Series(d[:, 1:].ravel(), index=pd.date_range(f"{int(d[0, 0])}-01-01", periods=d[:, 1:].size, freq="MS")).replace(-9999, np.nan)
dmi = dmi[dmi > -90]
# past-only BSISO forecasts (baseline two-stage model)
T = os.path.join(nc.WORK, "bsiso_forecast_tests")
F = pd.concat([pd.read_parquet(os.path.join(T, "C2", "chrono_bsiso_forecasts.parquet")),
               pd.read_parquet(os.path.join(T, "C1", "chrono_bsiso_forecasts.parquet"))])
F = F[F.scheme == "B"][["issue", "lead", "f1", "f2"]].reset_index(drop=True)
F.to_parquet(os.path.join(OUT, "bsiso_pastonly_forecasts.parquet"), index=False)
np.savez_compressed(os.path.join(OUT, "base_inputs.npz"), dates=dates.values.astype("datetime64[D]").astype(str), R1=R1, M1=M1,
                    lat1=lat1, lon1=lon1, band=band, band_lat=lat1[rows], cir=cir.astype("float32"),
                    bs_dates=bs.index.values.astype("datetime64[D]").astype(str), bs=bs.values.astype("float32"),
                    nino_dates=nino.index.values.astype("datetime64[D]").astype(str), nino=nino.values.astype("float32"),
                    dmi_dates=dmi.index.values.astype("datetime64[D]").astype(str), dmi=dmi.values.astype("float32"))
print("\nsaved", R1.shape, band.shape, "M1 cells", int(M1.sum()), "BSISO", bs.shape, "nino", nino.shape, "dmi", dmi.shape, "forecasts", F.shape)
print("check: JJAS days", int(((dates.month >= 6) & (dates.month <= 9)).sum()), "first/last", dates[0].date(), dates[-1].date())
