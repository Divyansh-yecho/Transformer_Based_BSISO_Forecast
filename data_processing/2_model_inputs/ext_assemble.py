# Night run 13-14 Nov 2026 (file 38), step P5: assemble the extended inputs (April-September 1959-2024) in the SAME formats the
# models already read, as NEW files (the originals are not touched):
#   other/intermediate/modelA_ext/base_inputs.npz  : IMD R1 / band / central-India rain (1959-1978 from imd_early.npz + the existing
#        1979-2024), BSISO PC1-4 (official from 1981, NCEP rebuild before; bs_source), Nino3.4, DMI, plus bsiso_pastonly_forecasts.parquet copied.
#   other/intermediate/profA_ext/maps.npz         : OLR (2.5 deg; 1959-1978 = calibrated NCEP proxy), u850, u200, MSLP (ERA5 WB2; 1959-1978
#        from era5_early_inputs.npz), same grids as profA/maps.npz;  maps_vo.npz : vorticity, NaN before 1981 (masked group).
#   data/olr_merged/olr_daily_1959_2024_proxy_merged.nc (made by ext_olr_bsiso.py) is the year-round OLR for the all-months BSISO path.
# Checks: identical values to the original files on every 1979-2024 day; continuity of every field at the 1978/1979 join (Apr-Sep means
# and sds of 1969-1978 vs 1979-1988); complete calendars; NaN counts.
import os, sys, shutil
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd, xarray as xr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as nc
I = nc.WORK; OA, OP = os.path.join(I, "modelA_ext"), os.path.join(I, "profA_ext"); os.makedirs(OA, exist_ok=True); os.makedirs(OP, exist_ok=True)
REP = []; say = lambda m: (print(m, flush=True), REP.append(m))
B0 = np.load(os.path.join(I, "modelA", "base_inputs.npz")); bd = pd.DatetimeIndex(B0["dates"])
IE = np.load(os.path.join(I, "ext", "imd_early.npz")); ie = pd.DatetimeIndex(IE["dates"]); sel = ie.year >= 1959
assert (IE["M1"] == B0["M1"]).all()
dates = ie[sel].append(bd); assert dates.is_monotonic_increasing and not dates.duplicated().any()
exp = pd.DatetimeIndex([d for d in pd.date_range("1959-04-01", "2024-09-30") if 4 <= d.month <= 9]); assert (dates == exp).all(), "calendar"
R1 = np.concatenate([IE["R1"][sel], B0["R1"]]); band = np.concatenate([IE["band"][sel], B0["band"]]); cir = np.concatenate([IE["cir"][sel], B0["cir"]])
X = pd.read_csv(os.path.join(nc.DATA, "bsiso_extended", "BSISO_extended_1959_2026.csv"), index_col=0, parse_dates=True)
X = X[X.index <= "2024-12-31"]
keep = {k: B0[k] for k in ("M1", "lat1", "lon1", "band_lat", "nino_dates", "nino", "dmi_dates", "dmi")}
np.savez_compressed(os.path.join(OA, "base_inputs.npz"), dates=dates.values.astype("datetime64[D]").astype(str), R1=R1, band=band, cir=cir,
                    bs_dates=X.index.values.astype("datetime64[D]").astype(str), bs=X[["p1", "p2", "q1", "q2"]].values.astype("float32"),
                    bs_source=X.source.values.astype("U16"), **keep)
shutil.copy(os.path.join(I, "modelA", "bsiso_pastonly_forecasts.parquet"), OA)
o = np.isin(dates, bd); assert np.array_equal(np.nan_to_num(R1[o], nan=-9), np.nan_to_num(B0["R1"], nan=-9))
bo = pd.Series(range(len(X)), index=X.index); b_old = pd.DataFrame(B0["bs"], index=pd.DatetimeIndex(B0["bs_dates"])); b_old = b_old[b_old.index >= "1981-01-01"]
same = np.allclose(X.loc[b_old.index, ["p1", "p2", "q1", "q2"]].values, b_old.values, equal_nan=True)
say(f"base_inputs (ext): {len(dates)} days {dates[0].date()}..{dates[-1].date()} | 1979-2024 rain identical to the original: True | official BSISO identical: {same}")
say(f"   BSISO sources: {X.source.value_counts().to_dict()}; NaN in PC1-4 from 1959: {int(X[['p1','p2','q1','q2']].isna().any(axis=1).sum())}")
# ---------------- maps
MP = np.load(os.path.join(I, "profA", "maps.npz")); MV = np.load(os.path.join(I, "profA", "maps_vo.npz")); md = pd.DatetimeIndex(MP["dates"]); assert (md == bd).all()
EA = np.load(os.path.join(I, "ext", "era5_early_inputs.npz")); ea = pd.DatetimeIndex(EA["dates"]); assert (ea == ie[sel]).all(), "ERA5 early calendar"
O = xr.open_dataset(os.path.join(nc.DATA, "olr_merged", "olr_daily_1959_2024_proxy_merged.nc")).olr.sortby("lat").sel(lat=slice(-10, 30), lon=slice(40, 160))
olr_e = O.sel(time=ea.values).values.astype("float32"); assert np.allclose(O.lat.values, MP["olr_lat"]) and np.allclose(O.lon.values, MP["olr_lon"])
out = {"dates": dates.strftime("%Y-%m-%d").values.astype("U10"), "olr": np.concatenate([olr_e, MP["olr"]]), "olr_lat": MP["olr_lat"], "olr_lon": MP["olr_lon"]}
for k in ("u850", "u200", "mslp"):
    out[k] = np.concatenate([EA[k], MP[k]]).astype("float32")
for k in ("u_lat", "u_lon", "mslp_lat", "mslp_lon"):
    out[k] = MP[k]
np.savez_compressed(os.path.join(OP, "maps.npz"), **out)
vo = {k: MV[k] for k in MV.files}
for k in ("vo850", "vo500"):
    vo[k] = np.concatenate([np.full((len(ea),) + MV[k].shape[1:], np.nan, "float32"), MV[k]])
np.savez_compressed(os.path.join(OP, "maps_vo.npz"), **vo)
for k in ("olr", "u850", "u200", "mslp"):
    a, b_ = out[k][(dates.year >= 1969) & (dates.year <= 1978)], out[k][(dates.year >= 1979) & (dates.year <= 1988)]
    say(f"maps (ext) {k:5s}: identical to the original on 1979-2024: {np.array_equal(out[k][o], MP[k])} | Apr-Sep mean/sd 1969-78 {a.mean():9.2f}/{a.std():7.2f} vs 1979-88 {b_.mean():9.2f}/{b_.std():7.2f} | NaN {np.isnan(out[k]).mean():.4f}")
open(os.path.join(I, "ext", "assemble_report.txt"), "w").write("\n".join(REP)); say("saved modelA_ext/base_inputs.npz, profA_ext/maps.npz, profA_ext/maps_vo.npz")
