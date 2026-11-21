# Night run 13-14 Nov 2026 (file 38), processing step P2: new ERA5 (WeatherBench2) files -> the model's field inputs on the grids of
# other/intermediate/profA/maps.npz:  u850, u200 : 5.625 deg, 10.5S-30.5N 39-121E (7 x 15);  mslp : 1.5 deg, 10-30N 70-95E (14 x 17, Pa).
# (The night's link was ~0.25 MB/s, so only the fields the model uses were downloaded; no vorticity: the model masks that group for
# these days.)  Outputs (other/intermediate/ext/): era5_early_inputs.npz (Apr-Sep 1959-1978), era5_offseason_inputs.npz (Jan-Mar +
# Oct-Dec 1979-2022, only the years downloaded).
# Checks: complete daily calendar for each year present, no NaN, no duplicates, magnitudes against maps.npz 1979-2013 (Apr-Sep mean and
# sd of every field), and for the off-season file the jump across the joins with the Apr-Sep data on disk (31 Mar->1 Apr, 30 Sep->1 Oct).
import os, sys, glob
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd
D = DATA
OUT = os.path.join(WORK, "ext"); os.makedirs(OUT, exist_ok=True)
MP = np.load(os.path.join(WORK, "profA", "maps.npz")); md = pd.DatetimeIndex(MP["dates"])
z0 = np.load(os.path.join(D, "era5_wb2", "wb2_1979.npz")); lo_lat, lo_lon, hi_lat, hi_lon = z0["lo_lat"], z0["lo_lon"], z0["hi_lat"], z0["hi_lon"]
mla = (lo_lat >= -10.5) & (lo_lat <= 30.5); mlo = (lo_lon >= 39) & (lo_lon <= 121); hla = (hi_lat >= 10) & (hi_lat <= 30); hlo = (hi_lon >= 70) & (hi_lon <= 95)
assert np.allclose(lo_lat[mla], MP["u_lat"]) and np.allclose(lo_lon[mlo], MP["u_lon"]) and np.allclose(hi_lat[hla], MP["mslp_lat"]) and np.allclose(hi_lon[hlo], MP["mslp_lon"])
t = lambda a: np.asarray(a, "float32").transpose(0, 2, 1)               # (time, lon, lat) -> (time, lat, lon)
REP = []; say = lambda m: (print(m, flush=True), REP.append(m))


def build(tag, months):
    files = sorted(glob.glob(os.path.join(D, "era5_wb2_ext", tag, "wb2_*.npz"))); out = {k: [] for k in ("u850", "u200", "mslp", "dates")}; yrs = []
    for f in files:
        z = np.load(f); assert np.allclose(z["hi_lat"], hi_lat) and np.allclose(z["lo_lat"], lo_lat) and np.allclose(z["lo_lon"], lo_lon)
        out["u850"].append(t(z["lo_u_component_of_wind_850"])[:, mla][:, :, mlo]); out["u200"].append(t(z["lo_u_component_of_wind_200"])[:, mla][:, :, mlo])
        out["mslp"].append(t(z["hi_mean_sea_level_pressure"])[:, hla][:, :, hlo]); out["dates"].append(z["dates"]); yrs.append(int(os.path.basename(f)[4:8]))
    out = {k: np.concatenate(v) for k, v in out.items()}; d = pd.DatetimeIndex(out["dates"])
    exp = pd.DatetimeIndex(np.concatenate([[x for x in pd.date_range(f"{y}-01-01", f"{y}-12-31") if x.month in months] for y in yrs]))
    say(f"\n{tag}: years {yrs[0]}-{yrs[-1]} ({len(yrs)} files), {len(d)} days | calendar complete: {len(d) == len(exp) and (d == exp).all()} | duplicates {d.duplicated().sum()}")
    ref = (md.year >= 1979) & (md.year <= 2013); am = d.month.isin([4, 5, 6, 7, 8, 9])
    for k in ("u850", "u200", "mslp"):
        a = out[k]; s = f"  {k:5s} NaN {np.isnan(a).mean():.4f} | all-days mean {np.nanmean(a):10.2f} sd {np.nanstd(a):8.2f}"
        if am.any():
            s += f" | Apr-Sep {np.nanmean(a[am]):10.2f} (sd {np.nanstd(a[am]):7.2f}) vs maps.npz 1979-2013 {np.nanmean(MP[k][ref]):10.2f} (sd {np.nanstd(MP[k][ref]):7.2f})"
        say(s)
    return out, yrs


if __name__ == "__main__":
    for tag, months, fn in (("early_inputs", [4, 5, 6, 7, 8, 9], "era5_early_inputs.npz"), ("offseason_inputs", [1, 2, 3, 10, 11, 12], "era5_offseason_inputs.npz")):
        if not glob.glob(os.path.join(D, "era5_wb2_ext", tag, "wb2_*.npz")):
            say(f"\n{tag}: no files yet"); continue
        out, yrs = build(tag, months)
        if tag == "offseason_inputs":                                     # joins with the Apr-Sep data on disk
            od = pd.DatetimeIndex(out["dates"]); J, Ty = [], np.abs(np.diff(MP["mslp"], axis=0)).mean()
            for y in yrs:
                for a_, b_, A, Bm, da, db in ((f"{y}-03-31", f"{y}-04-01", out["mslp"], MP["mslp"], od, md), (f"{y}-09-30", f"{y}-10-01", MP["mslp"], out["mslp"], md, od)):
                    ia, ib = np.where(da == a_)[0], np.where(db == b_)[0]
                    if len(ia) and len(ib):
                        J.append(np.abs(Bm[ib[0]] - A[ia[0]]).mean())
            say(f"  join check (MSLP): mean |change| across the file joins {np.mean(J):.1f} Pa vs typical day-to-day {Ty:.1f} Pa ({len(J)} joins)")
        np.savez_compressed(os.path.join(OUT, fn), u_lat=lo_lat[mla], u_lon=lo_lon[mlo], mslp_lat=hi_lat[hla], mslp_lon=hi_lon[hlo], **out)
        say(f"  saved other/intermediate/ext/{fn}")
    open(os.path.join(OUT, "era5_prep_report.txt"), "w").write("\n".join(REP))
