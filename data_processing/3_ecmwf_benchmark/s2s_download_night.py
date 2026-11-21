# Model C data: ECMWF S2S reforecasts (ECMWF Data Store, dataset s2s-reforecasts), model year 2025 (hindcast years 2005-2024,
# one start every second day). Start dates 25 May - 21 Sep (odd days), 00 UTC, leads 0-21 days, 2.5 deg grid.
#   pl  : u at 850 and 200 hPa, 10S-40N 40-160E        (ECMWF BSISO via regression on ERA5 u850/u200 -> observed PCs)
#   ttr : top net thermal radiation (OLR), same domain
#   tp  : total precipitation (accumulated), India 5-35N 65-100E
# control and perturbed forecasts separately -> 6 requests per start date. Logs to data/s2s_ecmwf/download.log.
import os, time, threading, queue
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import cdsapi

OUT = os.path.join(DATA, "s2s_ecmwf")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "download.log")
DATES = [(m, d) for m in ("05", "06", "07", "08", "09") for d in range(1, 32, 2)
         if (m, d) >= ("05", 25) and (m, d) <= ("09", 21) and not (m in ("06", "09") and d == 31)]
HY = [str(y) for y in range(2005, 2025)]
LEADS = [str(24 * k) for k in range(0, 22)]
BIG, IND = [40, 40, -10, 160], [35, 65, 5, 100]
JOBS = []
for m, d in DATES:
    for ft, ftag in (("perturbed_forecast", "pf"), ("control_forecast", "cf")):
        base = {"origin": "ecmwf", "year": "2025", "month": m, "day": f"{d:02d}", "time": "00:00", "hyear": HY, "hmonth": [m],
                "hday": [f"{d:02d}"], "forecast_type": ft, "grid": [2.5, 2.5], "data_format": "grib"}
        JOBS.append((dict(base, level_type="pressure", variable=["u_component_of_wind"], level_value=["850_hpa", "200_hpa"],
                          leadtime_hour=LEADS, area=BIG), f"pl_{ftag}_{m}{d:02d}.grib"))
        JOBS.append((dict(base, level_type="single_level", variable=["top_net_thermal_radiation"], leadtime_hour=LEADS, area=BIG),
                     f"ttr_{ftag}_{m}{d:02d}.grib"))
        JOBS.append((dict(base, level_type="single_level", variable=["total_precipitation"], leadtime_hour=LEADS, area=IND),
                     f"tp_{ftag}_{m}{d:02d}.grib"))
JOBS.sort(key=lambda j: ({'tp': 0, 'pl': 1, 'ttr': 2}[j[1].split('_')[0]], j[1]))
lock = threading.Lock()


def log(msg):
    with lock:
        with open(LOG, "a") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")


q = queue.Queue()
for j in JOBS:
    if not (os.path.exists(os.path.join(OUT, j[1])) and os.path.getsize(os.path.join(OUT, j[1])) > 10_000):
        q.put(j)
log(f"start: {q.qsize()} requests to do ({len(JOBS)} total, {len(DATES)} start dates)")


def worker():
    c = cdsapi.Client(url="https://ecds.ecmwf.int/api", quiet=True, progress=False)
    while True:
        try:
            req, fn = q.get_nowait()
        except queue.Empty:
            return
        path = os.path.join(OUT, fn)
        for attempt in range(1, 4):
            t0 = time.time()
            try:
                c.retrieve("s2s-reforecasts", req, path + ".part")
                os.replace(path + ".part", path)
                log(f"OK {fn} {os.path.getsize(path) / 1e6:.1f} MB in {(time.time() - t0) / 60:.1f} min")
                break
            except Exception as e:
                msg = " | ".join(l for l in str(e).splitlines()[:4] if "key" not in l.lower())
                log(f"FAIL {fn} attempt {attempt}: {msg}")
                time.sleep(60 * attempt)
        q.task_done()


th = [threading.Thread(target=worker) for _ in range(10)]
[t.start() for t in th]
[t.join() for t in th]
log("all done")
