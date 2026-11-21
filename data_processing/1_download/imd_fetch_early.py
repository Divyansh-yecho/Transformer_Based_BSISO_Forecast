# Night run 13-14 Nov 2026 (file 38): IMD 0.25 deg daily gridded rainfall (Pai et al. 2014), binary ind<YYYY>_rfp25.grd, from the
# public IMD Pune page (imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html: form 'rainfall.php', field 'rain' = year).
#   usage: python imd_fetch_early.py 1951 1978   -> D:\...\data\imd_rain_025_early\ind<year>_rfp25.grd
# Checks: exact file size (days x 129 x 135 x 4 bytes); 1979 is also downloaded once into a check folder and compared byte-for-byte
# with the copy already on disk (same source, same format).
import os, sys, time, hashlib, calendar
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import requests

Y0, Y1 = int(sys.argv[1]), int(sys.argv[2])
R = DATA
OUT = os.path.join(R, "imd_rain_025_early"); CHK = os.path.join(OUT, "check_1979"); os.makedirs(CHK, exist_ok=True)
LOG = os.path.join(OUT, "fetch.log")
log = lambda m: open(LOG, "a").write(time.strftime("%Y-%m-%d %H:%M:%S ") + m + "\n")
URL = "https://imdpune.gov.in/cmpg/Griddata/rainfall.php"
size = lambda y: (366 if calendar.isleap(y) else 365) * 129 * 135 * 4


def get(y, folder, tries=10):
    fn = os.path.join(folder, f"ind{y}_rfp25.grd")
    if os.path.exists(fn) and os.path.getsize(fn) == size(y):
        return True
    for a in range(tries):
        try:
            with requests.post(URL, data={"rain": str(y)}, timeout=(60, 600), stream=True) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"status {r.status_code}")
                with open(fn + ".tmp", "wb") as fh:
                    for ch in r.iter_content(1 << 20):
                        fh.write(ch)
            n = os.path.getsize(fn + ".tmp")
            if n == size(y):
                os.replace(fn + ".tmp", fn); log(f"OK {y} {n} bytes"); return True
            log(f"BAD {y} try {a + 1}: {n} bytes (expected {size(y)})")
        except Exception as e:
            log(f"FAIL {y} try {a + 1}: {str(e)[:160]}")
        time.sleep(min(600, 45 * (a + 1)))
    return False


log(f"start {Y0}-{Y1}")
if get(1979, CHK):
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    same = h(os.path.join(CHK, "ind1979_rfp25.grd")) == h(os.path.join(R, "imd_rain_025", "ind1979_rfp25.grd"))
    log(f"CHECK 1979 re-download identical to the file on disk: {same}")
for rnd in range(3):                                                   # up to 3 passes over the years still missing
    miss = [y for y in range(Y0, Y1 + 1) if not get(y, OUT)]
    log(f"pass {rnd + 1}: missing {miss}")
    if not miss:
        break
    time.sleep(1800)
log("all done")
