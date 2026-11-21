# Round TX (user, 15 Nov 2026): can the transformer alone be pushed? Cross-validation 1981-2013 only (no test years).
# Rounds TX1 (TXA, TXL) and TX2 (TXD, TXP, TXM); 5 folds x 3 seeds, 40 epochs scored every epoch. Same scoring as w3_eval.py.
# Reference rows (same folds): W3T (round TOCV, transformer only without OLR / u200), the linear parts (W3N epoch 0, round W3B2)
# and Model P (W3N epoch 12). For each config and head: the best epoch in CV (same rule as to_eval.py) and the score there.
# TXM: daily rain-map head (283 cells, leads 1-14) vs the per-cell linear baseline (lin_map_scores), mean over folds.
import os, sys, io, contextlib
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import WORK, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x", "W3B2", "W3N"]
src = open(os.path.join(ROOT, "models", "logistic_transformer", "cv_eval.py"), encoding="utf-8").read().split('pd.set_option("display.width", 250)')[0]
E = {"__file__": os.path.join(ROOT, "models", "logistic_transformer", "cv_eval.py")}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "w3_eval_defs", "exec"), E)
Run, all_scores, bcor = E["Run"], E["all_scores"], E["bcor"]
RUNS = {"W3N": E["RUNS"]["W3N"], "W3T": Run("TOCV", "W3T")}
for rnd, cs in (("TX1", ["TXA", "TXL"]), ("TX2", ["TXD", "TXP", "TXM"]), ("TY1", ["TYE", "TYW"]), ("TY2", ["TYEW", "TYEW48"])):
    for c in cs:
        try:
            RUNS[c] = Run(rnd, c)
        except AssertionError as e_:
            print("missing:", c, e_)
crit = {"bsiso": lambda s: s["bsiso"], "ere": lambda s: (s["ere_w1"] + s["ere_w2"]) / 2, "map": lambda s: (s["acc_w1"] + s["acc_w2"]) / 2}
COLS = ["bsiso_d1_7", "bsiso", "horizon", "ere_w1", "ere_w2", "acc_w1", "acc_w2"]
SC = {}
for c, R in RUNS.items():
    for e in R.eps:
        s = all_scores(R, e); s["bsiso_d1_7"] = bcor(R.S(e)[..., 2:], R.AUX)[:7].mean(); SC[(c, e)] = s
pd.set_option("display.width", 250)
for c, R in RUNS.items():
    if c in ("W3N",):
        continue
    T = pd.DataFrame({e: SC[(c, e)] for e in R.eps}).T[COLS]; T.index.name = "epoch"
    print(f"\nCV 1981-2013, {c}, seeds {len(R.seeds)}"); print(T.iloc[[i for i in range(len(T)) if (i + 1) % 2 == 0 or i == 0]].round(3).to_string())
rows = {"linear parts only (W3N epoch 0)": SC[("W3N", 0)], "Model P (W3N epoch 12)": SC[("W3N", 12)]}
best = {}
for c in [c for c in RUNS if c != "W3N"]:
    r = {}
    for h, f in crit.items():
        e = max(RUNS[c].eps, key=lambda e: f(SC[(c, e)])); best[(c, h)] = e; s = SC[(c, e)]
        if h == "bsiso":
            r.update(bsiso_d1_7=s["bsiso_d1_7"], bsiso=s["bsiso"], horizon=s["horizon"])
        elif h == "ere":
            r.update(ere_w1=s["ere_w1"], ere_w2=s["ere_w2"])
        else:
            r.update(acc_w1=s["acc_w1"], acc_w2=s["acc_w2"])
    r["epochs (bsiso/ere/map)"] = f"{best[(c, 'bsiso')]}/{best[(c, 'ere')]}/{best[(c, 'map')]}"
    rows[c + " (transformer only)"] = r
T = pd.DataFrame(rows).T[COLS + ["epochs (bsiso/ere/map)"]]
print("\nCV 1981-2013 at each head's best epoch (transformer-only rows) vs the linear parts and Model P:"); print(T.to_string(float_format=lambda v: f"{v:.3f}"))
T.to_csv(os.path.join(WORK, "maphead", "tx_cv_summary.csv"))
# daily rain-map head (TXM) vs the per-cell linear baseline
DM = {}
for c in [c for c in RUNS if c.startswith(("TXM", "TY"))]:
    R = RUNS[c]; ms = {}
    for e in R.eps:
        ms[e] = np.array([[r["map_scores"][e][k] for k in range(1, 15)] for r in R.recs]).mean(0)         # (14, 2) BSS, pattern corr
    lin = np.array([[f0["lin_map_scores"][k][:2] for k in range(1, 15)] for f0 in {r["fold"]: r for r in R.recs}.values()]).mean(0)
    eb = max(ms, key=lambda e: ms[e][:, 0][:7].mean()); DM[c] = (eb, ms[eb], lin, ms)
if DM:
    K_ = [1, 2, 3, 5, 7, 10, 14]
    T = pd.DataFrame({**{f"{c} BSS (ep {DM[c][0]})": DM[c][1][[k - 1 for k in K_], 0] for c in DM},
                      "linear BSS": DM[list(DM)[0]][2][[k - 1 for k in K_], 0],
                      **{f"{c} corr": DM[c][1][[k - 1 for k in K_], 1] for c in DM},
                      "linear corr": DM[list(DM)[0]][2][[k - 1 for k in K_], 1]}, index=pd.Index(K_, name="lead day"))
    print("\nDaily rain map (283 cells), each config at its best epoch (by BSS days 1-7) vs the per-cell linear baseline, CV mean over folds:")
    print(T.round(3).to_string()); T.to_csv(os.path.join(WORK, "maphead", "tx_daily_map_cv.csv"))
    for c in DM:                                                         # is the map still improving at the last epoch?
        ms = DM[c][3]; e_ = sorted(ms); print(f"  {c}: BSS d1-7 at epochs {e_[len(e_) // 2]}/{e_[-10]}/{e_[-1]} =",
                                            [round(ms[e][:7, 0].mean(), 4) for e in (e_[len(e_) // 2], e_[-10], e_[-1])])
