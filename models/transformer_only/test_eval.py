# Best transformer only (TXM, file 45) on the test monsoons 2014-2024 (user's go: 16 Nov 2026). Round TXMF: final fit 1960-2013,
# 10 seeds, test predictions at every epoch. Rule fixed BEFORE the test was read (tx_eval.py, CV 1981-2013 best epoch per head):
#   BSISO head epoch 20, weekly ERE head epoch 4, weekly map head epoch 40.
# Scored exactly as to_eval.py (transformer-only test) and compared with the poster rows (linear part, Model P, persistence, ECMWF)
# read from the saved test tables, plus a season bootstrap (11 summers) of TXM against the old transformer only (W3T, TOF) and
# against Model P / the linear part (W3BF snapshots: epoch 12 / epoch 0).
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); K = RUNS; IM = WORK
CHOICE = {"bsiso": 20, "ere": 4, "map": 40}
sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -30, 30))); rps_w = lambda pa, pl, c: (pa - (c >= 1)) ** 2 + (pl - (c >= 2)) ** 2


def bcor(F, O):
    return (F * O).sum((0, 2)) / np.sqrt((F ** 2).sum((0, 2)) * (O ** 2).sum((0, 2)))


def horizon(c):
    b = np.where(c < 0.5)[0]
    return 14.0 if not len(b) else (1.0 if b[0] == 0 else b[0] + (c[b[0] - 1] - 0.5) / (c[b[0] - 1] - c[b[0]]))


def acc(f, y):
    fc, yc = f - f.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True); den = np.sqrt((fc ** 2).sum(-1) * (yc ** 2).sum(-1))
    return np.where(den > 0, (fc * yc).sum(-1) / np.maximum(den, 1e-12), 0.0)


T = pd.read_pickle(os.path.join(K, "TXMF", "out", "out", "final_TXM.pkl"))
assert len(T["deep_snaps"]) == 10 and len(T["deep_w_snaps"]) == 10, f"{len(T['deep_snaps'])} seeds"
print("epochs saved:", min(T["deep_snaps"][0]), "-", max(T["deep_snaps"][0]))
AUX, yrs = T["AUX"], T["IY"]; Y = sorted(set(yrs)); W, WT = T["W"], T["W_train"]; ZW = W["ZW"].astype("float32")
ref_rows = WT["IY"] >= 1981


def wclim(doy, w):                                                         # 1981-2013 weekly normal chance (issue day +-15 d, large | any +-30 d)
    c_, d_ = WT["YWc"][ref_rows, w], WT["doy"][ref_rows]; pa, pl = np.zeros(len(doy)), np.zeros(len(doy))
    for i, d in enumerate(doy):
        m = np.isfinite(c_) & (np.abs(d_ - d) <= 15); pa[i] = (c_[m] >= 1).mean()
        m2 = np.isfinite(c_) & (c_ >= 1) & (np.abs(d_ - d) <= 30); pl[i] = pa[i] * (c_[m2] >= 2).mean()
    return pa, pl


REFW = [wclim(W["doy"], w) for w in range(2)]


def parts(Tm, eb, ew, ec):
    Wc = Tm["W"]; F = np.mean([s[eb][..., 2:4] for s in Tm["deep_snaps"]], 0)
    D = np.mean([s[ew][..., :2] for s in Tm["deep_w_snaps"]], 0); pa = sig(Wc["LW"][..., 0] + D[..., 0]); pl = pa * sig(Wc["LW"][..., 1] + D[..., 1])
    Dm = np.mean([s[ec][..., 2:] for s in Tm["deep_w_snaps"]], 0); ZH = Wc["LOCW"].astype("float32") + np.einsum("nwp,pc->nwc", Wc["GW"] + Dm, Wc["V"])
    return F, pa, pl, ZH


def stats(M, idx):
    F, pa, pl, ZH = M; cc = bcor(F[idx], AUX[idx]); o = dict(bsiso_d1_7=cc[:7].mean(), bsiso=cc[7:].mean(), horizon=horizon(cc))
    for w in range(2):
        ok = idx[W["VW"][idx, w]]; cl = W["YWc"][ok, w]
        o[f"ere_w{w+1}"] = 1 - rps_w(pa[ok, w], pl[ok, w], cl).sum() / rps_w(REFW[w][0][ok], REFW[w][1][ok], cl).sum()
        o[f"acc_w{w+1}"] = acc(ZH[ok, w], ZW[ok, w]).mean()
    return o


YI = {y: np.where(yrs == y)[0] for y in Y}; ALL = np.concatenate([YI[y] for y in Y])
MX = parts(T, CHOICE["bsiso"], CHOICE["ere"], CHOICE["map"]); SX = stats(MX, ALL)
print("\nTEST 2014-2024, TXM (best transformer only) at the CV-chosen epochs:"); print(pd.Series(SX).round(3).to_string())
row = {"BSISO COR d1-7": SX["bsiso_d1_7"], "BSISO COR d8-14": SX["bsiso"], "BSISO useful days": SX["horizon"], "ERE RPSS w1": SX["ere_w1"],
       "ERE RPSS w2": SX["ere_w2"], "map corr w1": SX["acc_w1"], "map corr w2": SX["acc_w2"]}
pd.DataFrame([row]).to_csv(os.path.join(IM, "poster", "test_transformer_only_TXM.csv"), index=False)

# comparison rows from the saved poster tables (same test rows and scoring)
PD = os.path.join(IM, "poster")
B = pd.read_csv(os.path.join(PD, "test_W3BF_bsiso.csv"), index_col=0); TW = pd.read_csv(os.path.join(PD, "test_W3BF_weekly.csv"), index_col=0)
TO = pd.read_csv(os.path.join(PD, "test_transformer_only.csv")).iloc[0]
cols = list(row)
comp = pd.DataFrame({
    "Best transformer only (TXM)": row,
    "Old transformer only (W3T)": {c: TO[c] for c in cols},
    "Linear part only": dict(zip(cols, [B.loc["linear part", "d1-7"], B.loc["linear part", "d8-14"], B.loc["linear part", "horizon"],
                                        TW.loc["ere_w1", "model_P"], TW.loc["ere_w2", "model_P"], TW.loc["acc_w1", "linear_part"], TW.loc["acc_w2", "linear_part"]])),
    "Model P": dict(zip(cols, [B.loc["Model P (new)", "d1-7"], B.loc["Model P (new)", "d8-14"], B.loc["Model P (new)", "horizon"],
                               TW.loc["ere_w1", "P_epoch12"], TW.loc["ere_w2", "P_epoch12"], TW.loc["acc_w1", "model_P"], TW.loc["acc_w2", "model_P"]])),
    "Persistence": dict(zip(cols, [B.loc["persistence", "d1-7"], B.loc["persistence", "d8-14"], B.loc["persistence", "horizon"], np.nan, np.nan,
                                   TW.loc["acc_w1", "last_week_persists"], TW.loc["acc_w2", "last_week_persists"]])),
    "ECMWF": dict(zip(cols, [B.loc["ECMWF", "d1-7"], B.loc["ECMWF", "d8-14"], B.loc["ECMWF", "horizon"], TW.loc["ere_w1", "ecmwf"],
                             TW.loc["ere_w2", "ecmwf"], TW.loc["acc_w1", "ecmwf"], TW.loc["acc_w2", "ecmwf"]]))}).T[cols]
pd.set_option("display.width", 250)
print("\nTEST 2014-2024 comparison:"); print(comp.round(3).to_string()); comp.to_csv(os.path.join(PD, "test_TXM_comparison.csv"))

# season bootstrap (11 test summers): TXM minus each reference, 90% range, summers TXM better
REF = {}
TOF = os.path.join(K, "TOF", "out", "out", "final_W3T.pkl")
if os.path.exists(TOF):
    REF["old transformer only (W3T 20/5/20)"] = parts(pd.read_pickle(TOF), 20, 5, 20)
PF = os.path.join(K, "MANUAL_W3BF", "out", "out", "final_W3N.pkl")
if os.path.exists(PF):
    P = pd.read_pickle(PF)
    for name, e in (("Model P (epoch 12)", 12), ("linear part (epoch 0)", 0)):
        if e in P["deep_snaps"][0]:
            REF[name] = parts(P, e, e, e)
keys = ["bsiso", "ere_w1", "ere_w2", "acc_w1", "acc_w2"]; rng = np.random.default_rng(0); BOOTI = [rng.choice(Y, len(Y)) for _ in range(1000)]
rows = []
for name, M in REF.items():
    d0 = {k: SX[k] - stats(M, ALL)[k] for k in keys}
    bs = np.array([[stats(MX, ix)[k] - stats(M, ix)[k] for k in keys] for ix in (np.concatenate([YI[y] for y in b]) for b in BOOTI)])
    per = np.array([[stats(MX, YI[y])[k] - stats(M, YI[y])[k] for k in keys] for y in Y])
    for j, k in enumerate(keys):
        rows.append(dict(comparison=f"TXM - {name}", score=k, diff=d0[k], lo90=np.percentile(bs[:, j], 5), hi90=np.percentile(bs[:, j], 95),
                         TXM_better_of_11=int((per[:, j] > 0).sum())))
BT = pd.DataFrame(rows); print("\nSeason bootstrap, TXM minus reference (test 2014-2024):"); print(BT.round(3).to_string(index=False))
BT.to_csv(os.path.join(PD, "test_TXM_bootstrap.csv"), index=False)
# all test epochs (diagnostic only, never used for the choice)
diag = pd.DataFrame({e: stats(parts(T, e, e, e), ALL) for e in sorted(T["deep_snaps"][0])}).T
diag.to_csv(os.path.join(IM, "maphead", "test_TXM_all_epochs_DIAGNOSTIC.csv")); print("saved")
