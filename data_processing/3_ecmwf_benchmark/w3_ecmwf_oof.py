# Round W3: our out-of-fold forecasts against ECMWF on 2005-2013 (the cross-validation years that ECMWF's reforecasts cover; folds 3-4).
# No test year is used. ECMWF is a benchmark only. Issue days with an ECMWF start on T (else T-1), as in maphead_test.py / ecmwf_bsiso.py.
#   BSISO      : ECMWF's own BSISO1 forecast (ecmwf_bsiso.py, saved PCs per start), bivariate COR days 8-14 and per lead
#   weekly map : ECMWF ensemble-mean rain (2.5 deg) bilinear to the 283 cells, log1p, anomaly from ECMWF's own climatology per lead,
#                cell and calendar day (+-15 d, leave-one-year-out over 2005-2024), / its sd; week mean of leads 1-7 / 8-14;
#                scaled per week by a factor fitted leave-one-year-out on 2005-2013 (ACC does not depend on it; MSSS does)
#   weekly ERE : "ECMWF calibrated alone": logistic with our weekly normal-chance offset on ECMWF's week-mean central-India rain
#                numbers (log member-mean rain, fraction of members above the 2005-2013 90th percentile, log max-cell rain) and their
#                week maxima; fitted leave-one-year-out on 2005-2013
# usage: python w3_ecmwf_oof.py ROUND CFG EPOCH [K_DIR]       (EPOCH 0 = the frozen linear parts)
import os, sys, glob
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common.py
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd
from scipy.interpolate import RegularGridInterpolator
HERE = os.path.dirname(os.path.abspath(__file__)); IM = WORK
RND, CFG, EP = sys.argv[1], sys.argv[2], int(sys.argv[3]); KD = sys.argv[4] if len(sys.argv) > 4 else os.path.join(RUNS, RND, "out", "out")
sig = lambda x: 1 / (1 + np.exp(-x)); bv = lambda a, b: (a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum())
def acc(f, y):
    fc, yc = f - f.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True); den = np.sqrt((fc ** 2).sum(-1) * (yc ** 2).sum(-1))
    return np.where(den > 0, (fc * yc).sum(-1) / np.maximum(den, 1e-12), 0.0)
rps_w = lambda pa, pl, c: (pa - (c >= 1)) ** 2 + (pl - (c >= 2)) ** 2
# ---------------- issue days (same rule as kaggle_modelA.py with MIN_ISS_YEAR=1960)
z = np.load(os.path.join(IM, "modelA_ext", "base_inputs.npz")); DATES = pd.DatetimeIndex(z["dates"]); M1 = z["M1"]; lat1, lon1 = z["lat1"], z["lon1"]
ISS = np.array([i for i, d in enumerate(DATES) if (d.month >= 5) and i + 14 < len(DATES) and DATES[i + 14] - d == pd.Timedelta(days=14)
                and 6 <= DATES[i + 1].month <= 9 and d.year >= 1960]); ISS = ISS[(DATES[ISS].month < 9) | (DATES[ISS].day <= 29)]
assert len(ISS) == 7085, len(ISS)
LA1, LO1 = np.meshgrid(lat1, lon1, indexing="ij"); PTS = np.c_[LA1[M1], LO1[M1]]
# ---------------- our out-of-fold forecasts on 2005-2013 (mean over seeds per fold)
recs = [pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(KD, f"cv_{CFG}_f*_s*.pkl")))]; folds = sorted({r["fold"] for r in recs})
cat = lambda g: np.concatenate([g([r for r in recs if r["fold"] == f]) for f in folds])
rows = cat(lambda rs: rs[0]["va_rows"]); IY = cat(lambda rs: rs[0]["IY"]); AUX = cat(lambda rs: rs[0]["AUX"])
S = cat(lambda rs: np.mean([r["snaps"][EP] for r in rs], 0)); Dw = cat(lambda rs: np.mean([r["w_snaps"][EP] for r in rs], 0))
W = {k: cat(lambda rs, k=k: rs[0]["W"][k]) for k in ("YWc", "LW", "OFFW", "VW", "GW", "ZW", "LOCW")}
FD = cat(lambda rs: np.full(len(rs[0]["IY"]), rs[0]["fold"])); Vf = {f: [r for r in recs if r["fold"] == f][0]["W"]["V"] for f in folds}
ZH = np.zeros(W["ZW"].shape, "float32")
for f in folds:
    m = FD == f; ZH[m] = W["LOCW"][m].astype("float32") + np.einsum("nwp,pc->nwc", W["GW"][m] + Dw[m, :, 2:], Vf[f])
PA = sig(W["LW"][..., 0] + Dw[..., 0]); PL = PA * sig(W["LW"][..., 1] + Dw[..., 1])
TI = DATES[ISS[rows]]
# ---------------- ECMWF: start on T, else T-1
eb = np.load(os.path.join(IM, "maphead", "ecmwf_bsiso_forecasts.npz")); ST = pd.DatetimeIndex(eb["starts"]); PC = eb["PC"]
ez = np.load(os.path.join(IM, "profB", "ecmwf_rain.npz")); assert (pd.DatetimeIndex(ez["starts"]) == ST).all()
spos = pd.Series(np.arange(len(ST)), index=ST); s_ = np.full(len(rows), -1); o_ = np.zeros(len(rows), int)
for o in (1, 0):
    p = spos.reindex(TI - pd.Timedelta(days=o)).values; m = np.isfinite(p); s_[m] = p[m].astype(int); o_[m] = o
win = (IY >= 2005) & (IY <= 2013) & (s_ >= 0); print(f"2005-2013 issue days {((IY >= 2005) & (IY <= 2013)).sum()}, with an ECMWF start on T or T-1: {win.sum()}")
J = o_[:, None] + np.arange(1, 15)[None] + 1                           # rain step index of target day T+k (as maphead_test.py)
JB = o_[:, None] + np.arange(1, 15)[None]                              # BSISO lead index of day T+k
EB = np.full((len(rows), 14, 2), np.nan); EB[win] = np.stack([PC[s_[i], JB[i], :2] for i in np.where(win)[0]])
# ECMWF rain z on the 283 cells, all 2005-2024 starts (own climatology, leave one year out)
lat, lon = ez["lat"][::-1], ez["lon"]; MN = ez["mean"]
XE = np.stack([np.stack([RegularGridInterpolator((lat, lon), MN[s][j][::-1], bounds_error=False, fill_value=None)(PTS) for j in range(1, 17)]) for s in range(len(ST))])
XE = np.log1p(np.maximum(XE, 0))                                         # (starts, steps 1..16, cells): step j -> calendar day S + j - 1
sy = ST.year.values; ZE = np.zeros_like(XE)
for j in range(XE.shape[1]):
    dk = (ST + pd.Timedelta(days=j)).dayofyear.values
    for y in np.unique(sy):
        for i in np.where(sy == y)[0]:
            m = (sy != y) & (np.abs(dk - dk[i]) <= 15); mu, sd = XE[m, j].mean(0), XE[m, j].std(0)
            ZE[i, j] = (XE[i, j] - mu) / np.maximum(sd, 0.05)
ZEW = np.full(W["ZW"].shape, np.nan, "float32")
for i in np.where(win)[0]:
    zz = ZE[s_[i], J[i] - 1]                                             # (14, cells) days T+1..T+14
    ZEW[i, 0], ZEW[i, 1] = zz[:7].mean(0), zz[7:].mean(0)
# ECMWF central-India rain numbers -> weekly ERE probabilities
CIM, CMX = ez["ci_mem"], ez["cimax"]; trw = win & (IY <= 2013)
thr = np.percentile(np.concatenate([CIM[s_[i]][:, J[i]].ravel() for i in np.where(trw)[0]]), 90)
def efeat(i, w):
    jj = J[i][7 * w:7 * w + 7]; cm = CIM[s_[i]][:, jj]; mx = CMX[s_[i]][jj]
    a = np.log1p(cm.mean(0)); b = (cm > thr).mean(0); c = np.log1p(mx)
    return np.r_[a.mean(), b.mean(), c.mean(), a.max(), b.max(), c.max()]
EF = np.full((len(rows), 2, 6), np.nan)
for i in np.where(win)[0]:
    EF[i] = [efeat(i, 0), efeat(i, 1)]
def fit_off(X, y, off, l2=1.0):
    X1 = np.c_[np.ones(len(X)), X]; w = np.zeros(X1.shape[1]); R_ = l2 * np.diag(np.r_[0.0, np.ones(X.shape[1])])
    for _ in range(60):
        p = sig(off + X1 @ w); st = np.linalg.solve((X1 * (p * (1 - p))[:, None]).T @ X1 + R_, X1.T @ (p - y) + R_ @ w); w -= st
        if np.abs(st).max() < 1e-8:
            break
    return w
EPA, EPL = np.full(PA.shape, np.nan), np.full(PA.shape, np.nan); BETA = {}
for w in range(2):
    ok = win & W["VW"][:, w]; mu_, sd_ = EF[ok, w].mean(0), EF[ok, w].std(0) + 1e-6
    for y in range(2005, 2014):
        tr = ok & (IY != y); te = ok & (IY == y); X = (EF[:, w] - mu_) / sd_
        ya = (W["YWc"][:, w] >= 1).astype(float); wa = fit_off(X[tr], ya[tr], W["OFFW"][tr, w, 0])
        trl = tr & (W["YWc"][:, w] >= 1); yl = (W["YWc"][:, w] >= 2).astype(float); wl = fit_off(X[trl], yl[trl], W["OFFW"][trl, w, 1])
        EPA[te, w] = sig(W["OFFW"][te, w, 0] + np.c_[np.ones(te.sum()), X[te]] @ wa)
        EPL[te, w] = EPA[te, w] * sig(W["OFFW"][te, w, 1] + np.c_[np.ones(te.sum()), X[te]] @ wl)
        yz, fz = W["ZW"][tr, w].astype("float64"), ZEW[tr, w].astype("float64"); b = (yz * fz).sum() / (fz * fz).sum()
        ZEW[te, w] *= b; BETA[(w, y)] = b
print("ECMWF weekly map scaling (leave-one-year-out), mean per week:", [round(np.mean([v for (w_, _), v in BETA.items() if w_ == w]), 2) for w in range(2)])
# ---------------- scores on the common 2005-2013 days; reference = the weekly normal chance (offset) and climatology (maps)
YRS = list(range(2005, 2014))
def scores(sel):
    out = {}
    okb = sel & np.isfinite(AUX).all((1, 2))
    cO = np.array([bv(S[okb, k, 2:], AUX[okb, k]) for k in range(14)]); cE = np.array([bv(EB[okb, k], AUX[okb, k]) for k in range(14)])
    out.update(bsiso_ours=cO[7:].mean(), bsiso_ecmwf=cE[7:].mean(), bsiso_ours_d5=cO[4], bsiso_ecmwf_d5=cE[4])
    for w in range(2):
        ok = sel & W["VW"][:, w]; c = W["YWc"][ok, w]; oa = sig(W["OFFW"][ok, w, 0]); ol = oa * sig(W["OFFW"][ok, w, 1]); r0 = rps_w(oa, ol, c).sum()
        out[f"ere_w{w+1}_ours"] = 1 - rps_w(PA[ok, w], PL[ok, w], c).sum() / r0; out[f"ere_w{w+1}_ecmwf"] = 1 - rps_w(EPA[ok, w], EPL[ok, w], c).sum() / r0
        y = W["ZW"][ok, w].astype("float32")
        out[f"acc_w{w+1}_ours"] = acc(ZH[ok, w], y).mean(); out[f"acc_w{w+1}_ecmwf"] = acc(ZEW[ok, w], y).mean()
        out[f"msss_w{w+1}_ours"] = 1 - ((y - ZH[ok, w]) ** 2).sum() / (y ** 2).sum(); out[f"msss_w{w+1}_ecmwf"] = 1 - ((y - ZEW[ok, w]) ** 2).sum() / (y ** 2).sum()
    return out
ALL = scores(win); PER = {y: scores(win & (IY == y)) for y in YRS}; rng = np.random.default_rng(0)
rows_ = []
for key in ("bsiso", "ere_w1", "ere_w2", "acc_w1", "acc_w2", "msss_w1", "msss_w2"):
    if key is None:
        continue
    o, e = ALL[f"{key}_ours"], ALL[f"{key}_ecmwf"]; d = np.array([PER[y][f"{key}_ours"] - PER[y][f"{key}_ecmwf"] for y in YRS])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    rows_.append(dict(score=key, ours=o, ecmwf=e, share=o / e if e > 0 else np.nan, ours_minus_ecmwf=f"{o - e:+.3f} [{np.percentile(bs, 5):+.3f}, {np.percentile(bs, 95):+.3f}]",
                      ours_better=f"{int((d > 0).sum())}/{len(YRS)}"))
T = pd.DataFrame(rows_); print(f"\nOUT-OF-FOLD 2005-2013, {RND}:{CFG} epoch {EP} vs ECMWF (benchmark), {win.sum()} issue days, 9 summers")
print(T.round(3).to_string(index=False)); print("BSISO day 5:", round(ALL["bsiso_ours_d5"], 3), "ours vs", round(ALL["bsiso_ecmwf_d5"], 3), "ECMWF")
T.to_csv(os.path.join(IM, "maphead", f"w3_ecmwf_oof_{RND}_{CFG}_ep{EP}.csv"), index=False)
