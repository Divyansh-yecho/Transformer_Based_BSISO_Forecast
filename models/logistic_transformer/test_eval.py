# Test 2014-2024 of the new Model P (= config W3N: transformer with 3 heads - BSISO, weekly ERE, weekly rain map - on top of the frozen
# linear parts; daily ERE = its linear part; round W3BF, 10 seeds, epoch 12). User's go for the test run: 15 Nov 2026.
# Choices fixed in cross-validation BEFORE the test (round W3B2):
#   stop epoch 12 (best and last BSISO epoch in CV); weekly ERE head = its linear part (the transformer correction lowered weekly
#   ERE in CV at every epoch, best epoch 1 = 0.0000); weekly map = epoch 12; daily ERE = linear part (no daily loss), Platt-
#   calibrated on the CV out-of-fold forecasts as in modelP_test_eval.py. Epoch-12 weekly ERE is shown as a secondary line.
# References: BSISO persistence; daily ERE 1981-2013 normal chance (PSCA G0, as every earlier test); weekly ERE 1981-2013 weekly normal
#   chance (issue day +-15 d; large | any +-30 d) from the training rows; weekly maps climatology = 0 and "last week persists".
# ECMWF (benchmark only): BSISO from ecmwf_bsiso.csv (same test days); weekly maps and weekly ERE as w3_ecmwf_oof.py, with the scale
#   factor and the calibration fitted on 2005-2013 and applied to 2014-2024.
import os, sys, glob, runpy
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd
from scipy.interpolate import RegularGridInterpolator
HERE = os.path.dirname(os.path.abspath(__file__)); K = RUNS; IM = WORK
FD, CVD = os.path.join(K, os.environ.get("W3_TEST_ROUND", "MANUAL_W3BF"), "out", "out"), os.path.join(K, "W3B2", "out", "out"); EP = 12
sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -30, 30))); rps_w = lambda pa, pl, c: (pa - (c >= 1)) ** 2 + (pl - (c >= 2)) ** 2
def acc(f, y):
    fc, yc = f - f.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True); den = np.sqrt((fc ** 2).sum(-1) * (yc ** 2).sum(-1))
    return np.where(den > 0, (fc * yc).sum(-1) / np.maximum(den, 1e-12), 0.0)
sys.argv = ["x", "none"]; e = runpy.run_path(os.path.join(ROOT, "src", "profA_eval.py")); G_ = e["score_test"].__globals__
T = pd.read_pickle(os.path.join(FD, "final_W3N.pkl")); assert len(T["deep_snaps"]) == 10 and len(T["deep_w_snaps"]) == 10
REF = pd.read_pickle(os.path.join(K, "PSCA", "out", "out", "final_G0.pkl")); Oa, Ol = REF["OFF"]; Yc, yrs, AUX, PERS = REF["Yc"], REF["IY"], REF["AUX"], REF["PERS"]
assert np.array_equal(np.nan_to_num(T["Yc"], nan=-1), np.nan_to_num(Yc, nan=-1)) and np.array_equal(T["IY"], yrs)
Y = sorted(set(yrs)); rng = np.random.default_rng(0)
# ---------------- issue days / ECMWF starts
z = np.load(os.path.join(IM, "modelA_ext", "base_inputs.npz")); DATES = pd.DatetimeIndex(z["dates"]); M1 = z["M1"]; lat1, lon1 = z["lat1"], z["lon1"]
ISS = np.array([i for i, d in enumerate(DATES) if (d.month >= 5) and i + 14 < len(DATES) and DATES[i + 14] - d == pd.Timedelta(days=14)
                and 6 <= DATES[i + 1].month <= 9 and d.year >= 1960]); ISS = ISS[(DATES[ISS].month < 9) | (DATES[ISS].day <= 29)]
te = T["te_rows"]; TI = DATES[ISS[te]]; assert (TI.year.values == yrs).all()
LA1, LO1 = np.meshgrid(lat1, lon1, indexing="ij"); PTS = np.c_[LA1[M1], LO1[M1]]
ez = np.load(os.path.join(IM, "profB", "ecmwf_rain.npz")); ST = pd.DatetimeIndex(ez["starts"]); spos = pd.Series(np.arange(len(ST)), index=ST)
def starts(ti):
    s_ = np.full(len(ti), -1); o_ = np.zeros(len(ti), int)
    for o in (1, 0):
        p = spos.reindex(ti - pd.Timedelta(days=o)).values; m = np.isfinite(p); s_[m] = p[m].astype(int); o_[m] = o
    return s_, o_
# ---------------- 1. BSISO
bs = lambda snaps, ep: np.mean([s[ep][..., 2:4] for s in snaps], 0)
BS = {"Model P (new)": bs(T["deep_snaps"], EP), "linear part": bs(T["deep_snaps"], 0), "persistence": np.repeat(PERS[:, None], 14, 1)}
old = pd.read_csv(os.path.join(IM, "maphead", "modelP_test_bsiso.csv"), index_col=0); ecb = pd.read_csv(os.path.join(IM, "maphead", "ecmwf_bsiso.csv")).ecmwf_cor.values
rowsB = {}
for m, F in BS.items():
    c = e["bcor"](F, AUX); rowsB[m] = list(c) + [c[:7].mean(), c[7:].mean(), e["horizon"](c)]
c = old.loc["Model P (epoch 2, chosen in CV)", [f"d{k}" for k in range(1, 15)]].values.astype(float); rowsB["Model P (old, daily heads)"] = list(c) + [c[:7].mean(), c[7:].mean(), e["horizon"](c)]
rowsB["ECMWF"] = list(ecb) + [ecb[:7].mean(), ecb[7:].mean(), e["horizon"](ecb)]
RB = pd.DataFrame(rowsB, index=[f"d{k}" for k in range(1, 15)] + ["d1-7", "d8-14", "horizon"]).T
print("TEST 2014-2024, BSISO bivariate COR"); print(RB.round(3).to_string())
def boot_b(Fa, Fb):
    d = []
    for _ in range(2000):
        ii = np.concatenate([np.where(yrs == y)[0] for y in rng.choice(Y, len(Y))]); d.append(e["bcor"](Fa[ii], AUX[ii])[7:].mean() - e["bcor"](Fb[ii], AUX[ii])[7:].mean())
    nb = sum(e["bcor"](Fa[yrs == y], AUX[yrs == y])[7:].mean() > e["bcor"](Fb[yrs == y], AUX[yrs == y])[7:].mean() for y in Y)
    return np.mean(d), np.percentile(d, 5), np.percentile(d, 95), nb
print("  new Model P - its linear part, d8-14: %+.4f [%+.4f, %+.4f] %d/11" % boot_b(BS["Model P (new)"], BS["linear part"]))
# ---------------- 2. daily ERE (= linear part, Platt from CV out-of-fold)
G_["OUTD"] = CVD; FH = e["load_cv"]("W3N")
hO = [np.concatenate([f["lin"][i] for f in FH]) for i in (0, 1)]; hD = np.concatenate([f["snaps"][EP] for f in FH]); hY = np.concatenate([f["Yc"] for f in FH])
cal = e["fit_cal"](*e["probs"](hO[0] + hD[..., 0], hO[1] + hD[..., 1]), hY)
dd = np.mean([s[EP] for s in T["deep_snaps"]], 0); PD_ = e["calibrate"](*e["probs"](T["lin"][0] + dd[..., 0], T["lin"][1] + dd[..., 1]), cal)
rc = e["rps"](*e["probs"](Oa, Ol), Yc); r = e["rps"](*PD_, Yc)
ere_d = (1 - np.nanmean(r[:, :7]) / np.nanmean(rc[:, :7]), 1 - np.nanmean(r[:, 7:]) / np.nanmean(rc[:, 7:])); print("\ndaily ERE RPSS d1-7 / d8-14: %.4f / %.4f" % ere_d)
# ---------------- 3. weekly heads
W, WT = T["W"], T["W_train"]; Dw = {ep: np.mean([s[ep] for s in T["deep_w_snaps"]], 0) for ep in (0, EP)}
ref_rows = WT["IY"] >= 1981
def wclim(doy, w):
    c_, d_ = WT["YWc"][ref_rows, w], WT["doy"][ref_rows]; pa, pl = np.zeros(len(doy)), np.zeros(len(doy))
    for i, d in enumerate(doy):
        m = np.isfinite(c_) & (np.abs(d_ - d) <= 15); pa[i] = (c_[m] >= 1).mean()
        m2 = np.isfinite(c_) & (c_ >= 1) & (np.abs(d_ - d) <= 30); pl[i] = pa[i] * (c_[m2] >= 2).mean()
    return pa, pl
REFW = [wclim(W["doy"], w) for w in range(2)]
PA = {ep: sig(W["LW"][..., 0] + Dw[ep][..., 0]) for ep in Dw}; PL = {ep: PA[ep] * sig(W["LW"][..., 1] + Dw[ep][..., 1]) for ep in Dw}
ZH = {ep: W["LOCW"].astype("float32") + np.einsum("nwp,pc->nwc", W["GW"] + Dw[ep][..., 2:], W["V"]) for ep in Dw}
ZW = W["ZW"].astype("float32")
# last week persists: the observed mean of days T-6..T = week-1 target of the issue day 7 days earlier (same summer)
ZP = np.full(ZW.shape, np.nan, "float32"); pos = {d: i for i, d in enumerate(TI)}
for i, d in enumerate(TI):
    j = pos.get(d - pd.Timedelta(days=7))
    if j is not None:
        ZP[i, 0] = ZP[i, 1] = ZW[j, 0]
# ECMWF weekly maps and weekly ERE (fitted on 2005-2013 out-of-fold rows of round W3B2, applied to the test)
recs = [pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(CVD, "cv_W3N_f*_s0.pkl")))]
cvW = {k: np.concatenate([r["W"][k] for r in recs]) for k in ("YWc", "OFFW", "VW", "ZW")}; cvIY = np.concatenate([r["IY"] for r in recs])
cvT = DATES[ISS[np.concatenate([r["va_rows"] for r in recs])]]; m05 = (cvIY >= 2005) & (cvIY <= 2013)
cvW = {k: v[m05] for k, v in cvW.items()}; cvT = cvT[m05]; cvIY = cvIY[m05]
s_tr, o_tr = starts(cvT); s_te, o_te = starts(TI); assert (s_tr >= 0).all() and (s_te >= 0).all()
lat, lon = ez["lat"][::-1], ez["lon"]; MN = ez["mean"]
XE = np.stack([np.stack([RegularGridInterpolator((lat, lon), MN[s][j][::-1], bounds_error=False, fill_value=None)(PTS) for j in range(1, 17)]) for s in range(len(ST))])
XE = np.log1p(np.maximum(XE, 0)); sy = ST.year.values; ZE = np.zeros_like(XE)
for j in range(XE.shape[1]):
    dk = (ST + pd.Timedelta(days=j)).dayofyear.values
    for i in range(len(ST)):
        m = (sy != sy[i]) & (np.abs(dk - dk[i]) <= 15); ZE[i, j] = (XE[i, j] - XE[m, j].mean(0)) / np.maximum(XE[m, j].std(0), 0.05)
def wk_maps(s_, o_):
    J = o_[:, None] + np.arange(1, 15)[None] + 1; out = np.zeros((len(s_), 2, XE.shape[2]), "float32")
    for i in range(len(s_)):
        zz = ZE[s_[i], J[i] - 1]; out[i, 0], out[i, 1] = zz[:7].mean(0), zz[7:].mean(0)
    return out, J
ZEtr, Jtr = wk_maps(s_tr, o_tr); ZEte, Jte = wk_maps(s_te, o_te)
for w in range(2):
    ok = cvW["VW"][:, w]; yz, fz = cvW["ZW"][ok, w].astype("float64"), ZEtr[ok, w].astype("float64"); ZEte[:, w] *= (yz * fz).sum() / (fz * fz).sum()
CIM, CMX = ez["ci_mem"], ez["cimax"]; thr = np.percentile(np.concatenate([CIM[s_tr[i]][:, Jtr[i]].ravel() for i in range(len(s_tr))]), 90)
def efeat(s_, J, w):
    out = []
    for i in range(len(s_)):
        jj = J[i][7 * w:7 * w + 7]; cm = CIM[s_[i]][:, jj]; mx = CMX[s_[i]][jj]; a = np.log1p(cm.mean(0)); b = (cm > thr).mean(0); c = np.log1p(mx)
        out.append(np.r_[a.mean(), b.mean(), c.mean(), a.max(), b.max(), c.max()])
    return np.array(out)
def fit_off(X, y, off, l2=1.0):
    X1 = np.c_[np.ones(len(X)), X]; wv = np.zeros(X1.shape[1]); R_ = l2 * np.diag(np.r_[0.0, np.ones(X.shape[1])])
    for _ in range(60):
        p = sig(off + X1 @ wv); st = np.linalg.solve((X1 * (p * (1 - p))[:, None]).T @ X1 + R_, X1.T @ (p - y) + R_ @ wv); wv -= st
        if np.abs(st).max() < 1e-8:
            break
    return wv
EPA, EPL = np.zeros(PA[0].shape), np.zeros(PA[0].shape)
for w in range(2):
    Xtr, Xte = efeat(s_tr, Jtr, w), efeat(s_te, Jte, w); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ok = cvW["VW"][:, w]; c_ = cvW["YWc"][:, w]
    wa = fit_off(Xtr[ok], (c_[ok] >= 1).astype(float), cvW["OFFW"][ok, w, 0]); okl = ok & (c_ >= 1); wl = fit_off(Xtr[okl], (c_[okl] >= 2).astype(float), cvW["OFFW"][okl, w, 1])
    EPA[:, w] = sig(W["OFFW"][:, w, 0] + np.c_[np.ones(len(Xte)), Xte] @ wa); EPL[:, w] = EPA[:, w] * sig(W["OFFW"][:, w, 1] + np.c_[np.ones(len(Xte)), Xte] @ wl)
def wscore(sel):
    o = {}
    for w in range(2):
        ok = sel & W["VW"][:, w]; c_ = W["YWc"][ok, w]; r0 = rps_w(REFW[w][0][ok], REFW[w][1][ok], c_).sum()
        for nm, (a, l) in (("P", (PA[0], PL[0])), ("P_ep12", (PA[EP], PL[EP])), ("EC", (EPA, EPL))):
            o[f"ere_w{w+1}_{nm}"] = 1 - rps_w(a[ok, w], l[ok, w], c_).sum() / r0
        y = ZW[ok, w]
        for nm, F in (("P", ZH[EP]), ("lin", ZH[0]), ("EC", ZEte), ("pers", ZP)):
            g = np.isfinite(F[ok, w]).all(-1); o[f"acc_w{w+1}_{nm}"] = acc(F[ok, w][g], y[g]).mean()
            o[f"msss_w{w+1}_{nm}"] = 1 - ((y[g] - F[ok, w][g]) ** 2).sum() / (y[g] ** 2).sum()
    return o
ALL = wscore(np.ones(len(te), bool)); PER = {y: wscore(yrs == y) for y in Y}
print("\nTEST 2014-2024, weekly heads (ERE: RPSS of 3 weekly classes vs the 1981-2013 weekly normal chance; maps: pattern corr. / MSSS)")
rows = []
for key in ("ere_w1", "ere_w2", "acc_w1", "acc_w2", "msss_w1", "msss_w2"):
    d = np.array([PER[y][f"{key}_P"] - PER[y][f"{key}_EC"] for y in Y]); b_ = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    row = dict(score=key, model_P=ALL[f"{key}_P"], ecmwf=ALL[f"{key}_EC"], P_minus_EC=f"{ALL[f'{key}_P'] - ALL[f'{key}_EC']:+.3f} [{np.percentile(b_, 5):+.3f}, {np.percentile(b_, 95):+.3f}]",
               P_better=f"{int((d > 0).sum())}/11")
    if key.startswith("ere"):
        row["P_epoch12"] = ALL[f"{key}_P_ep12"]
    else:
        row["linear_part"] = ALL[f"{key}_lin"]; row["last_week_persists"] = ALL[f"{key}_pers"]
    rows.append(row)
TW = pd.DataFrame(rows); pd.set_option("display.width", 250); print(TW.round(3).to_string(index=False))
OUT = os.path.join(IM, "poster"); os.makedirs(OUT, exist_ok=True)
RB.to_csv(os.path.join(OUT, "test_W3BF_bsiso.csv")); TW.to_csv(os.path.join(OUT, "test_W3BF_weekly.csv"), index=False)
pd.DataFrame({"daily_ere_d1_7": [ere_d[0]], "daily_ere_d8_14": [ere_d[1]]}).to_csv(os.path.join(OUT, "test_W3BF_daily_ere.csv"), index=False)
np.savez_compressed(os.path.join(OUT, "weekly_maps_test.npz"), dates=np.array(TI.strftime("%Y-%m-%d")), ours=ZH[EP], ecmwf=ZEte, obs=ZW, pts=PTS, VW=W["VW"])
print("saved to", os.path.abspath(OUT))
