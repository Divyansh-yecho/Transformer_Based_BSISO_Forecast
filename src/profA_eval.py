# Prof model A evaluation.
#   python profA_eval.py cv            -> summarise every cross-validation config in out/ (out-of-fold, 1981-2013 only):
#                                          RPS skill by epoch vs the linear link, best epoch, shrinkage nu, BSISO correlation by lead
#   python profA_eval.py final A1 EPOCH NU_W1 NU_W2 NU_B -> score out/final_A1.pkl on 2014-2024 (rules: file 24, section 7)
import os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUTD = os.environ.get("PROFA_OUT", os.path.join(HERE, "..", "intermediate", "profA", "out"))
NL = 14; NUS = [0.0, 0.25, 0.5, 0.75, 1.0]
sig = lambda x: 1 / (1 + np.exp(-x))


def probs(La, Ll):
    pa = sig(La); return pa, pa * sig(Ll)


def rps(pa, pl, Yc):
    r = (pa - (Yc >= 1)) ** 2 + (pl - (Yc >= 3)) ** 2; r = r.astype("float64"); r[~np.isfinite(Yc)] = np.nan; return r


def bcor(F, O):
    """bivariate correlation per lead: F, O (N, 14, 2)"""
    ok = np.isfinite(O).all(2)
    num = np.nansum(np.where(ok[..., None], F * O, 0), (0, 2))
    den = np.sqrt(np.nansum(np.where(ok[..., None], F ** 2, 0), (0, 2)) * np.nansum(np.where(ok[..., None], O ** 2, 0), (0, 2)))
    return num / den


def horizon(c):
    below = np.where(c < 0.5)[0]
    if not len(below):
        return float(NL)
    k = below[0]
    return float(k + 1) if k == 0 else k + (c[k - 1] - 0.5) / (c[k - 1] - c[k])   # leads are 1-based: index k-1 = lead k


def load_cv(cfg):
    recs = [pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(OUTD, f"cv_{cfg}_f*_s*.pkl"))) if ("_fast" not in f) == (os.environ.get("EVAL_FAST") != "1")]
    folds = sorted({r["fold"] for r in recs})
    out = []
    for fi in folds:
        rs = [r for r in recs if r["fold"] == fi]; eps = sorted(rs[0]["snaps"])
        snap = {e: np.mean([r["snaps"][e] for r in rs], 0) for e in eps}         # seed average (logit-space corrections)
        out.append(dict(rs[0], snaps=snap, nseeds=len(rs)))
    if os.environ.get("EVAL_EC") == "1":                               # Prof B: score only held-out rows that have ECMWF forecasts
        sub = []
        for f in out:
            m = np.asarray(f.get("ECOK", np.zeros(len(f["va_rows"]), bool)))
            if m.any():
                g = dict(f, va_rows=f["va_rows"][m], snaps={e_: v[m] for e_, v in f["snaps"].items()}, lin=[x[m] for x in f["lin"]], Yc=f["Yc"][m],
                         OFF=tuple(x[m] for x in f["OFF"]), AUX=f["AUX"][m], PERS=f["PERS"][m], IY=f["IY"][m], ECOK=m[m])
                sub.append(g)
        out = sub
    return out


def cv_summary(cfg):
    F = load_cv(cfg)
    if not F:
        return None
    eps = sorted(F[0]["snaps"]); Yc = np.concatenate([f["Yc"] for f in F]); AUX = np.concatenate([f["AUX"] for f in F])
    La = np.concatenate([f["lin"][0] for f in F]); Ll = np.concatenate([f["lin"][1] for f in F]); Pc = np.concatenate([f["lin"][2] for f in F])
    Oa = np.concatenate([f["OFF"][0] for f in F]); Ol = np.concatenate([f["OFF"][1] for f in F])
    rc = np.nanmean(rps(*probs(Oa, Ol), Yc), 0); rl = np.nanmean(rps(*probs(La, Ll), Yc), 0)
    rows = []
    for e in eps:
        o = np.concatenate([f["snaps"][e] for f in F])
        for nu in NUS:
            rd = np.nanmean(rps(*probs(La + nu * o[..., 0], Ll + nu * o[..., 1]), Yc), 0)
            cb = bcor(Pc + nu * o[..., 2:], AUX)
            rows.append({"cfg": cfg, "epoch": e, "nu": nu, "RPSS all": 1 - rd.mean() / rc.mean(), "RPSS w1": 1 - rd[:7].mean() / rc[:7].mean(), "RPSS w2": 1 - rd[7:].mean() / rc[7:].mean(),
                         "COR d8-14": cb[7:].mean(), "horizon": horizon(cb)})
    S = pd.DataFrame(rows)
    lin = {"RPSS w1": 1 - rl[:7].mean() / rc[:7].mean(), "RPSS w2": 1 - rl[7:].mean() / rc[7:].mean(), "COR d8-14": bcor(Pc, AUX)[7:].mean(),
           "horizon": horizon(bcor(Pc, AUX))}
    return S, lin, dict(folds=len(F), seeds=F[0]["nseeds"], nparams=F[0]["nparams"])


if __name__ == "__main__" and sys.argv[1] == "cv":
    cfgs = sorted({os.path.basename(f).split("_")[1] for f in glob.glob(os.path.join(OUTD, "cv_*_f*_s*.pkl")) if ("_fast" not in f) == (os.environ.get("EVAL_FAST") != "1")})
    pd.set_option("display.width", 200)
    for c in cfgs:
        r = cv_summary(c)
        if r is None:
            continue
        S, lin, info = r
        print(f"\n=== {c}: {info} | linear link: " + ", ".join(f"{k} {v:.4f}" for k, v in lin.items()))
        piv = S[S.nu == 1.0].set_index("epoch")[["RPSS w1", "RPSS w2", "COR d8-14", "horizon"]]
        print("nu = 1 by epoch:\n" + piv.round(4).to_string())
        best = S.loc[S.groupby("nu")["RPSS w1"].idxmax()].set_index("nu")
        print("best epoch per nu (by RPSS w1):\n" + best[["epoch", "RPSS w1", "RPSS w2", "COR d8-14", "horizon"]].round(4).to_string())
        S.to_csv(os.path.join(OUTD, f"cv_summary_{c}.csv"), index=False)
        # pre-declared selection rule (file 24, section 9): epoch = best out-of-fold RPSS over days 1-14 at nu = 1; then nu per band
        ep = int(S[S.nu == 1.0].set_index("epoch")["RPSS all"].idxmax()); Se = S[S.epoch == ep].set_index("nu")
        n1, n2, nb = Se["RPSS w1"].idxmax(), Se["RPSS w2"].idxmax(), Se["COR d8-14"].idxmax()
        print(f"SELECTION {c}: epoch {ep} | nu days 1-7 = {n1} (RPSS {Se.loc[n1, 'RPSS w1']:.4f} vs linear {lin['RPSS w1']:.4f}) | "
              f"nu days 8-14 = {n2} (RPSS {Se.loc[n2, 'RPSS w2']:.4f} vs {lin['RPSS w2']:.4f}) | nu BSISO = {nb} (COR {Se.loc[nb, 'COR d8-14']:.4f} vs {lin['COR d8-14']:.4f})")


# ---------------------------------------------------------------- final test (2014-2024), rules in file 24 section 7
def platt_fit(p, y):
    """1-D logistic recalibration on logit(p); returns (a, b)."""
    x = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))); w = np.array([0.0, 1.0])
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(50):
        q = sig(X @ w); g = X.T @ (q - y); H = (X * (q * (1 - q))[:, None]).T @ X + 1e-6 * np.eye(2); st = np.linalg.solve(H, g); w -= st
        if np.abs(st).max() < 1e-8:
            break
    return w


def platt_apply(p, w):
    x = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))); return sig(w[0] + w[1] * x)


BANDS = [range(0, 5), range(5, 10), range(10, 14)]


def calibrate(pa, pl, cal):
    pa, pl = pa.copy(), pl.copy()
    for bi, b in enumerate(BANDS):
        pa[:, b] = platt_apply(pa[:, b], cal[("any", bi)]); pl[:, b] = platt_apply(pl[:, b], cal[("lg", bi)])
    return pa, np.minimum(pl, pa)


def fit_cal(pa, pl, Yc):
    cal = {}
    for bi, b in enumerate(BANDS):
        y = Yc[:, b]; ok = np.isfinite(y)
        cal[("any", bi)] = platt_fit(pa[:, b][ok], (y[ok] >= 1).astype(float)); cal[("lg", bi)] = platt_fit(pl[:, b][ok], (y[ok] >= 3).astype(float))
    return cal


def auc(p, y):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(y); return roc_auc_score(y[ok], p[ok]) if 0 < np.nanmean(y[ok]) < 1 else np.nan


def score_test(cfg, epoch, nu_w1, nu_w2, nu_b):
    T = pd.read_pickle(os.path.join(OUTD, f"final_{cfg}{os.environ.get('FINAL_SUFFIX', '')}.pkl"))
    F = load_cv(cfg)                                    # out-of-fold forecasts for the calibration (same epoch and nu)
    nu_v = np.r_[np.full(7, nu_w1), np.full(7, nu_w2)][None, :]
    def ere(La, Ll, o):
        return probs(La + (0 if o is None else nu_v * o[..., 0]), Ll + (0 if o is None else nu_v * o[..., 1]))
    oof = {"linear": ere(np.concatenate([f["lin"][0] for f in F]), np.concatenate([f["lin"][1] for f in F]), None),
           "profA": ere(np.concatenate([f["lin"][0] for f in F]), np.concatenate([f["lin"][1] for f in F]), np.concatenate([f["snaps"][epoch] for f in F]))}
    Yoof = np.concatenate([f["Yc"] for f in F]); cals = {k: fit_cal(*v, Yoof) for k, v in oof.items()}
    La, Ll, Pc = T["lin"]; Yc = T["Yc"]; Oa, Ol = T["OFF"]; AUXt = T["AUX"]; yrs = T["IY"]
    deep = np.mean(T["deep"], 0); ctrl = np.mean(T["ctrl"], 0) if T["ctrl"] else None
    P = {"normal chance": probs(Oa, Ol), "linear (logistic)": calibrate(*ere(La, Ll, None), cals["linear"]),
         "Prof A": calibrate(*ere(La, Ll, deep), cals["profA"])}
    if ctrl is not None:
        P["control"] = calibrate(*ere(La, Ll, ctrl), cals["profA"])
    B = {"persistence": np.repeat(T["PERS"][:, None, :], NL, 1), "ridge": Pc, "Prof A": Pc + nu_b * deep[..., 2:]}
    if ctrl is not None:
        B["control"] = Pc + nu_b * ctrl[..., 2:]
    return dict(P=P, B=B, Yc=Yc, AUX=AUXt, yrs=yrs, clim=P["normal chance"])


def season_stats(sc_by_year_a, sc_by_year_b, higher_better=True, nb=4000, seed=0):
    """paired season bootstrap of (b - a) for per-season sums; returns mean diff, 90% range, seasons better."""
    a, b = np.asarray(sc_by_year_a), np.asarray(sc_by_year_b); n = len(a)
    idx = np.random.default_rng(seed).integers(0, n, (nb, n))
    d = (b[idx].mean(1) - a[idx].mean(1)) * (1 if higher_better else -1)
    better = int(((b - a) * (1 if higher_better else -1) > 0).sum())
    return float(d.mean()), float(np.percentile(d, 5)), float(np.percentile(d, 95)), better


if __name__ == "__main__" and sys.argv[1] == "final":
    cfg, ep, n1, n2, nb_ = sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
    R = score_test(cfg, ep, n1, n2, nb_); Yc, yrs = R["Yc"], R["yrs"]; Y = sorted(set(yrs)); rc = rps(*R["clim"], Yc)
    rows = []
    for m, (pa, pl) in R["P"].items():
        r = rps(pa, pl, Yc); sk = 1 - np.nanmean(r, 0) / np.nanmean(rc, 0)
        row = {"model": m, **{f"RPSS d{k + 1}": sk[k] for k in range(NL)}, "RPSS d1-7": 1 - np.nanmean(r[:, :7]) / np.nanmean(rc[:, :7]),
               "RPSS d8-14": 1 - np.nanmean(r[:, 7:]) / np.nanmean(rc[:, 7:])}
        for k in (0, 2, 4, 6, 9, 13):
            row[f"AUC any d{k + 1}"] = auc(pa[:, k], (Yc[:, k] >= 1).astype(float) + 0 * Yc[:, k]); row[f"AUC large d{k + 1}"] = auc(pl[:, k], (Yc[:, k] >= 3).astype(float) + 0 * Yc[:, k])
        rows.append(row)
    S = pd.DataFrame(rows); S.to_csv(os.path.join(OUTD, f"test_scores_{cfg}.csv"), index=False)
    print(S[["model", "RPSS d1-7", "RPSS d8-14", "RPSS d1", "RPSS d5", "RPSS d10", "RPSS d14", "AUC any d1", "AUC any d7", "AUC large d1"]].round(4).to_string(index=False))
    cb = {m: bcor(f, R["AUX"]) for m, f in R["B"].items()}
    for m, c in cb.items():
        print(f"BSISO {m}: COR d1 {c[0]:.3f} d5 {c[4]:.3f} d10 {c[9]:.3f} d14 {c[13]:.3f} | mean d8-14 {c[7:].mean():.3f} | horizon {horizon(c):.1f} d")
    # primary comparisons, per season (Holm over 3)
    def per_season_rpss(pa, pl, cols):
        return [1 - np.nanmean(rps(pa, pl, Yc)[yrs == y][:, cols]) / np.nanmean(rc[yrs == y][:, cols]) for y in Y]
    out = []
    for name, cols in (("RPSS d1-7", slice(0, 7)), ("RPSS d8-14", slice(7, 14))):
        a = per_season_rpss(*R["P"]["linear (logistic)"], cols); b = per_season_rpss(*R["P"]["Prof A"], cols)
        out.append((name, *season_stats(a, b)))
        if "control" in R["P"]:
            c = per_season_rpss(*R["P"]["control"], cols); out.append((name + " (Prof A vs control)", *season_stats(c, b)))
    sc = lambda F: [bcor(F[yrs == y], R["AUX"][yrs == y])[7:].mean() for y in Y]
    out.append(("BSISO COR d8-14", *season_stats(sc(R["B"]["ridge"]), sc(R["B"]["Prof A"]))))
    if "control" in R["B"]:
        out.append(("BSISO COR d8-14 (Prof A vs control)", *season_stats(sc(R["B"]["control"]), sc(R["B"]["Prof A"]))))
    D = pd.DataFrame(out, columns=["comparison", "gain", "lo (5%)", "hi (95%)", f"seasons better of {len(Y)}"])
    print(D.round(4).to_string(index=False)); D.to_csv(os.path.join(OUTD, f"test_primary_{cfg}.csv"), index=False)
