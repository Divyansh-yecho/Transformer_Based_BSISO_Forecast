# Round W3 (three heads: daily BSISO, weekly ERE, weekly rain map; PT1R training). Cross-validation 1981-2013, 5 folds x seeds.
# For every epoch: transformer gain = score minus epoch 0 (the frozen linear parts). Season bootstrap over the 33 held-out summers.
#   weekly ERE : RPSS of 3 weekly classes vs the out-of-fold 1981-2013 weekly normal chance (as w3_linear_cv.py)
#   weekly map : pattern anomaly correlation (ACC) and MSSS of week-1 / week-2 mean z (climatology = 0); damped persistence shown
#   BSISO      : bivariate COR days 8-14 and horizon; compared with PT1R (round PT1B, same folds and training)
#   daily ERE  : RPSS d1-7 / d8-14 vs PH7CV A0's normal chance (as pt_eval.py)
# usage: python w3_eval.py ROUND CFG [CFG ...]
import os, sys, glob
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import WORK, RUNS as RUN_DIR  # repo-wide paths, see paths.py
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); K = RUN_DIR
RND, CFGL = sys.argv[1], sys.argv[2:] or ["W3", "W3N"]
sig = lambda x: 1 / (1 + np.exp(-x))
recs_of = lambda rnd, cfg: [pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(K, rnd, "out", "out", f"cv_{cfg}_f*_s*.pkl")))]
def bcor(F, O):
    return (F * O).sum((0, 2)) / np.sqrt((F ** 2).sum((0, 2)) * (O ** 2).sum((0, 2)))
def horizon(c):
    b = np.where(c < 0.5)[0]
    return 14.0 if not len(b) else (1.0 if b[0] == 0 else b[0] + (c[b[0] - 1] - 0.5) / (c[b[0] - 1] - c[b[0]]))
def rps_d(pa, pl, Y):
    r = ((pa - (Y >= 1)) ** 2 + (pl - (Y >= 3)) ** 2).astype(float); r[~np.isfinite(Y)] = np.nan; return r
def acc(f, y):
    fc, yc = f - f.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True); den = np.sqrt((fc ** 2).sum(-1) * (yc ** 2).sum(-1))
    return np.where(den > 0, (fc * yc).sum(-1) / np.maximum(den, 1e-12), 0.0)
REFR = recs_of("PH7CV", "A0"); FOLDS = sorted({r["fold"] for r in REFR}); ref0 = {f: [r for r in REFR if r["fold"] == f][0] for f in FOLDS}
IYd = np.concatenate([ref0[f]["IY"] for f in FOLDS]); Ycd = np.concatenate([ref0[f]["Yc"] for f in FOLDS])
OFFR = np.concatenate([np.stack(ref0[f]["OFF"], -1) for f in FOLDS]); RCd = rps_d(sig(OFFR[..., 0]), sig(OFFR[..., 0]) * sig(OFFR[..., 1]), Ycd)
YS = sorted(set(IYd)); w1, w2 = slice(0, 7), slice(7, 14)


class Run:
    def __init__(self, rnd, cfg):
        self.recs = recs_of(rnd, cfg); assert self.recs, f"no records for {rnd}:{cfg}"
        self.seeds = sorted({r["seed"] for r in self.recs}); self.eps = sorted(self.recs[0]["snaps"]); self.wk = "w_snaps" in self.recs[0]
        f0 = lambda f: [r for r in self.recs if r["fold"] == f]
        for f in FOLDS:
            assert np.array_equal(f0(f)[0]["IY"], ref0[f]["IY"]), "held-out rows differ"
        cat = lambda g: np.concatenate([g(f0(f)[0]) for f in FOLDS])
        self.IY, self.AUX, self.Yc = cat(lambda r: r["IY"]), cat(lambda r: r["AUX"]), cat(lambda r: r["Yc"])
        self.OFF = cat(lambda r: np.stack(r["OFF"], -1)); self.fold = cat(lambda r: np.full(len(r["IY"]), r["fold"]))
        if self.wk:
            self.W = {k: cat(lambda r, k=k: r["W"][k]) for k in ("YWc", "LW", "OFFW", "VW", "RW", "GW", "ZW", "LOCW", "doy")}
            self.V = {f: f0(f)[0]["W"]["V"] for f in FOLDS}
            self.ZW32 = self.W["ZW"].astype("float32")
        self._c = {}
    def S(self, e):
        if ("S", e) not in self._c:
            self._c[("S", e)] = np.concatenate([np.mean([r["snaps"][e] for r in self.recs if r["fold"] == f], 0) for f in FOLDS])
        return self._c[("S", e)]
    def D(self, e):                                                     # weekly corrections (rows, 2, 32), mean over seeds
        if e not in self._c:
            self._c[e] = np.concatenate([np.mean([r["w_snaps"][e] for r in self.recs if r["fold"] == f], 0) for f in FOLDS])
        return self._c[e]
    def zhat(self, e):
        if ("Z", e) not in self._c:
            d = self.D(e); g = self.W["GW"] + d[..., 2:]; out = np.zeros(self.W["ZW"].shape, "float32")
            for f in FOLDS:
                m = self.fold == f; out[m] = self.W["LOCW"][m].astype("float32") + np.einsum("nwp,pc->nwc", g[m], self.V[f])
            self._c[("Z", e)] = out
        return self._c[("Z", e)]
    def wprob(self, e):
        d = self.D(e); pa = sig(self.W["LW"][..., 0] + d[..., 0]); return pa, pa * sig(self.W["LW"][..., 1] + d[..., 1])


RUNS = {c: Run(RND, c) for c in CFGL}; CTRL = Run("PT1B", "PT1R")
R0 = RUNS[CFGL[0]]; YW, DY, FD, VW = R0.W["YWc"], R0.W["doy"], R0.fold, R0.W["VW"]
# out-of-fold 1981-2013 weekly normal chance (from the other folds' held-out rows; issue day of year +-15 d, large | any +-30 d)
RA, RL = np.zeros(YW.shape), np.zeros(YW.shape)
for w in range(2):
    for i in range(len(DY)):
        o = (FD != FD[i]) & np.isfinite(YW[:, w]); m = o & (np.abs(DY - DY[i]) <= 15); RA[i, w] = (YW[m, w] >= 1).mean()
        m2 = o & (YW[:, w] >= 1) & (np.abs(DY - DY[i]) <= 30); RL[i, w] = RA[i, w] * (YW[m2, w] >= 2).mean()
rps_w = lambda pa, pl, c: (pa - (c >= 1)) ** 2 + (pl - (c >= 2)) ** 2
RCW = rps_w(RA, RL, YW)                                                   # reference RPS (rows, 2); NaN where the week is invalid
for R in RUNS.values():
    assert np.array_equal(np.nan_to_num(R.W["YWc"], nan=-1), np.nan_to_num(YW, nan=-1)), "weekly targets differ between configs"
rng = np.random.default_rng(0); BOOT = [rng.choice(YS, len(YS)) for _ in range(1000)]
yi = {y: np.where(R0.IY == y)[0] for y in YS}


def wk_scores(R, e, sel=None):
    """weekly ERE RPSS (w1, w2) and map ACC / MSSS (w1, w2) on the rows sel (default all)"""
    sel = np.arange(len(R.IY)) if sel is None else sel; pa, pl = R.wprob(e); zh = R.zhat(e); out = {}
    for w in range(2):
        ok = sel[VW[sel, w]]
        out[f"ere_w{w+1}"] = 1 - rps_w(pa[ok, w], pl[ok, w], YW[ok, w]).sum() / RCW[ok, w].sum()
        y = R.ZW32[ok, w]; f = zh[ok, w]
        out[f"acc_w{w+1}"] = acc(f, y).mean(); out[f"msss_w{w+1}"] = 1 - ((y - f) ** 2).sum() / (y ** 2).sum()
    return out


def bs_scores(R, e, sel=None):
    sel = np.arange(len(R.IY)) if sel is None else sel; Sx = R.S(e)[sel]; c = bcor(Sx[..., 2:], R.AUX[sel])
    r = rps_d(sig(R.OFF[sel, :, 0] + Sx[..., 0]), sig(R.OFF[sel, :, 0] + Sx[..., 0]) * sig(R.OFF[sel, :, 1] + Sx[..., 1]), R.Yc[sel])
    return dict(bsiso=c[w2].mean(), horizon=horizon(c), ere_d1_7=1 - np.nanmean(r[:, w1]) / np.nanmean(RCd[sel][:, w1]),
                ere_d8_14=1 - np.nanmean(r[:, w2]) / np.nanmean(RCd[sel][:, w2]))


def all_scores(R, e, sel=None):
    out = bs_scores(R, e, sel)
    if R.wk:
        out.update(wk_scores(R, e, sel))
    return out


def boot(Ra, ea, Rb, eb, key):
    """key score of (Rb, eb) minus (Ra, ea): mean, 90% season-bootstrap range, summers better of 33"""
    d = all_scores(Rb, eb)[key] - all_scores(Ra, ea)[key]
    dd = [all_scores(Rb, eb, np.concatenate([yi[y] for y in b]))[key] - all_scores(Ra, ea, np.concatenate([yi[y] for y in b]))[key] for b in BOOT[:300]]
    nb = sum(all_scores(Rb, eb, yi[y])[key] > all_scores(Ra, ea, yi[y])[key] for y in YS)
    return f"{d:+.4f} [{np.percentile(dd, 5):+.4f}, {np.percentile(dd, 95):+.4f}] {nb}/{len(YS)}"


pd.set_option("display.width", 250)
EP = [e for e in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12] if e in R0.eps]
print(f"seeds: { {c: len(R.seeds) for c, R in RUNS.items()} }; control PT1R seeds {len(CTRL.seeds)}")
for c, R in RUNS.items():
    T = pd.DataFrame({e: all_scores(R, e) for e in EP}).T; T.index.name = "epoch"
    print(f"\n{c}: scores by epoch (epoch 0 = the frozen linear parts)"); print(T.round(4).to_string())
# damped persistence (local part only) for the maps, and the weekly ERE normal chance of 1960-2013 (the model's own offset)
loc = {}
for w in range(2):
    ok = VW[:, w]; y = R0.W["ZW"][ok, w].astype("float32"); f = R0.W["LOCW"][ok, w].astype("float32")
    loc[f"acc_w{w+1}"] = acc(f, y).mean(); loc[f"msss_w{w+1}"] = 1 - ((y - f) ** 2).sum() / (y ** 2).sum()
    oa = sig(R0.W["OFFW"][ok, w, 0]); ol = oa * sig(R0.W["OFFW"][ok, w, 1]); loc[f"ere_w{w+1}_clim1960"] = 1 - rps_w(oa, ol, YW[ok, w]).sum() / RCW[ok, w].sum()
print("\nreference rows: map damped persistence (local part only) and the 1960-2013 weekly normal chance:", {k: round(v, 4) for k, v in loc.items()})
T = pd.DataFrame({e: bs_scores(CTRL, e) for e in [e for e in EP if e in CTRL.eps]}).T; print("\nPT1R (control, PT1B):"); print(T.round(4).to_string())
rows = []
for c, R in RUNS.items():
    g = {e: all_scores(R, e) for e in R.eps if e > 0}
    for key in (["ere_w1", "ere_w2", "acc_w1", "acc_w2"] if R.wk else []) + ["bsiso", "ere_d1_7", "ere_d8_14"]:
        eb = max(g, key=lambda e: g[e][key]); el = max(R.eps)
        rows.append(dict(config=c, score=key, epoch0=all_scores(R, 0)[key], best_epoch=eb, gain_best=boot(R, 0, R, eb, key),
                         gain_last=boot(R, 0, R, el, key)))
    for e in (8, 12):
        if e in R.eps and e in CTRL.eps:
            rows.append(dict(config=c, score=f"bsiso vs PT1R ep{e}", epoch0=np.nan, best_epoch=e, gain_best=boot(CTRL, e, R, e, "bsiso"), gain_last=""))
T = pd.DataFrame(rows); print("\nGAINS (mean, 90% season-bootstrap range, summers better of 33)"); print(T.round(4).to_string(index=False))
T.to_csv(os.path.join(WORK, "maphead", f"w3_cv_{RND}.csv"), index=False)
