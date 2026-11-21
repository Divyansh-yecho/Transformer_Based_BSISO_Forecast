# Model A: one multi-input transformer -> probabilities of central-India ERE outcomes (IMD 1-deg EREs), genuine past-only.
# Inputs (token groups): BSISO1/2 history (60 d), observed IMD rain band + central-India rain (10 d), ERA5 field EOFs
# (30 d, when available), slow background (Nino3.4, DMI, season, trend) via FiLM.  Learned query tokens -> heads:
#   week 1 / week 2: P(>=1 ERE day), P(>=3 ERE days), P(large ERE in week); daily days 1-7: P(>=medium), P(>=large);
#   auxiliary (training only): observed BSISO PC1/PC2 for days 1-14.
# Every logit = climatological log-odds of that outcome for the issue date (training years only) + network correction
# (final layer zero-initialised, so an untrained model = the normal chance).  Loss = masked BCE + aux_w * MSE.
# Baselines on the same test days: normal chance, two-stage BSISO model (option B, past-only forecasts), logistic on the same
# inputs, capacity-matched control (same model, non-BSISO groups taken from a random other year).
ROUND = "MA3"
# %% CELL 1 -- settings
import os, glob, json, time, math, warnings
warnings.filterwarnings("ignore")
FAST = os.environ.get("FAST") == "1"
PHASE = os.environ.get("PHASE", "dev")                            # dev = test 1990-2007 ; final = test 2008-2024
TEST_YEARS = list(range(1990, 2008)) if PHASE == "dev" else list(range(2008, 2025))
SEEDS = 3
LOG_L2 = 300.0                                                     # chosen in the dev screening (log 02:12)
ERA = ["dryair", "jet", "mslp", "soil", "sst", "tcwv", "v200"]
VARIANTS = {   # residual transformers on top of the logistic + all ERA5 (l2 = 300)
    "R resid on ERA5-logistic": dict(groups=["bsiso", "rain"] + ERA, d=48, layers=3, ff=96, drop=0.2, wd=1e-2, aux_w=0.5, resid=True),
    "R resid on ERA5-logistic, small": dict(groups=["bsiso", "rain"] + ERA, d=32, layers=2, ff=64, drop=0.2, wd=1e-2, aux_w=0.5, resid=True),
    "R resid CONTROL": dict(groups=["bsiso", "rain"] + ERA, d=48, layers=3, ff=96, drop=0.2, wd=1e-2, aux_w=0.5, resid=True, shuffle=True),
}
if FAST:
    TEST_YEARS, SEEDS = TEST_YEARS[:2], 1

# %% CELL 2 -- data
import numpy as np, pandas as pd
import torch, torch.nn as nn
from scipy import ndimage
from sklearn.metrics import roc_auc_score
ROOTS = [r for r in os.environ.get("MA_DATA", "/kaggle/input").split(os.pathsep) if r]
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs_ma"
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find(pat, required=True):
    for r in ROOTS:
        h = sorted(glob.glob(os.path.join(r, "**", pat), recursive=True))
        if h:
            return h[0]
    if required:
        raise FileNotFoundError(pat)
    return None


z = np.load(find("base_inputs.npz"))
DATES = pd.DatetimeIndex(z["dates"]); R1 = z["R1"]; M1 = z["M1"]; lat1, lon1 = z["lat1"], z["lon1"]
BAND = np.log1p(z["band"]); CIR = np.log1p(z["cir"])
BS = pd.DataFrame(z["bs"], index=pd.DatetimeIndex(z["bs_dates"]), columns=["p1", "p2", "q1", "q2"])
NINO = pd.Series(z["nino"], index=pd.DatetimeIndex(z["nino_dates"])); DMI = pd.Series(z["dmi"], index=pd.DatetimeIndex(z["dmi_dates"]))
FC = pd.read_parquet(find("bsiso_pastonly_forecasts.parquet"))
FCI = {L: FC[FC.lead == L].set_index("issue")[["f1", "f2"]] for L in range(1, 15)}
E5 = None
f5 = find("era5_feats.npz", required=False)
if f5:
    e = np.load(f5)
    E5 = {"dates": pd.DatetimeIndex(e["dates"]), "groups": {k[2:]: e[k] for k in e.files if k.startswith("g_")}}
    print("ERA5 groups:", {k: v.shape for k, v in E5["groups"].items()})
YEARS = DATES.year.values
JJAS = (DATES.month >= 6) & (DATES.month <= 9)
LA1, LO1 = np.meshgrid(lat1, lon1, indexing="ij")
print(f"device {DEV} | days {len(DATES)} | test years {TEST_YEARS[0]}-{TEST_YEARS[-1]} | seeds {SEEDS}", flush=True)


def ci_box(la, lo):
    return (la >= 17) & (la < 26.5) & (lo >= 74) & (lo <= 88.5) & ~((lo <= 77.5) & (la <= 21))


def classes(train_years):
    """1-deg ERE class per day (0 none, 1 one cell, 2 2-5 cells, 3 6+), p99 of wet days from JJAS of train_years."""
    s = JJAS & np.isin(YEARS, train_years)
    W = np.where(R1[s][:, M1] > 1, R1[s][:, M1], np.nan)
    T = np.full(M1.shape, np.inf); T[M1] = np.nanpercentile(W, 99, axis=0)
    out = np.zeros(len(DATES))
    ev = R1 > T[None]
    for t in np.where(JJAS)[0]:
        L, n = ndimage.label(M1 & ev[t], structure=np.ones((3, 3)))
        if n:
            idx = np.arange(1, n + 1)
            sz = np.asarray(ndimage.sum(np.ones(M1.shape), L, idx))
            ok = ci_box(np.asarray(ndimage.mean(LA1, L, idx)), np.asarray(ndimage.mean(LO1, L, idx)))
            if ok.any():
                out[t] = sz[ok].max()
    return np.digitize(out, [1, 2, 6])


# %% CELL 3 -- samples: one per issue day; targets; input arrays
POS = {d: i for i, d in enumerate(DATES)}
ISS = np.array([i for i, d in enumerate(DATES) if (d.month >= 5) and i + 14 < len(DATES) and DATES[i + 14] - d == pd.Timedelta(days=14)
                and DATES[i + 1].month >= 6 and DATES[i + 1].month <= 9 and d.year >= int(os.environ.get("MIN_ISS_YEAR", "1981"))])   # file 39: 1960 with the extended data
ISS = ISS[(DATES[ISS].month < 9) | (DATES[ISS].day <= 29)]
IY = YEARS[ISS]
TNAMES = ["w1_ge1", "w1_ge3", "w1_lg", "w2_ge1", "w2_ge3", "w2_lg"] + [f"d{k}_med" for k in range(1, 8)] + [f"d{k}_lg" for k in range(1, 8)]
NT = len(TNAMES)


def targets(C):
    Y = np.full((len(ISS), NT), np.nan, "float32")
    for n, i in enumerate(ISS):
        for w, o in ((0, 1), (1, 8)):
            idx = np.arange(i + o, i + o + 7)
            if JJAS[idx].all():
                c = C[idx]
                Y[n, 3 * w:3 * w + 3] = [(c >= 1).sum() >= 1, (c >= 1).sum() >= 3, (c >= 3).any()]
        for k in range(1, 8):
            if JJAS[i + k]:
                Y[n, 5 + k] = C[i + k] >= 2; Y[n, 12 + k] = C[i + k] >= 3
    return Y


# BSISO history (60 d) and aux targets (BSISO PC1/PC2 days 1-14)
bsv = BS.reindex(pd.date_range(DATES[0] - pd.Timedelta(days=80), DATES[-1] + pd.Timedelta(days=20))).interpolate(limit=3)
X_BS = np.stack([bsv.loc[DATES[i] - pd.Timedelta(days=59):DATES[i]].values for i in ISS]).astype("float32")      # (N,60,4)
AUX = np.stack([bsv.loc[DATES[i] + pd.Timedelta(days=1):DATES[i] + pd.Timedelta(days=14), ["p1", "p2"]].values for i in ISS]).astype("float32")
OBS5 = bsv.reindex(DATES - pd.Timedelta(days=5))[["p1", "p2"]].values                                         # observed BSISO 5 d before each day
FCV = {L: FCI[L].reindex(DATES[ISS]).values for L in range(1, 10)}                                               # past-only forecast, issue day -> lead L
# observed rain: band profile + CI rain, last 10 days (issue day included)
RAIN = np.concatenate([BAND, CIR[:, None]], 1)
X_RN = np.stack([RAIN[i - 9:i + 1] for i in ISS]).astype("float32")                                              # (N,10,26)
# slow background: Nino3.4 and DMI of the previous month, season (sin/cos), trend
prevm = lambda s, d: s.reindex([pd.Timestamp(d.year, d.month, 1) - pd.offsets.MonthBegin(1)]).values[0]
doy = DATES[ISS].dayofyear.values
X_SL = np.stack([[prevm(NINO, DATES[i]) for i in ISS], [prevm(DMI, DATES[i]) for i in ISS], np.sin(2 * np.pi * doy / 365.25),
                 np.cos(2 * np.pi * doy / 365.25), (IY - 2000) / 20.0], 1).astype("float32")
X_SL = np.nan_to_num(X_SL)
print(f"samples {len(ISS)} | X_BS {X_BS.shape} X_RN {X_RN.shape} X_SL {X_SL.shape}", flush=True)


def era5_tokens(train_years, names, k=5, ndays=30):
    """per ERA5 group: daily anomaly (training-years calendar climatology), PCA on training JJAS days, last 30 days of PCs."""
    out = {}
    ed = E5["dates"]; epos = {d: i for i, d in enumerate(ed)}
    for nm in names:
        A = E5["groups"][nm].astype("float32")                     # (days, points)
        doy_e = ed.dayofyear.values
        tr = np.isin(ed.year.values, train_years)
        clim = np.zeros((367, A.shape[1]), "float32")
        for dd in range(1, 367):
            m = tr & (np.abs(doy_e - dd) <= 7)
            if m.any():
                clim[dd] = A[m].mean(0)
        An = A - clim[doy_e]
        sd = An[tr].std(0) + 1e-6; An = An / sd
        trj = tr & (ed.month >= 6) & (ed.month <= 9)
        U, S, Vt = np.linalg.svd(An[trj] - An[trj].mean(0), full_matrices=False)
        P = (An - An[trj].mean(0)) @ Vt[:k].T
        P = P / (P[trj].std(0) + 1e-6)
        Z = np.zeros((ndays, P.shape[1]), "float32")              # missing ERA5 dates -> climatology (0 after standardising)
        out[nm] = np.stack([P[epos[DATES[i]] - ndays + 1:epos[DATES[i]] + 1] if (DATES[i] in epos and epos[DATES[i]] >= ndays - 1
                            and ed[epos[DATES[i]] - ndays + 1].year == DATES[i].year) else Z for i in ISS]).astype("float32")
    return out


# %% CELL 4 -- the model
class ModelA(nn.Module):
    def __init__(self, gdims, d=48, layers=3, ff=96, heads=4, drop=0.2, patch=5, n_slow=5, n_aux=28):
        super().__init__()
        self.patch, self.names = patch, list(gdims)
        self.emb = nn.ModuleDict({g: nn.Linear(patch * dim, d) for g, (dim, ln) in gdims.items()})
        self.ntok = {g: ln // patch for g, (dim, ln) in gdims.items()}
        self.pos = nn.ParameterDict({g: nn.Parameter(torch.zeros(1, self.ntok[g], d)) for g in gdims})
        self.typ = nn.ParameterDict({g: nn.Parameter(torch.randn(1, 1, d) * 0.02) for g in gdims})
        self.q = nn.Parameter(torch.randn(1, 4, d) * 0.02)          # queries: week 1, week 2, daily, aux BSISO
        self.film = nn.Linear(n_slow, 2 * d); nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True), layers)
        self.norm = nn.LayerNorm(d)
        self.h1, self.h2, self.hd = nn.Linear(d, 3), nn.Linear(d, 3), nn.Linear(d, 14)
        self.ha = nn.Linear(d, n_aux)
        for h in (self.h1, self.h2, self.hd):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

    def forward(self, xs, slow, gmask=None):
        toks = []
        for g in self.names:
            x = xs[g]; B = x.shape[0]
            t = self.emb[g](x.reshape(B, self.ntok[g], -1)) + self.pos[g] + self.typ[g]
            if gmask is not None and g in gmask:
                t = t * gmask[g][:, None, None]
            toks.append(t)
        h = torch.cat([self.q.expand(B, -1, -1)] + toks, 1)
        gb = self.film(slow); g_, b_ = gb.chunk(2, -1)
        h = h * (1 + g_[:, None]) + b_[:, None]
        h = self.norm(self.enc(h))
        return torch.cat([self.h1(h[:, 0]), self.h2(h[:, 1]), self.hd(h[:, 2])], 1), self.ha(h[:, 3])


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# %% CELL 5 -- per-fold helpers: climatology offsets, baselines, training
def clim_offsets(Y, tr):
    """logit of the smoothed calendar climatology of each target over training issue days, for every sample."""
    pos = doy - doy.min()
    off = np.zeros_like(Y)
    for j in range(NT):
        c = np.array([np.nanmean(np.where(tr & (pos == k), Y[:, j], np.nan)) for k in range(pos.max() + 1)])
        c = pd.Series(c).interpolate(limit_direction="both").rolling(15, center=True, min_periods=1).mean().values
        c = np.clip(c, 0.003, 0.997)
        off[:, j] = np.log(c / (1 - c))[pos]
    return off.astype("float32")


def fit_w(X, y, l2=1e-2):
    X = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X.shape[1]); R_ = l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
    for _ in range(40):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        st = np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + R_, X.T @ (p - y) + R_ @ w); w -= st
        if np.abs(st).max() < 1e-7:
            break
    return w


pr_ = lambda w, X: 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(X)), X]) @ w, -30, 30)))


def twostage(C, tr_years, Y, off, trs, tes):
    """option-B two-stage baseline: daily logistic on BSISO T-5 (observed for training; forecast when T-5 > issue day), weekly
    products from summed daily chances (logistic on [clim logit, log sum], fitted on training issue days)."""
    jj = JJAS & np.isin(YEARS, tr_years)
    pos_all = DATES.dayofyear.values
    feat = lambda v: np.column_stack([v, np.hypot(v[:, 0], v[:, 1])])
    P = np.full((len(ISS), NT), np.nan)
    daily_w = {}
    for thr in (1, 2, 3):
        y = (C >= thr).astype(float)
        cdy = np.array([np.mean(y[jj & (pos_all == k)]) if (jj & (pos_all == k)).any() else np.nan for k in range(367)])
        cdy = pd.Series(cdy).interpolate(limit_direction="both").rolling(15, center=True, min_periods=1).mean().values
        cdy = np.clip(cdy, 0.003, 0.997); cl = np.log(cdy / (1 - cdy))
        v5 = bsv.reindex(DATES - pd.Timedelta(days=5))[["p1", "p2"]].values
        ok = jj & np.isfinite(v5).all(1)
        w = fit_w(np.column_stack([cl[pos_all], feat(v5)])[ok], y[ok])
        daily_w[thr] = (w, cl)
    DP = {}                                                         # (thr, lead) -> daily chance for every sample
    for L in range(1, 15):
        t = ISS + L
        v = OBS5[t].copy()
        if L > 5:
            fv = FCV[L - 5]; use = ~trs & np.isfinite(fv).all(1)
            v[use] = fv[use]
        v = np.nan_to_num(v)
        for thr in (1, 2, 3):
            w, cl = daily_w[thr]
            DP[thr, L] = pr_(w, np.column_stack([cl[pos_all[t]], v, np.hypot(v[:, 0], v[:, 1])]))
    S = np.full((len(ISS), 4), np.nan)                              # summed daily P(>=1), P(>=3) for weeks 1, 2
    for k in range(1, 8):
        okk = JJAS[ISS + k]
        P[okk, 5 + k] = DP[2, k][okk]; P[okk, 12 + k] = np.minimum(DP[3, k], DP[2, k])[okk]
    for w_, o in ((0, 1), (1, 8)):
        okw = np.all([JJAS[ISS + o + k] for k in range(7)], 0)
        S[okw, 2 * w_] = np.sum([DP[1, o + k] for k in range(7)], 0)[okw]
        S[okw, 2 * w_ + 1] = np.sum([DP[3, o + k] for k in range(7)], 0)[okw]
    for j, (col, sc) in enumerate(((0, 0), (1, 0), (2, 1), (3, 2), (4, 2), (5, 3))):
        ok = trs & np.isfinite(Y[:, col]) & np.isfinite(S[:, sc])
        X = np.column_stack([off[:, col], np.log(S[:, sc] + 1e-3)])
        w = fit_w(X[ok], Y[ok, col]); m = np.isfinite(S[:, sc])
        P[m, col] = pr_(w, X[m])
    return P


def compact(Xd, tr, ncomp=3):
    """compact features for the logistic baseline: BSISO now/-5/-10 d, amplitude, rain-band EOFs today, CI rain, slow."""
    bsf = np.concatenate([Xd["bsiso"][:, -1], Xd["bsiso"][:, -6], Xd["bsiso"][:, -11], np.hypot(Xd["bsiso"][:, -1, :1], Xd["bsiso"][:, -1, 1:2])], 1)
    rn = Xd["rain"][:, -1, :25]; mu = rn[tr].mean(0); U, S, Vt = np.linalg.svd(rn[tr] - mu, full_matrices=False)
    feats = [bsf, (rn - mu) @ Vt[:ncomp].T, Xd["rain"][:, -1, 25:], Xd["slow"][:, :2]]
    for g in Xd:
        if g not in ("bsiso", "rain", "slow"):
            feats += [Xd[g][:, -1, :], Xd[g][:, -7:, :].mean(1)]
    F = np.concatenate(feats, 1)
    return (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-6)


def train_model(cfg, Xd, Y, off, tr_idx, va_idx, seed, gdims, ctrl_perm=None):
    torch.manual_seed(seed); np.random.seed(seed)
    m = ModelA(gdims, d=cfg["d"], layers=cfg["layers"], ff=cfg["ff"], drop=cfg["drop"]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg["wd"])
    T = {g: torch.tensor(v, device=DEV) for g, v in Xd.items() if g != "slow"}
    SL = torch.tensor(Xd["slow"], device=DEV); YY = torch.tensor(np.nan_to_num(Y), device=DEV)
    MK = torch.tensor(np.isfinite(Y).astype("float32"), device=DEV); OF = torch.tensor(off, device=DEV)
    AX = torch.tensor(AUXn, device=DEV)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def loss_on(idx, train=True):
        xs = {g: T[g][idx] for g in T}
        gm = None
        if train and cfg.get("gdrop", 0.1) > 0:
            gm = {g: (torch.rand(len(idx), device=DEV) > cfg.get("gdrop", 0.1)).float() for g in T if g != "bsiso"}
        lo, ax = m(xs, SL[idx], gm)
        l = (bce(lo + OF[idx], YY[idx]) * MK[idx]).sum() / MK[idx].sum().clamp(min=1)
        return l + cfg["aux_w"] * ((ax - AX[idx]) ** 2).mean(), l

    best, bad, state, ep = np.inf, 0, None, 0
    tri = torch.tensor(tr_idx, device=DEV); vai = torch.tensor(va_idx, device=DEV)
    maxep, pat = (3, 2) if FAST else (300, 25)
    for ep in range(maxep):
        m.train()
        perm = tri[torch.randperm(len(tri), device=DEV)]
        for b in range(0, len(perm), 128):
            opt.zero_grad(); L_, _ = loss_on(perm[b:b + 128]); L_.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            v = loss_on(vai, train=False)[1].item()
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: w.detach().clone() for k, w in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        allidx = torch.arange(len(Y), device=DEV)
        lo = torch.cat([m({g: T[g][allidx[b:b + 1024]] for g in T}, SL[allidx[b:b + 1024]])[0] for b in range(0, len(Y), 1024)])
        P = torch.sigmoid(lo + OF).cpu().numpy()
    return P, ep + 1, best


# %% CELL 6 -- run: every test year, every variant, seeds; baselines on the same days
AUXn = ((AUX - AUX.mean((0, 1))) / AUX.std((0, 1))).reshape(len(AUX), -1).astype("float32")
REC, LOG = [], []
t0 = time.time()
for Yt in TEST_YEARS:
    tr_years = np.arange(1981, Yt)
    C = classes(tr_years)
    Y = targets(C)
    trs, tes = np.isin(IY, tr_years), IY == Yt
    off = clim_offsets(Y, trs)
    rng = np.random.default_rng(Yt)
    va_years = rng.choice(tr_years, max(2, len(tr_years) // 7), replace=False)
    tr_idx = np.where(trs & ~np.isin(IY, va_years))[0]; va_idx = np.where(trs & np.isin(IY, va_years))[0]
    # inputs, normalised with training statistics
    era = era5_tokens(tr_years, sorted({g for v in VARIANTS.values() for g in v["groups"] if g not in ("bsiso", "rain")})) if E5 else {}
    base_X = {"bsiso": X_BS, "rain": X_RN, **era}
    Xn = {}
    for g, A in base_X.items():
        mu, sd = A[trs].mean((0, 1)), A[trs].std((0, 1)) + 1e-6
        Xn[g] = ((A - mu) / sd).astype("float32")
    sl = X_SL.copy(); sl[:, :2] = (sl[:, :2] - sl[trs, :2].mean(0)) / (sl[trs, :2].std(0) + 1e-6); Xn["slow"] = sl
    # baselines
    P_norm = 1 / (1 + np.exp(-off))
    P_two = twostage(C, tr_years, Y, off, trs, tes)
    Fc = compact({g: Xn[g] for g in Xn}, trs)                     # BSISO, rain, slow + every ERA5 group (issue day, 7-day mean)
    P_log = np.full_like(P_norm, np.nan)
    for j in range(NT):
        ok = trs & np.isfinite(Y[:, j])
        w = fit_w(np.column_stack([off[:, j], Fc])[ok], Y[ok, j], l2=LOG_L2)
        P_log[:, j] = pr_(w, np.column_stack([off[:, j], Fc]))
    preds = {"normal chance": P_norm, "two-stage BSISO": P_two, "logistic + all ERA5 l2=300": P_log}
    Pc = np.clip(P_log, 0.003, 0.997); off_res = np.log(Pc / (1 - Pc)).astype("float32")   # residual design: start from the logistic
    for vn, cfg in VARIANTS.items():
        Xd = {g: Xn[g] for g in cfg["groups"]}; Xd["slow"] = Xn["slow"]
        if cfg.get("shuffle"):                                     # capacity-matched control: non-BSISO groups from another year
            for g in Xd:
                if g in ("bsiso", "slow"):
                    continue
                A = Xd[g].copy(); dpos = doy - doy.min()
                for yy in np.unique(IY):
                    src = rng.choice([u for u in np.unique(IY[trs]) if u != yy])
                    a_i, b_i = np.where(IY == yy)[0], np.where(IY == src)[0]
                    mp = {p: k for k, p in zip(b_i, dpos[b_i])}
                    A[a_i] = np.stack([Xd[g][mp.get(p, b_i[0])] for p in dpos[a_i]])
                Xd[g] = A
        gdims = {g: (Xd[g].shape[2], Xd[g].shape[1] - Xd[g].shape[1] % 5) for g in Xd if g != "slow"}
        Xd = {g: (v[:, -gdims[g][1]:] if g in gdims else v) for g, v in Xd.items()}
        Ps = []
        for s in range(SEEDS):
            P, ne, vb = train_model(cfg, Xd, Y, off_res if cfg.get("resid") else off, tr_idx, va_idx, 1000 * s + Yt, gdims)
            Ps.append(P); LOG.append({"year": Yt, "variant": vn, "seed": s, "epochs": ne, "val_bce": vb})
        preds[vn] = np.mean(Ps, 0)
        if Yt == TEST_YEARS[0]:
            print(f"{vn}: {nparams(ModelA(gdims, d=cfg['d'], layers=cfg['layers'], ff=cfg['ff'])):,} parameters", flush=True)
    te = np.where(tes)[0]
    for nm, P in preds.items():
        REC.append({"year": Yt, "model": nm, "P": P[te].astype("float32"), "Y": Y[te], "iss": DATES[ISS[te]].values})
    print(f"test {Yt} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
pd.to_pickle(REC, os.path.join(OUT, "records.pkl")); pd.DataFrame(LOG).to_csv(os.path.join(OUT, "training_log.csv"), index=False)

# %% CELL 7 -- scores: BSS per target, RPSS (weekly count 0/1-2/3+, daily 3-class), AUC, season bootstrap vs two-stage
R = pd.DataFrame(REC)
YRS = np.array(TEST_YEARS); BI = np.random.default_rng(0).integers(0, len(YRS), (2000, len(YRS)))


def yearly_scores(g):
    g = g.set_index("year").loc[YRS]
    out = {}
    for j, t in enumerate(TNAMES):
        out[t] = np.array([np.nansum((p[:, j] - y[:, j]) ** 2) for p, y in zip(g.P, g.Y)])
    def rps(cols, p, y):
        P_ = np.minimum.accumulate(p[:, cols], 1); Y_ = y[:, cols]; ok = np.isfinite(Y_).all(1)
        return np.sum(((P_ - Y_) ** 2)[ok])
    out["count_w1"] = np.array([rps([0, 1], p, y) for p, y in zip(g.P, g.Y)])
    out["count_w2"] = np.array([rps([3, 4], p, y) for p, y in zip(g.P, g.Y)])
    out["daily3"] = np.array([sum(rps([5 + k, 12 + k], p, y) for k in range(1, 8)) for p, y in zip(g.P, g.Y)])
    return out


YS = {m: yearly_scores(g) for m, g in R.groupby("model")}
ref = YS["normal chance"]
rows = []
for m in YS:
    r = {"model": m}
    for t in ["w2_ge3", "w1_ge3", "w2_lg", "w1_lg", "count_w2", "count_w1", "daily3"]:
        sk = 1 - YS[m][t][BI].sum(1) / ref[t][BI].sum(1)
        r[t] = 1 - YS[m][t].sum() / ref[t].sum(); r[t + " lo"] = np.percentile(sk, 5); r[t + " hi"] = np.percentile(sk, 95)
        r[t + " yrs>0"] = int((YS[m][t] < ref[t]).sum())
    g = R[R.model == m]; P = np.concatenate(g.P.values); Yy = np.concatenate(g.Y.values)
    for t in ["w2_ge3", "w1_ge3", "w2_lg"]:
        j = TNAMES.index(t); ok = np.isfinite(Yy[:, j]); r[t + " AUC"] = roc_auc_score(Yy[ok, j], P[ok, j])
    rows.append(r)
S = pd.DataFrame(rows); S.to_csv(os.path.join(OUT, "scores.csv"), index=False)
diffs = []
for base in ("two-stage BSISO", "logistic + all ERA5 l2=300"):
    for m in YS:
        if m in ("normal chance", base):
            continue
        for t in ["w2_ge3", "w1_ge3", "w2_lg", "w1_lg", "count_w2", "daily3"]:
            a = 1 - YS[base][t][BI].sum(1) / ref[t][BI].sum(1); b = 1 - YS[m][t][BI].sum(1) / ref[t][BI].sum(1)
            diffs.append({"vs": base, "model": m, "target": t, "diff": (b - a).mean(), "lo": np.percentile(b - a, 5),
                          "hi": np.percentile(b - a, 95), "yrs better": int((YS[m][t] < YS[base][t]).sum())})
D = pd.DataFrame(diffs); D.to_csv(os.path.join(OUT, "diff_vs_baselines.csv"), index=False)
pd.set_option("display.width", 250)
print(S[["model"] + [c for c in S.columns if c in ("w2_ge3", "w2_ge3 lo", "w2_ge3 hi", "w2_ge3 yrs>0", "w1_ge3", "w2_lg", "count_w2", "daily3", "w2_ge3 AUC")]].round(3).to_string(index=False))
print(D.round(4).to_string(index=False))
L = pd.DataFrame(LOG); print(L.groupby("variant")[["epochs", "val_bce"]].mean().round(3))
print(f"total {(time.time() - t0) / 60:.1f} min")
