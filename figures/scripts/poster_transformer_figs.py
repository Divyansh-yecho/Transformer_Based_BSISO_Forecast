# Poster figures with the best transformer-only model (TXM, called "Transformer" on the poster), 16 Nov 2026.
#   9_rain_maps_transformer_20080613.png : weekly rain-anomaly maps, transformer vs ECMWF vs IMD (out-of-fold 2008: TXM cross-validation
#                                          round TX2, map head at its CV epoch 40, mean of the 3 seeds of the fold that held out 2008)
#   10_bsiso_correlation_transformer.png : BSISO COR by lead, test 2014-2024: Model P, ECMWF, transformer (TXMF epoch 20, 10 seeds), persistence
#   11_table_transformer.png             : test table: transformer, logistic (linear part), persistence, ECMWF (no ECMWF ERE score)
#   12_loss_curves_transformer.png       : training / held-out loss, transformer (TX2, 40 epochs) and Model P (W3B2), CV 1981-2013
import os, glob
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
HERE = os.path.dirname(os.path.abspath(__file__)); K = RUNS; PD = os.path.join(WORK, "poster")
FIG = os.path.join(ROOT, "figures"); NL = chr(10)
BLUE, ORANGE, GREY, LGREY, INK, INK2, GREEN = "#2a78d6", "#eb6834", "#6f6e69", "#b5b4ae", "#0b0b0b", "#52514e", "#1a9e77"
plt.rcParams.update({"font.size": 14, "axes.edgecolor": LGREY, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ebeae5", "grid.linewidth": 0.8,
                     "lines.linewidth": 2.2, "lines.markersize": 6.5, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), facecolor="white"); plt.close(fig); print("saved", name)


def bcor(F, O):
    return (F * O).sum((0, 2)) / np.sqrt((F ** 2).sum((0, 2)) * (O ** 2).sum((0, 2)))


def acc1(f, y):
    fc, yc = f - f.mean(), y - y.mean(); return (fc * yc).sum() / np.sqrt((fc ** 2).sum() * (yc ** 2).sum())


# ================================================================ 1. rain maps, 13 Jun 2008 (out-of-fold)
CASE = pd.Timestamp("2008-06-13")
M = np.load(os.path.join(PD, "weekly_maps_2005_2013.npz"), allow_pickle=True); dates = pd.DatetimeIndex(M["dates"].astype(str)); pts = M["pts"]
i = int(np.where(dates == CASE)[0][0])
R = [pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(K, "TX2", "out", "out", "cv_TXM_f*_s*.pkl")))]
R = [r for r in R if 2008 in set(r["IY"])]; assert len(R) == 3, len(R)
W = R[0]["W"]; rdates = pd.to_datetime([f"{y}-01-01" for y in R[0]["IY"]]) + pd.to_timedelta(np.asarray(W["doy"]) - 1, unit="D")
j = int(np.where(rdates == CASE)[0][0])
D = np.mean([r["w_snaps"][40][j] for r in R], 0)                                   # (2 weeks, 2 + 30)
ZH = W["LOCW"][j].astype("float32") + (W["GW"][j] + D[:, 2:]) @ W["V"]               # (2, 283)
ZO = W["ZW"][j].astype("float32")
assert np.allclose(ZO, M["obs"][i], atol=2e-2), "cell order / date mismatch between the CV record and the saved maps"
la, lo = np.unique(pts[:, 0]), np.unique(pts[:, 1])


def grid(v):
    g = np.full((len(la), len(lo)), np.nan); g[np.searchsorted(la, pts[:, 0]), np.searchsorted(lo, pts[:, 1])] = v; return g


cmap = LinearSegmentedColormap.from_list("drywet", ["#8c510a", "#d8b365", "#f4f3ee", "#92c5de", "#2166ac", "#0b3c78"]); V = 1.5; VF = 0.75
# Forecast columns (transformer and ECMWF alike) use a colour scale half as wide as the observed column: averaged forecasts have weaker
# anomalies (spread over all 2005-2013 forecasts: ours 0.29, ECMWF 0.44, observed 0.65). Stated on the figure; pattern corr. is unaffected.
fig, axs = plt.subplots(2, 3, figsize=(11.5, 9.0), constrained_layout=True)
cols = [("Transformer forecast", ZH), ("ECMWF forecast", M["ecmwf"][i]), ("What happened (IMD)", M["obs"][i])]
for w in range(2):
    d0, d1 = CASE + pd.Timedelta(days=1 + 7 * w), CASE + pd.Timedelta(days=7 + 7 * w)
    for c, (nm, X) in enumerate(cols):
        Vc = VF if c < 2 else V                                              # forecasts: half-width colour scale (they are averages, so weaker)
        ax = axs[w, c]; im = ax.pcolormesh(lo, la, grid(X[w]), cmap=cmap, vmin=-Vc, vmax=Vc, shading="nearest")
        if c < 2:
            im_f = im
        else:
            im_o = im
        ax.set_aspect("equal"); ax.set_xticks([70, 80, 90]); ax.set_yticks([10, 20, 30]); ax.grid(False)
        ax.set_xticklabels(["70°E", "80°E", "90°E"], fontsize=10); ax.set_yticklabels(["10°N", "20°N", "30°N"], fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
        t = nm + (NL + f"pattern corr. {acc1(X[w], M['obs'][i][w]):.2f}" if c < 2 else NL + f"{d0:%d %b}–{d1:%d %b %Y}")
        ax.set_title(t, fontsize=12, color=INK)
        if c == 0:
            ax.set_ylabel(f"Week {w + 1}" + NL + f"(days {1 + 7 * w}–{7 + 7 * w})", fontsize=13, color=INK)
cb1 = fig.colorbar(im_f, ax=axs[:, :2], orientation="horizontal", shrink=0.55, pad=0.03, extend="both", aspect=30)
cb1.set_label("Forecast rain anomaly (standardised, scale ±0.75)", fontsize=11)
cb2 = fig.colorbar(im_o, ax=axs[:, 2], orientation="horizontal", shrink=0.95, pad=0.03, extend="both", aspect=15)
cb2.set_label("Observed rain anomaly (scale ±1.5)", fontsize=11)
fig.text(0.02, -0.035, "Brown = drier than normal, blue = wetter. Forecast maps use a colour scale half as wide as the observed map: a forecast is an "
         "average over many possible outcomes," + NL + "so its anomalies are weaker. Pattern correlation does not depend on the colour scale.",
         fontsize=10, color=INK2, va="top")
fig.suptitle(f"Weekly rain-anomaly forecast issued {CASE:%d %b %Y} (out-of-fold example)", x=0.02, ha="left", fontsize=14, color=INK)
save(fig, f"9_rain_maps_transformer_{CASE:%Y%m%d}.png")
print("pattern corr transformer w1/w2", [round(acc1(ZH[w], M["obs"][i][w]), 3) for w in range(2)])

# ================================================================ 2. BSISO correlation by lead, test 2014-2024
dcols = [f"d{k}" for k in range(1, 15)]
B = pd.read_csv(os.path.join(PD, "test_W3BF_bsiso.csv"), index_col=0)
T = pd.read_pickle(os.path.join(K, "TXMF", "out", "out", "final_TXM.pkl"))
tx = np.r_[1, bcor(np.mean([s[20][..., 2:4] for s in T["deep_snaps"]], 0), T["AUX"])]
mp, ec, pe = (np.r_[1, B.loc["Model P (new)", dcols].values], np.r_[1, B.loc["ECMWF", dcols].values], np.r_[1, B.loc["persistence", dcols].values])
L = np.arange(0, 15)
fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.axhline(0.5, color=LGREY, lw=1.3, ls=(0, (4, 3)), zorder=1)
for y, c, nm in ((ec, ORANGE, "ECMWF"), (mp, BLUE, "Logistic + Transformer"), (tx, GREEN, "Transformer"), (pe, GREY, "Persistence")):
    ax.plot(L, y, color=c, marker="o", label=nm, zorder=3, markeredgecolor="white", markeredgewidth=1.2)
ax.set_xlim(-0.3, 14.3); ax.set_ylim(-0.1, 1.04); ax.set_xticks(L); ax.set_xlabel("Lead time (days)"); ax.set_ylabel("BSISO correlation")
ax.legend(loc="lower left"); ax.text(14.2, 0.515, "COR = 0.5", ha="right", va="bottom", fontsize=11, color=INK2)
ax.set_title("BSISO forecast skill, test monsoons 2014–2024", loc="left", color=INK, fontsize=15)
save(fig, "10_bsiso_correlation_transformer.png")
print("transformer COR by lead", np.round(tx, 3))

# ================================================================ 3. table: transformer, logistic, persistence, ECMWF
TW = pd.read_csv(os.path.join(PD, "test_W3BF_weekly.csv"), index_col=0); TX = pd.read_csv(os.path.join(PD, "test_transformer_only_TXM.csv")).iloc[0]
f3, f2, f1 = (lambda v: f"{v:.3f}"), (lambda v: f"{v:.2f}"), (lambda v: f"{v:.1f}")
brow = lambda r: [f3(B.loc[r, "d1-7"]), f3(B.loc[r, "d8-14"]), f1(B.loc[r, "horizon"])]
rows = [["Transformer", f3(TX["BSISO COR d1-7"]), f3(TX["BSISO COR d8-14"]), f1(TX["BSISO useful days"]), f3(TX["ERE RPSS w1"]), f3(TX["ERE RPSS w2"]),
         f2(TX["map corr w1"]), f2(TX["map corr w2"])],
        ["Logistic (linear model)"] + brow("linear part") + [f3(TW.loc["ere_w1", "model_P"]), f3(TW.loc["ere_w2", "model_P"]), f2(TW.loc["acc_w1", "linear_part"]),
                                                              f2(TW.loc["acc_w2", "linear_part"])],
        ["Persistence"] + brow("persistence") + ["–", "–", f2(TW.loc["acc_w1", "last_week_persists"]), f2(TW.loc["acc_w2", "last_week_persists"])],
        ["ECMWF", f3(B.loc["ECMWF", "d1-7"]), f3(B.loc["ECMWF", "d8-14"]), "> 14", "–", "–", f2(TW.loc["acc_w1", "ecmwf"]), f2(TW.loc["acc_w2", "ecmwf"])]]
hdr = ["", "BSISO COR" + NL + "days 1–7", "BSISO COR" + NL + "days 8–14", "BSISO useful" + NL + "days", "ERE RPSS" + NL + "week 1", "ERE RPSS" + NL + "week 2",
       "Rain map corr." + NL + "week 1", "Rain map corr." + NL + "week 2"]
fig, ax = plt.subplots(figsize=(14, 3.0)); ax.axis("off"); ax.set_position([0.01, 0.24, 0.98, 0.68])
tb = ax.table(cellText=rows, colLabels=hdr, loc="upper center", cellLoc="center", colLoc="center", colWidths=[0.22] + [0.111] * 7)
tb.auto_set_font_size(False); tb.set_fontsize(13); tb.scale(1, 2.4)
for (r_, c_), cell in tb.get_celld().items():
    cell.set_edgecolor("#d9d8d2"); cell.set_linewidth(0.8)
    if r_ == 0:
        cell.set_facecolor("#f1f0ec"); cell.set_text_props(color=INK, fontweight="bold", fontsize=11.5); cell.set_height(cell.get_height() * 1.35)
    if c_ == 0:
        cell.set_text_props(ha="left"); cell._loc = "left"; cell.PAD = 0.05
    if r_ == 1:
        cell.set_facecolor("#f4f8fd"); cell.set_text_props(fontweight="bold")
ax.set_title("Test monsoons 2014–2024 (11 summers)", loc="left", color=INK, fontsize=13.5, pad=4)
fig.text(0.01, 0.0, "Transformer: no linear part; epochs chosen per head in 1981–2013 cross-validation. Logistic: the linear models alone." + NL +
         "ERE: RPSS of the weekly class against the 1981–2013 normal chance. Rain: pattern correlation of weekly-mean rain-anomaly maps. "
         "BSISO useful days = lead where the correlation falls to 0.5." + NL + "Persistence: BSISO state on the issue day kept; last week's rain map repeated.",
         fontsize=9.8, color=INK2, va="bottom")
save(fig, "11_table_transformer.png")

# ================================================================ 4. loss curves (CV 1981-2013)
bce = lambda z, y: np.logaddexp(0, z) - y * z


def curves(rnd, cfg):
    Rr = [pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(K, rnd, "out", "out", f"cv_{cfg}_f*_s*.pkl")))]
    sd = np.concatenate([r["AUX"] for r in {r["fold"]: r for r in Rr}.values()]).reshape(-1, 2).std(0)
    eps = sorted(Rr[0]["snaps"]); out = {k: [] for k in ("bs", "ere", "map")}
    for e in eps:
        b, er, mpl = [], [], []
        for r in Rr:
            Wr = r["W"]; S = r["snaps"][e][..., 2:4]; Dd = r["w_snaps"][e]
            b.append(np.nanmean(((S - r["AUX"]) / sd) ** 2))
            m = np.isfinite(Wr["YWa"]); l1 = bce(Wr["LW"][..., 0] + Dd[..., 0], np.nan_to_num(Wr["YWa"]))[m].mean()
            m2 = np.isfinite(Wr["YWl"]); l2 = bce(Wr["LW"][..., 1] + Dd[..., 1], np.nan_to_num(Wr["YWl"]))[m2].mean(); er.append(l1 + l2)
            Z = Wr["LOCW"].astype("float32") + np.einsum("nwp,pc->nwc", Wr["GW"] + Dd[..., 2:], Wr["V"])
            mpl.append(((Z - Wr["ZW"].astype("float32")) ** 2)[Wr["VW"]].mean())
        out["bs"].append(np.mean(b)); out["ere"].append(np.mean(er)); out["map"].append(np.mean(mpl))
    tr = np.mean([r["hist"][:, 2] for r in Rr], 0)
    return np.array(eps), {k: np.array(v) for k, v in out.items()}, tr


P = curves("W3B2", "W3N"); X = curves("TX2", "TXM")
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.1)); plt.subplots_adjust(wspace=0.28, left=0.06, right=0.99, top=0.80, bottom=0.19)
titles = ["BSISO head (daily PC1/PC2)", "Weekly ERE head", "Weekly rain-map head"]; ylabs = ["normalised MSE", "cross-entropy", "MSE of map (z units)"]
CHO = {"bs": 20, "ere": 4, "map": 40}
for k_, (k, ax) in enumerate(zip(["bs", "ere", "map"], axs)):
    for (eps, o, tr), c, nm in ((X, GREEN, "Transformer"), (P, BLUE, "Model P")):
        ax.plot(eps, o[k], color=c, lw=2, marker="o", ms=3.5, label=f"{nm}, held-out", zorder=3)
        if k_ == 0:
            ax.plot(np.arange(1, len(tr) + 1), tr, color=c, lw=2, ls="--", label=f"{nm}, training", zorder=3)
    e_ = CHO[k]; y_ = X[1][k][list(X[0]).index(e_)]
    ax.plot([e_], [y_], marker="o", ms=11, mfc="none", mec=INK, mew=1.6, zorder=4)
    ax.annotate(f"epoch used: {e_}", (e_, y_), xytext=(14, 14) if k == "ere" else (0, 12), textcoords="offset points",
                ha="left" if k == "ere" else "center", fontsize=9.5, color=INK2)
    ax.set_title(titles[k_], loc="left", fontsize=12, color=INK); ax.set_xlabel("epoch", color=INK2); ax.set_ylabel(ylabs[k_], color=INK2)
    ax.set_xticks(range(0, 41, 5))
axs[0].legend(frameon=False, fontsize=9.5, loc="upper right")
fig.suptitle("Training and held-out loss, cross-validation 1981–2013 (mean over 5 folds and all seeds)", x=0.06, ha="left", fontsize=13, color=INK)
fig.text(0.06, 0.005, "Training loss was logged for the BSISO head only; the weekly heads show the held-out loss. "
         "Circles: the epoch each head of the transformer uses (chosen in cross-validation).", fontsize=9.2, color=INK2)
save(fig, "12_loss_curves_transformer.png")
