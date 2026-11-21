# Poster figures for the weekly ERE head, test monsoons 2014-2024 (user's go, 16 Nov 2026). Only re-sorts forecasts that were already
# scored (txm_test_eval.py): nothing is trained or tuned here, epochs fixed from CV before the test was read.
#   13_ere_skill_bars.png     : RPSS week 1 / week 2 vs the 1981-2013 normal chance: Transformer (TXMF, ERE epoch 4), Logistic
#                               (W3BF epoch 0 = linear part), Logistic + Transformer (W3BF epoch 12 = Model P), persistence (past week repeated)
#   14_ere_forecast_groups.png: test weeks split into three equal groups by the Transformer's forecast chance (low / middle / high);
#                               bars = how often the event happened, diamond = normal chance of the same weeks, dot = mean forecast
#   15_ere_one_summer.png     : one test summer, forecasts issued every 7 days: forecast chances of the three week classes (stacked)
#                               and what happened. Summer chosen by a fixed rule: median week-1 RPSS of the Transformer over the 11 summers.
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import DATA, WORK, RUNS, KDATA, ROOT  # repo-wide paths, see paths.py
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE = os.path.dirname(os.path.abspath(__file__)); K = RUNS; PD = os.path.join(WORK, "poster")
FIG = os.path.join(ROOT, "figures")
BLUE, ORANGE, GREY, LGREY, INK, INK2, GREEN, PURPLE = "#2a78d6", "#eb6834", "#6f6e69", "#b5b4ae", "#0b0b0b", "#52514e", "#1a9e77", "#7b5cc4"
C_NONE, C_MED, C_LARGE = "#dcdbd5", "#8db8ea", "#1f4e8c"
plt.rcParams.update({"font.size": 14, "axes.edgecolor": LGREY, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ebeae5", "grid.linewidth": 0.8,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False})
sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -30, 30))); rps_w = lambda pa, pl, c: (pa - (c >= 1)) ** 2 + (pl - (c >= 2)) ** 2


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), facecolor="white"); plt.close(fig); print("saved", name)


T = pd.read_pickle(os.path.join(K, "TXMF", "out", "out", "final_TXM.pkl"))
P = pd.read_pickle(os.path.join(K, "MANUAL_W3BF", "out", "out", "final_W3N.pkl"))
W, WT, yrs = T["W"], T["W_train"], T["IY"]; Y = sorted(set(yrs)); YI = {y: np.where(yrs == y)[0] for y in Y}
assert (P["IY"] == yrs).all() and np.allclose(P["W"]["doy"], W["doy"])
ref_rows = WT["IY"] >= 1981


def wclim(doy, w):                                                         # same normal chance as txm_test_eval.py
    c_, d_ = WT["YWc"][ref_rows, w], WT["doy"][ref_rows]; pa, pl = np.zeros(len(doy)), np.zeros(len(doy))
    for i, d in enumerate(doy):
        m = np.isfinite(c_) & (np.abs(d_ - d) <= 15); pa[i] = (c_[m] >= 1).mean()
        m2 = np.isfinite(c_) & (c_ >= 1) & (np.abs(d_ - d) <= 30); pl[i] = pa[i] * (c_[m2] >= 2).mean()
    return pa, pl


REF = [wclim(W["doy"], w) for w in range(2)]


def ere(Tm, e):
    D = np.mean([s[e][..., :2] for s in Tm["deep_w_snaps"]], 0); pa = sig(Tm["W"]["LW"][..., 0] + D[..., 0])
    return pa, pa * sig(Tm["W"]["LW"][..., 1] + D[..., 1])


def rpss(pa, pl, w, idx):
    ok = idx[W["VW"][idx, w]]; c = W["YWc"][ok, w]
    return 1 - rps_w(pa[ok, w], pl[ok, w], c).sum() / rps_w(REF[w][0][ok], REF[w][1][ok], c).sum()


ALL = np.arange(len(yrs))
M = {"Transformer": ere(T, 4), "Logistic": ere(P, 0), "Logistic + Transformer": ere(P, 12)}
S = {k: [rpss(*v, w, ALL) for w in range(2)] for k, v in M.items()}
pers = pd.read_csv(os.path.join(PD, "test_ere_persistence.csv"), index_col=0).iloc[:, 0]
S["Persistence"] = [pers["ere_w1"], pers["ere_w2"]]
print(pd.DataFrame(S, index=["w1", "w2"]).T.round(3))

# ================================================================ 13. skill bars
fig, ax = plt.subplots(figsize=(8.4, 5.0))
names, cols = list(S), [GREEN, PURPLE, BLUE, GREY]; bw = 0.19; LOW = -0.30
for j, (nm, c) in enumerate(zip(names, cols)):
    for w in range(2):
        x = w + (j - 1.5) * (bw + 0.02); v = S[nm][w]; h = max(v, LOW)
        ax.bar(x, h, bw, color=c, label=nm if w == 0 else None, zorder=3)
        if v < LOW:
            ax.text(x, LOW + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=11, color="white", rotation=90, zorder=4)
        else:
            ax.text(x, v + (0.008 if v >= 0 else -0.008), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=11, color=INK2)
ax.axhline(0, color=INK2, lw=1.2, zorder=2)
ax.text(1.62, 0.006, "normal chance", ha="right", va="bottom", fontsize=11, color=INK2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Week 1 (days 1–7)", "Week 2 (days 8–14)"]); ax.set_xlim(-0.55, 1.65)
ax.set_ylim(LOW, 0.27); ax.set_ylabel("ERE skill (RPSS)"); ax.grid(axis="x", visible=False)
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.55), fontsize=12)
ax.set_title("Weekly ERE skill, test monsoons 2014–2024", loc="left", color=INK, fontsize=15)
fig.text(0.01, -0.06, "Above 0: better than the 1981–2013 normal chance for the same dates." + chr(10) + "Persistence (last week's class repeated): bars cut at the axis, value printed.",
         fontsize=10, color=INK2, ha="left")
save(fig, "13_ere_skill_bars.png")

# ================================================================ 14. forecast groups (Transformer)
pa, pl = M["Transformer"]; rng = np.random.default_rng(0); BOOT = [rng.choice(Y, len(Y)) for _ in range(1000)]
fig, axs = plt.subplots(2, 2, figsize=(10, 7.6), sharey="row")
rows_out = []
for r, (lab, F, Rf, k) in enumerate((("any ERE", pa, 0, 1), ("large ERE", pl, 1, 2))):
    for w in range(2):
        ax = axs[r, w]; ok = np.where(W["VW"][:, w])[0]; f = F[ok, w]; o = (W["YWc"][ok, w] >= k).astype(float); nc = REF[w][Rf][ok]
        q = np.quantile(f, [1 / 3, 2 / 3]); g = np.digitize(f, q); yo = yrs[ok]
        for j, gn in enumerate(("Low", "Middle", "High")):
            m = g == j; obs = o[m].mean()
            bs = [np.concatenate([o[m & (yo == y)] for y in b]).mean() for b in BOOT]
            lo, hi = np.percentile(bs, [5, 95])
            ax.bar(j, obs, 0.6, color=GREEN, zorder=3)
            ax.errorbar(j, obs, yerr=[[obs - lo], [hi - obs]], color=INK, capsize=5, lw=1.4, zorder=4)
            ax.plot(j, nc[m].mean(), "D", color=GREY, ms=10, zorder=5, markeredgecolor="white")
            ax.plot(j, f[m].mean(), "o", color="white", ms=8, zorder=6, markeredgecolor=INK, markeredgewidth=1.5)
            ax.text(j + 0.33, obs, f"{100 * obs:.0f}%", va="center", ha="left", fontsize=12, color=INK)
            rows_out.append(dict(target=lab, week=w + 1, group=gn, n_weeks=int(m.sum()), forecast_mean=f[m].mean(), observed=obs, lo90=lo, hi90=hi,
                                 normal_chance=nc[m].mean()))
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Low", "Middle", "High"]); ax.grid(axis="x", visible=False); ax.set_xlim(-0.5, 2.75)
        ax.set_ylim(0, 1.0 if r == 0 else 0.5)
        ax.set_title(f"Chance of {lab}, week {w + 1}", loc="left", fontsize=13, color=INK)
        if w == 0:
            ax.set_ylabel("How often it happened")
h = [Patch(color=GREEN, label="How often it happened (90% range: resampled summers)"),
     plt.Line2D([], [], marker="o", ls="", color="white", markeredgecolor=INK, markeredgewidth=1.5, ms=8, label="Mean forecast chance"),
     plt.Line2D([], [], marker="D", ls="", color=GREY, ms=9, label="Normal chance (1981–2013, same dates)")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12)
fig.suptitle("When the Transformer says high, does the ERE happen?  Test monsoons 2014–2024", x=0.02, ha="left", fontsize=15, color=INK)
fig.supxlabel("Transformer forecast chance (test weeks split into three equal groups)", fontsize=14, color=INK2, y=0.0)
fig.tight_layout()
save(fig, "14_ere_forecast_groups.png")
pd.DataFrame(rows_out).to_csv(os.path.join(PD, "test_ere_forecast_groups.csv"), index=False)
print(pd.DataFrame(rows_out).round(3).to_string(index=False))

# ================================================================ 15. one summer
per = {y: rpss(pa, pl, 0, YI[y]) for y in Y}; srt = sorted(Y, key=lambda y: per[y]); YS = srt[len(srt) // 2]
print("week-1 RPSS by summer:", {y: round(v, 3) for y, v in per.items()}, "-> median summer", YS)
idx = YI[YS]; drow = dict(zip(W["doy"][idx], idx)); d1 = W["doy"][idx]
S0 = [d + 1 for d in d1 if (d - d1.min()) % 7 == 0]                                     # first target days, tiling the summer
TGT = [s_ for s_ in S0 if all((s_ - 1 - 7 * w) in drow and W["VW"][drow[s_ - 1 - 7 * w], w] for w in range(2))]
start = pd.to_datetime(f"{YS}-01-01") + pd.to_timedelta(np.array(TGT) - 1, "D"); x = np.arange(len(TGT))
fig, axs = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
for w, ax in enumerate(axs):
    ii = np.array([drow[s_ - 1 - 7 * w] for s_ in TGT]); a, l = pa[ii, w], pl[ii, w]
    ax.bar(x, 1 - a, 0.7, color=C_NONE, zorder=3); ax.bar(x, a - l, 0.7, bottom=1 - a, color=C_MED, zorder=3)
    ax.bar(x, l, 0.7, bottom=1 - l, color=C_LARGE, zorder=3)
    ax.plot(x, 1 - REF[w][0][ii], "_", color=INK, ms=22, mew=2.2, zorder=4)
    if w == 0:
        c = W["YWc"][ii, w]
        ax.scatter(x, np.full(len(x), 1.09), s=150, marker="s", c=[[C_NONE, C_MED, C_LARGE][int(v)] for v in c], edgecolor=INK2, lw=0.8,
                   zorder=5, clip_on=False)
        ax.text(-0.75, 1.09, "What happened", ha="right", va="center", fontsize=11, color=INK)
    ax.set_ylim(0, 1); ax.set_xlim(-0.6, len(x) - 0.4); ax.grid(axis="x", visible=False); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel(["Forecast made" + chr(10) + "1 week ahead", "Forecast made" + chr(10) + "2 weeks ahead"][w])
axs[1].set_xticks(x); axs[1].set_xticklabels([d.strftime("%d %b") for d in start], fontsize=11)
axs[1].set_xlabel("Forecast week (first day)")
h = [Patch(color=C_NONE, label="No ERE"), Patch(color=C_MED, label="Small–medium ERE"), Patch(color=C_LARGE, label="Large ERE"),
     plt.Line2D([], [], marker="_", ls="", color=INK, ms=18, mew=2.2, label="Normal chance of no ERE")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=12)
fig.suptitle(f"Transformer weekly ERE forecasts, monsoon {YS} (the median test summer by week-1 skill)", x=0.02, ha="left", fontsize=15, color=INK)
fig.tight_layout(h_pad=1.2)
save(fig, "15_ere_one_summer.png")
