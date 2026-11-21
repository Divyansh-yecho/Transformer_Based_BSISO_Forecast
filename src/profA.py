# Prof model A (design and rules: Markdown_files/24_Prof_model_A_design_and_protocol_Nov10.md).
#   linear link (logistic for P(any ERE), logistic for P(large | any), ridge for BSISO PC1/PC2; one per lead day 1-14) fitted
#   first and frozen; a transformer over BSISO / IMD-rain / field-map tokens learns a zero-initialised residual; post-hoc
#   shrinkage nu chosen in cross-validation. Training data 1981-2013 only; test 2014-2024 only in MODE=final.
# Usage (env vars):
#   MODE=cv     CFG=A1 FOLD=0..4 SEED=0      -> one fold of 5-block cross-validation (writes out/cv_<CFG>_f<FOLD>_s<SEED>.pkl)
#   MODE=final  CFG=A1 SEEDS=10 EPOCHS=..    -> fit on 1981-2013, predict 2014-2024 (+ control)  (out/final_<CFG>.pkl)
#   MODE=lin    -> linear link only, all folds and the l2 grid (out/lin_cv.pkl)
#   FAST=1 -> tiny smoke test (2 epochs, few samples)
import os, sys, time, json, glob, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MA_DATA", os.pathsep.join([os.path.join(HERE, "..", "intermediate", d) for d in ("modelA", "modelA_era5")]))
os.environ.setdefault("PHASE", "dev")
PIN = os.environ.get("PROFA_IN", os.path.join(HERE, "..", "intermediate", "profA"))          # maps.npz, maps_vo.npz
POLR = os.environ.get("PROFA_OLR", os.path.join(HERE, "..", "..", "..", "data", "olr_merged", "olr_daily_1979_2024_merged.nc"))
src = open(os.environ.get("MODELA_SRC", os.path.join(HERE, "kaggle_modelA.py")), encoding="utf-8").read()
exec(src.split("# %% CELL 6")[0])                     # DATES, classes(), ISS, IY, X_BS, AUX, X_RN, X_SL, fit_w, JJAS ...
import time
import numpy as np, pandas as pd, xarray as xr, torch, torch.nn as nn, torch.nn.functional as Fnn
t0 = time.time()

torch.set_num_threads(int(os.environ.get("THREADS", "2")))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODE = os.environ.get("MODE", "cv"); CFGN = os.environ.get("CFG", "A1"); FAST = os.environ.get("FAST") == "1"
OUTD = os.environ.get("PROFA_OUT", os.path.join(HERE, "..", "intermediate", "profA", "out")); os.makedirs(OUTD, exist_ok=True)
TRAIN_Y = np.arange(1981, 2014); TEST_Y = np.arange(2014, 2025)
if os.environ.get("TEST_YEARS"):                                  # independent test (file 32): TEST_YEARS=2025 (training stays 1981-2013)
    TEST_Y = np.array([int(y) for y in os.environ["TEST_YEARS"].split(",")])
if os.environ.get("SMOKE_FINAL") == "1":                         # code test of MODE=final without touching 2014-2024
    TRAIN_Y = np.arange(1981, 2011); TEST_Y = np.arange(2011, 2014)
BLOCKS = [range(1981, 1988), range(1988, 1994), range(1994, 2001), range(2001, 2008), range(2008, 2014)]
NL = 14
CFGS = {
    "A1": dict(enc="eof", d=48, layers=3, lam=0.5),
    "A2": dict(enc="cnn", d=48, layers=3, lam=0.5),
    "A3": dict(enc="eof", d=32, layers=2, lam=0.5),
    "A4": dict(enc="eof", d=48, layers=3, lam=0.0),
    "A5": dict(enc="eof", d=48, layers=3, lam=0.5, lr=2.5e-4, drop=0.2),
    # round 2 (PA2): PA1 peaked at the first checkpoint (epoch 5 of a 60-epoch schedule) -> short schedules, scored every epoch
    "B1": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1),
    "B2": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, lr=2.5e-4, drop=0.2, gdrop=0.3),
    "B3": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, mu=0.1, wd=0.1),
    # PA2 additions (file 26): against "year-ID" memorisation and noise fitting
    #   hp: deep field inputs minus their own trailing mean over T-89..T-30 (same year; causal) -> intraseasonal only
    #   film_enso=False: Nino3.4 / DMI removed from the deep path (they stay in the linear link)
    #   noise: Gaussian noise (SD units) on field EOFs and rain inputs (half on BSISO) while training
    #   jitter: field windows shifted back by 0..jitter days while training (backward only: no future data)
    #   swa: weights averaged over epochs >= swa (snapshots come from the averaged model)
    #   mreg: auxiliary ERE (any) heads for 5 other Indian regions, weight mreg (scored: central India only)
    "B4": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, hp=True, film_enso=False),
    "B5": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, hp=True, film_enso=False, noise=0.2, jitter=2),
    "B6": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, hp=True, film_enso=False, noise=0.2, jitter=2, swa=2),
    "B7": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, hp=True, film_enso=False, noise=0.2, jitter=2, mreg=0.5),
    # PA3 (file 27): the BSISO chain. C1 = B1 on the chain linear link; C2 = + deep chain on Prof A's own BSISO forecast;
    # C3 = C2 with a stronger BSISO loss (lam 1.0); C4 = C2 with B3's stronger penalty / weight decay
    "C1": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1),
    "C2": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, chain_deep=True),
    "C3": dict(enc="eof", d=48, layers=3, lam=1.0, epochs=12, eval_every=1, chain_deep=True),
    "C4": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, chain_deep=True, mu=0.1, wd=0.1),
    # Prof model B v1 (file 28): C1 + ECMWF S2S reforecast RAIN (no ECMWF BSISO / winds). Frozen C1 settings, no new tuning.
    "PB1": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, ec=True),
    # "Prof B1" (user's name, file 29): PB1 + ECMWF u850 / u200 forecasts (12 + 12 fixed-basis PCs per lead); still no ECMWF BSISO
    "PBW": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, ec=True, ecw=True),
    # Prof A improvement tests (file 30). D0 = C1 on the chosen improved linear link (env LINVAR);
    #   map: rain-map output head per lead (283 Indian 1-deg cells: P(rain > cell p90) + standardised log rain), loss weight = map
    #   map2ere: the predicted central-India cells feed the ERE logits (zero-init linear)
    #   ana: analog token per lead (rain maps that followed the K nearest past states in other training years + analog ERE rates)
    "D0": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1),
    "D1": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, map=1.0),
    "D2": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, map=1.0, map2ere=True),
    "D3": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, map=3.0, map2ere=True),
    "D4": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, ana=True),
    "D5": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, map=1.0, map2ere=True, ana=True),
    # PI2 (file 30): try 2 of the rain map. rain_in: the IMD 1-deg rain MAP (log1p) as an input field (daily tokens T-4..T);
    #   map_mlp: MLP decoder for the map head; pretrain: epochs of dense-target training (maps + BSISO) before the ERE training
    "E0": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, rain_in=True),
    "E1": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, rain_in=True, map=1.0, map2ere=True, map_mlp=True),
    "E2": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, rain_in=True, map=1.0, map2ere=True, map_mlp=True, pretrain=8),
    "E3": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, rain_in=True, map=3.0, map2ere=True, map_mlp=True, pretrain=8),
    "E4": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, map=1.0, map2ere=True, map_mlp=True, pretrain=8),
    # PI3 (file 30): all on the LR link (env RAINLIN=10, CHAIN_PEN=1). F0 = D0; F1 = D3 (map head w3 + map2ere); F2 = D4 (analog);
    #   F3 = E0 (rain-map input); F4 = D0 with half the learning rate (the residual over LR is smaller)
    "F0": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1),
    "F1": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, map=3.0, map2ere=True),
    "F2": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, ana=True),
    "F3": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, rain_in=True),
    "F4": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, lr=2.5e-4),
    # PI4 (file 30): the same deep variants on the LS link (env RAINLIN=10 SYNLIN=10 RAIN_LEADS=10 CHAIN_PEN=1); G5 = smaller transformer
    "G0": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1),
    "G1": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, map=3.0, map2ere=True),
    "G2": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, ana=True),
    "G3": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, rain_in=True),
    "G4": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, lr=2.5e-4),
    "G5": dict(enc="eof", d=32, layers=2, lam=0.5, epochs=12, eval_every=1),
    # transformer ONLY (no linear link; user request 11 Nov): logits start at normal chance, BSISO at 0; the transformer learns all.
    #   30-epoch schedule (it has to learn everything), no logit penalty. H1 also gets the rain-map input (all the map info of LS).
    "H0": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0),
    "H1": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True),
    "H2": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, map=1.0),
    # bs_daily (file 31 fix): BSISO as 11 five-day patches (T-59..T-5) + 5 DAILY tokens (T-4..T), so the exact current state is seen
    "H3": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, bs_daily=True),
    "H4": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True),
    "G6": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, bs_daily=True),
    # H5: ONE end-to-end network (professor's "day-T inputs added directly to every lead"): H4 + a learned per-lead linear skip
    #   from the day-T inputs (compact features, map EOFs for leads 1-10, BSISO on T+k-5) to the ERE logits, and from the BSISO
    #   history (+ OLR EOFs) to the BSISO head, trained jointly with the transformer (own lr 5e-3, L2 penalty skip_l2)
    "H5": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1),
    "H6": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.3,
               skip_lr=1e-3, lr=2.5e-4),
    "B8": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=12, eval_every=1, hp=True, film_enso=False, noise=0.2, jitter=2, swa=2, mreg=0.5),
    # file 33 (user, 12 Nov): BSISO head trained on ALL months. olr_y: the OLR tokens use EOFs of year-round anomalies; skb_bo: the
    #   BSISO skip uses only the ridge's inputs (daily PC1-4 T-29..T + OLR EOFs at lags 0..25 d), which exist on every day of the year.
    #   H8 = H5 + olr_y + skb_bo (control, JJAS samples only). H7 = H8 + the off-season days of the training years as extra samples
    #   (BSISO + OLR tokens real, rain / ERA5 tokens masked, BSISO loss only; allyr_b of them per step next to the 64 JJAS samples)
    "H8": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
               olr_y=True, skb_bo=True),
    "H7": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
               olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0),
    # file 36 (13 Nov): warm start. The in-network linear skips start at their closed-form solution (logistic per lead with the
    #   normal-chance offset for ERE, Prof A's penalties; ridge per lead on all training days for BSISO); the transformer starts at
    #   zero, so epoch 0 = the linear model. H7W: skips frozen (only the transformer learns). H7WF: skips fine-tuned at a small
    #   learning rate with the L2 penalty pulling them back to the closed-form start.
    "H7W": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
                olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True),
    "H7WF": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
                 olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_lr=1e-4),
    # file 37 (13 Nov): Model Y = H7W + an in-network rain-map head. Target: 30 EOF coefficients of the standardised log-rain anomaly
    #   (calendar-day climatology of the training years) minus its frozen closed-form linear part (per-cell persistence ridge +
    #   large-scale ridge, precomputed per fold by modelY_prep.py); the transformer learns the correction from its lead states.
    #   Y0: the map head reads the transformer but its loss does not train it (stop-gradient). Y1 / Y2: joint, map weight 1 / 3.
    "Y0": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
               olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, ymap=True, ymap_sg=True, ymapw=1.0),
    "Y1": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
               olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, ymap=True, ymapw=1.0),
    "Y2": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
               olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, ymap=True, ymapw=3.0),
    # Y3 / Y4 (night 13-14 Nov): the frozen linear map part = Model X's ridge (it also uses H7W's own BSISO / ERE forecasts, precomputed
    #   out-of-fold), so epoch 0 = Model X's maps; the transformer must add on top. Y4: gentler learning rate (2.5e-4).
    "Y3": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
               olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, ymap=True, ymapw=1.0, ymap_tag="X"),
    "Y4": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
               olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, ymap=True, ymapw=1.0, ymap_tag="X", lr=2.5e-4),
    # file 39 (13 Nov): the extended data (1959-2024 inputs; BSISO rebuilt from NCEP before 1981, OLR proxy before 1979; run with
    #   NO_VO=1 because vorticity does not exist before 1981). A0 = H7W trained on 1981-2013 only (the reference without vorticity);
    #   B = H7W trained on 1960-2013 (the extra years added directly); C = the transformer pre-trained on 1960-1980 (H7W set-up, pt_epochs
    #   epochs), then H7W fine-tuned on each fold's 1981-2013 training years starting from those transformer weights (the linear skips
    #   are refitted in closed form on the fold's years, as always). CV blocks and test years are unchanged.
    "A0": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
               olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, train_start=1981),
    "B": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
              olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, train_start=1960),
    "C": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
              olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, train_start=1981, pretrain_years=(1960, 1980), pt_epochs=3),
    # file 40 (13 Nov): H7W+ = B (1960-2013, no vorticity) + real ERA5 inputs (u850, u200, MSLP) for the off-season BSISO samples
    #   (era_y: year-round anomalies and EOFs from all training days, as olr_y does for OLR; used by monsoon and off-season samples
    #   alike). Widths 48 (main), 32 and 64 to see which direction the extra data favours.
    "HWP": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
                olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, train_start=1960, era_y=True),
    "HWP32": dict(enc="eof", d=32, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
                  olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, train_start=1960, era_y=True),
    "HWP64": dict(enc="eof", d=64, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
                  olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, train_start=1960, era_y=True),
    # file 41: input importance of the transformer arm (H7W+ width 32): at epochs 2-4 the held-out rows are re-predicted with one
    #   transformer input group removed (switched off: rain band, IMD rain map, OLR, u850, u200, MSLP, slow FiLM background;
    #   shuffled across rows: the BSISO history, which group dropout never removed). The linear link is untouched.
    "HWP32I": dict(enc="eof", d=32, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
                   olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, train_start=1960, era_y=True, imp_eps=[2, 3, 4]),
    # file 42 (13 Nov): Maeda's extra fields (precipitable water, SST, 850 hPa geopotential; profA_ext/maps_new.npz) as transformer inputs.
    #   All on H7W+ width 32 with the ERE definition fixed to the 1981-2013 thresholds (ere_thr_1981). New fields: per-sample trailing
    #   background (mean of T-89..T-30) removed (hp_fields) so trends and the 1979 reanalysis step do not reach the transformer.
    #   HX0 = our inputs (baseline); HXC = ours + the 3 new fields (= all of Maeda's six + ours); HXCP = HXC without OLR and u200 in the
    #   transformer; HXM = Maeda's six + BSISO history (no rain band, IMD rain map, MSLP). The linear links are the same in all four.
    "HX0": dict(enc="eof", d=32, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
                olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, train_start=1960, era_y=True, ere_thr_1981=True),
    "HXC": dict(enc="eof", d=32, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
                olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, train_start=1960, era_y=True, ere_thr_1981=True,
                new_fields=["tcwv", "sst", "z850"], hp_fields=["tcwv", "sst", "z850"]),
    "HXCP": dict(enc="eof", d=32, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
                olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, train_start=1960, era_y=True, ere_thr_1981=True,
                 new_fields=["tcwv", "sst", "z850"], hp_fields=["tcwv", "sst", "z850"], drop_fields=["olr", "u200"]),
    "HXM": dict(enc="eof", d=32, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1,
                olr_y=True, skb_bo=True, allyr=True, allyr_b=160, allyr_w=1.0, warm=True, skip_freeze=True, train_start=1960, era_y=True, ere_thr_1981=True,
                new_fields=["tcwv", "sst", "z850"], hp_fields=["tcwv", "sst", "z850"], drop_fields=["rain1", "mslp"], no_rainband=True),
    # learning curve (file 33): H5 trained on a random 50% / 75% of each fold's training years (same held-out blocks)
    "H5y50": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1, yfrac=0.5),
    "H5y75": dict(enc="eof", d=48, layers=3, lam=0.5, epochs=30, eval_every=1, nolin=True, mu=0.0, rain_in=True, bs_daily=True, skip=True, skip_l2=0.1, yfrac=0.75),
}
# file 43 (14 Nov): the transformer's own gain in Model P (HXCP). Inputs, targets and the frozen linear part are HXCP's in every
#   config below; only the transformer's training and output heads change.
#   sched="flat": learning rate rises over the first epoch, then stays constant (no one-cycle peak).  ema: the snapshots come from
#   an exponential moving average of the weights (decay per step); the un-averaged model's snapshots are kept as snaps_raw.
#   pen_b / pen_e: penalty on the mean square size of the transformer's own BSISO (PC / sd) / ERE-logit correction.
#   sam: sharpness-aware minimisation, radius rho (two gradient evaluations per step).   ere_sg: the ERE loss does not train the
#   transformer body (the ERE head reads the lead states without gradient).
#   mult=R: state-dependent propagator head: BSISO correction += C(lead state) (2 x R) @ z, z = the BSISO skip inputs projected
#   on R fixed directions (mult_basis: where a JJAS-only ridge differs from the frozen all-months ridge); C is zero-initialised.
#   dense=w: extra targets, future OLR and u850 EOF coefficients (10 + 10, year-round anomalies) on T+1..T+14, loss weight w.
_T = dict(CFGS["HXCP"], epochs=12, sched="flat", lr=1.5e-4, ema=0.99)
CFGS.update({
    "PTA": dict(_T),
    "PT1": dict(_T, pen_b=0.25, pen_e=0.3),
    "PT1P": dict(_T, pen_b=1.0, pen_e=0.3),
    "PT1S": dict(_T, pen_b=0.25, pen_e=0.3, sam=0.05),
    "PT1R": dict(_T, pen_b=0.25, pen_e=0.3, noise=0.2, jitter=2, drop=0.2),
    "PT1X": dict(_T, pen_b=0.25, pen_e=0.3, ere_sg=True),
    "PTM": dict(_T, pen_b=0.25, pen_e=0.3, mult=8),
    "PTD": dict(_T, pen_b=0.25, pen_e=0.3, dense=0.5),
})
# round PT2 (file 43): on PT1R (the most stable PT1 config). The dense-field head was still improving at epoch 12 -> 30 epochs, two
#   weights, and with the propagator head. PT1S (sam 0.05) learned nothing: the radius was larger than the near-zero head weights ->
#   sam_ad: adaptive sharpness-aware step (perturbation scaled by each weight's size, Kwon et al. 2021).
_R = dict(_T, pen_b=0.25, pen_e=0.3, noise=0.2, jitter=2, drop=0.2)
CFGS.update({
    "PD1": dict(_R, dense=0.5, epochs=30),
    "PD2": dict(_R, dense=2.0, epochs=30),
    "PD3": dict(_R, dense=0.5, mult=8, epochs=30),
    "PS1": dict(_R, sam=0.5, sam_ad=True),
})
# round W3 (user, 14 Nov): Model P with three heads, all on the PT1R training: BSISO PC1/PC2 (daily leads 1-14, as before), WEEKLY ERE
#   (week 1 = T+1..T+7, week 2 = T+8..T+14; 3 classes: no ERE in the week / small or medium only / at least one large day, modelled as
#   P(any in week), P(large | any)) and the WEEKLY rain-anomaly map (mean z over the week, 283 IMD 1-deg cells). Each head = a frozen
#   closed-form linear part fitted on the fold's training years (weekly_fit) + a zero-initialised transformer correction read from the
#   mean lead state of that week (head_w: 2 ERE logits + 30 map EOF coefficients per week). One loss: daily BSISO (lam) + daily ERE
#   (ere_d) + weekly ERE BCE (wk_e) + weekly map MSE in EOF space / its training variance (wk_m) + size penalties (pen_w, pen_m).
#   W3 keeps the daily ERE head of Model P; W3N switches its loss off (exactly three heads).
CFGS.update({
    "W3": dict(_R, wk=True, wk_e=1.0, wk_m=0.5, pen_w=0.3, pen_m=0.25),
    "W3N": dict(_R, wk=True, wk_e=1.0, wk_m=0.5, pen_w=0.3, pen_m=0.25, ere_d=0.0),
})
# Night run 15-16 Nov (rain head): R1 = W3N with 4x the rain-map loss weight and a 5x smaller penalty on the map correction;
#   R2 = R1 without the weekly ERE loss (rain map + BSISO only).
CFGS.update({
    "W3R1": dict(CFGS["W3N"], wk_m=2.0, pen_m=0.05),
    "W3R2": dict(CFGS["W3N"], wk_m=2.0, pen_m=0.05, wk_e=0.0),
})
# Night run, closed-form finding (rain_exp.py): all 20 issue-day EOFs of u200, z850, tcwv, OLR, u850, MSLP in the linear rain head's
#   predictors raise CV week-1/2 ACC by +0.021/+0.015. G1 = W3N + those predictors; G1T = G1 with OLR and u200 back as transformer inputs.
_GF = ["u200", "z850", "tcwv", "olr", "u850", "mslp"]
CFGS.update({
    "W3G1": dict(CFGS["W3N"], wk_g_full=_GF),
    "W3G1T": dict(CFGS["W3N"], wk_g_full=_GF, drop_fields=[]),
    "W3G2": dict(CFGS["W3N"], wk_g_full=_GF, wk_g_tend=True),                              # + 5-day EOF tendencies (closed form: +0.002 week 2)
})
# Transformer only (user, 15 Nov): Model P (W3N) inputs, architecture, three heads and PT1R regularisation, but NO linear part anywhere:
#   no BSISO/ERE skip, weekly ERE starts from the weekly normal chance (OFFW), weekly map from 0 (no damped persistence, no ridge;
#   map EOFs of the weekly z itself). Size penalties off (they would shrink the whole forecast). 20 epochs, scored every epoch;
#   W3T: lr 5e-4 (from-scratch rate of the H runs), W3TS: lr 1.5e-4 (Model P's rate). Variant and epoch chosen per head in CV.
_TO = dict(CFGS["W3N"], skip=False, warm=False, skip_freeze=False, pen_b=0.0, pen_e=0.0, pen_w=0.0, pen_m=0.0, wk_nolin=True, epochs=20)
CFGS.update({
    "W3T": dict(_TO, lr=5e-4),
    "W3TS": dict(_TO, lr=1.5e-4),
})
# Round TX (user, 15 Nov): can the transformer alone be pushed? All on W3T (transformer only, lr 5e-4), 40 epochs scored every epoch,
#   variant / epoch chosen per head in CV. TXA: all input fields back (W3T had OLR and u200 removed, as Model P's transformer).
#   TXL: TXA + one token per input vector of the linear parts (BSISO-skip inputs, compact + rain/MSLP EOFs, the weekly ridge's
#   predictors, the rain persistence projected on the weekly target EOFs; no linear OUTPUT is given). TXD: TXA + future OLR / u850
#   EOFs (T+1..T+14) as extra targets on JJAS and off-season days. TXP: TXA, first 10 epochs pretraining on BSISO + future fields
#   only (JJAS and off-season days), then the 40 normal epochs. TXM: TXL + a DAILY rain-map head (283 cells, leads 1-14: above the
#   cell's 90th percentile, standardised log rain), scored against a per-cell linear baseline on the same predictors (map_lin).
_TA = dict(_TO, drop_fields=[], epochs=40, lr=5e-4)
CFGS.update({
    "TXA": dict(_TA),
    "TXL": dict(_TA, lin_tok=True),
    "TXD": dict(_TA, dense=0.5),
    "TXP": dict(_TA, pre_ep=10, pre_dense=1.0),
    "TXM": dict(_TA, lin_tok=True, map=1.0),
})
# Round TY (user, 16 Nov): is the transformer's map gap in its architecture? On TXM (best transformer only), 60 epochs.
#   TYE: no squeeze of the per-day field encoding (20 EOFs -> 20 numbers instead of 12). TYW: wider output heads (daily map and
#   weekly heads read the lead state through a 128-unit MLP). TYEW: both. TYEW48: both + model width 48 instead of 32.
_TM = dict(CFGS["TXM"], epochs=60)
CFGS.update({
    "TYE": dict(_TM, enc_e=20),
    "TYW": dict(_TM, map_mlp=True, wk_mlp=True),
    "TYEW": dict(_TM, enc_e=20, map_mlp=True, wk_mlp=True),
    "TYEW48": dict(_TM, enc_e=20, map_mlp=True, wk_mlp=True, d=48),
})
# Round TC (user, 16 Nov): which weekly ERE OUTPUT works best? All = TXM (40 epochs) + extra weekly heads (zero-initialised, from
#   the normal chance). TXC: count of ERE days in the week as a chain P(n>=k | n>=k-1), k = 2..6 (any binning derivable).
#   TXC3: the same with only the user's bins 0-2 / 3-4 / 5+ (thresholds 3, 5). TXS: P(any small/medium day), independent of
#   large days. TXCS: TXC + TXS. The 3-class head (none / small-medium / large) stays in every config for comparison.
CFGS.update({
    "TXC": dict(CFGS["TXM"], wk_c=1.0),
    "TXC3": dict(CFGS["TXM"], wk_c=1.0, wk_c_ks=[3, 5]),
    "TXS": dict(CFGS["TXM"], wk_s=1.0),
    "TXCS": dict(CFGS["TXM"], wk_c=1.0, wk_s=1.0),
})
# Round DY (user, 17 Nov): DAILY yes/no ERE ("any central-India ERE on day T+k", k = 1..14) as an extra trained head, scored per lead
#   against the logistic (Model P's frozen linear part, epoch 0) and ECMWF. TXY: transformer only (TXM) + daily yes/no loss.
#   W3Y: logistic + transformer correction (Model P, W3N) + daily yes/no loss. The weekly heads and BSISO stay as they are.
CFGS.update({
    "TXY": dict(CFGS["TXM"], ere_d=1.0, ere_yn=True),
    "W3Y": dict(CFGS["W3N"], ere_d=1.0, ere_yn=True),
})
# other regions for the auxiliary ERE heads (cluster centroid inside the box; central India = ci_box, never among these)
REGIONS = {"westcoast": lambda la, lo: (la >= 8) & (la < 21) & (lo <= 77.5),
           "south": lambda la, lo: (la >= 8) & (la < 17) & (lo > 77.5) & (lo <= 88.5),
           "northwest": lambda la, lo: (la >= 21) & (la < 30) & (lo < 74),
           "north": lambda la, lo: (la >= 26.5) & (la < 33) & (lo >= 74) & (lo <= 88.5),
           "northeast": lambda la, lo: (la >= 20) & (la < 30) & (lo > 88.5)}
CFG = dict(dict(enc="eof", d=48, layers=3, lam=0.5, lr=5e-4, drop=0.1, gdrop=0.15, wd=0.05, mu=0.01, epochs=60, batch=64,
                hp=False, film_enso=True, noise=0.0, jitter=0, swa=0, mreg=0.0, ec=False, ecw=False, map=0.0, map2ere=False, ana=False), **CFGS.get(CFGN, {}))
if "CFG_JSON" in os.environ:
    CFG.update(json.loads(os.environ["CFG_JSON"]))
if FAST:
    CFG["epochs"] = 2

# ---------------------------------------------------------------- inputs
MP = np.load(os.path.join(PIN, "maps.npz"))
assert (pd.DatetimeIndex(MP["dates"]) == DATES).all()
FIELDS = {"olr": MP["olr"], "u850": MP["u850"], "u200": MP["u200"], "mslp": MP["mslp"] / 100.0}
fvo = os.path.join(PIN, "maps_vo.npz")
if os.path.exists(fvo) and os.environ.get("NO_VO") != "1":
    V = np.load(fvo); FIELDS["vo850"], FIELDS["vo500"] = V["vo850"], V["vo500"]
if CFG.get("rain_in"):                                                 # IMD 1-deg rain map (log1p, 0 outside India) as an input field
    FIELDS["rain1"] = np.where(M1[None], np.log1p(R1), 0.0).astype("float32")
if CFG.get("new_fields"):                                                # file 42: Maeda's extra fields (transformer inputs only)
    _mn = np.load(os.path.join(PIN, "maps_new.npz")); assert (pd.DatetimeIndex(_mn["dates"]) == DATES).all(), "maps_new calendar"
    for _f in CFG["new_fields"]:
        FIELDS[_f] = _mn[_f].astype("float32")
if CFG.get("sm_fields"):                                                 # night run 15-16 Nov: ERA5 soil moisture (layers 1-2), maps_sm.npz
    _sm = np.load(os.path.join(PIN, "maps_sm.npz")); assert (pd.DatetimeIndex(_sm["dates"]) == DATES).all(), "maps_sm calendar"
    for _f in CFG["sm_fields"]:
        FIELDS[_f] = _sm[_f].astype("float32")
FAST_F = [f for f in ("mslp", "vo850", "vo500", "rain1") if f in FIELDS]
print("fields:", {k: v.shape for k, v in FIELDS.items()}, flush=True)
W30 = np.stack([np.arange(i - 29, i + 1) for i in ISS])                      # date indices of the 30-day window (issue day last)
assert (YEARS[W30[:, 0]] == IY).all(), "30-day window crosses a year"
LAT = {"u": MP["u_lat"], "mslp": MP["mslp_lat"]}; LON = {"u": MP["u_lon"], "mslp": MP["mslp_lon"]}
if "vo850" in FIELDS:
    LAT["vo"], LON["vo"] = V["vo_lat"], V["vo_lon"]
DOY = DATES.dayofyear.values

# ---------------------------------------------------------------- year-round OLR and BSISO (file 33: all-months BSISO training)
OD = OLR_Y = BSY = POS_D = None
YR_ON = bool(CFG.get("olr_y") or CFG.get("skb_bo") or CFG.get("allyr"))
if YR_ON:
    assert CFG["enc"] == "eof" and (not CFG.get("allyr") or CFG.get("nolin")), "year-round options: eof encoder; allyr needs nolin"
    _o = xr.open_dataset(POLR).olr.sortby("lat").sel(lat=slice(-10, 30), lon=slice(40, 160)).load()
    OD = pd.DatetimeIndex(_o.time.values); OLR_Y = _o.values.astype("float32"); del _o
    assert (np.diff(OD.values).astype("timedelta64[D]").astype(int) == 1).all() and np.isfinite(OLR_Y).all()
    POS_D = pd.Series(np.arange(len(OD)), index=OD).reindex(DATES).values
    assert np.isfinite(POS_D).all(); POS_D = POS_D.astype(int)
    assert OLR_Y.shape[1:] == FIELDS["olr"].shape[1:] and np.allclose(OLR_Y[POS_D], FIELDS["olr"], atol=1e-2)
    BSY = BS.reindex(OD).interpolate(limit=3).values.astype("float32")             # year-round BSISO PC1-4 (NaN before 1981)
    ERA_Y = {}
    if CFG.get("era_y"):                                                # file 40: year-round u850 / u200 / MSLP (NaN where absent)
        _z = np.load(os.path.join(PIN, "maps_yr.npz")); _zd = pd.DatetimeIndex(_z["dates"]); _ix = _zd.get_indexer(OD)
        for _f in ("u850", "u200", "mslp"):
            _a = np.full((len(OD),) + _z[_f].shape[1:], np.nan, "float32"); _ok = _ix >= 0; _a[_ok] = _z[_f][_ix[_ok]]; ERA_Y[_f] = _a
        print("year-round ERA5:", {k: (v.shape, round(float(np.isfinite(v).all((1, 2)).mean()), 3)) for k, v in ERA_Y.items()}, flush=True)
    assert np.allclose(BSY[POS_D[ISS]], X_BS[:, -1], atol=1e-5)

# ---------------------------------------------------------------- rain maps (IMD 1 deg, Indian cells) for the map head / analogs
NCELL = int(M1.sum()); CIC = ci_box(LA1[M1], LO1[M1])                     # central-India cells among the 283 Indian cells
RM = R1[:, M1].astype("float32")                                         # (days, 283) daily rain, mm

# ---------------------------------------------------------------- Prof B: ECMWF rain forecasts per sample and lead
ECOK = np.zeros(len(ISS), bool); EC_MAP = EC_CIM = EC_MAX = EC_W = None
if CFG.get("ec"):
    PEC = os.environ.get("PROFB_EC", os.path.join(HERE, "..", "intermediate", "profB", "ecmwf_rain.npz"))
    ez = np.load(PEC); est = pd.DatetimeIndex(ez["starts"]); spos = pd.Series(np.arange(len(est)), index=est)
    EC_MAP = np.zeros((len(ISS), NL, ez["mean"].shape[2] * ez["mean"].shape[3]), "float32")
    EC_CIM = np.zeros((len(ISS), NL, ez["ci_mem"].shape[1]), "float32"); EC_MAX = np.zeros((len(ISS), NL), "float32")
    TI = DATES[ISS]; p0 = spos.reindex(TI).values; p1 = spos.reindex(TI - pd.Timedelta(days=1)).values
    for o, p in ((1, p1), (0, p0)):                                       # latest start on T (o = 0) or T-1 (o = 1); T wins
        m = np.isfinite(p); s_ = p[m].astype(int); j = o + np.arange(1, NL + 1) + 1   # j: rain on calendar day S + j - 1 = T + k
        EC_MAP[m] = np.log1p(ez["mean"][s_][:, j].reshape(m.sum(), NL, -1)); EC_CIM[m] = ez["ci_mem"][s_][:, :, j].transpose(0, 2, 1)
        EC_MAX[m] = ez["cimax"][s_][:, j]; ECOK |= m
    if CFG.get("ecw"):                                                   # ECMWF winds: ensemble-mean u850 / u200 PCs (basis 2005-2010)
        wz = np.load(os.environ.get("PROFB_WIND", os.path.join(HERE, "..", "intermediate", "modelC", "s2s_feats.npz")))
        assert (wz["starts"] == ez["starts"]).all() and wz["have_u850"].all() and wz["have_u200"].all()
        WP = np.concatenate([wz["pc_u850"], wz["pc_u200"]], -1)            # (starts, 22 steps at 00 UTC of start + step, 24)
        EC_W = np.zeros((len(ISS), NL, WP.shape[-1]), "float32")
        for o, p in ((1, p1), (0, p0)):
            m = np.isfinite(p); s_ = p[m].astype(int); j = o + np.arange(1, NL + 1) + 1    # target day T+k runs from step j-1 to step j
            EC_W[m] = 0.5 * (WP[s_][:, j - 1] + WP[s_][:, j])
    print(f"ECMWF rain: {ECOK.sum()} of {len(ISS)} samples have a start on T or T-1 (years {sorted(set(IY[ECOK]))[:1]}-{sorted(set(IY[ECOK]))[-1:]})", flush=True)


def box(field, key, la0, la1, lo0, lo1):
    la, lo = LAT[key], LON[key]
    return field[:, (la >= la0) & (la <= la1)][:, :, (lo >= lo0) & (lo <= lo1)].mean((1, 2))


def anomaly(A, tr_days):
    """day-of-year climatology from training days (+-15 d window), anomaly / one scalar std per field."""
    cl = np.zeros((367,) + A.shape[1:], "float32")
    for d in range(1, 367):
        m = tr_days & (np.abs(DOY - d) <= 15)
        if m.any():
            cl[d] = A[m].mean(0)
    An = A - cl[DOY]
    return (An / (An[tr_days].std() + 1e-9)).astype("float32")


def eofs(An, tr_days, k):
    X = An.reshape(len(An), -1); mu = X[tr_days].mean(0)
    U, S, Vt = np.linalg.svd(X[tr_days] - mu, full_matrices=False)
    P = (X - mu) @ Vt[:k].T
    return (P / (P[tr_days].std(0) + 1e-9)).astype("float32")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def build(tr_years):
    """everything that depends on the training years: classes, targets, offsets, anomalies, EOFs, compact features."""
    C = classes([y for y in tr_years if y >= 1981] if CFG.get("ere_thr_1981") else tr_years)   # file 42: same ERE events for every training period
    trs = np.isin(IY, tr_years); trd = np.isin(YEARS, tr_years) & JJAS
    Yany = np.full((len(ISS), NL), np.nan, "float32"); Ylg = Yany.copy(); Yc = Yany.copy()
    for k in range(1, NL + 1):
        ok = JJAS[ISS + k]; c = C[ISS + k]
        Yany[ok, k - 1] = c[ok] >= 1; Yc[ok, k - 1] = c[ok]
        m = ok & (c >= 1); Ylg[m, k - 1] = c[m] >= 3
    # normal chances by target day of year (training days only)
    def clim_p(ev, base, win):
        p = np.full(367, np.nan)
        for d in range(1, 367):
            m = trd & base & (np.abs(DOY - d) <= win)
            if m.sum() > 20:
                p[d] = ev[m].mean()
        return pd.Series(p).interpolate(limit_direction="both").values
    p_any = clim_p(C >= 1, np.ones(len(C), bool), 15); p_lg = clim_p(C >= 3, C >= 1, 30)
    OFF_any = np.stack([logit(p_any[DOY[ISS + k]]) for k in range(1, NL + 1)], 1).astype("float32")
    OFF_lg = np.stack([logit(p_lg[DOY[ISS + k]]) for k in range(1, NL + 1)], 1).astype("float32")
    # field anomalies and per-day encodings
    tra = np.isin(YEARS, tr_years)                                     # all training days (Apr-Sep): May days need a climatology too
    An = {f: anomaly(A, tra) for f, A in FIELDS.items()}
    for f in An:                                                       # guard: every calendar day used must have a climatology
        assert np.abs(An[f][tra].mean()) < 0.05, f"{f}: anomaly mean {An[f][tra].mean():.3f}"
    EOF20 = {f: eofs(An[f], trd, 20) for f in An}
    # compact features for the linear link
    bs = X_BS; amp1 = np.hypot(bs[:, :, 0], bs[:, :, 1]); amp2 = np.hypot(bs[:, :, 2], bs[:, :, 3])
    ph = np.arctan2(bs[:, -1, 1], bs[:, -1, 0])
    rn = X_RN[:, :, :25]; mu = rn[trs, -1].mean(0); _, _, Vt = np.linalg.svd(rn[trs, -1] - mu, full_matrices=False)
    re_t = (rn[:, -1] - mu) @ Vt[:3].T; re_4 = (rn[:, -4:].mean(1) - mu) @ Vt[:3].T
    ere7 = np.stack([(C[i - 6:i + 1] >= 1).sum() for i in ISS])
    olr5 = EOF20["olr"][:, :5]
    jet8 = box(An["u850"], "u", 5, 15, 50, 65); tej = box(An["u200"], "u", 5, 15, 40, 80)
    shr = box(An["u200"] - An["u850"], "u", 5, 20, 60, 95); ms = box(An["mslp"], "mslp", 15, 25, 75, 90)
    w5 = lambda s: np.stack([s[i - 4:i + 1].mean() for i in ISS]); w3 = lambda s: np.stack([s[i - 2:i + 1].mean() for i in ISS])
    feats = [bs[:, -1], bs[:, -6], bs[:, -11], amp1[:, -1:], amp2[:, -1:], np.sin(ph)[:, None], np.cos(ph)[:, None],
             re_t, re_4, X_RN[:, -1, 25:], ere7[:, None], olr5[ISS], olr5[ISS - 5], olr5[ISS - 10],
             jet8[ISS, None], w5(jet8)[:, None], tej[ISS, None], w5(tej)[:, None], shr[ISS, None], w5(shr)[:, None],
             ms[ISS, None], w3(ms)[:, None], X_SL[:, :4]]
    if "vo850" in An:
        v8 = box(An["vo850"], "vo", 15, 25, 75, 92); v5 = box(An["vo500"], "vo", 18, 28, 72, 88)
        feats += [v8[ISS, None], w3(v8)[:, None], v5[ISS, None]]
    Fc = np.concatenate(feats, 1).astype("float32")
    Fc = (Fc - Fc[trs].mean(0)) / (Fc[trs].std(0) + 1e-6)
    # BSISO ridge features (as the strongest earlier ridge, R8b): daily PC1-4 history T-29..T, 20 OLR EOFs at lags 0,5,...,25 d,
    # plus the compact features
    Fb = np.concatenate([bs[:, -30:].reshape(len(bs), -1)] + [EOF20["olr"][ISS - l] for l in range(0, 30, 5)] + [Fc], 1).astype("float32")
    Fb = (Fb - Fb[trs].mean(0)) / (Fb[trs].std(0) + 1e-6)
    # deep-path sequences
    bseq = np.concatenate([bs, amp1[:, :, None], amp2[:, :, None]], 2)
    bmu, bsd = bseq[trs].mean((0, 1)), bseq[trs].std((0, 1)) + 1e-6; bseq = (bseq - bmu) / bsd
    rseq = (X_RN - X_RN[trs].mean((0, 1))) / (X_RN[trs].std((0, 1)) + 1e-6)
    sl = X_SL[:, :4].copy(); slm, sls = sl[trs, :2].mean(0), sl[trs, :2].std(0) + 1e-6; sl[:, :2] = (sl[:, :2] - slm) / sls
    pcsd = AUX[trs].reshape(-1, 2).std(0)
    # per-sample trailing mean of each field's EOFs over T-89..T-30 (clipped to 1 Apr of the same year): known on day T
    EM = {}
    for f in EOF20:
        cs = np.vstack([np.zeros((1, 20)), np.cumsum(np.nan_to_num(EOF20[f]), 0)])     # NaN only in unused 1979-80 vorticity
        lo = np.maximum(ISS - 89, YSTART[ISS]); hi = ISS - 30 + 1
        EM[f] = ((cs[hi] - cs[lo]) / (hi - lo)[:, None]).astype("float32")
        assert np.isfinite(EM[f]).all() and (YEARS[lo] == IY).all() and (hi - lo >= 30).all()
    RAINLIN = int(os.environ.get("RAINLIN", "0"))                     # linear link "LR" (file 30): IMD rain-map EOFs for leads 1..RAIN_LEADS
    XR = None
    if RAINLIN > 0:
        LRm = np.log1p(RM); mu_ = LRm[trd].mean(0); _, _, Vt_ = np.linalg.svd(LRm[trd][::2] - mu_, full_matrices=False)
        E_ = (LRm - mu_) @ Vt_[:RAINLIN].T; XR = np.concatenate([E_[ISS], np.stack([E_[i - 2:i + 1].mean(0) for i in ISS])], 1)
        XR = ((XR - XR[trs].mean(0)) / XR[trs].std(0)).astype("float32")
        SYNLIN = int(os.environ.get("SYNLIN", "0"))                   # link "LS": + MSLP and vo850 map EOFs (T, 3-day mean)
        if SYNLIN > 0:
            w3_ = lambda A_: np.stack([A_[i - 2:i + 1].mean(0) for i in ISS])
            SYF = [f for f in ("mslp", "vo850") if f in EOF20]                # NO_VO=1 (extended data): MSLP only
            XS_ = np.concatenate([EOF20[f][ISS, :SYNLIN] for f in SYF] + [w3_(EOF20[f][:, :SYNLIN]) for f in SYF], 1)
            XR = np.concatenate([XR, ((XS_ - XS_[trs].mean(0)) / (XS_[trs].std(0) + 1e-6)).astype("float32")], 1)
    out = dict(XR=XR, C=C, Yany=Yany, Ylg=Ylg, Yc=Yc, OFF_any=OFF_any, OFF_lg=OFF_lg, An=An, EOF20=EOF20, EM=EM, Fc=Fc, Fb=Fb, bseq=bseq.astype("float32"),
               rseq=rseq.astype("float32"), slow=sl.astype("float32"), pcsd=pcsd, trs=trs)
    if CFG.get("ec"):                                                   # ECMWF rain: 10 map EOFs + 3 box numbers, from training ECMWF rows
        te_ = trs & ECOK
        thr = np.percentile(EC_CIM[te_], 90)                                  # member CI rain above the training 90th percentile
        sc = np.stack([np.log1p(EC_CIM.mean(-1)), (EC_CIM > thr).mean(-1), np.log1p(EC_MAX)], -1)   # (N, NL, 3)
        Xm = EC_MAP[te_].reshape(-1, EC_MAP.shape[2]); mu_m = Xm.mean(0)
        _, _, Vt = np.linalg.svd(Xm - mu_m, full_matrices=False); E = (EC_MAP - mu_m) @ Vt[:10].T
        X = np.concatenate([E, sc] + ([EC_W] if EC_W is not None else []), -1); X = (X - X[te_].mean((0, 1))) / (X[te_].std((0, 1)) + 1e-6)
        X[~ECOK] = 0.0; out["ECX"] = X.astype("float32")
        print(f"ECMWF rain features: training rows {te_.sum()} (years {sorted(set(IY[te_]))}), member threshold {thr:.1f} mm/d", flush=True)
    if CFG.get("map", 0) > 0:                                           # rain-map targets per lead (training-year thresholds / scaling)
        thr = np.percentile(RM[trd], 90, axis=0); lr_ = np.log1p(RM); mu_r, sd_r = lr_[trd].mean(0), lr_[trd].std(0) + 1e-6
        tix = ISS[:, None] + np.arange(1, NL + 1)[None]                    # (N, NL) target day indices
        out["Ymap"] = (RM[tix] > thr).astype("float32"); out["Amap"] = ((lr_[tix] - mu_r) / sd_r).astype("float32")
        out["Mmap"] = JJAS[tix].astype("float32"); out["OFFmap"] = logit((RM[trd] > thr).mean(0)).astype("float32")
    if CFG.get("ana") or os.environ.get("LINVAR", "") in ("V4", "V5"):
        out.update(analog_feats(Fc, trs, Yany, Ylg, trd))
    if CFG.get("mreg", 0) > 0:                                          # auxiliary labels: any ERE in other regions, per lead
        CR = region_sizes(tr_years); Yr = np.full((len(ISS), NL, len(REGIONS)), np.nan, "float32"); Or = Yr.copy()
        for r in range(len(REGIONS)):
            pr = clim_p(CR[:, r] >= 1, np.ones(len(C), bool), 15)
            for k in range(1, NL + 1):
                ok = JJAS[ISS + k]; Yr[ok, k - 1, r] = CR[ISS + k][ok, r] >= 1; Or[:, k - 1, r] = logit(pr[DOY[ISS + k]])
        out.update(Yreg=Yr, OFF_reg=Or.astype("float32"))
        print("regional ERE base rates (any, training JJAS):", dict(zip(REGIONS, np.round((CR[trd] >= 1).mean(0), 3))), flush=True)
    if YR_ON:
        out.update(year_round(tr_years, trs, bmu, bsd, slm, sls))
    return out


def year_round(tr_years, trs, bmu, bsd, slm, sls):
    """file 33: OLR EOFs of year-round anomalies (calendar climatology and EOFs from all training days), the BSISO skip features
    (daily PC1-4 T-29..T + OLR EOFs at lags 0, 5, ..., 25 d) for every sample, and (allyr) the off-season BSISO training samples:
    every day of the training years that is not a JJAS issue day, whose 60-day history and 14-day target lie in training years."""
    dy = OD.dayofyear.values; tr = np.isin(OD.year, tr_years); A = OLR_Y.reshape(len(OD), -1)
    cl = np.zeros((367, A.shape[1]), "float32")
    for d in range(1, 367):
        dd = np.abs(dy - d); cl[d] = A[tr & (np.minimum(dd, 366 - dd) <= 15)].mean(0)
    An = A - cl[dy]; mu = An[tr].mean(0)
    _, _, Vt = np.linalg.svd(An[tr][::3] - mu, full_matrices=False)
    EY = (An - mu) @ Vt[:20].T; EY = (EY / EY[tr].std(0)).astype("float32")
    fb = lambda t: np.concatenate([BSY[t - 29:t + 1].ravel()] + [EY[t - l] for l in range(0, 30, 5)])
    pj = POS_D[ISS]; FbJ = np.stack([fb(t) for t in pj]); fm, fs = FbJ[trs].mean(0), FbJ[trs].std(0) + 1e-6
    res = {"EY": EY, "olr_jjas": EY[POS_D], "FbY": ((FbJ - fm) / fs).astype("float32")}
    if CFG.get("era_y"):                                                # file 40: the same treatment for u850, u200, MSLP
        EYf = {}; res["EYok"] = {}
        for f_, Af in ERA_Y.items():
            A2 = Af.reshape(len(OD), -1).astype("float64"); okd = np.isfinite(A2).all(1); cl2 = np.zeros((367, A2.shape[1]))
            for d in range(1, 367):
                dd = np.abs(dy - d); cl2[d] = A2[tr & okd & (np.minimum(dd, 366 - dd) <= 15)].mean(0)
            An2 = A2 - cl2[dy]; mu2 = An2[tr & okd].mean(0); _, _, Vt2 = np.linalg.svd(An2[tr & okd][::3] - mu2, full_matrices=False)
            E2 = np.nan_to_num((An2 - mu2) @ Vt2[:20].T); E2 = (E2 / E2[tr & okd].std(0)).astype("float32"); EYf[f_] = E2
            res[f_ + "_jjas"] = E2[POS_D]; res["EYok"][f_] = okd
        res["EYf"] = EYf
    if CFG.get("allyr"):
        yr = OD.year.values; trset = set(int(y) for y in tr_years); iss_set = set(pj.tolist())
        T = np.array([t for t in range(59, len(OD) - NL) if t not in iss_set and {int(yr[t - 59]), int(yr[t]), int(yr[t + NL])} <= trset
                      and np.isfinite(BSY[t - 59:t + NL + 1]).all()])
        bq = np.stack([BSY[t - 59:t + 1] for t in T])
        bq = np.concatenate([bq, np.hypot(bq[..., 0], bq[..., 1])[..., None], np.hypot(bq[..., 2], bq[..., 3])[..., None]], 2)
        pm = (OD[T].to_period("M") - 1).to_timestamp()                     # first day of the previous month (as X_SL)
        nd = np.nan_to_num(np.column_stack([NINO.reindex(pm).values, DMI.reindex(pm).values]))
        d_ = OD[T].dayofyear.values; sl = np.column_stack([(nd - slm) / sls, np.sin(2 * np.pi * d_ / 365.25), np.cos(2 * np.pi * d_ / 365.25)])
        res.update(YT=T, Ybq=((bq - bmu) / bsd).astype("float32"), Ysl=sl.astype("float32"),
                   YSKB=((np.stack([fb(t) for t in T]) - fm) / fs).astype("float32"),
                   Yp=np.stack([BSY[t + 1:t + NL + 1, :2] for t in T]).astype("float32"))
        print(f"all-months BSISO samples: {len(T)} off-season days (months {np.bincount(OD[T].month, minlength=13)[1:].tolist()}) "
              f"next to {trs.sum()} JJAS issue days", flush=True)
    return res


KANA = int(os.environ.get("KANA", "20"))


def analog_feats(Fc, trs, Yany, Ylg, trd):
    """K nearest training-year issue days (other years, +-20 days of season) in compact-feature space (slow / vorticity columns
    dropped). Returns analog ERE rates per lead (any, large) and an analog rain-map token per lead (10 EOFs of the mean
    log-rain map that followed the analogs + the 2 rates), all from training years only."""
    S = Fc[:, :-7]; tr_idx = np.where(trs)[0]; St = S[tr_idx]
    D2 = (S ** 2).sum(1)[:, None] + (St ** 2).sum(1)[None] - 2 * S @ St.T
    D2[(np.abs(DOY[ISS][:, None] - DOY[ISS][tr_idx][None]) > 20) | (IY[:, None] == IY[tr_idx][None])] = np.inf
    J = tr_idx[np.argsort(D2, 1)[:, :KANA]]                              # (N, K) analog sample indices
    ya = Yany[J]; yl = np.where(np.isfinite(Ylg[J]), Ylg[J], 0.0) * np.where(np.isfinite(ya), 1.0, np.nan)
    AR = np.stack([np.nanmean(ya, 1), np.nanmean(yl, 1)], -1)             # (N, NL, 2)
    lr_ = np.log1p(RM); M = np.zeros((len(ISS), NL, NCELL), "float32")
    for k in range(1, NL + 1):
        M[:, k - 1] = lr_[ISS[J] + k].mean(1)
    Xm = M[trs].reshape(-1, NCELL); mu = Xm.mean(0); _, _, Vt = np.linalg.svd(Xm[::3] - mu, full_matrices=False)
    E = (M - mu) @ Vt[:10].T; X = np.concatenate([E, AR], -1); X = (X - X[trs].mean((0, 1))) / (X[trs].std((0, 1)) + 1e-6)
    return {"AR": AR.astype("float32"), "ANX": X.astype("float32")}


YSTART = np.searchsorted(YEARS, YEARS)                                 # index of each day's first calendar day (1 Apr) in DATES


def region_sizes(tr_years):
    """largest ERE cluster size per day whose centroid lies in each REGIONS box (same p99 wet-day thresholds as classes())."""
    s = JJAS & np.isin(YEARS, tr_years)
    W = np.where(R1[s][:, M1] > 1, R1[s][:, M1], np.nan)
    T = np.full(M1.shape, np.inf); T[M1] = np.nanpercentile(W, 99, axis=0)
    out = np.zeros((len(DATES), len(REGIONS))); ev = R1 > T[None]; fns = list(REGIONS.values())
    for t in np.where(JJAS)[0]:
        L, n = ndimage.label(M1 & ev[t], structure=np.ones((3, 3)))
        if n:
            idx = np.arange(1, n + 1); sz = np.asarray(ndimage.sum(np.ones(M1.shape), L, idx))
            cla, clo = np.asarray(ndimage.mean(LA1, L, idx)), np.asarray(ndimage.mean(LO1, L, idx))
            for r, fn in enumerate(fns):
                ok = fn(cla, clo)
                if ok.any():
                    out[t, r] = sz[ok].max()
    return out


# ---------------------------------------------------------------- BSISO ridge on all days of the year (training years only)
OLRY = None


def bsiso_ridge(tr_years, alpha):
    alpha = np.broadcast_to(np.asarray(alpha, float), (NL,))
    """ridge forecast of PC1/PC2 at leads 1-14, trained on every day of the training years (as the strongest earlier ridge, R8b):
    inputs = daily PC1-4 history T-29..T + 20 OLR EOFs (year-round anomalies) at lags 0, 5, ..., 25 days. Predicts the ISS days."""
    global OLRY
    if OLRY is None:
        o = xr.open_dataset(POLR).olr
        OLRY = o.sortby("lat").sel(lat=slice(-10, 30), lon=slice(40, 160)).load()
    od = pd.DatetimeIndex(OLRY.time.values); A = OLRY.values.reshape(len(od), -1).astype("float32"); dy = od.dayofyear.values
    tr = np.isin(od.year, tr_years)
    cl = np.stack([A[tr & (np.abs(dy - d) <= 15)].mean(0) if (tr & (np.abs(dy - d) <= 15)).any() else np.zeros(A.shape[1]) for d in range(1, 367)])
    An = A - cl[dy - 1]; mu = An[tr].mean(0)
    _, _, Vt = np.linalg.svd(An[tr][::3] - mu, full_matrices=False); E = (An - mu) @ Vt[:20].T; E /= E[tr].std(0)
    bs = BS.reindex(od).interpolate(limit=3).values.astype("float32")             # (days, 4) year-round index
    t_all = np.arange(29, len(od) - NL)

    def X_of(t):
        return np.concatenate([bs[t - 29:t + 1].ravel()] + [E[t - l] for l in range(0, 30, 5)])
    ok = lambda t: np.isfinite(bs[t - 29:t + NL + 1]).all()
    trt = np.array([t for t in t_all if tr[t] and tr[t + NL] and ok(t)])
    X = np.stack([X_of(t) for t in trt]); Y = np.stack([bs[t + 1:t + NL + 1, :2].ravel() for t in trt])
    m, s = X.mean(0), X.std(0) + 1e-6; Xs = np.column_stack([np.ones(len(X)), (X - m) / s])
    G = Xs.T @ Xs; XY = Xs.T @ Y; I_ = np.diag(np.r_[0, np.ones(X.shape[1])]); W = np.zeros((Xs.shape[1], NL * 2))
    for a in np.unique(alpha):                                          # Y columns are (lead, pc) in lead-major order
        cols = np.where(np.repeat(alpha == a, 2))[0]; W[:, cols] = np.linalg.solve(G + a * I_, XY[:, cols])
    pos = pd.Series(np.arange(len(od)), index=od).reindex(DATES[ISS]).values.astype(int)
    Xi = np.column_stack([np.ones(len(pos)), (np.stack([X_of(t) for t in pos]) - m) / s])
    return (Xi @ W).reshape(len(pos), NL, 2).astype("float32"), len(trt)


# ---------------------------------------------------------------- linear link
def envlist(name, default):
    v = os.environ.get(name, default)
    return np.asarray(json.loads(v), float) if v.strip().startswith("[") else float(v)


# ECMWF features used by the linear link: 3 rain numbers (+ the first 3 u850 and first 3 u200 PCs for Prof B1)
EC_LIN = [10, 11, 12] + ([13, 14, 15, 25, 26, 27] if CFG.get("ecw") else [])
CHAIN = os.environ.get("CHAIN", "none")                                # BSISO chain in the linear link: none | fc | obs
CHAIN_PEN = float(os.environ.get("CHAIN_PEN", "-1"))                     # >0: separate (small) ridge penalty on the 3 chain features (V1, file 30)


def fit_wv(X, y, pen):
    """logistic regression with a separate ridge penalty per column (intercept unpenalised)."""
    X1 = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X1.shape[1]); R_ = np.diag(np.r_[0.0, pen])
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(X1 @ w, -30, 30)))
        st = np.linalg.solve((X1 * (p * (1 - p))[:, None]).T @ X1 + R_, X1.T @ (p - y) + R_ @ w); w -= st
        if np.abs(st).max() < 1e-7:
            break
    return w


def fit_off(X, y, off, pen):
    """logistic regression with a fixed offset (the normal-chance logit) and a ridge penalty per column (intercept unpenalised)."""
    X1 = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X1.shape[1]); R_ = np.diag(np.r_[0.0, pen])
    for _ in range(60):
        p = 1 / (1 + np.exp(-np.clip(off + X1 @ w, -30, 30)))
        st = np.linalg.solve((X1 * (p * (1 - p))[:, None]).T @ X1 + R_, X1.T @ (p - y) + R_ @ w); w -= st
        if np.abs(st).max() < 1e-7:
            break
    return w


def warm_skip(model, D, tr_rows):
    """file 36: set the in-network linear skips to their closed-form solution on the training rows. ERE: per lead, logistic on the
    skip inputs with the normal-chance offset (penalty L2 per lead as the linear link; the 3 BSISO-chain columns CHAIN_PEN). BSISO:
    per lead, ridge (ALPHA) on the skip inputs of the training JJAS days plus, for all-months configs, the off-season days."""
    l2 = np.broadcast_to(np.asarray(envlist("L2", "300"), float), (NL,)); al = np.broadcast_to(np.asarray(envlist("ALPHA", "100"), float), (NL,))
    n_ = lambda t: t.detach().cpu().numpy().astype("float64")
    SK = n_(D.SK[tr_rows]); F = SK.shape[2]; We = np.zeros((NL, F, 2)); be = np.zeros((NL, 2))
    for k in range(NL):
        pen = np.r_[np.full(F - 3, l2[k]), np.full(3, CHAIN_PEN if CHAIN_PEN > 0 else l2[k])]
        for j, (Y, M, O) in enumerate(((D.Ya, D.Ma, D.La), (D.Yl, D.Ml, D.Ll))):
            m = n_(M[tr_rows, k]) > 0; X = SK[m, k]; nz = X.std(0) > 0                       # all-zero columns (map EOFs beyond lead 10) stay 0
            w = fit_off(X[:, nz], n_(Y[tr_rows, k])[m], n_(O[tr_rows, k])[m], pen[nz]); be[k, j] = w[0]; We[k, nz, j] = w[1:]
    Xb = n_(D.SKB[tr_rows]); Yb = n_(D.Yp[tr_rows]); Mb = n_(D.Mp[tr_rows])
    if YDATA[0] is not None:
        DY, yrows = YDATA[0]; yi = torch.as_tensor(yrows, device=DEV)
        Xb = np.concatenate([Xb, n_(DY.SKB[yi])]); Yb = np.concatenate([Yb, n_(DY.Yp[yi])]); Mb = np.concatenate([Mb, np.ones_like(n_(DY.Yp[yi]))])
    Fb = Xb.shape[1]; Wb = np.zeros((NL, Fb, 2)); bb = np.zeros((NL, 2))
    for k in range(NL):
        m = (Mb[:, k] > 0).all(1); X1 = np.column_stack([np.ones(m.sum()), Xb[m]]); R_ = al[k] * np.diag(np.r_[0.0, np.ones(Fb)])
        w = np.linalg.solve(X1.T @ X1 + R_, X1.T @ Yb[m, k]); bb[k] = w[0]; Wb[k] = w[1:]
    T_ = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEV)
    with torch.no_grad():
        model.skip_e.copy_(T_(We)); model.skip_eb.copy_(T_(be)); model.skip_b.copy_(T_(Wb)); model.skip_bb.copy_(T_(bb))
    model.skip_w0 = [p.detach().clone() for p in (model.skip_e, model.skip_b)]
    print(f"warm start: ERE skip {F} inputs, BSISO skip {Fb} inputs on {len(Xb)} days", flush=True)


def mult_basis(model, D, tr_rows, R):
    """file 43: input directions for the state-dependent propagator head. The frozen BSISO skip is one ridge for all months; a ridge
    on the JJAS training rows alone differs from it in a few input directions (bsiso_propagator_probe.py: 8 of them carry the
    seasonal part of the gain). Returns the top R left singular vectors of (JJAS weights - frozen weights), scaled so that the
    projected inputs have unit variance on the training rows. Needs the warm start (skip_b = the all-months ridge)."""
    al = np.broadcast_to(np.asarray(envlist("ALPHA", "100"), float), (NL,)); n_ = lambda t: t.detach().cpu().numpy().astype("float64")
    Xb = n_(D.SKB[tr_rows]); Yb = n_(D.Yp[tr_rows]); Mb = n_(D.Mp[tr_rows]); Fb = Xb.shape[1]; Wj = np.zeros((NL, Fb, 2))
    for k in range(NL):
        m = (Mb[:, k] > 0).all(1); X1 = np.column_stack([np.ones(m.sum()), Xb[m]])
        Wj[k] = np.linalg.solve(X1.T @ X1 + al[k] * np.diag(np.r_[0.0, np.ones(Fb)]), X1.T @ Yb[m, k])[1:]
    dW = (Wj - n_(model.skip_b)).transpose(1, 0, 2).reshape(Fb, -1); U = np.linalg.svd(dW, full_matrices=False)[0][:, :R]
    U = U / (Xb @ U).std(0); print(f"propagator head: {R} input directions from the JJAS-minus-all-months ridge weights", flush=True)
    return torch.as_tensor(U, dtype=torch.float32, device=DEV)


def chain_feats(Pc_chain):
    """per lead k: BSISO (PC1, PC2, amplitude) on day T+k-5. Observed when T+k-5 <= T (k <= 5); for k > 5 'fc' = the ridge
    forecast at lead k-5 (usable everywhere), 'obs' = the observed value (only for fitting, option B as the old two-stage model)."""
    zo = np.concatenate([X_BS[:, 55:60, :2], AUX[:, :NL - 5]], 1)        # observed BSISO on T+k-5, k = 1..14 (future part: AUX)
    zf = np.concatenate([X_BS[:, 55:60, :2], Pc_chain[:, :NL - 5]], 1)   # what is known on day T
    f = lambda z: np.concatenate([z, np.hypot(z[..., :1], z[..., 1:])], -1).astype("float32")
    return f(zo), f(zf)


def fit_linear(B, rows, l2a=300.0, l2l=300.0, alpha=100.0, chain=None, chain_alpha=None):
    """logistic P(any), logistic P(large|any), ridge PC1/PC2 per lead, fitted on rows. Returns logits / PCs for all samples.
    chain: 'fc' / 'obs' adds the BSISO on day T+k-5 to each lead's logistic (see chain_feats); predictions always use 'fc'."""
    chain = CHAIN if chain is None else chain
    Fc = B["Fc"]; N = len(Fc)
    if chain != "none":
        ca = alpha if chain_alpha is None else chain_alpha; key = ("Pchain", tuple(np.broadcast_to(np.asarray(ca, float), (NL,))))
        if key not in B:
            B[key] = bsiso_ridge(np.unique(IY[rows]), ca)[0]
        Zo, Zf = chain_feats(B[key])
    l2a = np.broadcast_to(np.asarray(l2a, float), (NL,)); l2l = np.broadcast_to(np.asarray(l2l, float), (NL,))
    La, Ll, Pc = np.zeros((N, NL), "float32"), np.zeros((N, NL), "float32"), np.zeros((N, NL, 2), "float32")
    Xr = np.column_stack([np.ones(N), B["Fb"]])
    Rg = float(np.mean(alpha)) * np.diag(np.r_[0, np.ones(B["Fb"].shape[1])])
    RL = int(os.environ.get("RAIN_LEADS", "7"))
    for k in range(NL):
        Fk = np.column_stack([Fc, B["XR"]]) if (B.get("XR") is not None and k < RL) else Fc   # rain-map EOFs: leads 1..RL only
        if chain == "none":
            Xfit = Xpred = Fk
        else:
            Xpred = np.column_stack([Fk, Zf[:, k]]); Xfit = np.column_stack([Fk, (Zo if chain == "obs" else Zf)[:, k]])
        for Yk, OFF, out_, l2 in ((B["Yany"], B["OFF_any"], La, l2a), (B["Ylg"], B["OFF_lg"], Ll, l2l)):
            y = Yk[:, k]; Xf = np.column_stack([OFF[:, k], Xfit]); ok = rows & np.isfinite(y) & np.isfinite(Xf).all(1)
            if chain != "none" and CHAIN_PEN > 0:                      # chain features (last 3 columns) almost unshrunk
                w = fit_wv(Xf[ok], y[ok], np.r_[np.full(Xf.shape[1] - 3, l2[k]), np.full(3, CHAIN_PEN)])
            else:
                w = fit_w(Xf[ok], y[ok], l2=l2[k])
            out_[:, k] = np.column_stack([np.ones(N), OFF[:, k], Xpred]) @ w
        if B.get("bsiso_mode") == "jjas":                              # alternative: ridge on the JJAS issue days only
            for c in range(2):
                y = AUX[:, k, c]; ok = rows & np.isfinite(y)
                w = np.linalg.solve(Xr[ok].T @ Xr[ok] + Rg, Xr[ok].T @ y[ok]); Pc[:, k, c] = Xr @ w
    if CFG.get("ec"):                                                   # Prof B linear link: logistic on [Prof A linear logit, ECMWF box numbers]
        re_ = rows & ECOK
        for k in range(NL):
            for Yk, out_ in ((B["Yany"], La), (B["Ylg"], Ll)):
                X = np.column_stack([out_[:, k], B["ECX"][:, k, EC_LIN]]); y = Yk[:, k]; ok = re_ & np.isfinite(y)
                w = fit_w(X[ok], y[ok], l2=1.0); out_[ECOK, k] = (np.column_stack([np.ones(N), X]) @ w)[ECOK]
    if B.get("bsiso_mode") != "jjas":
        Pc, n = bsiso_ridge(np.unique(IY[rows]), alpha)
    assert np.isfinite(La).all() and np.isfinite(Ll).all() and np.isfinite(Pc).all(), "linear link produced NaN"
    return La, Ll, Pc


def calclim(yrs, A, half=15):
    """calendar-day mean and sd of A (days, cells) over the days of years yrs, +-half-day window (as maphead.calclim)."""
    m = np.isin(YEARS, yrs); S1 = np.zeros((368, A.shape[1])); S2 = S1.copy(); N_ = np.zeros(368)
    np.add.at(S1, DOY[m], A[m]); np.add.at(S2, DOY[m], A[m] ** 2); np.add.at(N_, DOY[m], 1)
    c = lambda a: np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, 0)])
    C1, C2, CN = c(S1), c(S2), c(N_); lo = np.clip(np.arange(368) - half, 0, 368); hi = np.clip(np.arange(368) + half + 1, 0, 368)
    n = (CN[hi] - CN[lo])[:, None]; mu = (C1[hi] - C1[lo]) / np.maximum(n, 1); var = (C2[hi] - C2[lo]) / np.maximum(n, 1) - mu ** 2
    return mu, np.sqrt(np.maximum(var, 1e-6))


def ridge_fit(A, Y, lam):
    """ridge with an unpenalised intercept; returns W (features + 1, outputs)."""
    A1 = np.c_[A, np.ones(len(A))]; R_ = lam * np.eye(A1.shape[1]); R_[-1, -1] = 1e-6
    return np.linalg.solve(A1.T @ A1 + R_, A1.T @ Y)


WK_OFF = (1, 8); NW = 2; NWP = 30                                         # week starts (lead of first day), weeks, map EOFs
WK_CKS = (2, 3, 4, 5, 6)                                                    # round TC: count-head thresholds (ERE days in the week)
WK_TC_KEYS = ("CNT", "YWk", "YWs", "OFFC", "OFFS", "LCL", "LSL", "LC", "LS", "LWL")


def weekly_fit(B, rows):
    """round W3: frozen closed-form linear parts of the two weekly heads, fitted on the training rows (after fit_linear, which makes
    the linear BSISO forecast used as the chain). Adds to B:
      weekly ERE : YWa / YWl (N, 2) any ERE in the week / large | any, YWc class 0/1/2, OFFW (N, 2, 2) weekly normal-chance logits
                   (training issue days, +-15 d / +-30 d of issue day of year), LW (N, 2, 2) linear logits = OFFW + logistic on
                   [compact features, rain/MSLP map EOFs, week-mean BSISO chain] (fit_off; WK_L2, CHAIN_PEN on the chain columns)
      weekly map : z = (log1p rain - calendar mean) / calendar sd (training years), ZW (N, 2, 283) week-mean z, LOCW per-cell damped
                   persistence (ridge on z_T, z_T-2..T, z_T-6..T; LAM_L), V (30, 283) EOFs of the training weekly residual,
                   RW (N, 2, 30) target coefficients of ZW - LOCW, GW (N, 2, 30) ridge from issue-day predictors (rain-pattern history,
                   observed BSISO, OLR / u850 / MSLP EOFs, slow terms, the linear BSISO forecast over the week) to RW (WK_LAM x n)."""
    N = len(ISS); trs = rows; yrs = np.unique(IY[rows]); C = B["C"]
    TIXW = np.stack([ISS[:, None] + o + np.arange(7)[None] for o in WK_OFF], 1)            # (N, 2, 7) target days
    VW = (JJAS[TIXW] & (YEARS[TIXW] == IY[:, None, None])).all(-1)                          # (N, 2) whole week inside JJAS
    cw = C[TIXW]; YWa = np.where(VW, (cw >= 1).any(-1), np.nan).astype("float32")
    YWl = np.where(VW & (cw >= 1).any(-1), (cw >= 3).any(-1), np.nan).astype("float32")
    YWc = np.where(VW, np.where((cw >= 3).any(-1), 2, np.where((cw >= 1).any(-1), 1, 0)), np.nan).astype("float32")
    d0 = DOY[ISS]

    def clim_w(y, win):
        p = np.full((367, NW), np.nan)
        for w in range(NW):
            for d in range(1, 367):
                m = trs & np.isfinite(y[:, w]) & (np.abs(d0 - d) <= win)
                if m.sum() > 20:
                    p[d, w] = y[m, w].mean()
        return pd.DataFrame(p).interpolate(limit_direction="both").values
    OFFW = np.stack([logit(clim_w(YWa, 15)[d0]), logit(clim_w(YWl, 30)[d0])], -1).astype("float32")   # (N, 2, 2)
    # round TC (user, 16 Nov): extra weekly ERE outputs. Count head: n = number of ERE days in the week (0-7), as a chain of
    #   conditional chances P(n >= k_j | n >= k_{j-1}) for the thresholds KS (k_0 = 1, i.e. the existing "any ERE" output);
    #   small/medium head: P(at least one small or medium ERE day in the week), independent of large days. Targets, normal-chance
    #   offsets (training rows) and closed-form logistic baselines (same predictors and penalty as the 3-class head) are always built.
    KS = list(CFG.get("wk_c_ks", WK_CKS)); cnt = (cw >= 1).sum(-1)
    CNT = np.where(VW, cnt, np.nan).astype("float32"); prev = [1] + KS[:-1]
    YWk = np.stack([np.where(VW & (cnt >= kp), cnt >= k, np.nan) for kp, k in zip(prev, KS)], -1).astype("float32")   # (N, 2, len(KS))
    YWs = np.where(VW, ((cw >= 1) & (cw <= 2)).any(-1), np.nan).astype("float32")
    OFFC = np.stack([logit(clim_w(YWk[..., j], 30)[d0]) for j in range(len(KS))], -1).astype("float32")
    OFFS = logit(clim_w(YWs, 15)[d0]).astype("float32")
    LCL = np.zeros((N, NW, len(KS)), "float32"); LSL = np.zeros((N, NW), "float32")
    key = [k_ for k_ in B if isinstance(k_, tuple) and k_[0] == "Pchain"][0]; Pc = B[key]; Zf = chain_feats(Pc)[1]   # (N, NL, 3)
    XR = B["XR"] if B.get("XR") is not None else np.zeros((N, 0), "float32")
    l2w = np.broadcast_to(np.asarray(envlist("WK_L2", "[1000, 1000]"), float), (NW,)); LW = np.zeros((N, NW, 2), "float32")
    for w, o in enumerate(WK_OFF):
        X = np.column_stack([B["Fc"], XR, Zf[:, o - 1:o + 6].mean(1)]).astype("float64")
        pen = np.r_[np.full(X.shape[1] - 3, l2w[w]), np.full(3, CHAIN_PEN if CHAIN_PEN > 0 else l2w[w])]
        for j, Y in enumerate((YWa, YWl)):
            ok = trs & np.isfinite(Y[:, w]); wt = fit_off(X[ok], Y[ok, w], OFFW[ok, w, j], pen)
            LW[:, w, j] = OFFW[:, w, j] + np.column_stack([np.ones(N), X]) @ wt
        for j in range(len(KS)):
            ok = trs & np.isfinite(YWk[:, w, j]); wt = fit_off(X[ok], YWk[ok, w, j], OFFC[ok, w, j], pen)
            LCL[:, w, j] = OFFC[:, w, j] + np.column_stack([np.ones(N), X]) @ wt
        ok = trs & np.isfinite(YWs[:, w]); wt = fit_off(X[ok], YWs[ok, w], OFFS[ok, w], pen)
        LSL[:, w] = OFFS[:, w] + np.column_stack([np.ones(N), X]) @ wt
    # weekly rain-anomaly maps
    Xr = np.log1p(RM.astype("float64")); mu, sd = calclim(yrs, Xr); z = (Xr - mu[DOY]) / np.maximum(sd, 0.05)[DOY]
    ZW = z[TIXW].mean(2)                                                                     # (N, 2, cells)
    I = ISS; zt, z3, z7 = z[I], np.mean([z[I - j] for j in range(3)], 0), np.mean([z[I - j] for j in range(7)], 0)
    z15 = np.mean([z[I - j] for j in range(15)], 0)
    jj = np.isin(YEARS, yrs) & JJAS; Vd = np.linalg.svd(z[jj][::2], full_matrices=False)[2]
    Xl1 = np.stack([zt, z3, z7, np.ones_like(zt)], -1); lam_l = float(os.environ.get("WK_LAM_L", "300"))
    LOCW = np.zeros_like(ZW)
    for w in range(NW):
        ok = trs & VW[:, w]; R_ = lam_l * np.eye(4); R_[-1, -1] = 1e-6
        Wl = np.linalg.solve(np.einsum("nci,ncj->cij", Xl1[ok], Xl1[ok]) + R_, np.einsum("nci,nc->ci", Xl1[ok], ZW[ok, w])[..., None])[..., 0]
        LOCW[:, w] = np.einsum("nci,ci->nc", Xl1, Wl)
    if CFG.get("wk_nolin"):                                                                 # transformer only: no damped persistence
        LOCW[:] = 0.0
    okb = trs[:, None] & VW; V = np.linalg.svd((ZW - LOCW)[okb], full_matrices=False)[2][:NWP]
    RW = np.einsum("nwc,pc->nwp", ZW - LOCW, V)
    std_ = lambda X_: ((X_ - X_[trs].mean(0)) / (X_[trs].std(0) + 1e-6)).astype("float32")
    ZPV = std_(np.concatenate([zt @ V.T, z3 @ V.T, z7 @ V.T], 1))                          # lin_tok: rain persistence in target space
    ZLOC = np.stack([zt, z3, z7], -1).astype("float32")                                       # map_lin: each cell's own rain history
    G = [zt @ Vd[:15].T, z3 @ Vd[:15].T, z7 @ Vd[:15].T, z15 @ Vd[:10].T, X_BS[:, -1, :4], X_BS[:, -6, :4], X_BS[:, -11, :4], X_SL]
    for f in ("olr", "u850", "mslp") + tuple(CFG.get("wk_g_extra", ())):                 # wk_g_extra: night-run extra fields (e.g. soil moisture)
        if f in B["EOF20"]:
            E_ = B["EOF20"][f]; G += [E_[I, :8], np.mean([E_[I - j, :4] for j in range(3)], 0)]
    for f in CFG.get("wk_g_full", ()):                                                      # night run: all 20 issue-day EOFs of these fields
        if f in B["EOF20"]:
            G.append(B["EOF20"][f][I])
            if CFG.get("wk_g_tend"):                                                         # + 5-day tendency of the first 10 EOFs
                G.append(B["EOF20"][f][I, :10] - B["EOF20"][f][I - 5, :10])
    G = np.concatenate(G, 1); GWX = std_(G); lamw = np.broadcast_to(np.asarray(envlist("WK_LAM", "[0.3, 3]"), float), (NW,)); GW = np.zeros_like(RW)
    for w, o in enumerate(WK_OFF):
        pw = Pc[:, o - 1:o + 6]; Gw = np.c_[G, pw.reshape(N, -1), np.hypot(pw[..., 0], pw[..., 1]).mean(1)]
        Gw = (Gw - Gw[trs].mean(0)) / (Gw[trs].std(0) + 1e-6); ok = trs & VW[:, w]
        Wg = ridge_fit(Gw[ok], RW[ok, w], lamw[w] * ok.sum()); GW[:, w] = np.c_[Gw, np.ones(N)] @ Wg
    LC, LS, LWL = LCL.copy(), LSL.copy(), LW.copy()                                          # round TC: training base of the extra heads; LWL = logistic 3-class
    if CFG.get("wk_nolin"):                                                                 # transformer only: no linear parts at all
        LW = OFFW.copy(); GW[:] = 0.0                                                         # ERE starts from normal chance, map from 0
        LC, LS = OFFC.copy(), OFFS.copy()
    wsc = float(((RW - GW)[okb] ** 2).mean())                                               # residual variance (loss scale)
    B.update(YWa=YWa, YWl=YWl, YWc=YWc, OFFW=OFFW, LW=LW, VW=VW, ZW=ZW.astype("float32"), LOCW=LOCW.astype("float32"), V=V.astype("float32"),
             RW=RW.astype("float32"), GW=GW.astype("float32"), wsc=wsc, ZPV=ZPV, GWX=GWX, ZLOC=ZLOC,
             CNT=CNT, YWk=YWk, YWs=YWs, OFFC=OFFC, OFFS=OFFS, LCL=LCL, LSL=LSL, LC=LC, LS=LS, LWL=LWL)
    print(f"weekly heads: {trs.sum()} training rows | week classes (train, w1) {np.bincount(YWc[trs & VW[:, 0], 0].astype(int), minlength=3) / (trs & VW[:, 0]).sum()} "
          f"| map target variance {float((RW[okb] ** 2).mean()):.4f}, after linear part {wsc:.4f}", flush=True)


def probs(La, Ll):
    pa = 1 / (1 + np.exp(-La)); pl = pa / (1 + np.exp(-Ll))
    return pa, pl


def rps(pa, pl, Yc):
    """3-class RPS per sample and lead (cumulative: any, large); NaN where the target day is outside JJAS."""
    ya = (Yc >= 1).astype("float32"); yl = (Yc >= 3).astype("float32")
    r = (pa - ya) ** 2 + (pl - yl) ** 2
    r[~np.isfinite(Yc)] = np.nan
    return r


# ---------------------------------------------------------------- deep path
class FieldEnc(nn.Module):
    def __init__(self, kind, shape, e=12):
        super().__init__(); self.kind = kind
        if kind == "eof":
            self.net = nn.Linear(20, e)
        else:
            self.net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.GELU(),
                                     nn.AdaptiveAvgPool2d((2, 3)), nn.Flatten(), nn.Linear(96, e))

    def forward(self, x):
        return self.net(x if self.kind == "eof" else x[:, None])


class ProfA(nn.Module):
    def __init__(self, cfg, fshapes, nb=6, nr=26, ns=4, lt_dims=()):
        super().__init__(); d = cfg["d"]; e = int(cfg.get("enc_e", 12)); self.cfg = cfg     # enc_e (round TY): per-day field encoding width
        self.lt_in = None                                                  # TX round: the linear models' input vectors as tokens
        self.ltp = nn.ModuleList([nn.Linear(k_, d) for k_ in lt_dims]); self.pos_lt = nn.Parameter(torch.randn(1, max(1, len(lt_dims)), d) * 0.02)
        self.fnames = list(fshapes)
        self.enc = nn.ModuleDict({f: FieldEnc(cfg["enc"], s, e) for f, s in fshapes.items()})
        self.p5 = nn.ModuleDict({f: nn.Linear(5 * e, d) for f in self.fnames})                 # 5-day patch token
        self.p1 = nn.ModuleDict({f: nn.Linear(e, d) for f in self.fnames if f in FAST_F})       # daily token (fast fields)
        self.bs = nn.Linear(5 * nb, d); self.r1 = nn.Linear(nr, d); self.r5 = nn.Linear(5 * nr, d)
        if cfg.get("bs_daily"):
            self.bs1 = nn.Linear(nb, d)                                    # daily BSISO tokens for T-4..T
        ntok = {"bsiso": 16 if cfg.get("bs_daily") else 12, "rain": 6, **{f: (10 if f in FAST_F else 6) for f in self.fnames}}
        self.pos = nn.ParameterDict({g: nn.Parameter(torch.randn(1, n, d) * 0.02) for g, n in ntok.items()})
        self.q = nn.Parameter(torch.randn(1, NL, d) * 0.02)
        if cfg.get("ana"):                                                 # analog token for day T+k -> added to lead query k
            self.anq = nn.Linear(12, d)
        if cfg.get("map", 0) > 0:                                          # rain-map head (exceedance logit, standardised log rain)
            if cfg.get("map_mlp"):
                self.head_map = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 2 * NCELL))
                nn.init.zeros_(self.head_map[2].weight); nn.init.zeros_(self.head_map[2].bias)
            else:
                self.head_map = nn.Linear(d, 2 * NCELL); nn.init.zeros_(self.head_map.weight); nn.init.zeros_(self.head_map.bias)
            if cfg.get("map2ere"):
                self.m2e = nn.Linear(int(CIC.sum()), 2); nn.init.zeros_(self.m2e.weight); nn.init.zeros_(self.m2e.bias)
        if cfg.get("ec"):                                                  # ECMWF rain for day T+k -> added to lead query k
            self.ecq = nn.Linear(13 + (24 if cfg.get("ecw") else 0), d); self.ec_miss = nn.Parameter(torch.zeros(1, 1, d))
        self.film = nn.Linear(ns, 2 * d); nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, cfg["drop"], batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg["layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, 4)
        self.last_h = self.last_raw = self.last_dn = None                  # file 43: lead states / own correction / dense-head output
        if cfg.get("mult"):                                                 # file 43: state-dependent propagator head (zero-init)
            self.head_m = nn.Linear(d, 2 * int(cfg["mult"])); nn.init.zeros_(self.head_m.weight); nn.init.zeros_(self.head_m.bias)
        if cfg.get("dense") or cfg.get("pre_ep"):                           # file 43: future OLR / u850 EOF coefficients (extra targets)
            self.head_dn = nn.Linear(d, 20)
        self.last_w = None
        if cfg.get("wk"):                                                   # round W3: weekly ERE (2 logits) + weekly map (30 EOF coefs)
            if cfg.get("wk_mlp"):                                           # round TY: wider weekly head (MLP, zero-initialised output)
                self.head_w = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 2 + NWP))
                nn.init.zeros_(self.head_w[2].weight); nn.init.zeros_(self.head_w[2].bias)
            else:
                self.head_w = nn.Linear(d, 2 + NWP); nn.init.zeros_(self.head_w.weight); nn.init.zeros_(self.head_w.bias)
        self.last_c = None
        if cfg.get("wk") and (cfg.get("wk_c", 0) > 0 or cfg.get("wk_s", 0) > 0):   # round TC: count chain (len KS) + small/medium logit
            nc_ = len(cfg.get("wk_c_ks", WK_CKS)) + 1
            self.head_c = nn.Linear(d, nc_); nn.init.zeros_(self.head_c.weight); nn.init.zeros_(self.head_c.bias)
        if cfg.get("ymap"):                                                 # Model Y: rain-anomaly map correction (EOF coefficients)
            self.head_ym = nn.Linear(d, NYM); nn.init.zeros_(self.head_ym.weight); nn.init.zeros_(self.head_ym.bias)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        if cfg.get("skip"):                                                 # in-network linear skip (sizes set lazily from the data)
            self.skip_e = None; self.skip_b = None
        if cfg.get("chain_deep"):                                         # BSISO(T+k-5) -> ERE logit corrections, zero-init output
            self.chain = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 2))
            nn.init.zeros_(self.chain[2].weight); nn.init.zeros_(self.chain[2].bias)
        self.nreg = len(REGIONS) if cfg.get("mreg", 0) > 0 else 0
        if self.nreg:
            self.head_reg = nn.Linear(d, self.nreg); nn.init.zeros_(self.head_reg.weight); nn.init.zeros_(self.head_reg.bias)

    def forward(self, bseq, rseq, fseq, slow, gmask=None, ec=None, ana=None):
        B = bseq.shape[0]; toks = []
        if self.cfg.get("bs_daily"):
            toks.append(torch.cat([self.bs(bseq[:, :55].reshape(B, 11, -1)), self.bs1(bseq[:, 55:])], 1) + self.pos["bsiso"])
        else:
            toks.append(self.bs(bseq.reshape(B, 12, -1)) + self.pos["bsiso"])
        r = torch.cat([self.r5(rseq[:, :5].reshape(B, 1, -1)), self.r1(rseq[:, 5:])], 1) + self.pos["rain"]
        if not self.cfg.get("no_rainband"):                                # file 42 (HXM): rain-band tokens left out
            toks.append(r if gmask is None else r * gmask["rain"][:, None, None])
        for f in self.fnames:
            x = fseq[f]                                                    # (B, 30, e) encoded days
            if f in FAST_F:
                t = torch.cat([self.p5[f](x[:, :25].reshape(B, 5, -1)), self.p1[f](x[:, 25:])], 1)
            else:
                t = self.p5[f](x.reshape(B, 6, -1))
            t = t + self.pos[f]
            toks.append(t if gmask is None else t * gmask[f][:, None, None])
        if self.lt_in is not None:                                         # one token per linear-model input vector (masked: 0)
            for j_, (x_, m_) in enumerate(self.lt_in):
                toks.append(((self.ltp[j_](x_) + self.pos_lt[0, j_]) * m_[:, None])[:, None])
        q = self.q.expand(B, -1, -1)
        if ec is not None:
            X, ok = ec; ok = ok[:, None, None]
            q = q + ok * self.ecq(X) + (1 - ok) * self.ec_miss
        if ana is not None:
            q = q + self.anq(ana)
        h = torch.cat([q] + toks, 1)
        g, b = self.film(slow).chunk(2, -1); h = h * (1 + g[:, None]) + b[:, None]
        h = self.norm(self.encoder(h))[:, :NL]
        out = self.head(h)                                                 # (B, 14, 4): d_any, d_large, d_pc1, d_pc2
        if self.cfg.get("ere_sg"):                                         # file 43: the ERE loss does not reach the transformer body
            out = torch.cat([self.head(h.detach())[..., :2], out[..., 2:]], -1)
        self.last_h = h; self.last_dn = self.head_dn(h) if hasattr(self, "head_dn") else None
        if self.cfg.get("wk"):                                             # week w reads the mean lead state of its 7 days
            hw_ = torch.stack([h[:, o - 1:o + 6].mean(1) for o in WK_OFF], 1)
            self.last_w = self.head_w(hw_)                                 # (B, 2, 2 + NWP)
            if hasattr(self, "head_c"):
                self.last_c = self.head_c(hw_)                             # (B, 2, len(KS) + 1)
        out = torch.cat([out, self.head_reg(h)], -1) if self.nreg else out    # (+ regional any-ERE logit corrections)
        self.last_map = None
        self.last_ym = self.head_ym(h.detach() if self.cfg.get("ymap_sg") else h) if self.cfg.get("ymap") else None
        if self.cfg.get("map", 0) > 0:
            mp = self.head_map(h).reshape(B, NL, NCELL, 2); self.last_map = mp
            if self.cfg.get("map2ere"):
                out = torch.cat([out[..., :2] + self.m2e(mp[..., CIC_T, 0]), out[..., 2:]], -1)
        return out


NYM = 30; MAPY_KEY = [None]                                             # Model Y: EOF patterns; which precomputed build (f0..f4 / final)
PMAPY = os.environ.get("PROFA_MAPY", os.path.join(HERE, "..", "intermediate", "modelY"))
if not os.path.exists(os.path.join(PMAPY, "mapY_f0.npz")) and glob.glob("/kaggle/input/**/mapY_f0.npz", recursive=True):
    PMAPY = os.path.dirname(glob.glob("/kaggle/input/**/mapY_f0.npz", recursive=True)[0])


def dense_targets(B, pos):
    """file 43: future large-scale fields as extra targets. pos (n, NL) = year-round calendar positions of days T+1..T+14.
    Returns the first 10 OLR and first 10 u850 EOF coefficients (year-round anomalies, unit variance on training days) and a mask."""
    o = B["EY"][pos][..., :10]; u = B["EYf"]["u850"][pos][..., :10]; ok = B["EYok"]["u850"][pos]
    return (np.concatenate([o, u], -1).astype("float32"),
            np.concatenate([np.ones_like(o), np.repeat(ok[..., None], 10, -1)], -1).astype("float32"))


def aux_loss(model, D, ii, cfg, ere=True):
    """file 43: penalties on the size of the transformer's own correction and the dense-field loss, for the rows of the last
    run_model call on this model (0 for configs without these options)."""
    L = 0.0; raw = model.last_raw
    if cfg.get("pen_b", 0) > 0:
        L = L + cfg["pen_b"] * ((raw[..., 2:4] / D.pcsd) ** 2).mean()
    if ere and cfg.get("pen_e", 0) > 0:
        L = L + cfg["pen_e"] * (raw[..., :2] ** 2).mean()
    if cfg.get("dense", 0) > 0:
        m = D.DNM[ii]; L = L + cfg["dense"] * (((model.last_dn - D.DN[ii]) ** 2) * m).sum() / m.sum().clamp(min=1)
    return L


class Data:
    """tensors for one fold: per-day field inputs (EOF coefficients or anomaly maps) and per-sample sequences / targets."""
    def __init__(self, B, lin, cfg, perm=None):
        self.cfg = cfg; self.B = B; idx = np.arange(len(ISS)) if perm is None else perm   # perm: control (inputs from another year)
        T_ = lambda x: torch.as_tensor(np.ascontiguousarray(x)).to(DEV)
        self.day = {f: T_(B["EOF20"][f] if cfg["enc"] == "eof" else B["An"][f]) for f in FIELDS}
        if cfg.get("olr_y"):                                                # OLR tokens from the year-round EOFs (file 33)
            self.day["olr"] = T_(B["olr_jjas"])
        if cfg.get("era_y"):                                                # file 40: u850 / u200 / MSLP tokens from the year-round EOFs
            for f_ in B["EYf"]:
                self.day[f_] = T_(B[f_ + "_jjas"])
        self.W = T_(W30[idx]); self.bseq = T_(B["bseq"][idx]); self.rseq = T_(B["rseq"][idx])
        sl = B["slow"][idx].copy()
        if not cfg.get("film_enso", True):
            sl[:, :2] = 0.0                                                # Nino3.4 / DMI out of the deep path
        self.slow = T_(sl)
        self.EM = {f: T_(B["EM"][f][idx]) for f in FIELDS} if (cfg.get("hp") or cfg.get("hp_fields")) else None
        if cfg.get("mreg", 0) > 0:
            self.Yr = T_(np.nan_to_num(B["Yreg"])); self.Mr = T_(np.isfinite(B["Yreg"]).astype("float32")); self.Or = T_(B["OFF_reg"])
        La, Ll, Pc = lin
        self.La, self.Ll, self.Pc = T_(La), T_(Ll), T_(Pc)
        self.Ya = T_(np.nan_to_num(B["Yany"])); self.Ma = T_(np.isfinite(B["Yany"]).astype("float32"))
        self.Yl = T_(np.nan_to_num(B["Ylg"])); self.Ml = T_(np.isfinite(B["Ylg"]).astype("float32"))
        aux = AUX.copy(); self.Yp = T_(np.nan_to_num(aux)); self.Mp = T_(np.isfinite(aux).astype("float32"))
        self.pcsd = T_(np.asarray(B["pcsd"], "float32"))
        if cfg.get("ec"):
            self.ECX = T_(B["ECX"][idx]); self.ECOK = T_(ECOK[idx].astype("float32"))
        if cfg.get("ana"):
            self.ANX = T_(B["ANX"][idx])
        if cfg.get("map", 0) > 0:                                           # targets stay real (not permuted) like the other targets
            self.Ymap, self.Amap, self.Mmap = T_(B["Ymap"]), T_(B["Amap"]), T_(B["Mmap"]); self.OFFmap = T_(B["OFFmap"])
        if cfg.get("skip"):                                                 # day-T inputs for the in-network linear skip
            key = [k_ for k_ in B if isinstance(k_, tuple) and k_[0] == "Pchain"][0]; Zf = chain_feats(B[key])[1]       # (N, NL, 3)
            RLk = int(os.environ.get("RAIN_LEADS", "7")); XRm = B["XR"] if B.get("XR") is not None else np.zeros((len(ISS), 0), "float32")
            SK = np.concatenate([np.repeat(np.concatenate([B["Fc"], XRm], 1)[:, None], NL, 1), Zf], 2).astype("float32")
            SK[:, RLk:, B["Fc"].shape[1]:B["Fc"].shape[1] + XRm.shape[1]] = 0.0                  # map EOFs only for leads 1..RL (as LS)
            self.SK = T_(SK[idx]); self.SKB = T_((B["FbY"] if cfg.get("skb_bo") else B["Fb"])[idx])
        self.Zobs5 = T_(X_BS[idx, 55:60, :2])                              # observed BSISO on T-4..T = T+k-5 for k = 1..5
        if cfg.get("ymap"):                                                 # Model Y targets: map coefficients minus the frozen linear part
            z_ = np.load(os.path.join(PMAPY, f"mapY{cfg.get('ymap_tag', '')}_{MAPY_KEY[0]}.npz"))
            assert (z_["iss_dates"] == DATES[ISS].values.astype("datetime64[D]").astype(str)).all() and z_["G"].shape[2] == NYM
            Yt = (z_["R"] - z_["G"]).astype("float32"); Mt = z_["M"].astype("float32")
            self.ym_scale = float((Yt[B["trs"]] ** 2 * Mt[B["trs"]][..., None]).sum() / (Mt[B["trs"]].sum() * NYM))   # target variance (training rows)
            self.Ym, self.Mym = T_(Yt), T_(Mt)                         # targets stay real (not permuted)
        self.lead = T_((np.arange(1, NL + 1, dtype="float32") / NL)[None, :, None])
        if cfg.get("lin_tok"):                                              # TX round: inputs of the linear parts (no linear outputs)
            XRm_ = B["XR"] if B.get("XR") is not None else np.zeros((len(ISS), 0), "float32")
            self.LT = [T_(x_[idx]) for x_ in ((B["FbY"] if "FbY" in B else B["Fb"]), np.concatenate([B["Fc"], XRm_], 1), B["GWX"], B["ZPV"])]
            self.LTM = [1.0, 1.0, 1.0, 1.0]
        if cfg.get("dense") or cfg.get("pre_ep"):                          # targets stay real (not permuted)
            self.DN, self.DNM = [T_(a) for a in dense_targets(B, POS_D[ISS][:, None] + np.arange(1, NL + 1)[None])]
        if cfg.get("wk"):                                                   # round W3: weekly targets and frozen linear parts (real rows)
            self.YWa, self.MWa = T_(np.nan_to_num(B["YWa"])), T_(np.isfinite(B["YWa"]).astype("float32"))
            self.YWl, self.MWl = T_(np.nan_to_num(B["YWl"])), T_(np.isfinite(B["YWl"]).astype("float32"))
            self.LW, self.RW, self.GW = T_(B["LW"]), T_(B["RW"]), T_(B["GW"]); self.MW = T_(B["VW"].astype("float32")); self.wsc = float(B["wsc"])
            self.YWk, self.MWk, self.LC = T_(np.nan_to_num(B["YWk"])), T_(np.isfinite(B["YWk"]).astype("float32")), T_(B["LC"])
            self.YWs, self.MWs, self.LS = T_(np.nan_to_num(B["YWs"])), T_(np.isfinite(B["YWs"]).astype("float32")), T_(B["LS"])

    def fseq(self, model, ii, train=False):
        W = self.W[ii]; out = {}; c = self.cfg
        if train and c.get("jitter", 0) > 0:                               # backward shift only: never data after day T
            W = W - torch.randint(0, c["jitter"] + 1, (len(ii), 1), device=DEV)
        for f in model.fnames:
            if c["enc"] == "eof":
                x = self.day[f][W]                                         # (B, 30, 20)
                if self.EM is not None and (c.get("hp") or f in c.get("hp_fields", [])):
                    x = x - self.EM[f][ii][:, None]
                if train and c.get("noise", 0) > 0:
                    x = x + c["noise"] * torch.randn_like(x)
                out[f] = model.enc[f](x)
            else:
                assert not c.get("hp"), "hp is implemented for the eof encoder only"
                u, inv = torch.unique(W.reshape(-1), return_inverse=True)
                enc = model.enc[f](self.day[f][u])
                out[f] = enc[inv].reshape(W.shape[0], 30, -1)
        return out


class DataY(Data):
    """file 33: the off-season BSISO samples (all months). BSISO and OLR tokens are real; rain and ERA5 tokens are masked to zero
    (the same masking as group dropout), the ERE skip is off, and only the BSISO loss is used."""
    def __init__(self, B, cfg):
        self.cfg = cfg; T_ = lambda x: torch.as_tensor(np.ascontiguousarray(x)).to(DEV); n = len(B["YT"])
        z = torch.zeros((len(OD), 20), device=DEV)
        EYf = B.get("EYf", {}) if cfg.get("era_y") else {}
        self.day = {f: (T_(B["EY"]) if f == "olr" else (T_(EYf[f]) if f in EYf else z)) for f in FIELDS}
        self.W = T_(B["YT"][:, None] - np.arange(29, -1, -1)[None]); self.bseq = T_(B["Ybq"])
        self.rseq = torch.zeros((n,) + X_RN.shape[1:], device=DEV); self.slow = T_(B["Ysl"]); self.EM = None; self.SK = None
        self.SKB = T_(B["YSKB"]); self.Yp = T_(B["Yp"]); self.pcsd = T_(np.asarray(B["pcsd"], "float32"))
        self.gfix = {g: (1.0 if (g == "olr" or g in EYf) else 0.0) for g in ["rain"] + list(FIELDS)}   # file 40: ERA5 real if era_y
        if cfg.get("dense") or cfg.get("pre_ep"):
            self.DN, self.DNM = [T_(a) for a in dense_targets(B, B["YT"][:, None] + np.arange(1, NL + 1)[None])]
        if cfg.get("lin_tok"):                                              # off-season: BSISO-skip vector real, the JJAS-only ones off
            XRm_ = B["XR"] if B.get("XR") is not None else np.zeros((len(ISS), 0), "float32")
            dims_ = [B["Fc"].shape[1] + XRm_.shape[1], B["GWX"].shape[1], B["ZPV"].shape[1]]
            self.LT = [self.SKB] + [torch.zeros((n, k_), device=DEV) for k_ in dims_]; self.LTM = [1.0, 0.0, 0.0, 0.0]


YDATA = [None]                                                           # (DataY, rows) when the config trains on all months


IMP_MASK = {"g": None, "perm": None}                                     # file 41: group removed at evaluation (importance test)


def run_model(model, D, ii, train):
    gm = None
    if train and model.cfg["gdrop"] > 0:
        gm = {g: (torch.rand(len(ii), device=DEV) > model.cfg["gdrop"]).float() for g in ["rain"] + model.fnames}
    gf = getattr(D, "gfix", None)
    if gf is not None:                                                  # off-season samples: fixed masks on top of group dropout
        gm = {g: (gm[g] if gm is not None else torch.ones(len(ii), device=DEV)) * gf[g] for g in ["rain"] + model.fnames}
    g_ = None if train else IMP_MASK["g"]
    if g_ is not None and g_ in ["rain"] + list(model.fnames):               # importance test: switch one input group off
        gm = {g: (gm[g] if gm is not None else torch.ones(len(ii), device=DEV)) for g in ["rain"] + model.fnames}; gm[g_] = gm[g_] * 0.0
    bseq, rseq = (D.bseq[IMP_MASK["perm"][ii]] if g_ == "bsiso" else D.bseq[ii]), D.rseq[ii]; nz = model.cfg.get("noise", 0)
    if train and nz > 0:
        bseq = bseq + 0.5 * nz * torch.randn_like(bseq); rseq = rseq + nz * torch.randn_like(rseq)
    ec = None
    if model.cfg.get("ec"):
        ok = D.ECOK[ii]
        if train and model.cfg["gdrop"] > 0:                             # ECMWF token dropped like the other groups
            ok = ok * (torch.rand(len(ii), device=DEV) > model.cfg["gdrop"]).float()
        ec = (D.ECX[ii], ok)
    if model.cfg.get("lin_tok"):
        lt_ = []
        for x_, m0_ in zip(D.LT, D.LTM):
            m_ = torch.full((len(ii),), float(m0_), device=DEV)
            if train and model.cfg["gdrop"] > 0:
                m_ = m_ * (torch.rand(len(ii), device=DEV) > model.cfg["gdrop"]).float()
            lt_.append((x_[ii], m_))
        model.lt_in = lt_
    out = model(bseq, rseq, D.fseq(model, ii, train), D.slow[ii] * (0.0 if g_ == "slow" else 1.0), gm, ec, D.ANX[ii] if model.cfg.get("ana") else None)
    model.lt_in = None
    if model.cfg.get("mult"):                                           # file 43: state-dependent change of the linear BSISO map
        c_ = model.head_m(model.last_h).reshape(len(ii), NL, 2, -1)
        out = torch.cat([out[..., :2], out[..., 2:4] + torch.einsum("nkor,nr->nko", c_, D.SKB[ii] @ model.mU), out[..., 4:]], -1)
    model.last_raw = out                                                # the transformer's own correction (before the linear skips)
    if model.cfg.get("skip"):
        e_ = (torch.einsum("nkf,kfo->nko", D.SK[ii], model.skip_e) if D.SK is not None else 0.0) + model.skip_eb
        b_ = torch.einsum("nf,kfo->nko", D.SKB[ii], model.skip_b) + model.skip_bb
        out = torch.cat([out[..., :2] + e_, out[..., 2:4] + b_, out[..., 4:]], -1)
    if model.cfg.get("chain_deep"):                                   # own BSISO forecast (ridge + deep, no gradient into the BSISO head)
        zf = D.Pc[ii] + out[..., 2:4].detach()
        z = torch.cat([D.Zobs5[ii], zf[:, :NL - 5]], 1)                  # lead k uses the BSISO on day T+k-5
        f = torch.cat([z, z.norm(dim=-1, keepdim=True), D.lead.expand(len(ii), -1, -1)], -1)
        out = torch.cat([out[..., :2] + model.chain(f), out[..., 2:]], -1)
    return out


def loss_fn(out, D, ii, cfg, ere=True):
    la = D.La[ii] + out[..., 0]; ll = D.Ll[ii] + out[..., 1]
    ba = (Fnn.binary_cross_entropy_with_logits(la, D.Ya[ii], reduction="none") * D.Ma[ii]).sum() / D.Ma[ii].sum().clamp(min=1)
    bl = (Fnn.binary_cross_entropy_with_logits(ll, D.Yl[ii], reduction="none") * D.Ml[ii]).sum() / D.Ml[ii].sum().clamp(min=1)
    pc = D.Pc[ii] + out[..., 2:4]
    mp = ((((pc - D.Yp[ii]) / D.pcsd) ** 2) * D.Mp[ii]).sum() / D.Mp[ii].sum().clamp(min=1)
    pen = (out[..., :2] ** 2).mean()
    bl_w = 0.0 if cfg.get("ere_yn") else 1.0                            # round DY: daily yes/no only (no large | any term)
    L = (cfg.get("ere_d", 1.0) * (ba + bl_w * bl) + cfg["mu"] * pen if ere else 0.0) + cfg["lam"] * mp
    if cfg.get("wk") and MODEL_REF[0] is not None and MODEL_REF[0].last_w is not None and hasattr(D, "RW"):   # round W3 weekly heads
        dw = MODEL_REF[0].last_w; mw = D.MW[ii]
        wa = (Fnn.binary_cross_entropy_with_logits(D.LW[ii, :, 0] + dw[..., 0], D.YWa[ii], reduction="none") * D.MWa[ii]).sum() / D.MWa[ii].sum().clamp(min=1)
        wl = (Fnn.binary_cross_entropy_with_logits(D.LW[ii, :, 1] + dw[..., 1], D.YWl[ii], reduction="none") * D.MWl[ii]).sum() / D.MWl[ii].sum().clamp(min=1)
        wm = ((((D.GW[ii] + dw[..., 2:] - D.RW[ii]) ** 2) * mw[..., None]).sum() / (mw.sum() * NWP).clamp(min=1)) / D.wsc
        L = L + cfg.get("wk_e", 1.0) * (wa + wl) + cfg.get("wk_m", 0.5) * wm \
            + cfg.get("pen_w", 0.0) * (dw[..., :2] ** 2).mean() + cfg.get("pen_m", 0.0) * (dw[..., 2:] ** 2).mean() / D.wsc
        dc = MODEL_REF[0].last_c
        if dc is not None and cfg.get("wk_c", 0) > 0:                     # round TC: count chain, mean BCE over thresholds x 2
            m_ = D.MWk[ii]; bce_ = Fnn.binary_cross_entropy_with_logits(D.LC[ii] + dc[..., :-1], D.YWk[ii], reduction="none") * m_
            L = L + cfg["wk_c"] * 2.0 * (bce_.sum((0, 1)) / m_.sum((0, 1)).clamp(min=1)).mean()
        if dc is not None and cfg.get("wk_s", 0) > 0:                     # round TC: any small/medium day in the week
            m_ = D.MWs[ii]
            L = L + cfg["wk_s"] * (Fnn.binary_cross_entropy_with_logits(D.LS[ii] + dc[..., -1], D.YWs[ii], reduction="none") * m_).sum() / m_.sum().clamp(min=1)
    if cfg.get("skip") and MODEL_REF[0] is not None and MODEL_REF[0].skip_e is not None:
        m_ = MODEL_REF[0]; e0, b0 = getattr(m_, "skip_w0", (0.0, 0.0))                      # warm start: pull back to the closed form
        L = L + cfg.get("skip_l2", 0.1) * (((m_.skip_e - e0) ** 2).sum(1).mean() + 0.1 * ((m_.skip_b - b0) ** 2).sum(1).mean())
    if cfg.get("map", 0) > 0 and MODEL_REF[0] is not None and MODEL_REF[0].last_map is not None:
        mq = MODEL_REF[0].last_map; mm = D.Mmap[ii][..., None]
        be = (Fnn.binary_cross_entropy_with_logits(D.OFFmap + mq[..., 0], D.Ymap[ii], reduction="none") * mm).sum() / (mm.sum() * NCELL).clamp(min=1)
        ma = ((((mq[..., 1] - D.Amap[ii]) ** 2) * mm).sum() / (mm.sum() * NCELL).clamp(min=1))
        L = L + cfg["map"] * (be + 0.5 * ma)
    if cfg.get("ymap") and MODEL_REF[0] is not None and MODEL_REF[0].last_ym is not None and hasattr(D, "Ym"):
        mq = MODEL_REF[0].last_ym; mm = D.Mym[ii][..., None]
        L = L + cfg.get("ymapw", 1.0) * ((((mq - D.Ym[ii]) ** 2) * mm).sum() / (mm.sum() * NYM).clamp(min=1)) / D.ym_scale
    if cfg.get("mreg", 0) > 0:
        lr_ = D.Or[ii] + out[..., 4:]
        br = (Fnn.binary_cross_entropy_with_logits(lr_, D.Yr[ii], reduction="none") * D.Mr[ii]).sum() / D.Mr[ii].sum().clamp(min=1)
        L = L + cfg["mreg"] * br
    if MODEL_REF[0] is not None and (cfg.get("pen_b", 0) > 0 or cfg.get("pen_e", 0) > 0 or cfg.get("dense", 0) > 0):
        L = L + aux_loss(MODEL_REF[0], D, ii, cfg, ere)
    return L, (ba.item(), bl.item(), mp.item())


MODEL_REF = [None]                                                       # the model whose last_map the loss reads
CIC_T = torch.as_tensor(np.where(CIC)[0])


def map_scores(model, D, rows):
    """held-out rain-map skill per lead: Brier skill of the exceedance chance vs the cell base rate, and pattern correlation of
    the standardised log rain (configs with the map head only)."""
    model.eval(); P, A = [], []
    with torch.no_grad():
        for b in range(0, len(rows), 256):
            ii = torch.tensor(rows[b:b + 256], device=DEV); run_model(model, D, ii, False); mp = model.last_map
            P.append(torch.sigmoid(D.OFFmap + mp[..., 0]).cpu().numpy()); A.append(mp[..., 1].cpu().numpy())
    P, A = np.concatenate(P), np.concatenate(A); Y = D.Ymap[rows].cpu().numpy(); Am = D.Amap[rows].cpu().numpy(); M = D.Mmap[rows].cpu().numpy()
    p0 = 1 / (1 + np.exp(-D.OFFmap.cpu().numpy())); out = {}
    for k in range(NL):
        m = M[:, k] > 0
        bsk = ((P[m, k] - Y[m, k]) ** 2).mean(); b0k = ((p0[None] - Y[m, k]) ** 2).mean()
        a, o = A[m, k], Am[m, k]; a = a - a.mean(1, keepdims=True); o = o - o.mean(1, keepdims=True)
        out[k + 1] = (float(1 - bsk / b0k), float(np.mean((a * o).sum(1) / np.sqrt((a ** 2).sum(1) * (o ** 2).sum(1) + 1e-9))))
    return out


def lin_map_scores(B, tr_rows, va_rows):
    """TX round: per-cell linear baseline for the daily rain-map head, leads 1-14. Predictors: compact features + rain / MSLP map
    EOFs + the weekly ridge's predictors (shared by all cells) + the cell's own rain history (z on T, T-2..T, T-6..T). Exceedance:
    logistic on top of the cell base rate; log rain: least squares. Ridge weight from {1e-3, 1e-2, 0.1, 1} on every 5th training year,
    then refit on all training years. Scores as map_scores (Brier skill vs the cell base rate, pattern correlation)."""
    XRm = B["XR"] if B.get("XR") is not None else np.zeros((len(ISS), 0), "float32")
    T_ = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=DEV)
    Xs = T_(np.concatenate([B["Fc"], XRm, B["GWX"]], 1)); Zl = T_(B["ZLOC"]); Y, A, M = T_(B["Ymap"]), T_(B["Amap"]), T_(B["Mmap"])
    off = T_(B["OFFmap"]); C_ = Y.shape[2]; p_ = Xs.shape[1]
    yrs = np.unique(IY[tr_rows]); hold = np.isin(IY[tr_rows], yrs[::5]); itr, iva = tr_rows[~hold], tr_rows[hold]
    print(f"linear map baseline: {p_} shared predictors, {len(itr)} / {len(iva)} inner rows, finite {bool(torch.isfinite(Xs).all())}", flush=True)

    def fit(rows, k, lam, kind):
        W = torch.zeros(p_, C_, device=DEV, requires_grad=True); Wl = torch.zeros(C_, 3, device=DEV, requires_grad=True)
        b = torch.zeros(C_, device=DEV, requires_grad=True); r = torch.as_tensor(rows, device=DEV); m = M[r, k][:, None]
        opt = torch.optim.LBFGS([W, Wl, b], lr=1.0, max_iter=150, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); z = Xs[r] @ W + (Zl[r] * Wl).sum(-1) + b
            if kind == "bce":
                L = (Fnn.binary_cross_entropy_with_logits(off + z, Y[r, k], reduction="none") * m).sum() / (m.sum() * C_)
            else:
                L = (((z - A[r, k]) ** 2) * m).sum() / (m.sum() * C_)
            L = L + lam * ((W ** 2).sum() + (Wl ** 2).sum()) / C_; L.backward(); return L
        opt.step(closure)
        return lambda rr: (Xs[rr] @ W + (Zl[rr] * Wl).sum(-1) + b).detach()

    def score(f_bce, f_mse, rows, k):
        r = torch.as_tensor(rows, device=DEV); m = (M[r, k] > 0).cpu().numpy(); rr = r[torch.as_tensor(m, device=DEV)]
        P = torch.sigmoid(off + f_bce(rr)).cpu().numpy(); Yk = Y[rr, k].cpu().numpy(); p0 = torch.sigmoid(off).cpu().numpy()
        bss = 1 - ((P - Yk) ** 2).mean() / ((p0[None] - Yk) ** 2).mean()
        a, o = f_mse(rr).cpu().numpy(), A[rr, k].cpu().numpy(); a = a - a.mean(1, keepdims=True); o = o - o.mean(1, keepdims=True)
        return float(bss), float(np.mean((a * o).sum(1) / np.sqrt((a ** 2).sum(1) * (o ** 2).sum(1) + 1e-9)))
    out = {}
    for k in range(NL):
        lb = max([1e-3, 1e-2, 1e-1, 1.0], key=lambda l_: score(fit(itr, k, l_, "bce"), lambda rr: A[rr, k] * 0, iva, k)[0])
        lm = max([1e-3, 1e-2, 1e-1, 1.0], key=lambda l_: score(lambda rr: off.expand(len(rr), -1) * 0 - 50, fit(itr, k, l_, "mse"), iva, k)[1])
        out[k + 1] = score(fit(tr_rows, k, lb, "bce"), fit(tr_rows, k, lm, "mse"), va_rows, k) + (lb, lm)
        print(f"  linear map baseline lead {k + 1}: BSS {out[k + 1][0]:.3f} corr {out[k + 1][1]:.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return out


def predict(model, D, rows):
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(rows), 256):
            out.append(run_model(model, D, torch.tensor(rows[b:b + 256], device=DEV), False)[..., :4])
    return torch.cat(out).cpu().numpy()


def predict_w(model, D, rows):
    """round W3: the transformer's weekly corrections (rows, 2, 2 + NWP): ERE logits (any, large | any) and map EOF coefficients."""
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(rows), 256):
            run_model(model, D, torch.tensor(rows[b:b + 256], device=DEV), False); out.append(model.last_w)
    return torch.cat(out).cpu().numpy().astype("float32")


def predict_c(model, D, rows):
    """round TC: the transformer's corrections of the extra weekly heads (rows, 2, len(KS) + 1): count chain logits, small/medium."""
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(rows), 256):
            run_model(model, D, torch.tensor(rows[b:b + 256], device=DEV), False); out.append(model.last_c)
    return torch.cat(out).cpu().numpy().astype("float32")


def predict_ym(model, D, rows):
    """Model Y: the transformer's map correction (EOF coefficients) for rows, float16."""
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(rows), 256):
            run_model(model, D, torch.tensor(rows[b:b + 256], device=DEV), False); out.append(model.last_ym)
    return torch.cat(out).cpu().numpy().astype("float16")


def train(cfg, D, tr_rows, seed, eval_rows=None, eval_every=5, init_state=None):
    torch.manual_seed(seed); np.random.seed(seed)
    shapes = {f: tuple(FIELDS[f].shape[1:]) for f in FIELDS if f not in cfg.get("drop_fields", [])}   # file 42: groups left out of the transformer
    global CIC_T
    model = ProfA(cfg, shapes, lt_dims=[x_.shape[1] for x_ in D.LT] if cfg.get("lin_tok") else ()).to(DEV); MODEL_REF[0] = model; CIC_T = CIC_T.to(DEV)
    if cfg.get("skip"):
        Fe, Fb_ = D.SK.shape[2], D.SKB.shape[1]
        model.skip_e = nn.Parameter(torch.zeros(NL, Fe, 2, device=DEV)); model.skip_eb = nn.Parameter(torch.zeros(NL, 2, device=DEV))
        model.skip_b = nn.Parameter(torch.zeros(NL, Fb_, 2, device=DEV)); model.skip_bb = nn.Parameter(torch.zeros(NL, 2, device=DEV))
        sk_ids = {id(p) for p in (model.skip_e, model.skip_eb, model.skip_b, model.skip_bb)}
        if init_state is not None:                                       # file 39 (config C): pre-trained transformer weights; the
            sd = {k: v for k, v in init_state.items() if not k.startswith("skip_")}   # linear skips are fitted below on this fold's years
            miss = model.load_state_dict(sd, strict=False); print(f"loaded pre-trained weights ({len(sd)} tensors); missing {len(miss.missing_keys)}", flush=True)
        if cfg.get("warm"):
            warm_skip(model, D, tr_rows)
        groups = [{"params": [p for p in model.parameters() if id(p) not in sk_ids], "lr": cfg.get("lr", 5e-4), "weight_decay": cfg["wd"]}]
        if cfg.get("skip_freeze"):
            for p in (model.skip_e, model.skip_eb, model.skip_b, model.skip_bb):
                p.requires_grad_(False)
        else:
            groups.append({"params": [model.skip_e, model.skip_eb, model.skip_b, model.skip_bb], "lr": cfg.get("skip_lr", 5e-3), "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 5e-4), weight_decay=cfg["wd"])
    E = cfg["epochs"]; nb = int(np.ceil(len(tr_rows) / cfg["batch"]))
    mlr = [g_["lr"] for g_ in opt.param_groups]
    if cfg.get("sched") == "flat":                                      # file 43: linear rise over the first epoch, then constant
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s_: min(1.0, 0.1 + 0.9 * s_ / nb))
    else:
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=mlr if len(mlr) > 1 else mlr[0], total_steps=max(1, E * nb), pct_start=0.1, anneal_strategy="cos")
    if cfg.get("mult"):                                                 # file 43: fixed projection of the BSISO skip inputs (unit variance)
        model.register_buffer("mU", mult_basis(model, D, tr_rows, int(cfg["mult"])))
    ema = None
    if cfg.get("ema", 0) > 0:                                           # file 43: exponential moving average of the weights, every step
        dec_ = float(cfg["ema"]); ema = torch.optim.swa_utils.AveragedModel(model, avg_fn=lambda a_, p_, n_: dec_ * a_ + (1 - dec_) * p_)
    snaps, hist = {}, []; mscores = {}; ysn = {}; imp = {}; snaps_raw = {}; wsn = {}; csn = {}

    def step_loss(ii, jj):
        L, parts = loss_fn(run_model(model, D, ii, True), D, ii, cfg)
        if jj is not None:                                               # + a batch of off-season days, BSISO loss only (file 33)
            DY = YDATA[0][0]; oy = run_model(model, DY, jj, True)
            L = L + cfg["lam"] * cfg.get("allyr_w", 1.0) * (((oy[..., 2:4] - DY.Yp[jj]) / DY.pcsd) ** 2).mean() + aux_loss(model, DY, jj, cfg, ere=False)
        return L, parts
    if cfg.get("warm") and eval_rows is not None:                       # epoch 0 = the closed-form linear model (transformer output is 0)
        snaps[0] = predict(model, D, eval_rows)
        if cfg.get("ymap"):
            ysn[0] = predict_ym(model, D, eval_rows)
        if cfg.get("wk"):
            wsn[0] = predict_w(model, D, eval_rows)
            if hasattr(model, "head_c"):
                csn[0] = predict_c(model, D, eval_rows)
    swa = torch.optim.swa_utils.AveragedModel(model) if cfg.get("swa", 0) > 0 else None
    if cfg.get("pre_ep", 0) > 0:                                         # TX round: pretraining stage (same optimiser, no scheduler step)
        pcfg = dict(cfg, dense=cfg.get("pre_dense", 1.0), pen_b=0.0, pen_e=0.0)

        def bl_(o_, DD, r_):                                               # BSISO loss (JJAS rows: masked; off-season rows: all valid)
            pc_ = o_[..., 2:4] + (DD.Pc[r_] if hasattr(DD, "Pc") else 0.0); m_ = DD.Mp[r_] if hasattr(DD, "Mp") else torch.ones_like(pc_)
            return ((((pc_ - DD.Yp[r_]) / DD.pcsd) ** 2) * m_).sum() / m_.sum().clamp(min=1)
        for pe in range(cfg["pre_ep"]):
            model.train(); perm = np.random.permutation(tr_rows); pl_ = []
            for b in range(nb):
                ii = torch.tensor(perm[b * cfg["batch"]:(b + 1) * cfg["batch"]], device=DEV)
                if len(ii) == 0:
                    continue
                o_ = run_model(model, D, ii, True); L = cfg["lam"] * bl_(o_, D, ii) + aux_loss(model, D, ii, pcfg, ere=False)
                if cfg.get("allyr") and YDATA[0] is not None:
                    DY = YDATA[0][0]; yrows = YDATA[0][1]; jj = torch.as_tensor(yrows[np.random.randint(0, len(yrows), cfg["allyr_b"])], device=DEV)
                    oy = run_model(model, DY, jj, True); L = L + cfg["lam"] * cfg.get("allyr_w", 1.0) * bl_(oy, DY, jj) + aux_loss(model, DY, jj, pcfg, ere=False)
                opt.zero_grad(); L.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); pl_.append(L.item())
                if ema is not None:
                    ema.update_parameters(model)
            print(f"  pretraining epoch {pe + 1}: loss {np.mean(pl_):.4f}", flush=True)
    if cfg.get("pretrain", 0) > 0:                                       # stage 1: dense targets only (rain maps + BSISO), training years
        popt = torch.optim.AdamW(model.parameters(), lr=cfg.get("plr", 3e-4), weight_decay=cfg["wd"])
        for pe in range(cfg["pretrain"]):
            model.train(); perm = np.random.permutation(tr_rows)
            for b in range(nb):
                ii = torch.tensor(perm[b * cfg["batch"]:(b + 1) * cfg["batch"]], device=DEV)
                if len(ii) == 0:
                    continue
                L, _ = loss_fn(run_model(model, D, ii, True), D, ii, cfg, ere=False)
                popt.zero_grad(); L.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); popt.step()
        with torch.no_grad():                                              # stage 2 starts from zero ERE corrections again
            model.head.weight[:2].zero_(); model.head.bias[:2].zero_()
    for ep in range(1, E + 1):
        model.train(); perm = np.random.permutation(tr_rows); tl = []
        for b in range(nb):
            ii = torch.tensor(perm[b * cfg["batch"]:(b + 1) * cfg["batch"]], device=DEV)
            if len(ii) == 0:
                continue
            jj = None
            if cfg.get("allyr") and YDATA[0] is not None:
                yrows = YDATA[0][1]; jj = torch.as_tensor(yrows[np.random.randint(0, len(yrows), cfg["allyr_b"])], device=DEV)
            L, parts = step_loss(ii, jj); opt.zero_grad(); L.backward()
            if cfg.get("sam", 0) > 0:                                      # file 43: sharpness-aware step (gradient at the perturbed weights)
                ps_ = [p for p in model.parameters() if p.grad is not None]
                if cfg.get("sam_ad"):                                      # adaptive: scale-invariant perturbation
                    tw_ = [p.detach().abs() + 0.01 for p in ps_]; gn_ = torch.sqrt(sum(((t_ * p.grad) ** 2).sum() for t_, p in zip(tw_, ps_))) + 1e-12
                    eps_ = [cfg["sam"] * t_ * t_ * p.grad / gn_ for t_, p in zip(tw_, ps_)]
                else:
                    gn_ = torch.sqrt(sum((p.grad ** 2).sum() for p in ps_)) + 1e-12; eps_ = [cfg["sam"] * p.grad / gn_ for p in ps_]
                with torch.no_grad():
                    for p, e_ in zip(ps_, eps_):
                        p.add_(e_)
                L2_, _ = step_loss(ii, jj); opt.zero_grad(); L2_.backward()
                with torch.no_grad():
                    for p, e_ in zip(ps_, eps_):
                        p.sub_(e_)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); tl.append(parts)
            if ema is not None:
                ema.update_parameters(model)
        hist.append(np.mean(tl, 0))
        pm = model if ema is None else ema.module
        if ema is not None and eval_rows is not None and (ep % eval_every == 0 or ep == E):
            snaps_raw[ep] = predict(model, D, eval_rows)                  # the un-averaged model, to separate the effect of averaging
        if swa is not None and ep >= cfg["swa"]:                            # uniform weight average over epochs swa..ep
            swa.update_parameters(model); pm = swa.module
        if eval_rows is not None and (ep % eval_every == 0 or ep == E):
            snaps[ep] = predict(pm, D, eval_rows)
            if ep in cfg.get("imp_eps", []):                             # file 41: importance test on the held-out rows
                er = np.asarray(eval_rows); pmut = np.arange(len(D.bseq)); pmut[er] = er[np.random.default_rng(1000 + ep).permutation(len(er))]
                IMP_MASK["perm"] = torch.as_tensor(pmut, device=DEV)
                for g_ in ["bsiso", "rain"] + list(pm.fnames) + ["slow"]:
                    IMP_MASK["g"] = g_; imp[(g_, ep)] = predict(pm, D, eval_rows)
                IMP_MASK["g"] = None
            if cfg.get("ymap"):
                ysn[ep] = predict_ym(pm, D, eval_rows)
            if cfg.get("wk"):
                wsn[ep] = predict_w(pm, D, eval_rows)
                if hasattr(pm, "head_c"):
                    csn[ep] = predict_c(pm, D, eval_rows)
            if cfg.get("map", 0) > 0:
                MODEL_REF[0] = pm; mscores[ep] = map_scores(pm, D, eval_rows); MODEL_REF[0] = model
        if ep >= cfg.get("stop", E):                                     # final fits: same schedule as CV, stopped at the chosen epoch
            break
    train.mscores = mscores; train.ysnaps = ysn; train.imp = imp; train.snaps_raw = snaps_raw; train.wsnaps = wsn; train.csnaps = csn
    return pm, snaps, np.array(hist)


# ---------------------------------------------------------------- modes
def nparams(m):
    return sum(p.numel() for p in m.parameters())


if MODE == "lin":                                                      # linear link: l2 grid on all 5 folds (out-of-fold)
    GRID = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]; res = []
    for fi, blk in enumerate(BLOCKS):
        tr = np.array([y for y in TRAIN_Y if y not in blk]); B = build(tr); va = np.isin(IY, list(blk))
        for l2 in GRID:
            La, Ll, Pc = fit_linear(B, B["trs"], l2, l2, l2, chain_alpha=envlist("ALPHA_CHAIN", "100"))
            pa, pl = probs(La, Ll); pca, pcl = probs(B["OFF_any"], B["OFF_lg"])
            res.append({"fold": fi, "l2": l2, "rps": np.nanmean(rps(pa, pl, B["Yc"])[va], 0), "rps_clim": np.nanmean(rps(pca, pcl, B["Yc"])[va], 0),
                        "pc_mse": np.nanmean(((Pc - AUX) ** 2)[va], (0, 2))})
        print(f"fold {fi} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pd.to_pickle(res, os.path.join(OUTD, "lin_cv.pkl" if CHAIN == "none" else f"lin_cv_{CHAIN}.pkl"))
    R_ = pd.DataFrame(res)
    for l2, g in R_.groupby("l2"):
        sk = 1 - np.stack(g.rps).mean(0) / np.stack(g.rps_clim).mean(0)
        print(f"l2 {l2:7.0f}: RPSS days 1-7 {sk[:7].mean():.4f}  days 8-14 {sk[7:].mean():.4f}  PC MSE d8-14 {np.stack(g.pc_mse).mean(0)[7:].mean():.3f}")

if MODE == "cv":
    fi = int(os.environ.get("FOLD", "0")); SEEDL = [int(x) for x in os.environ.get("SEED", "0").split(",")]; L2 = envlist("L2", "300")
    blk = BLOCKS[fi]; tr = np.array([y for y in np.arange(CFG.get("train_start", 1981), 2014) if y not in blk])
    PT_STATE = {}
    if CFG.get("pretrain_years"):                                       # config C: pre-train on years that are never scored
        p0, p1 = CFG["pretrain_years"]; ptY = np.arange(p0, p1 + 1); assert not set(ptY) & set(range(1981, 2025))
        Bp = build(ptY); fit_linear(Bp, Bp["trs"], L2, L2, envlist("ALPHA", "100"))      # also makes the BSISO-chain input the skip needs
        linp = (Bp["OFF_any"].copy(), Bp["OFF_lg"].copy(), np.zeros((len(ISS), NL, 2), "float32")); Dp = Data(Bp, linp, CFG)
        if CFG.get("allyr"):
            yp_ = np.arange(len(Bp["YT"])); YDATA[0] = (DataY(Bp, CFG), yp_[::20] if FAST else yp_)
        rp = np.where(Bp["trs"])[0]; rp = rp[::20] if FAST else rp
        for seed in [int(x) for x in os.environ.get("SEED", "0").split(",")]:
            mp_, _, _ = train(dict(CFG, stop=CFG.get("pt_epochs", 3)), Dp, rp, 500 + seed)
            PT_STATE[seed] = {k: v.detach().clone() for k, v in mp_.state_dict().items()}
            print(f"pre-trained on {p0}-{p1}: {len(rp)} issue days, {CFG.get('pt_epochs', 3)} epochs (seed {seed})", flush=True)
        del Bp, Dp; YDATA[0] = None
    if CFG.get("yfrac", 1.0) < 1.0:                                    # learning curve: fixed random subset of the training years
        tr = np.sort(np.random.default_rng(100 + fi).choice(tr, int(round(CFG["yfrac"] * len(tr))), replace=False)); print("training years", tr.tolist(), flush=True)
    B = build(tr)
    lin = fit_linear(B, B["trs"], L2, L2, envlist("ALPHA", "100"))
    if CFG.get("wk"):                                                  # round W3: frozen weekly linear parts (training rows only)
        weekly_fit(B, B["trs"])
    if CFG.get("nolin"):
        lin = (B["OFF_any"].copy(), B["OFF_lg"].copy(), np.zeros((len(ISS), NL, 2), "float32"))
    va_rows = np.where(np.isin(IY, list(blk)))[0]; tr_rows = np.where(B["trs"])[0]
    if FAST:
        tr_rows = tr_rows[::20]; va_rows = va_rows[::10]
    for f, A_ in FIELDS.items():
        assert np.isfinite(A_[W30[np.r_[tr_rows, va_rows]]]).all(), f"missing {f} data in the cross-validation windows"
    MAPY_KEY[0] = f"f{fi}"; D = Data(B, lin, CFG)
    if CFG.get("allyr"):
        yr_ = np.arange(len(B["YT"])); YDATA[0] = (DataY(B, CFG), yr_[::20] if FAST else yr_)
    LMS = lin_map_scores(B, tr_rows, va_rows) if CFG.get("map", 0) > 0 and CFG.get("wk") else {}
    if LMS:
        print("linear daily rain-map baseline (lead: BSS, pattern corr):", {k_: tuple(round(v_, 3) for v_ in LMS[k_][:2]) for k_ in (1, 2, 3, 5, 7, 10, 14)}, flush=True)
    for seed in SEEDL:                                                 # several seeds share one fold build
        t1 = time.time(); model, snaps, hist = train(CFG, D, tr_rows, seed, va_rows, CFG.get("eval_every", 5), init_state=PT_STATE.get(seed))
        print(f"CV {CFGN} fold {fi} seed {seed}: {nparams(model)} params, {CFG['epochs']} epochs in {(time.time() - t1) / 60:.1f} min", flush=True)
        rec = {"cfg": CFGN, "CFG": CFG, "fold": fi, "seed": seed, "va_rows": va_rows, "snaps": snaps, "hist": hist, "nparams": nparams(model),
               "lin": [x[va_rows] for x in lin], "Yc": B["Yc"][va_rows], "OFF": (B["OFF_any"][va_rows], B["OFF_lg"][va_rows]), "AUX": AUX[va_rows], "PERS": X_BS[va_rows, -1, :2],
               "IY": IY[va_rows], "ECOK": ECOK[va_rows], "map_scores": getattr(train, "mscores", {}), "ym_snaps": getattr(train, "ysnaps", {}), "imp": getattr(train, "imp", {}),
               "snaps_raw": getattr(train, "snaps_raw", {}), "lin_map_scores": LMS}
        if CFG.get("wk"):                                               # round W3: weekly heads (corrections by epoch + frozen parts)
            rec["w_snaps"] = train.wsnaps; rec["c_snaps"] = train.csnaps
            rec["W"] = {k_: B[k_][va_rows] for k_ in ("YWa", "YWl", "YWc", "OFFW", "LW", "VW", "RW", "GW") + WK_TC_KEYS}
            rec["W"].update(ZW=B["ZW"][va_rows].astype("float16"), LOCW=B["LOCW"][va_rows].astype("float16"), V=B["V"], wsc=B["wsc"],
                            doy=DOY[ISS[va_rows]])
        pd.to_pickle(rec, os.path.join(OUTD, f"cv_{CFGN}_f{fi}_s{seed}{'_fast' if FAST else ''}.pkl"))
        La, Ll, Pc = rec["lin"]; pa0, pl0 = probs(La, Ll); r0 = np.nanmean(rps(pa0, pl0, rec["Yc"]))
        for ep, o in snaps.items():
            pa, pl = probs(La + o[..., 0], Ll + o[..., 1]); r = np.nanmean(rps(pa, pl, rec["Yc"]))
            m0 = np.nanmean((Pc - rec["AUX"]) ** 2); m1 = np.nanmean((Pc + o[..., 2:] - rec["AUX"]) ** 2)
            print(f"  epoch {ep:3d}: RPS linear {r0:.5f} -> deep {r:.5f} ({100 * (r0 - r) / r0:+.2f}%) | PC MSE {m0:.4f} -> {m1:.4f}", flush=True)

SEED_BASE = int(os.environ.get("SEED_BASE", "1000"))                   # final-fit seeds start here (seed check: 2000)
if MODE == "final":
    L2 = envlist("L2", "300"); E = int(os.environ.get("EPOCHS", str(CFG["epochs"]))); NS = int(os.environ.get("SEEDS", "10"))
    TRAIN_Y = np.arange(CFG.get("train_start", 1981), 2014)
    cfg = dict(CFG, stop=E); B = build(TRAIN_Y)                       # schedule length = CV's cfg epochs; stop at the chosen epoch
    lin = fit_linear(B, B["trs"], L2, L2, envlist("ALPHA", "100"))
    if CFG.get("wk"):
        weekly_fit(B, B["trs"])
    if CFG.get("nolin"):
        lin = (B["OFF_any"].copy(), B["OFF_lg"].copy(), np.zeros((len(ISS), NL, 2), "float32"))
    te_rows = np.where(np.isin(IY, TEST_Y))[0]; tr_rows = np.where(B["trs"])[0]
    if CFG.get("ec"):
        assert ECOK[te_rows].all(), "Prof B: a test sample has no ECMWF start on T or T-1"
    for f, A_ in FIELDS.items():                                       # every test input must exist (e.g. CDS 2023 vorticity)
        assert np.isfinite(A_[W30[np.r_[te_rows, tr_rows]]]).all(), f"missing {f} data in the training or test windows"
    # control: deep inputs from another training year, same issue day of year (linear link stays real)
    rng = np.random.default_rng(123); iss_doy = DOY[ISS]; perm = np.arange(len(ISS))
    for n in range(len(ISS)):
        cand = np.where(B["trs"] & (iss_doy == iss_doy[n]) & (IY != IY[n]))[0]
        if len(cand):
            perm[n] = rng.choice(cand)
    out = {"cfg": CFGN, "CFG": cfg, "te_rows": te_rows, "lin": [x[te_rows] for x in lin], "Yc": B["Yc"][te_rows],
           "OFF": (B["OFF_any"][te_rows], B["OFF_lg"][te_rows]), "AUX": AUX[te_rows], "PERS": X_BS[te_rows, -1, :2], "deep": [], "ctrl": [], "IY": IY[te_rows]}
    if CFG.get("wk"):                                                  # round W3 test: weekly heads on the test rows (+ training-row classes for the reference)
        out["W"] = {k_: B[k_][te_rows] for k_ in ("YWa", "YWl", "YWc", "OFFW", "LW", "VW", "RW", "GW") + WK_TC_KEYS}
        out["W"].update(ZW=B["ZW"][te_rows].astype("float16"), LOCW=B["LOCW"][te_rows].astype("float16"), V=B["V"], wsc=B["wsc"], doy=DOY[ISS[te_rows]])
        out["W_train"] = dict(YWc=B["YWc"][tr_rows], doy=DOY[ISS[tr_rows]], IY=IY[tr_rows])
    if CFG.get("allyr"):
        YDATA[0] = (DataY(B, CFG), np.arange(len(B["YT"])))
    MAPY_KEY[0] = "final"
    for s in range(NS):
        for tag, P in (("deep", None), ("ctrl", perm)):
            if tag == "ctrl" and s >= int(os.environ.get("CTRL_SEEDS", "5")):
                continue
            D = Data(B, lin, cfg, P)
            if os.environ.get("FINAL_SNAPS") == "1":
                model, sn, _ = train(cfg, D, tr_rows, SEED_BASE + s, te_rows, 1); out.setdefault(tag + "_snaps", []).append(sn)
                if cfg.get("ymap"):
                    out.setdefault(tag + "_ym_snaps", []).append(train.ysnaps)
                if cfg.get("wk"):
                    out.setdefault(tag + "_w_snaps", []).append(train.wsnaps); out.setdefault(tag + "_c_snaps", []).append(train.csnaps)
            else:
                model, _, _ = train(cfg, D, tr_rows, SEED_BASE + s)
            out[tag].append(predict(model, D, te_rows)); print(f"final {tag} seed {s} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
            pd.to_pickle(out, os.path.join(OUTD, f"final_{CFGN}{'_smoke' if os.environ.get('SMOKE_FINAL') == '1' else ''}.pkl"))
print(f"total {(time.time() - t0) / 60:.1f} min")
