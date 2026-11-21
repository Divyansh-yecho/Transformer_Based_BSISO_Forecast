# Model code

These are the files of the Kaggle dataset `divyanshyecho/profa-inputs-ext`, exactly as the final runs used them. The
notebooks in `models/*/notebooks/` find them in `/kaggle/input/` and run `profA.py` once per fold and seed.

`profA.py` holds every config ever tested, so it is long. To read one model, start with `models/logistic_transformer/model.py`
or `models/transformer_only/model.py`: each holds only the network, loss and training of that model, with the same weight
names, and `models/check_equivalence.py` checks that they match this file.

| File | What it does |
|---|---|
| `profA.py` | the whole model: data building (anomalies, EOFs, targets, thresholds from the training years of each fold), the linear fits (`warm_skip`, `fit_off`, the weekly ridge), the network (`ProfA`), the losses, training (`train`) and the run modes |
| `kaggle_modelA.py` | shared start-up run by `profA.py`: loads the inputs, builds the extreme-rain classes and the issue days |
| `profA_eval.py` | shared scoring helpers: bivariate correlation, useful days, RPS, calibration, season bootstrap |
| `linear_choice_fixed.json` | the fixed choices of the linear models |

## Configs

All settings are in the `CFGS` dictionary at the top of `profA.py`. Each config builds on an earlier one; the notes below the
dictionary say what every round tested. The two final models:

| Config | Built from | Model |
|---|---|---|
| `W3N` | `HXCP` (tokens, encoder, frozen linear parts) -> `_T` (flat learning rate, EMA) -> `_R` (size penalties, noise, dropout) -> `W3N` (weekly map and ERE heads) | Logistic + Transformer |
| `TXM` | `W3N` -> `_TO` (no linear parts) -> `_TA` (all input fields, learning rate 5e-4, 40 epochs) -> `TXM` (linear inputs as tokens, daily map head) | Transformer alone |

The other configs in the dictionary are earlier tests. They are kept so that every round can be re-run; see each config's
comment for what it tried.

## Running

`profA.py` is driven by environment variables, which the notebooks set:

| Variable | Meaning |
|---|---|
| `CFG` | config name, for example `W3N` |
| `MODE` | `lin` (linear parts only), `cv` (one fold of cross-validation) or `final` (fit on 1960-2013, forecast the test years) |
| `FOLD`, `SEED` | fold (0-4) and seed(s) for `cv`; `SEEDS` and `EPOCHS` for `final` |
| `FAST=1` | a two-epoch smoke test |
| `PROFA_IN`, `PROFA_OLR`, `MODELA_SRC`, `PROFA_OUT` | input folder, OLR file, path of `kaggle_modelA.py`, output folder |
| `MIN_ISS_YEAR` | first training year (1960) |

Each run writes a record (`cv_<CFG>_f<fold>_s<seed>.pkl` or `final_<CFG>.pkl`) with the forecasts at every epoch
(`snaps`, `w_snaps`), the targets and the held-out rows.
