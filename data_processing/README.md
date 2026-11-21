# Data processing

These scripts turn the raw downloads into the model input files, and turn the ECMWF reforecasts into the benchmark
(the ECMWF BSISO forecast and ECMWF rain-anomaly maps). Run them in the order below. Folders come from
[`../paths.py`](../paths.py); the datasets and their sources are in [`../data/README.md`](../data/README.md).
`common.py` holds shared helpers: the central-India box, reading the official BSISO index, the BSISO phase and the
extreme-rain cluster code.

## 1. Downloads (`1_download/`)

| Script | What it fetches |
|---|---|
| `imd_fetch_early.py` | IMD 0.25° daily rain, 1951-1978 (1979-2024 come from the same IMD page) |
| `era5_wb2_fetch.py` | ERA5 daily means from WeatherBench2, April-September 1979-2022, 30°S-60°N 0-180°E: MSLP, precipitable water, SST at 1.5°; u850, u200 at 5.625° |
| `era5_cds_2024.py` | the same fields for 2023 and 2024 from the Copernicus Climate Data Store (2022 too, as overlap to correct the CDS sampling) |
| `era5_wb2_fetch_ext.py`, `era5_inputs_fetch.py` | the same fields for 1959-1978 and for the off-season months (`python era5_inputs_fetch.py early_inputs 1959 1978`) |
| `era5_new_inputs_fetch.py`, `era5_cds_z850.py` | precipitable water, SST and 850 hPa height for the extended calendar; z850 for 2022-2024 from the CDS |
| `ncep_fetch_bsiso.py` | NCEP Reanalysis 1 OLR and u850, 1959-2022, to rebuild the BSISO index before 1981 |

All downloads log to a file next to their output and skip files that are already complete, so they can be restarted.

## 2. Model inputs (`2_model_inputs/`)

Outputs of steps a, b and f go to the raw-data folder (`BSISO_DATA`), the others to `BSISO_WORK`.

| Step | Script | Output | What it does |
|---|---|---|---|
| a | `imd_convert.py` | NetCDF of IMD rain 1979-2024 | reads the IMD binary files, puts them on the IMERG 0.25° lattice and shifts each date back one day (IMD days end at 08:30 IST) |
| b | `olr_merge.py` | `olr_merged/` | NOAA interpolated OLR to 2022, CPC blended OLR after, corrected to the first on the 1991-2022 overlap |
| c | `modelA_prep_base.py` | `modelA/base_inputs.npz` | IMD rain as 4x4 block means on the 283 Indian 1° cells, the rain band (1° rows 6-31°N averaged over 70-95°E), central-India rain, the BSISO index, Niño3.4, DMI, and the past-only BSISO forecasts of an earlier ridge model |
| d | `profA_prep_maps.py` | `profA/maps.npz` | the daily field maps on the model grids: OLR (2.5°, 10°S-30°N 40-160°E), u850 and u200 (5.625°, 10°S-30°N 40-120°E), MSLP (1.5°, 10-30°N 70-95°E) |
| e | `ext_prep_imd.py`, `ext_prep_era5.py` | `ext/` | the same inputs for 1951/1959-1978 |
| f | `ext_olr_bsiso.py` | `bsiso_extended/BSISO_extended_1959_2026.csv`, OLR proxy | the BSISO index before 1981, rebuilt from NCEP OLR and u850 (bivariate correlation 0.967 with the official index on 2014-2022), and an OLR proxy before 1979 |
| g | `ext_assemble.py` | `modelA_ext/base_inputs.npz`, `profA_ext/maps.npz` | joins everything into April-September 1959-2024 in the same file formats |
| h | `ext_build_maps_yr.py` | `profA_ext/maps_yr.npz` | year-round fields for the off-season BSISO samples |
| i | `ext_prep_new_fields.py` | `profA_ext/maps_new.npz` | precipitable water, SST and z850 on the extended calendar |

Only raw values are saved. Anomalies, scaling, EOFs and the extreme-rain thresholds are computed later inside the model code,
from the training years of each cross-validation fold.

## 3. ECMWF benchmark (`3_ecmwf_benchmark/`)

ECMWF does not issue a BSISO index or our rain maps, so both are computed from its reforecasts the same way as for the
observations. ECMWF is only compared with; it is never an input.

| Script | Output | What it does |
|---|---|---|
| `s2s_download_night.py` | `s2s_ecmwf/` | downloads the S2S reforecasts: u850, u200 and OLR over 10°S-40°N 40-160°E, rain over 5-35°N 65-100°E, control and 10 perturbed members |
| `profB_prep_ecmwf_rain.py` | `profB/ecmwf_rain.npz` | ensemble-mean daily rain map and the central-India rain of every member, per start and lead |
| `ecmwf_bsiso.py` | `maphead/ecmwf_bsiso.csv`, `ecmwf_bsiso_forecasts.npz` | the ECMWF BSISO forecast (below) and its correlation on the test issue days |
| `w3_ecmwf_oof.py` | `maphead/w3_ecmwf_oof_*.csv` | our out-of-fold forecasts against ECMWF on 2005-2013, the cross-validation years that the reforecasts cover |

**ECMWF BSISO.** The forecast OLR and u850 are turned into PC1-PC4 with the same mapping that rebuilds the official index
from observations (it reproduces BSISO1 at 0.988 on 2014-2022):
1. For forecast day D = S + k (start S): u850 is the mean of steps k and k+1, and OLR comes from the accumulated thermal
   radiation, -(ttr[k+1] - ttr[k]) / 86400. Both are averaged over the 11 members.
2. Anomaly = forecast minus ECMWF's own climatology for that lead and calendar day (±15 days, leave one year out over 2005-2024).
3. The mean of the previous 120 days is removed, using observed anomalies before S and forecast anomalies from S on.
4. Then come the same scaling, EOFs, ridge and variance factors as for the observed index.

**ECMWF rain-anomaly maps** (computed in `w3_ecmwf_oof.py` and in `models/logistic_transformer/test_eval.py`):
1. Ensemble-mean rain (2.5°) is interpolated bilinearly to the 283 IMD cells and log-transformed, log(1 + rain).
2. Anomaly from ECMWF's own climatology per lead, cell and calendar day (±15 days, leave one year out over 2005-2024),
   divided by its spread. This matches how the observed target maps are standardised.
3. Week 1 and week 2 are the means of leads 1-7 and 8-14. A scale factor per week is fitted on 2005-2013. The ACC does not
   depend on it, but the MSSS does.

**ECMWF weekly ERE.** A logistic model with our weekly normal-chance offset, on ECMWF's week-mean central-India rain
(log member-mean rain, the fraction of members above the 2005-2013 90th percentile, log largest-cell rain). It is fitted on
2005-2013 and applied to the test years.

Issue days are matched to an ECMWF start on the same day, or the day before when there is none (starts are every second day).
The reforecasts come from today's ECMWF model version, which flatters ECMWF a little in the early test years.
