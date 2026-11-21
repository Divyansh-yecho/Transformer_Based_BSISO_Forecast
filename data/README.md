# Data

No data is stored in this repository. This page lists every dataset the models use, where it comes from and where the
scripts expect it. The three folders are set in [`../paths.py`](../paths.py):

| Folder | Variable | Holds |
|---|---|---|
| raw data | `BSISO_DATA` | downloads, as listed below |
| processed inputs | `BSISO_WORK` | the model input files built by `data_processing/`, and result tables |
| Kaggle outputs | `BSISO_RUNS` | `<ROUND>/out/out/` with the run records (`cv_<CFG>_f<fold>_s<seed>.pkl`, `final_<CFG>.pkl`) |

## Datasets

| Data | Used for | Period | Source and access | Script |
|---|---|---|---|---|
| IMD gridded rainfall, 0.25°, daily (Pai et al. 2014) | rain maps (283 cells at 1°), rain band, central-India rain, ERE targets | 1951-2024 | India Meteorological Department, Pune: binary files `ind<YYYY>_rfp25.grd` from imdpune.gov.in/cmpg/Griddata/rainfall.php | `imd_fetch_early.py`, `imd_convert.py` |
| ERA5 reanalysis (Hersbach et al. 2020) | MSLP, 850 hPa wind, precipitable water, SST, 850 hPa height | 1959-2024 | 1959-2022: the public WeatherBench2 copy on Google Cloud (`gs://weatherbench2/datasets/era5/`, anonymous access, daily means of the 6-hourly fields). 2023-2024: the Copernicus Climate Data Store (account and `~/.cdsapirc` key needed) | `era5_*_fetch*.py`, `era5_cds_z850.py` |
| Outgoing longwave radiation (OLR) | BSISO linear model, BSISO index | 1959-2024 | NOAA Interpolated OLR (1979-2022) and the CPC Blended OLR (2023-2024), merged by `olr_merge.py`; before 1979 a proxy calibrated on NCEP Reanalysis 1 | `olr_merge.py`, `ext_olr_bsiso.py` |
| NCEP/NCAR Reanalysis 1 | rebuilding the BSISO index before 1981 | 1959-2022 | NOAA PSL OPeNDAP server, only the needed level and region | `ncep_fetch_bsiso.py` |
| BSISO index (Lee et al. 2013) | inputs and BSISO targets | 1981-2024 | the official index from the APEC Climate Center (file `BSISO.INDEX.NORM.LY.data`) | read by `common.load_bsiso` |
| Niño3.4 and Dipole Mode Index | slow background state (FiLM) | monthly | NOAA CPC (`ersst5.nino.mth.91-20.ascii`) and NOAA PSL (`dmi.had.long.data`) | `modelA_prep_base.py` |
| GPM IMERG, 0.25° | the common lattice and India mask that the IMD rain is placed on | 2000-2024 | NASA GES DISC | `imd_convert.py` |
| ECMWF S2S reforecasts | **benchmark only**, never an input | hindcasts 2005-2024, model year 2025 | ECMWF Data Store, dataset `s2s-reforecasts` (account and key needed): starts every second day 25 May-21 September, 11 members, leads 0-21 days, 2.5° | `s2s_download_night.py` |

## Expected folder layout

Raw data (`BSISO_DATA`):

```
imd_rain_025/              IMD 0.25° binary files 1979-2024
imd_rain_025_early/        IMD 0.25° binary files 1951-1978
imerg_india_025_jjas_2000_2024.nc
kaggle_bsiso_extras/india_mask_025.npy
era5_wb2/                  WeatherBench2 daily means 1979-2022 (wb2_<year>.npz) and CDS files for 2023-2024
era5_wb2_ext/              1959-1978, off-season months, new fields, z850
olr_noaa_interp/olr.day.mean.nc,  olr.cbo-2.5deg.day.mean.nc,  olr_merged/
ncep_r1/                   NCEP R1 OLR and u850 per year
BSISO.INDEX.NORM.LY.data,  bsiso_extended/BSISO_extended_1959_2026.csv
official_indices/enso_cpc/ersst5.nino.mth.91-20.ascii,  official_indices/iod_psl/dmi.had.long.data
s2s_ecmwf/                 ECMWF reforecast GRIB files
```

Processed inputs (`BSISO_WORK`), built by `data_processing/2_model_inputs/` and `3_ecmwf_benchmark/`:

| File | Content |
|---|---|
| `modelA_ext/base_inputs.npz` | IMD rain on 1° cells, the rain band (25 latitude rows), central-India rain, BSISO PC1-4, Niño3.4, DMI; April-September 1959-2024 |
| `profA_ext/maps.npz` | daily OLR, u850, u200 and MSLP maps on the model grids |
| `profA_ext/maps_yr.npz` | year-round u850, u200, MSLP for the off-season samples |
| `profA_ext/maps_new.npz` | precipitable water, SST (1.5°, 30°S-30°N, 30-180°E) and 850 hPa height |
| `profB/ecmwf_rain.npz` | ECMWF ensemble-mean rain maps and central-India rain per start and lead |
| `maphead/ecmwf_bsiso.csv`, `maphead/ecmwf_bsiso_forecasts.npz` | ECMWF's BSISO forecast and its skill on the test days |

## The Kaggle dataset

The training runs read one private Kaggle dataset, `divyanshyecho/profa-inputs-ext`. It holds the four files of
[`../src/`](../src/) and these inputs: `base_inputs.npz`, `maps.npz`, `maps_yr.npz`, `maps_new.npz`, `maps_vo.npz`,
`maps_sm.npz`, `olr_daily_1959_2024_proxy_merged.nc` and `bsiso_pastonly_forecasts.parquet`. The last one holds past-only
BSISO forecasts of an earlier ridge model; `kaggle_modelA.py` uses them to build the "ERE chance from the BSISO five days
earlier" features of the extreme-rain logistic model. `maps_vo.npz` and `maps_sm.npz` (vorticity, soil moisture) are not
used by the two final models.

## Kaggle run outputs used by the scoring scripts

| Round | Configs | Used by |
|---|---|---|
| `W3B2` | W3N, 5 folds x 5 seeds | Logistic + Transformer cross-validation |
| `MANUAL_W3BF` | W3N, final fit, 10 seeds | Logistic + Transformer test |
| `TX1`, `TX2` | TXA, TXL / TXD, TXP, TXM | transformer-alone cross-validation |
| `TY1` | TYE, TYW (60 epochs) | loss curves of the transformer alone |
| `TXMF` | TXM, final fit, 10 seeds | transformer-alone test |
| `PH7CV` | A0 | the normal-chance reference for all ERE scores |
| `PT1B`, `TOCV`, `TOF`, `PSCA` | PT1R, W3T, G0 | comparison rows in the scoring tables |

The run records are large (0.2-1 GB per round) and are not in the repository.
