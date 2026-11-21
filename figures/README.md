# Figures

The scripts in [`scripts/`](scripts/) make these figures. Diagram scripts need no data; the result figures read the Kaggle run
records and tables from the folders in [`../paths.py`](../paths.py).

## The BSISO and Indian rain

| Figure | Shows | Script |
|---|---|---|
| `bsiso_phase_diagram.png` | the BSISO1 phase diagram (PC2, PC1) with phases 1-8 | APCC monitoring plot |
| `rain_anomaly_phase_maps.png` | IMERG rain anomaly by BSISO1 phase, 2000-2024 | `background/BSISO-Rain` notebook |
| `p_ere_given_phase_screenshot.png` | chance of an extreme-rain day over central India in each phase | earlier analysis notebook |
| `active_spell_tracks.png`, `break_spell_tracks.png` | BSISO path during long active and break spells | earlier analysis notebook |

## Inputs and tokens

| Figure | Shows | Script |
|---|---|---|
| `raw_inputs.png` | the six raw input maps on one day | `report_raw_inputs.py` |
| `eof_diagram.png` | a daily map written as 20 EOF coefficients | `report_eof_diagram.py` |
| `token_diagram.png` | how 30 days of one field become 10 tokens | `report_token_diagram.py` |
| `tm_token_matrix.png`, `tm_position_embedding.png` | the 80 x 32 token matrix and how one row is built | `report_token_matrix_figs.py` |

## The network

| Figure | Shows | Script |
|---|---|---|
| `7_transformer_schematic.png` | the whole model: inputs, tokens, encoder, heads | `poster_transformer_schematic.py` |
| `enc_layer_flow.png`, `attn_steps.png`, `gelu_curve.png` | one encoder layer, attention for one query, the GELU activation | `report_encoder_figs.py` |
| `head_bsiso_flow.png`, `head_rain_flow.png` | from the encoder output to the BSISO forecast and to the weekly rain maps | `report_heads_flow_figs.py` |
| `logistic_plus_transformer.png` | the frozen linear models next to the transformer | `report_logistic_ere_figs.py` |
| `ere_head_flow.png` | the extreme-rain head and its three classes | `report_logistic_ere_figs.py` |

## Training and results

| Figure | Shows | Script |
|---|---|---|
| `loss_curves_report.png` | training and held-out loss of the transformer alone, 60 epochs | `loss_curves_report.py` |
| `ere_loss_terms.png` | the three held-out loss terms; the ERE term overfits from epoch 3 | `report_logistic_ere_figs.py` |
| `10_bsiso_correlation_transformer.png` | test BSISO correlation by lead: both models, persistence, ECMWF | `poster_transformer_figs.py` |
| `9_rain_maps_transformer_20080613.png` | forecast and observed weekly rain maps for one issue day | `poster_transformer_figs.py` |
| `13_ere_skill_bars.png` | weekly ERE skill of the models and ECMWF | `poster_ere_figs.py` |
| `15_ere_one_summer.png` | the weekly ERE forecasts of the 2023 monsoon | `poster_ere_figs.py` |
