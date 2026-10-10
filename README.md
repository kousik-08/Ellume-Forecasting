# LLume Forecast — Solar PV Day-Ahead Power Forecasting

This project forecasts hourly AC power output (kW) for a solar PV plant in Tumakuru, Karnataka. The model uses the four supplied day-ahead weather forecasts plus timestamp-derived features. The final predictor is an equal-weight ensemble of `HistGradientBoostingRegressor` and `RandomForestRegressor`.

## Quick start

Use Python 3.12 in a virtual environment if that is your installed version. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe solve.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If the `.venv` already exists, do not recreate it unless necessary. If Windows Application Control blocks a Python executable, use an approved installed interpreter or ask IT to authorize it; do not bypass organizational controls.

## How to navigate the code

Open `solve.py` and inspect these functions in order:

| Function / section | Responsibility |
|---|---|
| Constants near the top | Defines target, raw forecast columns, engineered features, and 10,000 kW capacity |
| `parse_timestamps()` | Parses ISO and day-first timestamps |
| `add_features()` | Adds cyclical time features, approximate solar elevation, and weather/solar interactions |
| `prepare_data()` | Cleans labels and inputs, records data audit, handles suspected unit mismatches, consolidates duplicate timestamps |
| `make_hgb()` / `make_rf()` | Model pipelines with median imputation |
| `make_final_model()` | Equal-weight ensemble of the two pipelines |
| `MonthHourMedianBaseline` | Reference prediction by month and hour |
| `evaluate()` | Five chronological validation folds, metrics, OOF diagnostics, worst-fold analysis |
| `seasonal_holdout()` | Supplementary July-August 2024 historical holdout |
| `run()` | Fits final model, checks output contract, writes predictions and reports |
| `main()` | Reads `data/train.csv` and `data/test.csv` and runs the pipeline |

`app.py` imports the shared functions from `solve.py`; it should not duplicate the model implementation. The Streamlit dashboard evaluates the model when the user clicks **Evaluate Model & Generate Forecast**.

## Streamlit visual metrics

The dashboard provides six tabs:

1. **Model Comparison** — CV MAE comparison for HistGradientBoosting, Random Forest, their ensemble, and the month/hour baseline; fold-by-fold metrics.
2. **Actual vs Predicted** — daily actual/predicted power trends, validation scatter plot, and fold explorer. Uses historical out-of-fold data.
3. **Error Analysis** — absolute and signed error histograms, daily MAE, and worst-fold diagnostics.
4. **Generation Patterns** — average output by hour/month and hourly error patterns.
5. **Test Forecast** — test-period forecast line, daily average, date/hour filters, and CSV download.
6. **Metrics & Data** — detailed metrics, feature list, and downloads for fold metrics and out-of-fold diagnostics.

**Metric interpretation:** MAE is in kW; lower is better. A positive signed error means overprediction. A negative signed error means underprediction. R² is dimensionless and should be read alongside MAE and fold-to-fold variation.

## Current recorded validation results

Latest recorded run using five chronological `TimeSeriesSplit` folds:

- Ensemble mean fold MAE: **271.09 kW**.
- Ensemble fold MAE standard deviation: **146.05 kW**.
- Fold MAEs: **297.33, 505.44, 316.92, 86.29, and 149.47 kW**.
- Month/hour baseline mean fold MAE: **354.94 kW**.
- Mean fold MAE reduction vs. baseline: **23.62%**.
- Supplementary July-August 2024 holdout: ensemble MAE **485.91 kW**, baseline MAE **758.95 kW**.

These are historical validation metrics, not measured test-period accuracy. Fold performance varies substantially; see `cv_fold_metrics.csv` and `worst_fold_diagnostics.csv`.

## Deliverables

- `solve.py` — end-to-end preprocessing, features, evaluation, final fitting, predictions, and reports.
- `app.py` — interactive Streamlit metrics dashboard.
- `predictions.csv` — 1,488 test-period forecasts with columns `timestamp,predicted_ac_power_kw`.
- `experiment_report.json` — data audit, validation results, seasonal holdout, final prediction summary, and limitations.
- `cv_fold_metrics.csv` — validation fold date ranges and metrics.
- `oof_diagnostics.csv` — historical out-of-fold actuals, predictions, and absolute errors.
- `worst_fold_diagnostics.csv`, `worst_fold_monthly_summary.csv`, `worst_fold_hourly_summary.csv` — diagnostics for the hardest fold.
- `label_correction_sensitivity.json` — metric comparison with and without the suspected unit correction.
- `approach_note.txt` — technical approach and code guide.
- `walkthrough_script.txt` — presentation script for explaining the code and dashboard.
- `sensitivity_check.py` — optional run to quantify sensitivity to the label correction.

## Data-quality assumption that must be disclosed

The source has 230 historical records marked `RUN` with measured POA irradiance above 200 W/m² but target values between 0.1 and 20 in a kW column. These appear consistent with MW values stored as kW, so `prepare_data()` multiplies them by 1,000. This is an inference, not a verified source quality flag. Disabling the correction produced mean fold MAE around **412.77 kW**, versus **271.09 kW** with it enabled. Confirm the assumption with the source data owner before relying on the model.

The plant status and measured POA fields are used only for historical label-quality checks, not as predictive features, because they are not supplied for the test period.

## Important limitation

The test file has no actual power labels, so test-period MAE, RMSE, and R² cannot be computed. The reported 271.09 kW MAE is a chronological cross-validation estimate, not test-set accuracy. The final predictions are constrained to the plant capacity range of 0–10,000 kW.
