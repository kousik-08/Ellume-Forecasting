"""LLume Forecast: reproducible day-ahead solar generation prediction.

Run from this directory:
    python solve.py

The final predictor averages a HistGradientBoostingRegressor and a RandomForestRegressor.
Both models use median imputation inside their own sklearn Pipeline, so every CV fold
fits imputation only on that fold's training portion.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TARGET = "ac_power_kw"
FORECAST_COLUMNS = [
    "forecast_cloud_cover",
    "forecast_temp_c",
    "forecast_wind_ms",
    "forecast_humidity_pct",
]
FEATURES = FORECAST_COLUMNS + [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
    "solar_elevation_sin", "solar_elevation_deg", "clear_sky_proxy",
    "cloud_solar_interaction", "temp_solar_interaction", "humidity_solar_interaction",
]
PLANT_AC_CAPACITY_KW = 10_000.0


def parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse the dataset's mixed ISO and day-first timestamp strings safely."""
    values = series.astype(str).str.strip()
    iso_mask = values.str.match(r"^\d{4}-\d{2}-\d{2}")
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    result.loc[iso_mask] = pd.to_datetime(values.loc[iso_mask], format="mixed", errors="coerce")
    result.loc[~iso_mask] = pd.to_datetime(
        values.loc[~iso_mask], format="mixed", dayfirst=True, errors="coerce"
    )
    return result


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only features available at forecast time (weather + timestamp)."""
    out = frame.copy()
    ts = out["ts"]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    day_of_year = ts.dt.dayofyear.to_numpy(dtype=float)

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.2425)
    out["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.2425)
    out["month_sin"] = np.sin(2 * np.pi * (ts.dt.month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (ts.dt.month - 1) / 12.0)

    # Approximate solar elevation for the documented plant coordinates in Tumakuru.
    latitude = np.deg2rad(14.10)
    longitude = 77.28
    b = 2 * np.pi * (day_of_year - 81.0) / 364.0
    equation_of_time = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    solar_minutes = hour.to_numpy() * 60.0 + 4.0 * (longitude - 82.5) + equation_of_time
    hour_angle = np.deg2rad(solar_minutes / 4.0 - 180.0)
    declination = np.deg2rad(23.44) * np.sin(2 * np.pi * (284.0 + day_of_year) / 365.2425)
    sin_elevation = (
        np.sin(latitude) * np.sin(declination)
        + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    )
    out["solar_elevation_sin"] = np.clip(sin_elevation, 0.0, 1.0)
    out["solar_elevation_deg"] = np.rad2deg(np.arcsin(np.clip(sin_elevation, -1.0, 1.0)))

    cloud = pd.to_numeric(out["forecast_cloud_cover"], errors="coerce").clip(0.0, 1.0)
    out["clear_sky_proxy"] = out["solar_elevation_sin"] * (1.0 - cloud)
    out["cloud_solar_interaction"] = cloud * out["solar_elevation_sin"]
    out["temp_solar_interaction"] = out["forecast_temp_c"] * out["solar_elevation_sin"]
    out["humidity_solar_interaction"] = out["forecast_humidity_pct"] * out["solar_elevation_sin"]
    return out


def prepare_data(train_raw: pd.DataFrame, test_raw: pd.DataFrame, correct_likely_mw_labels: bool = True):
    """Clean raw data and engineer features; return frames and an audit dictionary."""
    train = train_raw.copy()
    test = test_raw.copy()
    train["ts"] = parse_timestamps(train["timestamp"])
    test["ts"] = parse_timestamps(test["timestamp"])

    audit = {
        "raw_train_rows": int(len(train)),
        "raw_test_rows": int(len(test)),
        "missing_target_rows_removed": int(train[TARGET].isna().sum()),
        "target_sentinel_rows_removed": int(train[TARGET].eq(-999).sum()),
        "invalid_train_timestamps_removed": int(train["ts"].isna().sum()),
        "invalid_test_timestamps": int(test["ts"].isna().sum()),
    }

    for col in FORECAST_COLUMNS:
        train[col] = pd.to_numeric(train[col], errors="coerce").replace(-999, np.nan)
        test[col] = pd.to_numeric(test[col], errors="coerce").replace(-999, np.nan)
    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
    train = train.loc[train["ts"].notna() & train[TARGET].notna() & train[TARGET].ne(-999)].copy()
    test = test.loc[test["ts"].notna()].copy()

    # A specific data-quality correction: RUN rows with measured POA > 200 W/m² but
    # target between 0.1 and 20 are consistent with values recorded in MW rather than kW.
    # These measurements are used only to clean historical labels, never as predictors.
    status = train.get("status", pd.Series("", index=train.index)).astype(str).str.strip().str.upper()
    poa = pd.to_numeric(train.get("measured_poa_wm2", pd.Series(np.nan, index=train.index)), errors="coerce")
    likely_mw_mask = status.eq("RUN") & poa.gt(200) & train[TARGET].between(0.1, 20.0, inclusive="both")
    audit["likely_mw_to_kw_target_candidates"] = int(likely_mw_mask.sum())
    audit["likely_mw_to_kw_target_corrections"] = int(likely_mw_mask.sum()) if correct_likely_mw_labels else 0
    if correct_likely_mw_labels:
        train.loc[likely_mw_mask, TARGET] = train.loc[likely_mw_mask, TARGET] * 1000.0

    # The plant's documented AC inverter rating is 10,000 kW. Clip negative sensor noise
    # and impossible exports to physical bounds; record counts before clipping.
    audit["negative_target_rows_clipped_to_zero"] = int(train[TARGET].lt(0).sum())
    audit["above_capacity_target_rows_clipped"] = int(train[TARGET].gt(PLANT_AC_CAPACITY_KW).sum())
    train[TARGET] = train[TARGET].clip(0.0, PLANT_AC_CAPACITY_KW)

    # Resolve repeated hours by taking medians rather than discarding all repeated records.
    duplicate_mask = train["ts"].duplicated(keep=False)
    audit["rows_in_duplicate_timestamp_groups"] = int(duplicate_mask.sum())
    audit["duplicate_timestamp_groups"] = int(train.loc[duplicate_mask, "ts"].nunique())
    aggregation = {column: "median" for column in FORECAST_COLUMNS}
    aggregation[TARGET] = "median"
    train = train.groupby("ts", as_index=False).agg(aggregation).sort_values("ts").reset_index(drop=True)
    test = test.sort_values("ts").reset_index(drop=True)

    train = add_features(train)
    test = add_features(test)
    audit["clean_train_rows"] = int(len(train))
    audit["train_start"] = str(train["ts"].min())
    audit["train_end"] = str(train["ts"].max())
    audit["test_rows"] = int(len(test))
    audit["test_start"] = str(test["ts"].min())
    audit["test_end"] = str(test["ts"].max())
    return train, test, audit


def make_hgb() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=2.0, random_state=42,
        )),
    ])


def make_rf() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=220, max_depth=20, min_samples_leaf=2,
            min_samples_split=5, max_features=1.0, n_jobs=4, random_state=42,
        )),
    ])


def make_final_model() -> VotingRegressor:
    """Equal-weight average; it modestly improved OOF MAE over either model alone."""
    return VotingRegressor(estimators=[("hgb", make_hgb()), ("rf", make_rf())], n_jobs=1)


class MonthHourMedianBaseline:
    """Leakage-safe baseline: median by month/hour, fallback to hour, then global median."""
    def fit(self, frame: pd.DataFrame, y: pd.Series):
        work = pd.DataFrame({
            "month": frame["ts"].dt.month.to_numpy(),
            "hour": frame["ts"].dt.hour.to_numpy(),
            "target": np.asarray(y),
        })
        self.global_median = float(np.median(y))
        self.month_hour = work.groupby(["month", "hour"])["target"].median().to_dict()
        self.hour = work.groupby("hour")["target"].median().to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray([
            self.month_hour.get((int(ts.month), int(ts.hour)), self.hour.get(int(ts.hour), self.global_median))
            for ts in frame["ts"]
        ], dtype=float)


def evaluate(train: pd.DataFrame):
    """Evaluate candidates with identical chronological folds and return diagnostics."""
    X, y = train[FEATURES], train[TARGET]
    splitter = TimeSeriesSplit(n_splits=5)
    candidates = {"HistGradientBoosting": make_hgb(), "RandomForest": make_rf()}
    fold_metrics = []
    oof_by_model = {name: [] for name in candidates}
    oof_by_model["HGB_RF_ensemble"] = []
    oof_actual = []
    diagnostics = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        Xtr, Xva = X.iloc[train_idx], X.iloc[valid_idx]
        ytr, yva = y.iloc[train_idx], y.iloc[valid_idx]
        valid_dates = train.iloc[valid_idx]["ts"]
        oof_actual.extend(yva.to_numpy(dtype=float).tolist())
        row = {
            "fold": fold,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(valid_idx)),
            "train_end": str(train.iloc[train_idx]["ts"].max()),
            "validation_start": str(valid_dates.min()),
            "validation_end": str(valid_dates.max()),
        }
        fold_preds = {}
        for name, model in candidates.items():
            model.fit(Xtr, ytr)
            pred = np.clip(model.predict(Xva), 0.0, PLANT_AC_CAPACITY_KW)
            fold_preds[name] = pred
            oof_by_model[name].extend(pred.tolist())
            row[f"{name}_mae_kw"] = float(mean_absolute_error(yva, pred))
            row[f"{name}_r2"] = float(r2_score(yva, pred))

        # Blend out-of-fold predictions from models trained on the exact same fold.
        ensemble_pred = np.mean([fold_preds["HistGradientBoosting"], fold_preds["RandomForest"]], axis=0)
        oof_by_model["HGB_RF_ensemble"].extend(ensemble_pred.tolist())
        row["HGB_RF_ensemble_mae_kw"] = float(mean_absolute_error(yva, ensemble_pred))
        row["HGB_RF_ensemble_r2"] = float(r2_score(yva, ensemble_pred))

        baseline = MonthHourMedianBaseline().fit(train.iloc[train_idx], ytr)
        baseline_pred = np.clip(baseline.predict(train.iloc[valid_idx]), 0.0, PLANT_AC_CAPACITY_KW)
        row["month_hour_baseline_mae_kw"] = float(mean_absolute_error(yva, baseline_pred))
        fold_metrics.append(row)

        fold_diag = train.iloc[valid_idx][["ts"] + FORECAST_COLUMNS].copy()
        fold_diag["fold"] = fold
        fold_diag["actual_kw"] = yva.to_numpy(dtype=float)
        fold_diag["predicted_kw"] = ensemble_pred
        fold_diag["abs_error_kw"] = np.abs(fold_diag["actual_kw"] - ensemble_pred)
        diagnostics.append(fold_diag)

    fold_df = pd.DataFrame(fold_metrics)
    diag_df = pd.concat(diagnostics, ignore_index=True)
    actual = np.asarray(oof_actual, dtype=float)
    report_models = {}
    for name, pred_list in oof_by_model.items():
        pred = np.asarray(pred_list, dtype=float)
        fold_values = fold_df[f"{name}_mae_kw"].to_numpy()
        report_models[name] = {
            "fold_mae_kw": [float(x) for x in fold_values],
            "mean_fold_mae_kw": float(np.mean(fold_values)),
            "std_fold_mae_kw": float(np.std(fold_values)),
            "pooled_oof_mae_kw": float(mean_absolute_error(actual, pred)),
            "mean_fold_r2": float(np.mean(fold_df[f"{name}_r2"].to_numpy())),
        }
    baseline_values = fold_df["month_hour_baseline_mae_kw"].to_numpy()
    report_models["MonthHourMedianBaseline"] = {
        "fold_mae_kw": [float(x) for x in baseline_values],
        "mean_fold_mae_kw": float(np.mean(baseline_values)),
        "std_fold_mae_kw": float(np.std(baseline_values)),
        "pooled_oof_mae_kw": None,
    }
    ensemble_mae = report_models["HGB_RF_ensemble"]["mean_fold_mae_kw"]
    baseline_mae = report_models["MonthHourMedianBaseline"]["mean_fold_mae_kw"]
    report_models["HGB_RF_ensemble"]["improvement_vs_baseline_pct"] = float((baseline_mae - ensemble_mae) / baseline_mae * 100.0)

    worst_fold_number = int(fold_df.loc[fold_df["HGB_RF_ensemble_mae_kw"].idxmax(), "fold"])
    worst = diag_df.loc[diag_df["fold"].eq(worst_fold_number)].copy()
    worst["hour"] = pd.to_datetime(worst["ts"]).dt.hour
    worst["month"] = pd.to_datetime(worst["ts"]).dt.month
    monthly = worst.groupby("month").agg(
        rows=("abs_error_kw", "size"), mae_kw=("abs_error_kw", "mean"),
        actual_mean_kw=("actual_kw", "mean"), predicted_mean_kw=("predicted_kw", "mean"),
    ).reset_index()
    hourly = worst.groupby("hour").agg(
        rows=("abs_error_kw", "size"), mae_kw=("abs_error_kw", "mean"),
        actual_mean_kw=("actual_kw", "mean"), predicted_mean_kw=("predicted_kw", "mean"),
    ).reset_index()
    return fold_df, diag_df, worst, monthly, hourly, report_models

def seasonal_holdout(train: pd.DataFrame):
    """Supplemental summer-like holdout: train before July 2024, validate Jul-Aug 2024."""
    train_mask = train["ts"] < pd.Timestamp("2024-07-01")
    valid_mask = (train["ts"] >= pd.Timestamp("2024-07-01")) & (train["ts"] < pd.Timestamp("2024-09-01"))
    tr, va = train.loc[train_mask], train.loc[valid_mask]
    if len(tr) == 0 or len(va) == 0:
        return {"note": "Not enough data for the July-August 2024 seasonal holdout."}
    ytr, yva = tr[TARGET], va[TARGET]
    model = make_final_model().fit(tr[FEATURES], ytr)
    pred = np.clip(model.predict(va[FEATURES]), 0.0, PLANT_AC_CAPACITY_KW)
    baseline = MonthHourMedianBaseline().fit(tr, ytr)
    base_pred = np.clip(baseline.predict(va), 0.0, PLANT_AC_CAPACITY_KW)
    return {
        "training_rows": int(len(tr)), "validation_rows": int(len(va)),
        "validation_start": str(va["ts"].min()), "validation_end": str(va["ts"].max()),
        "ensemble_mae_kw": float(mean_absolute_error(yva, pred)),
        "month_hour_baseline_mae_kw": float(mean_absolute_error(yva, base_pred)),
        "improvement_vs_baseline_pct": float((mean_absolute_error(yva, base_pred) - mean_absolute_error(yva, pred)) / mean_absolute_error(yva, base_pred) * 100.0),
        "note": "Supplementary season-matched check; only one historical July-August period is available.",
    }


def run(train_raw: pd.DataFrame, test_raw: pd.DataFrame, output_dir: Path = BASE_DIR):
    train, test, audit = prepare_data(train_raw, test_raw)
    fold_df, diag_df, worst, monthly, hourly, metrics = evaluate(train)
    X, y, Xt = train[FEATURES], train[TARGET], test[FEATURES]
    final_model = make_final_model().fit(X, y)
    pred = np.clip(final_model.predict(Xt), 0.0, PLANT_AC_CAPACITY_KW)
    submission = pd.DataFrame({"timestamp": test["timestamp"], "predicted_ac_power_kw": pred})

    # Hard output-contract checks.
    assert list(submission.columns) == ["timestamp", "predicted_ac_power_kw"]
    assert len(submission) == len(test_raw) == 1488
    assert submission["timestamp"].tolist() == test.sort_values("ts")["timestamp"].tolist()
    assert np.isfinite(pred).all() and (pred >= 0).all() and (pred <= PLANT_AC_CAPACITY_KW).all()

    output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_dir / "predictions.csv", index=False, float_format="%.6f")
    fold_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    diag_df.to_csv(output_dir / "oof_diagnostics.csv", index=False)
    worst.to_csv(output_dir / "worst_fold_diagnostics.csv", index=False)
    monthly.to_csv(output_dir / "worst_fold_monthly_summary.csv", index=False)
    hourly.to_csv(output_dir / "worst_fold_hourly_summary.csv", index=False)

    report = {
        "final_model": "Equal-weight ensemble: HistGradientBoosting + RandomForest",
        "features": FEATURES,
        "data_quality_audit": audit,
        "cross_validation": {
            "method": "TimeSeriesSplit(n_splits=5), chronological, same folds for all candidates",
            "models": metrics,
            "fold_details": fold_df.to_dict(orient="records"),
            "worst_ensemble_fold": int(worst["fold"].iloc[0]),
        },
        "seasonal_holdout": seasonal_holdout(train),
        "final_fit": {
            "training_rows": int(len(train)),
            "training_r2_in_sample_not_generalization_metric": float(final_model.score(X, y)),
            "test_rows": int(len(submission)),
            "test_start": str(test["ts"].min()), "test_end": str(test["ts"].max()),
            "prediction_min_kw": float(pred.min()), "prediction_max_kw": float(pred.max()),
            "prediction_mean_kw": float(pred.mean()),
            "predictions_all_finite": bool(np.isfinite(pred).all()),
            "predictions_within_plant_capacity": bool(((pred >= 0) & (pred <= PLANT_AC_CAPACITY_KW)).all()),
        },
        "limitations": [
            "No ground-truth target is provided for the July-August 2025 test period, so test MAE cannot be measured.",
            "The 230 likely MW-to-kW label corrections are rule-based data-quality inferences and should be documented, not treated as verified source metadata.",
            "Validation fold performance varies by time period; inspect worst_fold_diagnostics.csv.",
            "The July-August 2024 seasonal holdout is only one historical season and is supplementary, not a guarantee of test performance.",
        ],
    }
    (output_dir / "experiment_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("=== LLume Forecast final run ===")
    print(f"Clean training rows: {len(train):,}; test rows: {len(test):,}")
    print("Candidate CV results (mean fold MAE, kW):")
    for name, result in sorted(metrics.items(), key=lambda item: item[1]["mean_fold_mae_kw"]):
        print(f"  {name:26s} {result['mean_fold_mae_kw']:.2f} +/- {result['std_fold_mae_kw']:.2f}; folds={np.round(result['fold_mae_kw'], 2).tolist()}")
    ens = metrics["HGB_RF_ensemble"]
    base = metrics["MonthHourMedianBaseline"]
    print(f"Ensemble improvement vs baseline: {ens['improvement_vs_baseline_pct']:.2f}%")
    print(f"Worst ensemble fold: {report['cross_validation']['worst_ensemble_fold']}")
    print(f"Supplemental Jul-Aug 2024 holdout: {report['seasonal_holdout']}")
    print(f"Predictions saved: {output_dir / 'predictions.csv'}")
    print(f"Prediction range: {pred.min():.2f} to {pred.max():.2f} kW; mean={pred.mean():.2f} kW")
    return report


def main():
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    run(train_raw, test_raw, BASE_DIR)


if __name__ == "__main__":
    main()
