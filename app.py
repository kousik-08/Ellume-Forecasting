
from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st

from solve import (
    FEATURES,
    TARGET,
    PLANT_AC_CAPACITY_KW,
    evaluate,
    make_final_model,
    prepare_data,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LLume Forecast | Solar AI Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.20);
            padding: 16px;
            border-radius: 12px;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem;
        }
        .section-caption {
            color: #808080;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def load_default_csv(filename):
    """Load a default project CSV if it exists."""
    candidates = [
        BASE_DIR / "data" / filename,
        BASE_DIR / filename,
    ]

    for path in candidates:
        if path.exists():
            return pd.read_csv(path)

    return None


@st.cache_data(show_spinner=False)
def cached_evaluate(train_df):
    """Cache cross-validation results for unchanged training data."""
    return evaluate(train_df)


@st.cache_data(show_spinner=False)
def cached_predictions(train_df, test_df):
    """Fit the final model and generate test-period predictions."""
    model = make_final_model()
    model.fit(train_df[FEATURES], train_df[TARGET])

    predictions = np.clip(
        model.predict(test_df[FEATURES]),
        0,
        PLANT_AC_CAPACITY_KW,
    )

    result = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"].values,
            "predicted_ac_power_kw": predictions,
        }
    )

    return result


def format_kw(value):
    return f"{value:,.1f} kW"


def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def prepare_daily_series(frame, value_columns):
    """Convert hourly data into daily averages for readable charts."""
    temp = frame.copy()
    temp["ts"] = pd.to_datetime(temp["ts"], errors="coerce")
    temp = temp.dropna(subset=["ts"]).set_index("ts")

    available = [c for c in value_columns if c in temp.columns]
    if not available:
        return pd.DataFrame()

    return temp[available].resample("D").mean().dropna(how="all")


def build_histogram(series, bins=30):
    """Create a histogram table without requiring Plotly or Matplotlib."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()

    if len(values) == 0:
        return pd.DataFrame(columns=["Error range (kW)", "Hours"])

    counts, edges = np.histogram(values, bins=bins)

    labels = [
        f"{edges[i]:,.0f} to {edges[i + 1]:,.0f}"
        for i in range(len(counts))
    ]

    return pd.DataFrame(
        {
            "Error range (kW)": labels,
            "Hours": counts,
        }
    )


# ============================================================
# HEADER
# ============================================================

st.title("☀️ LLume Forecast")
st.markdown(
    "### Solar PV Power Forecasting & Model Performance"
)
st.markdown(
    """
    Monitor forecast accuracy, compare the AI model with a historical
    baseline, inspect prediction errors, and explore hourly solar generation.
    """
)

st.caption(
    "Plant: Tumakuru, Karnataka | "
    f"Rated AC capacity: {PLANT_AC_CAPACITY_KW:,.0f} kW | "
    "Model: HistGradientBoosting + Random Forest ensemble"
)

# ============================================================
# SIDEBAR: DATA INPUTS
# ============================================================

st.sidebar.title("⚙️ Dashboard Settings")
st.sidebar.markdown("### Input data")

train_upload = st.sidebar.file_uploader(
    "Upload training CSV",
    type=["csv"],
    key="train_upload",
)

test_upload = st.sidebar.file_uploader(
    "Upload test CSV",
    type=["csv"],
    key="test_upload",
)

use_defaults = st.sidebar.checkbox(
    "Use project data files automatically",
    value=True,
    help="Looks for data/train.csv and data/test.csv in the project folder.",
)

raw_train = None
raw_test = None

try:
    if train_upload is not None:
        raw_train = pd.read_csv(train_upload)
    elif use_defaults:
        raw_train = load_default_csv("train.csv")

    if test_upload is not None:
        raw_test = pd.read_csv(test_upload)
    elif use_defaults:
        raw_test = load_default_csv("test.csv")

except Exception as exc:
    st.error(f"Could not read the CSV files: {exc}")
    st.stop()

if raw_train is None or raw_test is None:
    st.info(
        "Upload both CSV files, or place the files in "
        "`data/train.csv` and `data/test.csv`."
    )
    st.stop()

try:
    train, test, audit = prepare_data(raw_train, raw_test)
except Exception as exc:
    st.error(f"Data preparation failed: {exc}")
    st.info(
        "Check that the CSV files contain the expected timestamp, "
        "weather forecast, and training target columns."
    )
    st.stop()

if train.empty or test.empty:
    st.error("The cleaned training or test dataset is empty.")
    st.stop()

st.sidebar.success("Training and test data loaded")

st.sidebar.markdown("### Dataset coverage")
st.sidebar.write(
    f"Training: {audit.get('train_start', 'N/A')} to "
    f"{audit.get('train_end', 'N/A')}"
)
st.sidebar.write(
    f"Test: {audit.get('test_start', 'N/A')} to "
    f"{audit.get('test_end', 'N/A')}"
)

# ============================================================
# DATA OVERVIEW
# ============================================================

st.header("1. Dataset Overview")

a, b, c, d = st.columns(4)

a.metric(
    "Clean training rows",
    f"{len(train):,}",
)

b.metric(
    "Test forecast hours",
    f"{len(test):,}",
)

c.metric(
    "Weather features",
    "4",
)

d.metric(
    "Plant AC capacity",
    f"{PLANT_AC_CAPACITY_KW:,.0f} kW",
)

with st.expander("View data quality audit"):
    audit_cols = st.columns(2)

    with audit_cols[0]:
        st.markdown("**Training data quality**")
        st.write(
            {
                "Raw training rows": audit.get("raw_train_rows"),
                "Clean training rows": audit.get("clean_train_rows"),
                "Missing targets removed": audit.get(
                    "missing_target_rows_removed"
                ),
                "Target sentinel rows removed": audit.get(
                    "target_sentinel_rows_removed"
                ),
                "Negative targets clipped": audit.get(
                    "negative_target_rows_clipped_to_zero"
                ),
            }
        )

    with audit_cols[1]:
        st.markdown("**Other checks**")
        st.write(
            {
                "Duplicate timestamp groups": audit.get(
                    "duplicate_timestamp_groups"
                ),
                "Likely unit corrections": audit.get(
                    "likely_mw_to_kw_target_corrections"
                ),
                "Test rows": audit.get("test_rows"),
                "Training start": audit.get("train_start"),
                "Training end": audit.get("train_end"),
            }
        )

    st.warning(
        "Data-quality note: the preprocessing pipeline heuristically "
        "corrects some historical target values that appear to be recorded "
        "in MW rather than kW. This affects training labels and validation "
        "results. Confirm this assumption against the source system before "
        "using the results for operational decisions."
    )

with st.expander("Preview input data"):
    tab_train, tab_test = st.tabs(
        ["Training data", "Test data"]
    )

    with tab_train:
        st.dataframe(
            raw_train.head(30),
            use_container_width=True,
            hide_index=True,
        )

    with tab_test:
        st.dataframe(
            raw_test.head(30),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# MODEL EXECUTION
# ============================================================

st.header("2. Model Evaluation & Forecasting")

st.markdown(
    """
    The evaluation uses **five chronological validation folds**.
    Each validation period comes after its corresponding training period,
    helping assess performance on future observations.
    """
)

run_model = st.button(
    "🚀 Evaluate Model & Generate Forecast",
    type="primary",
    use_container_width=True,
)

if run_model:
    try:
        with st.spinner(
            "Running cross-validation and fitting the final forecasting model..."
        ):
            fold_df, diag_df, worst_df, monthly_df, hourly_df, metrics = (
                cached_evaluate(train)
            )

            prediction_df = cached_predictions(train, test)

            st.session_state["llume_results"] = {
                "fold_df": fold_df,
                "diag_df": diag_df,
                "worst_df": worst_df,
                "monthly_df": monthly_df,
                "hourly_df": hourly_df,
                "metrics": metrics,
                "prediction_df": prediction_df,
            }

        st.success(
            "Model evaluation and prediction generation completed."
        )

    except Exception as exc:
        st.error(f"Model execution failed: {exc}")
        st.exception(exc)
        st.stop()

if "llume_results" not in st.session_state:
    st.info(
        "Click **Evaluate Model & Generate Forecast** to calculate metrics, "
        "display charts, and create the prediction CSV."
    )
    st.stop()

results = st.session_state["llume_results"]

fold_df = results["fold_df"]
diag_df = results["diag_df"].copy()
worst_df = results["worst_df"]
monthly_df = results["monthly_df"]
hourly_df = results["hourly_df"]
metrics = results["metrics"]
prediction_df = results["prediction_df"]

ensemble = metrics["HGB_RF_ensemble"]
baseline = metrics["MonthHourMedianBaseline"]
hgb = metrics["HistGradientBoosting"]
rf = metrics["RandomForest"]

# ============================================================
# KPI DASHBOARD
# ============================================================

st.divider()
st.header("3. Model Performance Dashboard")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Ensemble CV MAE",
    format_kw(ensemble["mean_fold_mae_kw"]),
    help="Mean absolute error averaged across the five chronological folds.",
)

k2.metric(
    "Baseline CV MAE",
    format_kw(baseline["mean_fold_mae_kw"]),
    help="Reference model using historical month-and-hour medians.",
)

k3.metric(
    "Improvement vs baseline",
    f"{ensemble['improvement_vs_baseline_pct']:.1f}%",
    help="Relative reduction in mean fold MAE versus the baseline.",
)

k4.metric(
    "Mean fold R²",
    f"{ensemble['mean_fold_r2']:.4f}",
    help="Average R² across chronological validation folds.",
)

k5, k6, k7, k8 = st.columns(4)

k5.metric(
    "Validation MAE variation",
    format_kw(ensemble["std_fold_mae_kw"]),
    help="Standard deviation of MAE across validation folds.",
)

k6.metric(
    "HistGradientBoosting MAE",
    format_kw(hgb["mean_fold_mae_kw"]),
)

k7.metric(
    "Random Forest MAE",
    format_kw(rf["mean_fold_mae_kw"]),
)

k8.metric(
    "Predictions generated",
    f"{len(prediction_df):,}",
)

st.caption(
    "Lower MAE is better. MAE represents average absolute prediction error "
    "in kW. R² measures how well predictions explain variation in the target. "
    "These are validation metrics, not test-period accuracy."
)

# ============================================================
# TABS FOR VISUALIZATIONS
# ============================================================

(
    tab_performance,
    tab_actual_predicted,
    tab_errors,
    tab_patterns,
    tab_forecast,
    tab_data,
) = st.tabs(
    [
        "📊 Model Comparison",
        "🎯 Actual vs Predicted",
        "📉 Error Analysis",
        "🌤️ Generation Patterns",
        "🔮 Test Forecast",
        "📋 Metrics & Data",
    ]
)

# ============================================================
# TAB 1: MODEL COMPARISON
# ============================================================

with tab_performance:
    st.subheader("Validation MAE by Model")

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "HistGradientBoosting",
                "Random Forest",
                "HGB + RF Ensemble",
                "Month-hour baseline",
            ],
            "Mean CV MAE (kW)": [
                hgb["mean_fold_mae_kw"],
                rf["mean_fold_mae_kw"],
                ensemble["mean_fold_mae_kw"],
                baseline["mean_fold_mae_kw"],
            ],
        }
    )

    left, right = st.columns([1.25, 1])

    with left:
        st.bar_chart(
            comparison_df,
            x="Model",
            y="Mean CV MAE (kW)",
            horizontal=True,
        )

    with right:
        st.markdown("**Model comparison table**")
        st.dataframe(
            comparison_df.sort_values("Mean CV MAE (kW)"),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Performance across chronological folds")

    chart_fold = fold_df[
        [
            "fold",
            "HGB_RF_ensemble_mae_kw",
            "month_hour_baseline_mae_kw",
        ]
    ].copy()

    chart_fold = chart_fold.rename(
        columns={
            "fold": "Fold",
            "HGB_RF_ensemble_mae_kw": "AI Ensemble MAE (kW)",
            "month_hour_baseline_mae_kw": "Baseline MAE (kW)",
        }
    )

    st.line_chart(
        chart_fold.set_index("Fold"),
        y=["AI Ensemble MAE (kW)", "Baseline MAE (kW)"],
    )

    st.dataframe(
        fold_df[
            [
                "fold",
                "train_rows",
                "validation_rows",
                "validation_start",
                "validation_end",
                "HGB_RF_ensemble_mae_kw",
                "HGB_RF_ensemble_r2",
                "month_hour_baseline_mae_kw",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    worst_fold = int(
        fold_df.loc[
            fold_df["HGB_RF_ensemble_mae_kw"].idxmax(),
            "fold",
        ]
    )

    worst_mae = float(
        fold_df["HGB_RF_ensemble_mae_kw"].max()
    )

    st.warning(
        f"Fold {worst_fold} has the highest ensemble MAE "
        f"({worst_mae:,.1f} kW). Forecast accuracy changes over time, "
        "so inspect the error-analysis tab rather than relying only on "
        "the overall average."
    )

# ============================================================
# TAB 2: ACTUAL VS PREDICTED
# ============================================================

with tab_actual_predicted:
    st.subheader("Out-of-fold actual vs predicted power")

    plot_df = diag_df.copy()
    plot_df["ts"] = pd.to_datetime(plot_df["ts"], errors="coerce")
    plot_df = plot_df.dropna(subset=["ts"])

    st.caption(
        "These are predictions for historical validation observations that "
        "were held out from model fitting. They are not test-set predictions."
    )

    if not plot_df.empty:
        daily = prepare_daily_series(
            plot_df.rename(
                columns={
                    "actual_kw": "Actual power (kW)",
                    "predicted_kw": "Predicted power (kW)",
                }
            ),
            ["Actual power (kW)", "Predicted power (kW)"],
        )

        st.markdown("**Daily average power: actual vs predicted**")

        if not daily.empty:
            st.line_chart(daily)

        st.markdown("**Prediction scatter plot**")

        scatter_df = plot_df[
            ["actual_kw", "predicted_kw"]
        ].rename(
            columns={
                "actual_kw": "Actual power (kW)",
                "predicted_kw": "Predicted power (kW)",
            }
        )

        max_points = 4000
        if len(scatter_df) > max_points:
            scatter_df = scatter_df.sample(
                max_points,
                random_state=42,
            )

        st.scatter_chart(
            scatter_df,
            x="Actual power (kW)",
            y="Predicted power (kW)",
        )

        st.caption(
            "Points closer to an imaginary 45-degree diagonal represent "
            "more accurate predictions. The chart is sampled when there "
            "are more than 4,000 observations."
        )

        st.markdown("**Validation prediction explorer**")

        selected_fold = st.selectbox(
            "Choose validation fold",
            sorted(plot_df["fold"].dropna().unique().tolist()),
            key="validation_fold",
        )

        fold_plot = plot_df[
            plot_df["fold"] == selected_fold
        ].sort_values("ts")

        fold_series = fold_plot[
            ["ts", "actual_kw", "predicted_kw"]
        ].rename(
            columns={
                "actual_kw": "Actual (kW)",
                "predicted_kw": "Predicted (kW)",
            }
        )

        fold_daily = prepare_daily_series(
            fold_series.rename(
                columns={
                    "Actual (kW)": "Actual (kW)",
                    "Predicted (kW)": "Predicted (kW)",
                }
            ),
            ["Actual (kW)", "Predicted (kW)"],
        )

        st.line_chart(fold_daily)

        fold_mae = float(
            fold_plot["abs_error_kw"].mean()
        )

        st.metric(
            f"Fold {selected_fold} MAE",
            format_kw(fold_mae),
        )

# ============================================================
# TAB 3: ERROR ANALYSIS
# ============================================================

with tab_errors:
    st.subheader("Prediction error distribution")

    error_df = diag_df.copy()

    # Signed error: positive means overprediction.
    error_df["signed_error_kw"] = (
        error_df["predicted_kw"] - error_df["actual_kw"]
    )

    e1, e2, e3 = st.columns(3)

    e1.metric(
        "Mean absolute error",
        format_kw(error_df["abs_error_kw"].mean()),
    )

    e2.metric(
        "Mean signed error",
        format_kw(error_df["signed_error_kw"].mean()),
        help=(
            "Positive values indicate average overprediction; "
            "negative values indicate average underprediction."
        ),
    )

    e3.metric(
        "90th percentile absolute error",
        format_kw(error_df["abs_error_kw"].quantile(0.90)),
    )

    st.markdown("**Absolute error distribution**")

    histogram = build_histogram(
        error_df["abs_error_kw"],
        bins=30,
    )

    st.bar_chart(
        histogram,
        x="Error range (kW)",
        y="Hours",
    )

    st.markdown("**Signed error distribution**")

    signed_histogram = build_histogram(
        error_df["signed_error_kw"],
        bins=30,
    )

    st.bar_chart(
        signed_histogram,
        x="Error range (kW)",
        y="Hours",
    )

    st.markdown("**Error over time**")

    daily_error = (
        error_df.set_index("ts")
        .sort_index()
        .resample("D")["abs_error_kw"]
        .mean()
        .dropna()
        .to_frame("Daily MAE (kW)")
    )

    st.line_chart(daily_error)

    st.markdown("**Worst-fold diagnostics**")

    if not worst_df.empty:
        worst_view = worst_df.copy()

        if "ts" in worst_view.columns:
            worst_view["ts"] = pd.to_datetime(
                worst_view["ts"],
                errors="coerce",
            )

        st.dataframe(
            worst_view.head(100),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Error charts use out-of-fold historical predictions. "
        "They do not show measured errors for the unlabeled test period."
    )

# ============================================================
# TAB 4: GENERATION PATTERNS
# ============================================================

with tab_patterns:
    st.subheader("Historical generation patterns")

    pattern_df = diag_df.copy()
    pattern_df["ts"] = pd.to_datetime(
        pattern_df["ts"],
        errors="coerce",
    )
    pattern_df = pattern_df.dropna(subset=["ts"])

    pattern_df["Hour"] = pattern_df["ts"].dt.hour
    pattern_df["Month"] = pattern_df["ts"].dt.month

    left, right = st.columns(2)

    with left:
        st.markdown("**Average actual power by hour**")

        hourly_actual = (
            pattern_df.groupby("Hour")["actual_kw"]
            .mean()
            .rename("Actual average (kW)")
            .to_frame()
        )

        st.bar_chart(hourly_actual)

    with right:
        st.markdown("**Average predicted power by hour**")

        hourly_predicted = (
            pattern_df.groupby("Hour")["predicted_kw"]
            .mean()
            .rename("Predicted average (kW)")
            .to_frame()
        )

        st.bar_chart(hourly_predicted)

    st.markdown("**Monthly actual vs predicted generation**")

    monthly_pattern = (
        pattern_df.groupby("Month")
        .agg(
            Actual=("actual_kw", "mean"),
            Predicted=("predicted_kw", "mean"),
        )
    )

    st.line_chart(monthly_pattern)

    st.markdown("**Hourly prediction error**")

    hourly_error = (
        pattern_df.groupby("Hour")
        .agg(
            MAE=("abs_error_kw", "mean"),
            Rows=("abs_error_kw", "size"),
        )
        .reset_index()
    )

    st.bar_chart(
        hourly_error,
        x="Hour",
        y="MAE",
    )

    st.dataframe(
        hourly_error.rename(
            columns={
                "MAE": "Mean absolute error (kW)",
                "Rows": "Validation observations",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These patterns are computed from historical validation folds. "
        "They are useful for understanding daylight generation patterns "
        "and identifying hours where the model struggles."
    )

# ============================================================
# TAB 5: TEST FORECAST
# ============================================================

with tab_forecast:
    st.subheader("Generated test-period forecasts")

    forecast = prediction_df.copy()
    forecast["timestamp"] = pd.to_datetime(
        forecast["timestamp"],
        errors="coerce",
    )
    forecast = forecast.dropna(subset=["timestamp"]).sort_values(
        "timestamp"
    )

    f1, f2, f3, f4 = st.columns(4)

    f1.metric(
        "Minimum forecast",
        format_kw(forecast["predicted_ac_power_kw"].min()),
    )

    f2.metric(
        "Maximum forecast",
        format_kw(forecast["predicted_ac_power_kw"].max()),
    )

    f3.metric(
        "Mean forecast",
        format_kw(forecast["predicted_ac_power_kw"].mean()),
    )

    f4.metric(
        "Hours forecasted",
        f"{len(forecast):,}",
    )

    st.markdown("**Forecasted AC power over time**")

    forecast_series = forecast.set_index("timestamp")[
        ["predicted_ac_power_kw"]
    ].rename(
        columns={
            "predicted_ac_power_kw": "Predicted AC power (kW)"
        }
    )

    st.line_chart(forecast_series)

    st.markdown("**Daily average forecast**")

    daily_forecast = (
        forecast.set_index("timestamp")[
            "predicted_ac_power_kw"
        ]
        .resample("D")
        .mean()
        .dropna()
        .to_frame("Daily average power (kW)")
    )

    st.bar_chart(daily_forecast)

    st.markdown("**Filter predictions**")

    min_date = forecast["timestamp"].min().date()
    max_date = forecast["timestamp"].max().date()

    date_range = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="forecast_date_range",
    )

    filtered = forecast.copy()

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range

        filtered = filtered[
            filtered["timestamp"].dt.date.between(
                start_date,
                end_date,
            )
        ]

    hour_range = st.slider(
        "Filter by hour of day",
        min_value=0,
        max_value=23,
        value=(0, 23),
    )

    filtered = filtered[
        filtered["timestamp"].dt.hour.between(
            hour_range[0],
            hour_range[1],
        )
    ]

    st.write(f"Showing **{len(filtered):,}** forecast rows.")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = prediction_df.to_csv(
        index=False,
        float_format="%.6f",
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download complete predictions.csv",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    st.info(
        "The test dataset has no ground-truth AC power labels. "
        "Therefore, the charts display forecasts only; test-period MAE, "
        "RMSE, and R² cannot be calculated until actual measurements "
        "become available."
    )

# ============================================================
# TAB 6: METRICS AND DATA
# ============================================================

with tab_data:
    st.subheader("Complete validation metrics")

    metric_rows = []

    for model_name, model_metrics in metrics.items():
        metric_rows.append(
            {
                "Model": model_name,
                "Mean fold MAE (kW)": safe_float(
                    model_metrics.get("mean_fold_mae_kw")
                ),
                "Std fold MAE (kW)": safe_float(
                    model_metrics.get("std_fold_mae_kw")
                ),
                "Mean fold R²": safe_float(
                    model_metrics.get("mean_fold_r2")
                ),
                "Improvement vs baseline (%)": safe_float(
                    model_metrics.get("improvement_vs_baseline_pct")
                ),
            }
        )

    metric_table = pd.DataFrame(metric_rows)

    st.dataframe(
        metric_table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Feature list")

    st.write(
        "The model uses weather forecast variables, cyclical time "
        "features, an approximate solar elevation calculation, and "
        "weather-solar interaction features."
    )

    feature_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Description": [
                "Forecast cloud cover",
                "Forecast temperature (°C)",
                "Forecast wind speed (m/s)",
                "Forecast humidity (%)",
                "Hour sine encoding",
                "Hour cosine encoding",
                "Day-of-year sine encoding",
                "Day-of-year cosine encoding",
                "Month sine encoding",
                "Month cosine encoding",
                "Approximate solar elevation, sine",
                "Approximate solar elevation, degrees",
                "Clear-sky solar proxy",
                "Cloud cover × solar elevation",
                "Temperature × solar elevation",
                "Humidity × solar elevation",
            ][:len(FEATURES)],
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Export validation results")

    fold_csv = fold_df.to_csv(index=False).encode("utf-8")
    diagnostics_csv = diag_df.to_csv(index=False).encode("utf-8")
    metrics_csv = metric_table.to_csv(index=False).encode("utf-8")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.download_button(
            "Download fold metrics",
            data=fold_csv,
            file_name="cv_fold_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "Download OOF diagnostics",
            data=diagnostics_csv,
            file_name="oof_diagnostics.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d3:
        st.download_button(
            "Download model comparison",
            data=metrics_csv,
            file_name="model_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LLume Forecast | Solar PV AC-power forecasting | "
    "Chronological cross-validation | "
    "Predictions constrained to plant capacity"
)
