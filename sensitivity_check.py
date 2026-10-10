"""Optional sensitivity run for the heuristic MW-to-kW target correction.

Run from this directory with: python sensitivity_check.py
This reruns five-fold validation without correcting the 230 candidate labels.
"""
import json
import pandas as pd
from solve import BASE_DIR, DATA_DIR, evaluate, prepare_data

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
train, _, audit = prepare_data(train_raw, test_raw, correct_likely_mw_labels=False)
_, _, _, _, _, metrics = evaluate(train)
uncorrected = metrics["HGB_RF_ensemble"]
main_report = json.loads((BASE_DIR / "experiment_report.json").read_text(encoding="utf-8"))
corrected = main_report["cross_validation"]["models"]["HGB_RF_ensemble"]
result = {
    "candidate_rows": int(audit["likely_mw_to_kw_target_candidates"]),
    "with_correction": {
        "mean_fold_mae_kw": corrected["mean_fold_mae_kw"],
        "std_fold_mae_kw": corrected["std_fold_mae_kw"],
        "fold_mae_kw": corrected["fold_mae_kw"],
    },
    "without_correction": {
        "mean_fold_mae_kw": uncorrected["mean_fold_mae_kw"],
        "std_fold_mae_kw": uncorrected["std_fold_mae_kw"],
        "fold_mae_kw": uncorrected["fold_mae_kw"],
    },
    "warning": "The unit correction is inferred from status, irradiance, and target magnitude, not confirmed by a source quality flag. Review it with the data owner.",
}
(BASE_DIR / "label_correction_sensitivity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("Rule-based target correction DISABLED")
print("Candidate labels not rescaled:", audit["likely_mw_to_kw_target_candidates"])
print("Mean CV MAE:", round(uncorrected["mean_fold_mae_kw"], 2), "kW")
print("CV standard deviation:", round(uncorrected["std_fold_mae_kw"], 2), "kW")
print("Fold MAEs:", [round(x, 2) for x in uncorrected["fold_mae_kw"]])
print("Sensitivity results saved to label_correction_sensitivity.json")
