"""Reproducible evaluation of the regression models and the early-warning bands.

Run from the project root:
    python evaluate_models.py

Writes results/evaluation.md. It does not train or save the production model
(final_student_performance_model.pkl) and does not change app.py.
"""

import os
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ============================================================
# DATA (same dataset, features and target as the project)
# ============================================================

DATA_PATH = "data/Final_Marks_Data.csv"
RESULTS_PATH = "results/evaluation.md"

features = [
    "Attendance (%)",
    "Internal Test 1 (out of 40)",
    "Internal Test 2 (out of 40)",
    "Assignment Score (out of 10)",
    "Daily Study Hours"
]

target = "Final Exam Marks (out of 100)"

df = pd.read_csv(DATA_PATH)
X = df[features]
y = df[target]

# Same 5-fold split as cross_validation.py and final_model.py
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Early-warning thresholds, identical to classify_early_warning in app.py
HIGH_CONCERN_BELOW = 40
BORDERLINE_UP_TO = 44
BANDS = ["High Concern", "Borderline", "Lower Concern"]


def to_band(marks):
    marks = np.asarray(marks, dtype=float)
    return np.where(
        marks < HIGH_CONCERN_BELOW, "High Concern",
        np.where(marks <= BORDERLINE_UP_TO, "Borderline", "Lower Concern")
    )


# ============================================================
# MODELS (configurations copied from the existing model scripts)
# ============================================================

models = {
    # Production pipeline (final_model.py)
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ]),
    # decision_tree.py
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=5),
    # random_forest.py
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
    # random_forest_tuned.py
    "Random Forest (tuned)": RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),
    # svr.py
    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1))
    ]),
}

if XGBOOST_AVAILABLE:
    # xgboost_model.py
    models["XGBoost"] = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42, objective="reg:squarederror"
    )
    # xgboost_tuned.py
    models["XGBoost (tuned)"] = XGBRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.03, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, objective="reg:squarederror"
    )

# ============================================================
# CROSS-VALIDATION (manual loop so fold coverage can be checked)
# ============================================================

folds = list(kf.split(X))

# Every student appears in exactly one test fold, and train/test never overlap
test_counts = np.zeros(len(df), dtype=int)
for train_idx, test_idx in folds:
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "train/test overlap"
    assert len(train_idx) + len(test_idx) == len(df), "fold does not cover the dataset"
    test_counts[test_idx] += 1
assert (test_counts == 1).all(), "each row must be tested exactly once"


def cross_validate_model(model):
    oof = np.full(len(df), np.nan)
    fold_scores = []
    for train_idx, test_idx in folds:
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fitted.predict(X.iloc[test_idx])
        oof[test_idx] = pred
        y_true = y.iloc[test_idx]
        fold_scores.append({
            "MAE": mean_absolute_error(y_true, pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, pred)),
            "R2": r2_score(y_true, pred),
        })
    assert not np.isnan(oof).any(), "missing out-of-fold predictions"
    return pd.DataFrame(fold_scores), oof


print("===== 5-FOLD CROSS-VALIDATION =====")
regression_rows = []
fold_tables = {}
oof_predictions = {}
for name, model in models.items():
    scores, oof = cross_validate_model(model)
    fold_tables[name] = scores
    oof_predictions[name] = oof
    # Population standard deviation, as in cross_validation.py
    regression_rows.append({
        "Model": name,
        "MAE mean": scores["MAE"].mean(), "MAE std": scores["MAE"].std(ddof=0),
        "RMSE mean": scores["RMSE"].mean(), "RMSE std": scores["RMSE"].std(ddof=0),
        "R2 mean": scores["R2"].mean(), "R2 std": scores["R2"].std(ddof=0),
    })
    print(f"{name:24s} MAE {scores['MAE'].mean():.2f} ± {scores['MAE'].std(ddof=0):.2f} | "
          f"RMSE {scores['RMSE'].mean():.2f} ± {scores['RMSE'].std(ddof=0):.2f} | "
          f"R² {scores['R2'].mean():.4f} ± {scores['R2'].std(ddof=0):.4f}")

regression = pd.DataFrame(regression_rows)

# Plain LinearRegression (linear_regression.py) gives the same predictions as
# the scaled production pipeline; scaling does not change an OLS fit
plain_lr_scores, plain_lr_oof = cross_validate_model(LinearRegression())
lr_equivalence = float(np.abs(plain_lr_oof - oof_predictions["Linear Regression"]).max())

# ============================================================
# EARLY-WARNING EVALUATION (production model, out-of-fold predictions)
# ============================================================

production_oof = oof_predictions["Linear Regression"]
actual_band = to_band(y)
predicted_band = to_band(production_oof)

band_cm = confusion_matrix(actual_band, predicted_band, labels=BANDS)
actual_counts = pd.Series(actual_band).value_counts().reindex(BANDS, fill_value=0)
predicted_counts = pd.Series(predicted_band).value_counts().reindex(BANDS, fill_value=0)

per_band = []
for band in BANDS:
    a = actual_band == band
    p = predicted_band == band
    per_band.append({
        "Band": band,
        "Precision": precision_score(a, p, zero_division=0),
        "Recall": recall_score(a, p, zero_division=0),
        "F1": f1_score(a, p, zero_division=0),
    })
per_band = pd.DataFrame(per_band)


def binary_evaluation(actual_positive, predicted_positive):
    tn, fp, fn, tp = confusion_matrix(actual_positive, predicted_positive, labels=[False, True]).ravel()
    return {
        "positives": int(actual_positive.sum()),
        "flagged": int(predicted_positive.sum()),
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "precision": precision_score(actual_positive, predicted_positive, zero_division=0),
        "recall": recall_score(actual_positive, predicted_positive, zero_division=0),
        "f1": f1_score(actual_positive, predicted_positive, zero_division=0),
        # Threshold-free ranking quality of the out-of-fold predicted marks
        # (lower predicted marks = higher concern)
        "roc_auc": roc_auc_score(actual_positive, -production_oof),
    }


below_40 = binary_evaluation(y.to_numpy() < HIGH_CONCERN_BELOW, production_oof < HIGH_CONCERN_BELOW)
up_to_44 = binary_evaluation(y.to_numpy() <= BORDERLINE_UP_TO, production_oof <= BORDERLINE_UP_TO)

per_fold_positives = [int((y.iloc[test_idx] < HIGH_CONCERN_BELOW).sum()) for _, test_idx in folds]

print("\n===== EARLY-WARNING (production model, out-of-fold) =====")
print(pd.DataFrame(band_cm, index=[f"actual {b}" for b in BANDS], columns=[f"predicted {b}" for b in BANDS]))
print(f"Actual < 40: {below_40['positives']} | identified as High Concern: {below_40['TP']} | "
      f"precision {below_40['precision']:.3f} recall {below_40['recall']:.3f} F1 {below_40['f1']:.3f} ROC-AUC {below_40['roc_auc']:.3f}")
print(f"Actual <= 44: {up_to_44['positives']} | flagged High Concern or Borderline: {up_to_44['TP']} | "
      f"precision {up_to_44['precision']:.3f} recall {up_to_44['recall']:.3f} F1 {up_to_44['f1']:.3f} ROC-AUC {up_to_44['roc_auc']:.3f}")

# ============================================================
# WRITE RESULTS
# ============================================================


def md_table(frame, float_formats=None):
    float_formats = float_formats or {}
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, row in frame.iterrows():
        cells = []
        for col in headers:
            value = row[col]
            fmt = float_formats.get(col)
            cells.append(fmt.format(value) if fmt and isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


summary = pd.DataFrame({
    "Model": regression["Model"],
    "MAE (marks)": [f"{m:.2f} ± {s:.2f}" for m, s in zip(regression["MAE mean"], regression["MAE std"])],
    "RMSE (marks)": [f"{m:.2f} ± {s:.2f}" for m, s in zip(regression["RMSE mean"], regression["RMSE std"])],
    "R²": [f"{m:.4f} ± {s:.4f}" for m, s in zip(regression["R2 mean"], regression["R2 std"])],
})

lr_folds = fold_tables["Linear Regression"].copy()
lr_folds.insert(0, "Fold", [str(i) for i in range(1, len(lr_folds) + 1)])
lr_folds["Test rows"] = [str(len(test_idx)) for _, test_idx in folds]
lr_folds["Actual < 40 in fold"] = [str(n) for n in per_fold_positives]

cm_table = pd.DataFrame(band_cm, columns=[f"Predicted {b}" for b in BANDS])
cm_table.insert(0, "Actual \\ Predicted", [f"Actual {b}" for b in BANDS])
cm_table = cm_table.astype(str)

counts_table = pd.DataFrame({
    "Band": BANDS,
    "Actual students (real final marks)": [str(actual_counts[b]) for b in BANDS],
    "Predicted students (out-of-fold)": [str(predicted_counts[b]) for b in BANDS],
})


def binary_table(result):
    return pd.DataFrame([{
        "Actual positives": str(result["positives"]),
        "Flagged": str(result["flagged"]),
        "TP": str(result["TP"]), "FN": str(result["FN"]), "FP": str(result["FP"]), "TN": str(result["TN"]),
        "Precision": f"{result['precision']:.3f}",
        "Recall": f"{result['recall']:.3f}",
        "F1": f"{result['f1']:.3f}",
        "ROC-AUC": f"{result['roc_auc']:.3f}",
    }])


best_mae = regression.loc[regression["MAE mean"].idxmin(), "Model"]
best_rmse = regression.loc[regression["RMSE mean"].idxmin(), "Model"]
best_r2 = regression.loc[regression["R2 mean"].idxmax(), "Model"]
lr_row = regression.set_index("Model").loc["Linear Regression"]

xgboost_note = (
    "XGBoost and XGBoost (tuned) were evaluated with the configurations from xgboost_model.py and xgboost_tuned.py."
    if XGBOOST_AVAILABLE else
    "XGBoost and XGBoost (tuned) were **not evaluated**: the `xgboost` package is not installed in this "
    "environment and is not listed in requirements.txt, so xgboost_model.py and xgboost_tuned.py cannot be "
    "reproduced from the current project."
)

report = f"""# Model and Early-Warning Evaluation

Generated by `evaluate_models.py`. Every number below is produced by that script from the current project;
no earlier results are reused. The script does not train or save the production model.

## Dataset

- File: `{DATA_PATH}`
- Records: {len(df)} (each student appears once; no missing values: {int(df.isna().sum().sum()) == 0})
- Features ({len(features)}): {", ".join(features)}
- Target: {target}
- Target range in the data: {y.min()}–{y.max()} (mean {y.mean():.2f})

## Cross-validation methodology

- 5-fold `KFold(n_splits=5, shuffle=True, random_state=42)` — the same split as `cross_validation.py` and `final_model.py`.
- The same folds, features and target are used for every model.
- In each fold a fresh copy of the model is fitted on the training rows only and predicts the held-out rows.
  Scaling (Linear Regression, SVR) happens inside the pipeline, so it is fitted on training rows only.
- Checked by the script: train and test rows never overlap, and every student is predicted exactly once
  (out-of-fold predictions for all {len(df)} students).
- Metrics are computed per fold and reported as mean ± standard deviation (population standard deviation, as in `cross_validation.py`).
- Model configurations are copied from `linear_regression.py`/`final_model.py`, `decision_tree.py`,
  `random_forest.py`, `random_forest_tuned.py` and `svr.py`.
- Linear Regression is the production pipeline (StandardScaler + LinearRegression). The plain
  `LinearRegression()` of `linear_regression.py` gives the same predictions (maximum difference {lr_equivalence:.2e} marks).
- {xgboost_note}

## Regression results (prediction of final-exam marks)

These metrics measure how closely the predicted marks match the actual final-exam marks.
MAE and RMSE are in marks out of 100.

{md_table(summary)}

Per-fold results for the production model (Linear Regression):

{md_table(lr_folds, {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"})}

## Early-warning evaluation (derived categories)

This section evaluates the early-warning categories, not the marks. The production model's
**out-of-fold** predicted marks (each student's prediction comes from a model that did not train on that
student) are mapped to the current thresholds, and compared with the category of the student's actual
final-exam marks:

- predicted < {HIGH_CONCERN_BELOW} → High Concern
- predicted {HIGH_CONCERN_BELOW}–{BORDERLINE_UP_TO} → Borderline
- predicted > {BORDERLINE_UP_TO} → Lower Concern

The thresholds are unchanged from `app.py`.

### Class distribution

{md_table(counts_table)}

### Confusion matrix (rows = actual category, columns = predicted category)

{md_table(cm_table)}

### Per-category precision, recall and F1

{md_table(per_band, {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"})}

### Actual below {HIGH_CONCERN_BELOW} identified as High Concern

{md_table(binary_table(below_40))}

### Actual at or below {BORDERLINE_UP_TO} flagged as High Concern or Borderline

{md_table(binary_table(up_to_44))}

ROC-AUC is computed from the continuous out-of-fold predicted marks (lower marks = higher concern). It
measures how well the predictions rank students, independent of the fixed thresholds. Precision, recall
and F1 measure the fixed thresholds. A multi-class ROC-AUC is not reported, because the three categories
are cut from a single predicted value.

## Interpretation and limitations

- The two evaluations answer different questions: the regression metrics describe the error of the
  predicted marks; the early-warning metrics describe how the fixed thresholds on those predictions
  identify students by their actual category.
- The at-risk groups are small: {below_40['positives']} of {len(df)} students ({below_40['positives'] / len(df):.2%})
  scored below {HIGH_CONCERN_BELOW}, and {up_to_44['positives']} ({up_to_44['positives'] / len(df):.2%}) scored at or below {BORDERLINE_UP_TO}.
  Per test fold, between {min(per_fold_positives)} and {max(per_fold_positives)} students scored below {HIGH_CONCERN_BELOW},
  so the early-warning metrics rest on few cases and would change noticeably with a few different students.
- Actual categories use integer final-exam marks; predicted categories use unrounded predictions, as `app.py` does.
- The results describe this dataset and this model only. The model is fitted to observational data, so
  the results describe associations, not the effect of changing any input.
- Package versions used: Python {platform.python_version()}, scikit-learn {sklearn.__version__},
  numpy {np.__version__}, pandas {pd.__version__}. `requirements.txt` pins scikit-learn 1.8.0.

## Key findings

- Lowest MAE: {best_mae} ({regression.set_index("Model").loc[best_mae, "MAE mean"]:.2f} marks).
- Lowest RMSE: {best_rmse} ({regression.set_index("Model").loc[best_rmse, "RMSE mean"]:.2f} marks).
- Highest R²: {best_r2} ({regression.set_index("Model").loc[best_r2, "R2 mean"]:.4f}).
- Production model (Linear Regression): MAE {lr_row["MAE mean"]:.2f} ± {lr_row["MAE std"]:.2f}, RMSE {lr_row["RMSE mean"]:.2f} ± {lr_row["RMSE std"]:.2f}, R² {lr_row["R2 mean"]:.4f} ± {lr_row["R2 std"]:.4f}.
- Students who actually scored below {HIGH_CONCERN_BELOW}: {below_40['positives']}; identified as High Concern: {below_40['TP']}
  (recall {below_40['recall']:.3f}, precision {below_40['precision']:.3f}, F1 {below_40['f1']:.3f}); {below_40['FP']} student(s) flagged High Concern scored {HIGH_CONCERN_BELOW} or above.
- Students who actually scored at or below {BORDERLINE_UP_TO}: {up_to_44['positives']}; flagged High Concern or Borderline: {up_to_44['TP']}
  (recall {up_to_44['recall']:.3f}, precision {up_to_44['precision']:.3f}, F1 {up_to_44['f1']:.3f}).
- Of {actual_counts['Borderline']} students with actual Borderline marks, {band_cm[1][2]} were predicted Lower Concern.
- ROC-AUC of the out-of-fold predicted marks: {below_40['roc_auc']:.3f} for identifying actual < {HIGH_CONCERN_BELOW}; {up_to_44['roc_auc']:.3f} for actual ≤ {BORDERLINE_UP_TO}.
- Class imbalance: {below_40['positives'] / len(df):.2%} of students are in the actual High Concern category and {actual_counts['Lower Concern'] / len(df):.2%} in Lower Concern.
"""

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\nResults written to {RESULTS_PATH}")
