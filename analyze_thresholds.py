"""Early-warning threshold sensitivity analysis (analysis only).

Run from the project root:
    python analyze_thresholds.py

Uses the out-of-fold predictions produced by evaluate_models.py (5-fold, same
split as the project). Writes results/threshold_analysis.md. It does not change
the production thresholds in app.py, the model, or the frontend.

Note: running evaluate_models.py regenerates results/evaluation.md; its content
is deterministic, so the file is rewritten unchanged.
"""

import contextlib
import io
import math
import os
import runpy

import numpy as np
import pandas as pd

RESULTS_PATH = "results/threshold_analysis.md"

# Exact out-of-fold predictions from the Step 1 evaluation pipeline
with contextlib.redirect_stdout(io.StringIO()):
    evaluation = runpy.run_path("evaluate_models.py")

oof = np.asarray(evaluation["production_oof"], dtype=float)   # production model, out-of-fold
actual = evaluation["y"].to_numpy()
assert len(oof) == len(actual) == 2000 and not np.isnan(oof).any()

CURRENT_HIGH = evaluation["HIGH_CONCERN_BELOW"]    # 40 (from app.py)
CURRENT_BORDER = evaluation["BORDERLINE_UP_TO"]    # 44 (from app.py)

# Actual outcome groups are fixed; only the cut-offs applied to predictions vary
actual_below_40 = actual < CURRENT_HIGH
actual_up_to_44 = actual <= CURRENT_BORDER
N = len(actual)


def actual_band(values):
    return np.where(values < CURRENT_HIGH, "High Concern", np.where(values <= CURRENT_BORDER, "Borderline", "Lower Concern"))


def predicted_band(pred, high_below, border_up_to):
    return np.where(pred < high_below, "High Concern", np.where(pred <= border_up_to, "Borderline", "Lower Concern"))


def wilson(successes, total, z=1.96):
    # 95% Wilson score interval for a proportion
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return (centre - half, centre + half)


def binary(actual_positive, flagged):
    tp = int((actual_positive & flagged).sum())
    fp = int((~actual_positive & flagged).sum())
    fn = int((actual_positive & ~flagged).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    lo, hi = wilson(tp, tp + fn)
    return {"flagged": int(flagged.sum()), "TP": tp, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "f1": f1, "recall_ci": (lo, hi)}


def roc_auc(actual_positive, score):
    # Rank-based AUC (Mann-Whitney); higher score = higher concern
    pos, neg = score[actual_positive], score[~actual_positive]
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return (greater + 0.5 * ties) / (len(pos) * len(neg))


BANDS = ["High Concern", "Borderline", "Lower Concern"]
act_band = actual_band(actual)


def band_matrix(high_below, border_up_to):
    pred_band = predicted_band(oof, high_below, border_up_to)
    return pd.DataFrame(
        [[int(((act_band == a) & (pred_band == p)).sum()) for p in BANDS] for a in BANDS],
        index=[f"Actual {b}" for b in BANDS], columns=[f"Predicted {b}" for b in BANDS])


def fmt(x, d=3):
    return "n/a" if isinstance(x, float) and math.isnan(x) else f"{x:.{d}f}"


def md_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def matrix_md(frame):
    rows = [[idx] + [str(v) for v in frame.loc[idx]] for idx in frame.index]
    return md_table(rows, ["Actual \\ Predicted"] + list(frame.columns))


# ------------------------------------------------------------
# 1. Current thresholds
# ------------------------------------------------------------

current_high = binary(actual_below_40, oof < CURRENT_HIGH)
current_any_for_40 = binary(actual_below_40, oof <= CURRENT_BORDER)
current_any_for_44 = binary(actual_up_to_44, oof <= CURRENT_BORDER)
current_matrix = band_matrix(CURRENT_HIGH, CURRENT_BORDER)
auc_40 = roc_auc(actual_below_40, -oof)
auc_44 = roc_auc(actual_up_to_44, -oof)

cur_pred_band = predicted_band(oof, CURRENT_HIGH, CURRENT_BORDER)
below40_split = {b: int((cur_pred_band[actual_below_40] == b).sum()) for b in BANDS}
upto44_split = {b: int((cur_pred_band[actual_up_to_44] == b).sum()) for b in BANDS}

# ------------------------------------------------------------
# 2. Candidate thresholds
# ------------------------------------------------------------

HIGH_CANDIDATES = [30, 32, 34, 36, 38, 40, 42, 44, 46]
BORDER_CANDIDATES = [42, 44, 46, 48, 50, 52]

high_rows = []
for c in HIGH_CANDIDATES:
    r = binary(actual_below_40, oof < c)
    high_rows.append([f"< {c}" + (" (current)" if c == CURRENT_HIGH else ""), r["flagged"], f"{r['flagged'] / N:.2%}",
                      f"{r['TP']} of {int(actual_below_40.sum())}", r["FN"], r["FP"],
                      fmt(r["precision"]), fmt(r["recall"]), f"{fmt(r['recall_ci'][0], 2)}–{fmt(r['recall_ci'][1], 2)}", fmt(r["f1"])])

border_rows_40 = []
border_rows_44 = []
for c in BORDER_CANDIDATES:
    flagged = oof <= c
    a = binary(actual_below_40, flagged)
    b = binary(actual_up_to_44, flagged)
    label = f"≤ {c}" + (" (current)" if c == CURRENT_BORDER else "")
    border_rows_40.append([label, a["flagged"], f"{a['flagged'] / N:.2%}", f"{a['TP']} of {int(actual_below_40.sum())}",
                           a["FN"], fmt(a["precision"]), fmt(a["recall"]), f"{fmt(a['recall_ci'][0], 2)}–{fmt(a['recall_ci'][1], 2)}", fmt(a["f1"])])
    border_rows_44.append([label, b["flagged"], f"{b['TP']} of {int(actual_up_to_44.sum())}", b["FN"], b["FP"],
                           fmt(b["precision"]), fmt(b["recall"]), f"{fmt(b['recall_ci'][0], 2)}–{fmt(b['recall_ci'][1], 2)}", fmt(b["f1"])])

CONFIGS = [(38, 44), (40, 44), (42, 44), (40, 46), (40, 48), (42, 48)]
config_sections = []
for hb, bu in CONFIGS:
    title = f"High Concern < {hb}, Borderline {hb}–{bu}, Lower Concern > {bu}" + (" — current" if (hb, bu) == (CURRENT_HIGH, CURRENT_BORDER) else "")
    config_sections.append(f"#### {title}\n\n{matrix_md(band_matrix(hb, bu))}")

# Individual students who actually scored below 40
low_idx = np.where(actual_below_40)[0]
low_order = low_idx[np.argsort(oof[low_idx])]
student_rows = [[i + 1, int(actual[j]), f"{oof[j]:.1f}", cur_pred_band[j]] for i, j in enumerate(low_order)]

# Prediction distribution
flag_counts = [[f"< {c}", int((oof < c).sum()), f"{(oof < c).mean():.2%}"] for c in (30, 35, 40, 44, 46, 48, 50, 55)]
shrink_low = float(oof[actual_below_40].mean() - actual[actual_below_40].mean())
border_actual = (actual >= CURRENT_HIGH) & (actual <= CURRENT_BORDER)

# Values used in the interpretation
h42 = binary(actual_below_40, oof < 42)
h38 = binary(actual_below_40, oof < 38)
b46 = binary(actual_below_40, oof <= 46)
b48 = binary(actual_below_40, oof <= 48)
b46_44 = binary(actual_up_to_44, oof <= 46)
b48_44 = binary(actual_up_to_44, oof <= 48)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

report = f"""# Early-Warning Threshold Analysis

Analysis only. Generated by `analyze_thresholds.py` from the **out-of-fold** predictions of the production
model (StandardScaler + LinearRegression) produced by `evaluate_models.py`: 5-fold
`KFold(n_splits=5, shuffle=True, random_state=42)`, so every student's predicted mark comes from a model that
did not train on that student. No predictions from the production model trained on all {N} rows are used.

The actual outcome groups are fixed throughout (actual final-exam marks < {CURRENT_HIGH}, 40–44, > {CURRENT_BORDER});
only the cut-offs applied to the predicted marks vary.

## 1. Current thresholds

| Predicted final-exam marks | Early-warning category |
|---|---|
| < {CURRENT_HIGH} | High Concern |
| {CURRENT_HIGH}–{CURRENT_BORDER} | Borderline |
| > {CURRENT_BORDER} | Lower Concern |

## 2. Current performance

### Confusion matrix (rows = actual category from real marks, columns = predicted category)

{matrix_md(current_matrix)}

### Students who actually scored below {CURRENT_HIGH} ({int(actual_below_40.sum())} students)

- Classified High Concern: **{below40_split['High Concern']}**
- Classified Borderline: **{below40_split['Borderline']}**
- Classified Lower Concern: **{below40_split['Lower Concern']}**

| Measure | High Concern only (predicted < {CURRENT_HIGH}) | High Concern or Borderline (predicted ≤ {CURRENT_BORDER}) |
|---|---|---|
| Students flagged | {current_high['flagged']} | {current_any_for_40['flagged']} |
| Actual < {CURRENT_HIGH} detected | {current_high['TP']} of {int(actual_below_40.sum())} | {current_any_for_40['TP']} of {int(actual_below_40.sum())} |
| Recall (95% CI) | {fmt(current_high['recall'])} ({fmt(current_high['recall_ci'][0], 2)}–{fmt(current_high['recall_ci'][1], 2)}) | {fmt(current_any_for_40['recall'])} ({fmt(current_any_for_40['recall_ci'][0], 2)}–{fmt(current_any_for_40['recall_ci'][1], 2)}) |
| Precision (flagged students who actually scored < {CURRENT_HIGH}) | {fmt(current_high['precision'])} | {fmt(current_any_for_40['precision'])} |
| F1 | {fmt(current_high['f1'])} | {fmt(current_any_for_40['f1'])} |

### Students who actually scored at or below {CURRENT_BORDER} ({int(actual_up_to_44.sum())} students)

- Classified High Concern: **{upto44_split['High Concern']}**
- Classified Borderline: **{upto44_split['Borderline']}**
- Classified Lower Concern: **{upto44_split['Lower Concern']}**

High Concern or Borderline (predicted ≤ {CURRENT_BORDER}): {current_any_for_44['TP']} of {int(actual_up_to_44.sum())} detected,
recall {fmt(current_any_for_44['recall'])} (95% CI {fmt(current_any_for_44['recall_ci'][0], 2)}–{fmt(current_any_for_44['recall_ci'][1], 2)}),
precision {fmt(current_any_for_44['precision'])}, F1 {fmt(current_any_for_44['f1'])}.

ROC-AUC of the out-of-fold predicted marks (threshold-free ranking): {auc_40:.3f} for actual < {CURRENT_HIGH},
{auc_44:.3f} for actual ≤ {CURRENT_BORDER}.

## 3. Class distribution and prediction distribution

| Actual final-exam marks | Students | Share |
|---|---|---|
| < {CURRENT_HIGH} | {int(actual_below_40.sum())} | {actual_below_40.mean():.2%} |
| {CURRENT_HIGH}–{CURRENT_BORDER} | {int(border_actual.sum())} | {border_actual.mean():.2%} |
| > {CURRENT_BORDER} | {int((actual > CURRENT_BORDER).sum())} | {(actual > CURRENT_BORDER).mean():.2%} |

Out-of-fold predicted marks — how many students fall below each value:

{md_table(flag_counts, ["Predicted marks", "Students", "Share of all students"])}

The lowest out-of-fold prediction is {oof.min():.1f}; 5% of predictions are below {np.percentile(oof, 5):.1f}.

**Predictions for low scorers are pulled towards the average.** Students who actually scored below {CURRENT_HIGH}
averaged {actual[actual_below_40].mean():.2f} marks, but their out-of-fold predictions averaged {oof[actual_below_40].mean():.2f}
(+{shrink_low:.2f} marks). Students who actually scored {CURRENT_HIGH}–{CURRENT_BORDER} averaged {actual[border_actual].mean():.2f},
with predictions averaging {oof[border_actual].mean():.2f}. Across all students the average prediction error is
{float((actual - oof).mean()):+.3f} marks, so the shift is specific to the low end, as is typical for a least-squares regression.

Every student who actually scored below {CURRENT_HIGH}, ordered by predicted mark:

{md_table(student_rows, ["#", "Actual marks", "Out-of-fold predicted marks", "Current category"])}

## 4. Candidate threshold analysis

### 4.1 High Concern cut-off (flag if predicted < c), evaluated against actual < {CURRENT_HIGH}

{md_table(high_rows, ["Cut-off", "Flagged", "Share flagged", "Actual < 40 detected", "Missed (FN)", "Flagged but scored ≥ 40 (FP)", "Precision", "Recall", "Recall 95% CI", "F1"])}

Cut-offs below 34 are not meaningful for this data: only {int((oof < 34).sum())} students are predicted below 34 at all,
so they detect almost nobody.

### 4.2 Borderline upper cut-off (flag as High Concern or Borderline if predicted ≤ c)

Against actual < {CURRENT_HIGH} (is a genuinely low-scoring student flagged at all?):

{md_table(border_rows_40, ["Cut-off", "Flagged", "Share flagged", "Actual < 40 flagged", "Missed", "Precision", "Recall", "Recall 95% CI", "F1"])}

Against actual ≤ {CURRENT_BORDER}:

{md_table(border_rows_44, ["Cut-off", "Flagged", "Actual ≤ 44 flagged", "Missed", "Flagged but scored > 44", "Precision", "Recall", "Recall 95% CI", "F1"])}

### 4.3 Three-category confusion matrices for selected configurations

{(chr(10) * 2).join(config_sections)}

## 5. Precision / recall trade-offs (from the tables above)

- **Moving the High Concern cut-off down** (for example to < 38) flags {h38['flagged']} students and detects {h38['TP']} of
  {int(actual_below_40.sum())} (recall {fmt(h38['recall'])}, precision {fmt(h38['precision'])}): fewer false alarms, more
  genuinely low-scoring students missed.
- **Moving the High Concern cut-off up** (for example to < 42) flags {h42['flagged']} students and detects {h42['TP']} of
  {int(actual_below_40.sum())} (recall {fmt(h42['recall'])}, precision {fmt(h42['precision'])}), compared with {current_high['TP']} of
  {int(actual_below_40.sum())} and {current_high['flagged']} flagged at < {CURRENT_HIGH}.
- **Widening the Borderline band** raises coverage slowly and flags many more students: at ≤ 46, {b46['TP']} of
  {int(actual_below_40.sum())} low scorers are flagged with {b46['flagged']} students flagged in total; at ≤ 48,
  {b48['TP']} of {int(actual_below_40.sum())} with {b48['flagged']} flagged, compared with {current_any_for_40['TP']} of
  {int(actual_below_40.sum())} and {current_any_for_40['flagged']} flagged at ≤ {CURRENT_BORDER}. For actual ≤ {CURRENT_BORDER}, recall
  rises from {fmt(current_any_for_44['recall'])} to {fmt(b46_44['recall'])} (≤ 46) and {fmt(b48_44['recall'])} (≤ 48).
- The {int(actual_below_40.sum())} students below {CURRENT_HIGH} are few, so each additional detection moves recall by
  {1 / int(actual_below_40.sum()):.2f}; the 95% confidence interval for the current High Concern recall is
  {fmt(current_high['recall_ci'][0], 2)}–{fmt(current_high['recall_ci'][1], 2)}. Differences of one or two students between
  candidate cut-offs are within this uncertainty.

## 6. Interpretation

- The model ranks students well (ROC-AUC {auc_40:.3f}), but its predictions for low scorers are pulled upwards by
  about {shrink_low:.1f} marks on average. This is why a fixed cut-off at {CURRENT_HIGH} on the predicted mark catches only
  {current_high['TP']} of {int(actual_below_40.sum())} students who actually scored below {CURRENT_HIGH}.
- The early-warning design already has a second tier: {current_any_for_40['TP']} of {int(actual_below_40.sum())} students who
  actually scored below {CURRENT_HIGH} are flagged as either High Concern or Borderline, while {current_any_for_40['flagged']} students
  ({current_any_for_40['flagged'] / N:.2%} of the cohort) are flagged in total.
- Of the {below40_split['Lower Concern']} low scorers classified Lower Concern, the predictions are
  {", ".join(f"{oof[j]:.1f}" for j in low_order if cur_pred_band[j] == "Lower Concern")}. Reaching them requires a Borderline
  cut-off of about {math.ceil(oof[actual_below_40].max())}, which would flag {int((oof <= math.ceil(oof[actual_below_40].max())).sum())} students
  ({(oof <= math.ceil(oof[actual_below_40].max())).mean():.2%} of the cohort).
- Any cut-off chosen by looking at these tables is fitted to these {int(actual_below_40.sum())} students. Its measured
  performance would be an optimistic estimate. An unbiased estimate for a changed threshold would need a separate
  procedure, such as nested cross-validation (choosing the threshold inside each training split and measuring it on the
  held-out fold) or a new, independent set of students.
- None of these thresholds are "optimal" in a general sense: the right balance depends on how the institution weighs a
  missed student against an unnecessary check-in, which the data cannot decide.

## 7. Recommendation

### A. Keep current thresholds
Supported by: the thresholds are a fixed, pre-defined policy (not tuned on this data), so the reported performance is an
honest out-of-fold estimate; precision is {fmt(current_high['precision'])} for High Concern; and together High Concern and
Borderline flag {current_any_for_40['TP']} of {int(actual_below_40.sum())} actual low scorers while flagging only
{current_any_for_40['flagged']} students ({current_any_for_40['flagged'] / N:.2%}).

### B. Change thresholds
For the students who actually scored below {CURRENT_HIGH}, the data show a trade-off rather than a clearly better
configuration: raising the High Concern cut-off to < 42 detects {h42['TP'] - current_high['TP']} more of the {int(actual_below_40.sum())} for
{h42['flagged'] - current_high['flagged']} more flags, and widening Borderline to ≤ 46 or ≤ 48 flags {b46['TP'] - current_any_for_40['TP']} and
{b48['TP'] - current_any_for_40['TP']} more of them respectively, for {b46['flagged'] - current_any_for_40['flagged']} and
{b48['flagged'] - current_any_for_40['flagged']} more flags. These differences are within the uncertainty of a {int(actual_below_40.sum())}-student group.

The one materially different trade-off concerns the wider group who actually scored at or below {CURRENT_BORDER}: widening
Borderline to ≤ 46 raises their recall from {fmt(current_any_for_44['recall'])} to {fmt(b46_44['recall'])}
({b46_44['TP'] - current_any_for_44['TP']} more of {int(actual_up_to_44.sum())} students; 95% CIs {fmt(current_any_for_44['recall_ci'][0], 2)}–{fmt(current_any_for_44['recall_ci'][1], 2)}
and {fmt(b46_44['recall_ci'][0], 2)}–{fmt(b46_44['recall_ci'][1], 2)}), while precision falls from {fmt(current_any_for_44['precision'])} to
{fmt(b46_44['precision'])} and the number flagged rises from {current_any_for_40['flagged']} to {b46['flagged']}. This would be a policy change
to the definition of the Borderline band. Because it was identified on these same out-of-fold predictions, its
performance could only be reported after the separate validation described above.

### C. Keep current thresholds and report the limitation — recommended
This is the most defensible choice given: only {int(actual_below_40.sum())} actual < {CURRENT_HIGH} students
({actual_below_40.mean():.2%}, severe class imbalance); a strong ranking (ROC-AUC {auc_40:.3f}) combined with a High Concern
recall of {fmt(current_high['recall'])} (95% CI {fmt(current_high['recall_ci'][0], 2)}–{fmt(current_high['recall_ci'][1], 2)}) and precision of
{fmt(current_high['precision'])}; and the regression model's documented tendency to pull low predictions towards the average.
For early identification, the system should be described as flagging students as High Concern **or** Borderline
({current_any_for_40['TP']} of {int(actual_below_40.sum())} actual low scorers flagged), with High Concern alone as the most urgent,
highest-precision tier. The limitation to report: fixed thresholds on predicted marks miss some genuinely low-scoring
students ({below40_split['Lower Concern']} of {int(actual_below_40.sum())} were classified Lower Concern), and any threshold change, including
the wider Borderline band described in B, would need separate validation.

NO PRODUCTION THRESHOLDS WERE CHANGED.
"""

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print(f"Current: High Concern detects {current_high['TP']}/{int(actual_below_40.sum())} (recall {current_high['recall']:.3f}, precision {current_high['precision']:.3f}); "
      f"High Concern or Borderline flags {current_any_for_40['TP']}/{int(actual_below_40.sum())} with {current_any_for_40['flagged']} students flagged")
print(f"Results written to {RESULTS_PATH}")
