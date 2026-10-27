# AI-Based Student Performance Prediction and Decision Support System with Explainable AI

## 1. Overview

This project is a web-based decision-support system that uses five academic indicators to:

- **predict a student's final-exam marks** (0–100) with a regression model;
- assign an **early-warning category** (High Concern, Borderline or Lower Concern) from the predicted marks;
- explain each prediction with **SHAP** feature contributions (in marks);
- generate **feature-level recommendations** guided by the SHAP contributions;
- let users explore hypothetical inputs in a **What-If scenario simulator**;
- let faculty validate and import CSV records for existing students, then save predictions and explanations as one atomic batch;
- store each student's **academic record, predictions (with their SHAP values and recommendations) and prediction history** in a SQLite database, with **faculty and student accounts** and role-based access;
- let faculty deactivate and reactivate student accounts without deleting academic or intervention history;
- let faculty assign **improvement tasks (interventions)** that students mark as completed.

The system describes associations learned from data. It does not estimate the causal effect of changing any input, and it does not guarantee any future outcome.

## 2. Problem

Academic difficulties are often noticed only after final examinations, when there is little time left for support. Internal test marks, attendance, assignment scores and study habits are available earlier in the term. This project uses those indicators to identify students whose predicted final-exam performance falls into an early-warning category, so that educators can consider support sooner, and to explain which inputs contributed to each prediction.

## 3. Features

The model uses five inputs. The model endpoints, and prediction from a stored record, accept only values within the ranges seen in the training data. A stored academic record may hold any real academic value (attendance 0–100, tests 0–40, assignment 0–10, study hours 0–24); a record outside the ranges below is kept as entered, but no prediction is generated for it.

| Input | Range accepted |
|---|---|
| Attendance (%) | 52–100 |
| Internal Test 1 (out of 40) | 18–40 |
| Internal Test 2 (out of 40) | 16–40 |
| Assignment Score (out of 10) | 4–10 |
| Daily Study Hours | 1–5 |

**Target:** Final Exam Marks (out of 100).

## 4. Dataset

- File: `data/Final_Marks_Data.csv`
- 2,000 rows (one per student, identified by `Student_ID`), 5 input features, final-exam marks as the target.
- No missing values; all values are whole numbers.
- Actual final-exam marks range from 25 to 100 (mean 64.86).
- **Origin:** the exact external source of this dataset is not documented in the repository and is currently **not verified**.

Two evaluation set-ups exist in the repository:

- The original per-model scripts (`linear_regression.py`, `decision_tree.py`, `random_forest.py`, `random_forest_tuned.py`, `svr.py`) each use an **80/20 hold-out split** (`random_state=42`) and print their own results.
- The reproducible comparison in `evaluate_models.py` uses **5-fold cross-validation** for every model. The results reported below come from this evaluation.

## 5. Machine Learning

Models evaluated with the same 5 folds, features and target (configurations taken from the scripts above):

- Linear Regression
- Decision Tree (max depth 5)
- Random Forest (200 trees, max depth 10)
- Tuned Random Forest (500 trees, hand-set hyperparameters)
- SVR (RBF kernel, with scaling)

XGBoost scripts (`xgboost_model.py`, `xgboost_tuned.py`) exist in the repository, but the `xgboost` package is not part of the project's requirements, so XGBoost is **not** part of the reproducible comparison.

**Production model:** a scikit-learn pipeline of `StandardScaler` + `LinearRegression`, saved as `final_student_performance_model.pkl` by `final_model.py`. Linear Regression is used because it had the strongest reproducible 5-fold regression metrics (lowest MAE, lowest RMSE and highest R²).

Results from `results/evaluation.md` (mean ± standard deviation over 5 folds):

| Model | MAE (marks) | RMSE (marks) | R² |
|---|---|---|---|
| Linear Regression | 3.62 ± 0.07 | 4.59 ± 0.09 | 0.8357 ± 0.0084 |
| Decision Tree | 4.85 ± 0.17 | 6.07 ± 0.17 | 0.7123 ± 0.0158 |
| Random Forest | 3.99 ± 0.11 | 5.05 ± 0.10 | 0.8008 ± 0.0063 |
| Random Forest (tuned) | 3.85 ± 0.08 | 4.88 ± 0.09 | 0.8145 ± 0.0053 |
| SVR | 3.91 ± 0.11 | 4.94 ± 0.11 | 0.8099 ± 0.0069 |

## 6. Evaluation

### Regression performance (predicting marks)

- **MAE** (mean absolute error): the average size of the prediction error, in marks.
- **RMSE** (root mean squared error): also in marks, but gives larger errors more weight.
- **R²**: the proportion of variation in final-exam marks explained by the model (not a percentage accuracy).
- **5-fold cross-validation** (`KFold(n_splits=5, shuffle=True, random_state=42)`): the data is split into five parts; each model is trained on four and tested on the fifth, five times, so every student is predicted once by a model that did not see that student.

### Early-warning performance (derived categories)

The early-warning category is derived from the predicted marks:

| Predicted final-exam marks | Category |
|---|---|
| < 40 | High Concern |
| 40–44 | Borderline |
| > 44 | Lower Concern |

These thresholds are project-defined early-warning boundaries, not official university pass/fail rules. They were evaluated separately from the regression metrics, using the production model's **out-of-fold** predictions compared with each student's actual final-exam category (`results/evaluation.md`, `results/threshold_analysis.md`):

| Actual \ Predicted | High Concern | Borderline | Lower Concern |
|---|---|---|---|
| Actual < 40 (25 students) | 13 | 8 | 4 |
| Actual 40–44 (58 students) | 3 | 12 | 43 |
| Actual > 44 (1,917 students) | 1 | 10 | 1,906 |

| Measure | High Concern only (predicted < 40) | High Concern or Borderline (predicted ≤ 44) |
|---|---|---|
| Students who actually scored < 40 that were flagged | 13 of 25 | 21 of 25 |
| Recall (95% CI) | 0.520 (0.33–0.70) | 0.840 (0.65–0.94) |
| Precision (flagged students who actually scored < 40) | 0.765 | 0.447 |
| Students flagged in total | 17 | 47 (2.35% of all students) |

For students who actually scored at or below 44 (83 students), High Concern or Borderline flagged 36 (recall 0.434, precision 0.766).

ROC-AUC of the out-of-fold predicted marks is 0.990 for identifying students who scored below 40 (0.982 for at or below 44). ROC-AUC measures how well the predictions **rank** students; precision and recall measure the **fixed thresholds** above.

## 7. Explainable AI

`POST /explain` uses **SHAP** (`shap` 0.52, exact explainer with an independent masker over the 2,000 training rows) on the same production model. For each prediction it returns:

- a **base value** (the average model prediction, 64.855 marks), and
- a **contribution per input feature**, in marks. The base value plus all contributions equals the predicted marks.

A positive contribution means the input moved this prediction above the average prediction; a negative contribution means it moved it below. SHAP describes how the **model** arrived at its prediction. It does not show that an input caused a student's performance, and it does not establish what would happen if an input changed.

The shared SHAP explainer keeps internal buffers between calls, so explanations are computed one at a time (a lock in `app.py`). This keeps `/explain` and `/recommend` consistent when requests overlap.

## 8. Recommendation System

`POST /recommend` produces **rule-based, SHAP-guided, feature-level recommendations** from the student's actual inputs and the SHAP contributions of the same model:

| SHAP contribution of a feature | Priority |
|---|---|
| below −1.0 marks | HIGH |
| −1.0 to below −0.5 marks | MEDIUM |
| −0.5 to +0.5 marks | no recommendation |
| above +0.5 marks | POSITIVE (maintenance guidance) |

- Each recommendation includes the feature, priority, current value, a performance level relative to the dataset's quartiles, the SHAP contribution, an explanatory message and a predefined action for that feature and priority.
- Negative contributors are listed first, ordered by the size of their SHAP contribution; positive contributors follow as maintenance guidance. All qualifying recommendations are returned.
- The thresholds are project-defined, not universal standards. The risk category and the SHAP thresholds are separate: the risk level does not change which recommendations are produced.
- The actions are predefined academic guidance for each feature. They are not individually generated study plans.
- Internal tests are treated as already completed: their messages use the result to point to preparation before the final examination.

## 9. What-If Simulator

The What-If page (`/what-if`, in the Student Portal) lets a signed-in student adjust the five inputs within their accepted ranges and:

- sends each hypothetical input combination to the same `POST /predict` endpoint and production model (debounced, with earlier requests cancelled);
- displays the predicted marks and the early-warning category for the scenario;
- displays the **difference in model prediction** between the scenario and its starting scenario (the inputs of the student's latest practice prediction, or sample inputs); it does not start from, use or change the student's stored academic record;
- shows an updating state while a new scenario is being scored.

The difference shown is the difference between two model predictions. It does not estimate a causal effect and does not guarantee an actual improvement. Internal Test 1 and 2 are already completed assessments, so changing them represents a hypothetical scenario only. The simulator does not currently show SHAP explanations or recommendations for scenarios.

## 10. Early-Warning Limitations

- Only **25 of 2,000 students (1.25%)** actually scored below 40, and 83 (4.15%) scored at or below 44: a severe class imbalance. The early-warning figures rest on few students and are uncertain (for example, the 95% confidence interval for High Concern recall is 0.33–0.70).
- **High Concern alone** identified 13 of the 25 students who scored below 40. **High Concern or Borderline together** identified 21 of the 25; 4 were classified Lower Concern.
- ROC-AUC is high (0.990) because the model ranks students well, but recall at the fixed thresholds is lower.
- Predictions for students with low actual marks tend to be closer to the average: students who scored below 40 averaged 36.04 marks, while their out-of-fold predictions averaged 39.22.
- A threshold sensitivity analysis (`results/threshold_analysis.md`) showed trade-offs between catching more low-scoring students and flagging more students overall. The production thresholds were **not changed** based on this analysis, because any threshold chosen on the same data would need separate validation before its performance could be reported.

For early identification, the High Concern and Borderline categories are best read together; High Concern is the most urgent, highest-precision tier.

## 11. Reproducibility

### Tested environment

| Component | Version |
|---|---|
| Python | 3.14.7 |
| Flask | 3.1.3 |
| Flask-Cors | 6.0.5 |
| joblib | 1.6.0 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| shap | 0.52.0 |
| Node.js | 24.21.0 |
| npm | 11.19.0 |

Frontend packages (React 19, React Router, Vite, Tailwind CSS, framer-motion, lucide-react, jsPDF, oxlint) are defined in `package.json` and locked in `package-lock.json`.

The saved model was serialized with scikit-learn 1.9.1 by `final_model.py`.

### Install

```bash
# Backend (Windows)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Backend (macOS / Linux)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend
npm install
```

### Run

```bash
# 1. Backend API (http://127.0.0.1:5000)
python app.py

# 2. Frontend (in a second terminal; Vite prints the local URL, usually http://localhost:5173)
npm run dev
```

The frontend calls the backend at `http://127.0.0.1:5000` (set in `src/api.js`), so the backend must be running for sign-in and predictions.

On its first start the backend creates the SQLite database `data/student_records.db` and adds nine fictional sample students (not real people and not rows of the training data) and their accounts. Starting it again keeps the existing data; to start over, stop the backend and delete that file.

Development accounts (only password hashes are stored in the database):

| Role | Username | Password |
|---|---|---|
| Faculty | `faculty` | `Faculty@2026` |
| Student | roll number, e.g. `PBL-2026-006` | `Student@` + last three digits, e.g. `Student@006` |

### Evaluation

```bash
python evaluate_models.py      # writes results/evaluation.md
python analyze_thresholds.py   # writes results/threshold_analysis.md (re-runs evaluate_models.py)
```

Both scripts are deterministic and do not modify the production model.

Other scripts:

- `final_model.py` retrains the production model on all 2,000 rows and **overwrites** `final_student_performance_model.pkl`.
- `shap_global.py`, `shap_beeswarm.py` and `shap_visualization.py` regenerate the SHAP figures (`*.png`) and require `matplotlib`, which is not in `requirements.txt` and was not installed in the tested environment.

### Tests and checks

```bash
python -m unittest discover -s tests -v   # backend tests
npm run test:frontend                    # bulk-upload CSV export tests
npm run build                             # frontend production build
npm run lint                              # oxlint
```

## 12. API

### Model endpoints (no sign-in)

These `POST` endpoints take a JSON body with the five inputs and store nothing (they serve the practice Prediction page and What-If):

```json
{ "attendance": 91, "test1": 32, "test2": 36, "assignment": 7, "study_hours": 3 }
```

| Endpoint | Returns |
|---|---|
| `GET /` | Service status: `status`, `message`, `model_loaded`, `shap_loaded` |
| `POST /predict` | `status`, `predicted_marks`, `risk_level`, `risk_code`, `risk_basis`, `risk_disclaimer` |
| `POST /explain` | Everything from `/predict`, plus `base_value` and `shap_contributions` (marks per feature) |
| `POST /recommend` | Everything from `/predict`, plus `recommendations`: a list of `feature`, `priority`, `current_value`, `performance_level`, `shap_contribution`, `recommendation`, `action` |

Invalid or missing inputs return HTTP 400 with `{"status": "error", "message": ...}`.

Example (predicted 70.07, Lower Concern):

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"attendance\": 91, \"test1\": 32, \"test2\": 36, \"assignment\": 7, \"study_hours\": 3}"
```

### Account and student-record endpoints (sign-in required)

`POST /auth/login` returns a token; every other endpoint below needs the header `Authorization: Bearer <token>`. Permissions are checked by the backend: faculty can access every student; a student can access only their own record (other students' data returns 403).

| Endpoint | Who | Purpose |
|---|---|---|
| `POST /auth/login`, `POST /auth/logout`, `GET /auth/me` | anyone / signed in | Sign in (username and password), sign out, current user |
| `GET /students` | faculty | All students with their prediction status and latest prediction |
| `POST /students/accounts` | faculty | Create one student identity and login account |
| `PUT /students/<id>/account-status` | faculty | Deactivate or reactivate that student's account (`{"is_active": false}` or `true`) |
| `POST /students/registration/validate` | faculty | Validate and preview student accounts in a CSV file |
| `POST /students/registration/import` | faculty | Revalidate and atomically create valid student accounts from a CSV file |
| `GET /students/<id>` | faculty, own student | Record, prediction status and latest stored prediction |
| `PUT /students/<id>/academic-record` | faculty | Attendance, Internal Test 1 and 2, Assignment Score |
| `PUT /students/<id>/study-hours` | own student | Daily Study Hours |
| `POST /students/<id>/predictions` | faculty, own student | Predict from the **stored** record (no inputs in the request) and store the result with its SHAP values and recommendations |
| `GET /students/<id>/predictions` | faculty, own student | Prediction history, each with its input snapshot and whether it still matches the record |
| `POST /students/<id>/interventions` | faculty | Assign an improvement task, optionally linked to a stored prediction and recommendation |
| `GET /students/<id>/interventions` | faculty, own student | The student's improvement tasks |
| `GET /interventions` | faculty | Every student's improvement tasks, for the Teacher Portal overview pages |
| `PUT /interventions/<id>/complete` | own student | Mark a task completed |
| `POST /students/bulk-upload/validate` | faculty | Validate and preview academic records in a CSV file (multipart field `file`) |
| `POST /students/bulk-upload/import` | faculty | Revalidate and atomically save eligible academic rows with predictions (multipart field `file`) |

A prediction is refused (HTTP 409, with the reason) while the record is incomplete, is waiting for study hours, or is outside the model's ranges.

### Student account registration

The Faculty Portal has separate **Create Student**, **Bulk Student Registration**, and **Bulk Academic Upload** actions. Individual and CSV account creation require a student name, unique roll number/Student ID, unique username, and an initial password of at least 8 characters. Usernames are 3–80 letters, numbers, dots, underscores, or hyphens. Only faculty can use these endpoints.

Bulk Student Registration accepts UTF-8 CSV only, up to 2 MB and 500 rows. Its template columns are `Student Name`, `Student Roll Number`, `Username`, and `Password`. Validate and review the preview before confirming. Invalid rows are reported with their row number and reason; valid rows are inserted together in one transaction, and a failed batch leaves no partial accounts. Passwords supplied in the file are used only for account creation, hashed with Werkzeug, and never included in previews, results, or error reports. Keep the original CSV and credentials private; the application does not provide password recovery or display passwords again.

Registration creates only the student identity and student login account. It does not create an academic record, prediction, or intervention. Newly registered accounts can sign in with their chosen username and password and are recognized by Bulk Academic Upload using their `students.roll_no`. No additional dependency is required for registration.

### Student account status

The Faculty Portal's **Student Accounts** view displays each account as Active or Inactive and offers individual deactivation/reactivation. Deactivation requires browser confirmation, blocks future sign-ins, and revokes the student's existing authentication tokens. Reactivating restores the ability to sign in; the student signs in again to receive a new token. Both operations preserve academic records, prediction history, interventions, and audit references. Students without a login account are shown separately and cannot be changed through this workflow.

At startup, an idempotent SQLite migration adds `users.is_active` when it is missing, with a default of active for existing rows. It alters the existing database in place; it does not clear or rebuild tables. Student roles are unchanged, and the existing startup seeding checks continue to skip seeding while student/user rows exist. Bulk registration does not mark accounts as demo accounts: its input has no trusted demo designation, so no account is automatically classified, deactivated, or deleted.

### Faculty bulk academic upload

The Faculty Portal's **Bulk Academic Upload** view accepts CSV only (UTF-8, maximum 2 MB and 500 data rows). The required columns are `Student ID`, `Student Name`, `Attendance`, `Internal Test 1`, `Internal Test 2`, `Assignment Score`, and `Daily Study Hours`; common header variants such as `student_id`, `roll_no`, and `Internal Test 1 (out of 40)` are accepted.

The Student ID must match an existing `students.roll_no` and student account, and Student Name must match that existing identity. The account must have a blank academic record and no prediction or intervention history. The importer never creates login accounts or changes existing identities/records. It validates model-supported ranges, reports row-level errors in the preview, and imports only valid rows. All valid rows and their saved predictions, SHAP explanations, and recommendations are committed in one SQLite transaction; if the database commit fails or a record changes after validation, the full batch is rolled back. Faculty can download the successful prediction results and a CSV error report.

No database migration or new package is needed. The existing `academic_records`, `predictions`, and `interventions` tables are used.

## 13. Testing

`tests/test_recommendations.py` contains 23 recommendation and SHAP tests. `tests/test_bulk_upload.py` covers CSV validation, faculty access, atomic imports, prediction history, and existing student workflows. `tests/test_student_registration.py` covers faculty-only individual and CSV account creation, login, duplicate and required-field validation, rollback, privacy of passwords, and compatibility with academic upload (Python `unittest`, no extra dependencies):

| Test class | Tests | Covers |
|---|---|---|
| `ShapThresholdTests` | 2 | SHAP priority boundaries (−1.01, −1.00, −0.99, −0.51, −0.50, −0.49, +0.50, +0.51) |
| `OrderingTests` | 4 | Negative-first ordering by SHAP magnitude, unrounded sorting, positives after negatives, no recommendations dropped |
| `ActionTextTests` | 4 | Exact action text; every feature has HIGH/MEDIUM/POSITIVE actions |
| `CompletedTestAndStudyWordingTests` | 4 | Completed-test message wording, unchanged actions, Study Hours action wording |
| `RiskClassificationTests` | 1 | Risk boundaries (39.99, 40.00, 44.00, 44.01) |
| `RecommendEndpointTests` | 3 | `/recommend` response fields, ordering with real SHAP values, invalid input rejected |
| `ShapConsistencyTests` | 5 | Feature order, reference SHAP values, identical SHAP from `/explain` and `/recommend`, consistency under concurrent requests |
| `BulkUploadTests` | 12 | CSV validation, permissions, atomic import/rollback, prediction history, interventions, and existing workflows |
| `StudentRegistrationTests` | 12 | Student creation/login, duplicate and required-field checks, permissions, concurrent duplicate prevention, bulk registration rollback, and academic-upload compatibility |
| `StudentAccountStatusTests` | 7 | Faculty authorization, validation, idempotent status changes, inactive login/token rejection, data preservation, and legacy-schema migration |

`tests/bulk_upload_exports.test.js` and `tests/student_registration_csv.test.js` check CSV templates, frontend CSV validation, and error-report generation with Node's built-in test runner. `tests/student_account_management.test.js` checks account status labels, confirmation behavior, and action requests. The evaluation scripts (`evaluate_models.py`, `analyze_thresholds.py`) check fold coverage (every student predicted exactly once, no train/test overlap) with assertions.

## 14. Limitations

- Small number of genuinely low-scoring students (25 below 40) and severe class imbalance; early-warning results are uncertain (see Section 10).
- The model learns associations from observational data. It is not a causal model, and neither the predictions, SHAP values nor What-If differences show the effect of changing an input.
- Linear model with five academic inputs only.
- No demographic fields in the data, so no fairness or bias analysis is possible or performed.
- Authentication is deliberately simple: username and password with seeded development accounts and faculty-managed student registration; sign-in tokens expire after 8 hours. Faculty can deactivate or reactivate student accounts in Student Accounts. Deactivation rejects future sign-ins and revokes existing tokens without deleting student data. There is no password reset, institutional login or limit on repeated sign-in attempts.
- The SQLite database is a single local file with nine fictional sample students in one course; there are no departments, sections or semesters to choose between, and every faculty account can see every student.
- Daily Study Hours is self-reported by the student.
- Interventions record assignment and completion only; whether an intervention improved a student's performance is not measured.
- The Teacher Portal's Analytics and Reports summarise only what the database stores: there are no department records and no attendance history over time.
- The dataset's external origin is not verified.

## 15. Future Scope

Not currently implemented:

- Progress monitoring over the term (trends across a student's stored predictions).
- Integration with real faculty and institutional student data.
- Institutional login, password reset and production-grade account management.
- Measuring intervention outcomes, to study whether interventions are effective.
- Department-level and cohort analytics, once department records are available.
- Fairness monitoring, once suitable demographic data is available.
- Validation of alternative early-warning thresholds on new, independent data.
- A mobile application.
- NLP-based feedback.
