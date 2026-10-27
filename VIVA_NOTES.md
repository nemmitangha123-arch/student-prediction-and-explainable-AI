# Viva Notes — AI-Based Student Performance Prediction and Decision Support System with Explainable AI

Companion to `Claude V-AI-Based_Student_Performance_Prediction_System_Refined.pptx` (17 slides).
Every number below comes from the current project: `results/evaluation.md`, `results/threshold_analysis.md`,
`app.py`, and live calls to the running API.

---

## Key facts to remember

| Item | Value (verified) |
|---|---|
| Dataset | 2,000 records, 5 input features, target = Final Exam Marks (/100) |
| Inputs and model-supported ranges | Attendance 52–100 · Internal Test 1 18–40 · Internal Test 2 16–40 · Assignment 4–10 · Daily Study Hours 1–5 |
| Stored data | SQLite (`data/student_records.db`): students, accounts, academic records, predictions (with SHAP and recommendations), interventions |
| Accounts | `faculty` / `Faculty@2026`; students sign in with their roll number, e.g. `PBL-2026-006` / `Student@006` |
| Final model | Pipeline: StandardScaler → LinearRegression (`final_student_performance_model.pkl`) |
| Final evaluation | 5-fold cross-validation, `KFold(n_splits=5, shuffle=True, random_state=42)` |
| MAE / RMSE / R² | 3.62 ± 0.07 marks · 4.59 ± 0.09 marks · 0.8357 ± 0.0084 |
| Models compared (same folds) | Linear Regression, Decision Tree, Random Forest, Random Forest (tuned), SVR |
| Early-warning rules (`app.py`) | predicted < 40 High Concern · 40–44 Borderline · > 44 Lower Concern |
| SHAP base value | 64.855 marks; base value + Σ SHAP = prediction (checked for all five demo students) |
| Recommendation rules (`classify_shap_priority`) | SHAP < −1 → HIGH · −1 to −0.5 → MEDIUM · −0.5 to +0.5 → none · > +0.5 → POSITIVE |

### The five demo students (live API results)

| Demo student | Inputs (Att · T1 · T2 · Assign · Study) | Predicted marks | Band |
|---|---|---|---|
| Demo Student 1 | 92 · 36 · 37 · 9 · 4 | 80.71 | Lower Concern |
| Demo Student 2 | 80 · 30 · 28 · 8 · 3 | 58.32 | Lower Concern |
| Demo Student 3 | 70 · 24 · 25 · 7 · 2 | 42.12 | Borderline |
| Demo Student 4 | 65 · 24 · 22 · 6 · 2 | 35.98 | High Concern |
| Demo Student 5 | 58 · 20 · 19 · 5 · 1 | 22.70 | High Concern |

Demo Student 3 with attendance 85 (What-If or re-prediction): 47.97, Lower Concern (+5.85 marks in the model's prediction).

The demo profiles are now used only in the practice pages (Prediction and What-If). The stored sample student **Sana Khan (PBL-2026-006)** has the same record as Demo Student 3 (70 · 24 · 25 · 7 · 2), so she gives the same 42.12, and 47.97 after attendance is changed to 85.

---

## Likely questions and answers

**1. Why did you choose Linear Regression?**
It had the lowest error on every metric in the same 5-fold cross-validation: MAE 3.62, RMSE 4.59, R² 0.8357. The tuned Random Forest (MAE 3.85) and SVR (MAE 3.91) were close but worse. It is also the simplest and most transparent model, which suits an explainability project.

**2. Why is this a regression problem?**
The target, final-exam marks, is a continuous number from 0 to 100, so we predict the number itself. The early-warning levels are derived afterwards from that prediction.

**3. Why did you compare multiple models?**
To make sure the choice was evidence-based. All five models were evaluated on exactly the same folds, features and target, so the comparison is fair.

**4. What does MAE mean?**
Mean absolute error: the average size of the prediction error. An MAE of 3.62 means our predictions are, on average, about 3.6 final-exam marks away from the actual marks.

**5. What does RMSE mean?**
Root mean squared error. Like MAE it is in marks, but it squares the errors first, so large mistakes count more. Our RMSE is 4.59 marks.

**6. What does R² mean?**
The coefficient of determination: the share of the variation in final-exam marks that the model explains. R² = 0.8357 means about 84% of the variation is explained.

**7. Why can't we call R² "accuracy"?**
Accuracy is the percentage of correct predictions in a classification task. A regression prediction is never exactly "correct" or "wrong" — it is off by some number of marks. R² is explained variance, not a percentage of correct answers. The model also gives no probability or confidence score.

**8. What about XGBoost? It is in the project folder.**
There are experiment scripts (`xgboost_model.py`, `xgboost_tuned.py`), but the `xgboost` package is not installed and not in `requirements.txt`, so the result cannot be reproduced. Rather than quote an unverified number, it was left out of the verified comparison.

**9. How did you avoid data leakage?**
The StandardScaler is inside the pipeline, so in cross-validation it is fitted on the training folds only. Every student is predicted exactly once by a model that never saw that student (out-of-fold).

**10. Why did you use SHAP?**
A prediction alone tells us "what" (for example 42.12 marks). SHAP tells us "why": how many marks each input added or removed for that individual student. That makes the prediction explainable and is the basis for the recommendations.

**11. What does a positive (or negative) SHAP value mean?**
A positive value means that input pushed this student's predicted marks up compared with the model's average prediction (the base value, 64.855); a negative value pushed it down. The contributions add up exactly to the prediction: for Demo Student 3, 64.855 + (−22.73) = 42.12.

**12. Does SHAP prove causation?**
No. SHAP explains how the model arrived at its prediction. It shows association in our data, not that changing an input would cause a change in real marks.

**13. How are at-risk students identified?**
The model predicts final-exam marks, then fixed rules in `app.py` assign a level: below 40 is High Concern, 40–44 is Borderline, above 44 is Lower Concern.

**14. Is the early-warning classifier part of the ML model?**
No. The ML model is a regressor. The three levels are project-defined decision rules applied to its prediction. The app also states that the level is not an official pass/fail decision.

**15. How good is the early warning?**
Evaluated out-of-fold: only 25 of 2,000 students actually scored below 40. High Concern flagged 13 of them (recall 0.52, precision 0.77); High Concern or Borderline flagged 21 of the 25. Regression pulls low scores towards the average (about +3 marks), so some low scorers are predicted too high. We kept the thresholds fixed rather than tuning them on the same data, and we report this as a limitation.

**16. How are recommendations generated?**
By fixed rules on each feature's SHAP contribution: below −1 mark → HIGH, −1 to −0.5 → MEDIUM, between −0.5 and +0.5 → no recommendation, above +0.5 → POSITIVE maintenance guidance. Each recommendation has a message and an action, and negative contributors are listed first, strongest first. They are guidance, not guaranteed improvements.

**17. What does the What-If simulator actually do?**
It starts from a practice scenario (the inputs of the student's latest practice prediction, or sample inputs), not from the stored academic record, and it never changes the database. When you move a slider, it calls the same `/predict` endpoint and shows the current prediction, the What-If prediction and the difference. It is a comparison of two model predictions, not a causal simulation.

**18. Is the system using real student data?**
No real people's data. The model was trained and evaluated on the project's 2,000-record dataset. The web application stores nine fictional sample students in its database (not real people and not rows of the training data), each with an account, an academic record, predictions and improvement tasks. Only the practice pages (Prediction and What-If) still use five predefined demo profiles, labelled "Practice Mode — Sample Scenario"; every Teacher Portal page uses the database.

**19. Why are there five demo students?**
They were the original demonstration of the full range of outcomes — two Lower Concern (one with HIGH recommendations), one Borderline and two High Concern — without using or inventing real student records. Their inputs are only starting values; the predictions, SHAP values and recommendations are computed live by the model. They now remain only in the practice pages; the main workflow uses the database's sample students.

**20. Why is there no department analytics?**
There is no department or cohort data in the project. Instead of showing fabricated charts, the Analytics and Reports pages show the early-warning distribution and intervention summary of the stored students, and state that department reports need department records the project does not store.

**21. Why isn't intervention effectiveness shown?**
The system stores each intervention and whether the student marked it completed, but it does not record outcomes, so it cannot measure whether an intervention worked.

**22. Where is the data stored? Is there a database?**
Yes: a SQLite database, `data/student_records.db`, created by the backend on its first start. It stores students, faculty and student accounts (passwords only as hashes), academic records, every generated prediction with its input snapshot, SHAP values and recommendations (the prediction history), and interventions with their completion status. The data survives page reloads and sign-out. Only the practice pages keep their results in the browser.

**23. What happens if the backend is unavailable?**
The app shows "Prediction service unavailable — The Flask prediction service isn't currently running. Start the backend and try again." with a Retry button. It never substitutes fake predictions. The Teacher Portal checks the API status (`GET /`) and shows the same notice.

**24. What happens if someone enters an out-of-range value?**
There are two kinds of range. A stored academic record accepts any real academic value (for example attendance 45%), so true marks are never refused. Prediction uses the narrower ranges in the training data: a record outside them shows "Prediction unavailable — academic record is outside the model's supported range" with the reason, and no prediction is generated. The values are never clamped or changed, so the model is never asked to extrapolate. The model endpoints used by the practice pages reject out-of-range values with an error message.

**25. How do you know the system works correctly?**
There are 23 automated backend tests (`python -m unittest discover -s tests`), including checks that `/explain` and `/recommend` return identical SHAP values and stay consistent under concurrent requests. During development the Student → Teacher flow was also checked with automated browser tests; those scripts are not part of the repository.

**26. What would you implement next?**
Real student records with consent and governance, institutional login, progress trends across a student's stored predictions, department analytics, recording intervention outcomes to study effectiveness, and re-validating the model and thresholds on new cohorts before any deployment. None of these are implemented today; the database, accounts, prediction history and intervention tracking already are.

---

## Live demo checklist

1. Before class: start the backend (`python app.py`) and the frontend (`npm run dev`), open the landing page. For a clean demo, start from a fresh database (stop the backend, delete `data/student_records.db`, start it again).
2. **Sign in** as `faculty` / `Faculty@2026` → Teacher Portal Dashboard lists the nine students with their prediction status.
3. Student AI Analysis → select **Sana Khan (PBL-2026-006)** → record 70 · 24 · 25 · 7 · 2 → **Generate Prediction** → 42.12, Borderline.
4. Show "What contributed to this prediction?" (SHAP) and the Recommended Actions.
5. On the Internal Test 1 recommendation → **Create Intervention** → edit the text if wanted → **Assign intervention** ("Based on Prediction #…").
6. **Edit academic record** → attendance 70 → 85 → Save → status "Record changed — prediction needs to be regenerated" → **Generate Prediction** → 47.97, Lower Concern. Prediction History shows both predictions; the first is Outdated and keeps its original inputs.
7. Sign out → sign in as `PBL-2026-006` / `Student@006` → Dashboard shows her record and current prediction → My Improvement Tasks → **Mark as Complete**.
8. Sign out → sign in as faculty → Sana Khan → Interventions shows Completed with the completion time.
9. Optional: Arjun Rao (attendance 45) shows "Prediction unavailable — academic record is outside the model's supported range"; What-If (as a student) compares hypothetical inputs without changing the record.
10. If the backend stops: point out the "Server unavailable" / "Prediction service unavailable" notice and Retry — no fake data.

---

## References

Verified tool references (used on the Conclusion slide):

- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30* (NeurIPS 2017).
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

**REFERENCES NEEDED — still to be supplied by the team:**

- Literature-review sources on early identification of at-risk students and on explainable AI in education. The earlier draft's studies had no authors, titles, venues or DOIs and could not be verified, so none are cited here.
- For each source add: authors, year, title, venue (journal/conference), volume/pages and DOI, and check it against the original paper before adding it to the presentation.
- The source of the 2,000-record dataset (`data/Final_Marks_Data.csv`), if it came from a published or public source.
