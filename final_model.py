import pandas as pd
import joblib

from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/Final_Marks_Data.csv")

# Features
features = [
    "Attendance (%)",
    "Internal Test 1 (out of 40)",
    "Internal Test 2 (out of 40)",
    "Assignment Score (out of 10)",
    "Daily Study Hours"
]

# Target
target = "Final Exam Marks (out of 100)"

X = df[features]
y = df[target]

# Final Linear Regression pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# 5-fold cross-validation
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = cross_validate(
    model,
    X,
    y,
    cv=kf,
    scoring={
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }
)

# Convert error scores to positive
mae_scores = -results["test_MAE"]
rmse_scores = -results["test_RMSE"]
r2_scores = results["test_R2"]

print("===== FINAL MODEL VALIDATION =====")

print(f"\nMAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
print(f"RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
print(f"R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

# Train final model on the complete dataset
model.fit(X, y)

# Save model
joblib.dump(model, "final_student_performance_model.pkl")

print("\nFinal model trained on all 2,000 records.")
print("Model saved as: final_student_performance_model.pkl")