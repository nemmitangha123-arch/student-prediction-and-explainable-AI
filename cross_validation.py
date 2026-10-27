import pandas as pd

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

# 5-fold cross-validation
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Linear Regression model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# Evaluate multiple metrics
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

# Convert negative error scores to positive
mae_scores = -results["test_MAE"]
rmse_scores = -results["test_RMSE"]
r2_scores = results["test_R2"]

print("===== 5-Fold Cross-Validation =====")

print("\nMAE for each fold:")
for i, score in enumerate(mae_scores, 1):
    print(f"Fold {i}: {score:.2f}")

print(f"Average MAE: {mae_scores.mean():.2f}")

print("\nRMSE for each fold:")
for i, score in enumerate(rmse_scores, 1):
    print(f"Fold {i}: {score:.2f}")

print(f"Average RMSE: {rmse_scores.mean():.2f}")

print("\nR² for each fold:")
for i, score in enumerate(r2_scores, 1):
    print(f"Fold {i}: {score:.4f}")

print(f"Average R²: {r2_scores.mean():.4f}")
print(f"R² Standard Deviation: {r2_scores.std():.4f}")