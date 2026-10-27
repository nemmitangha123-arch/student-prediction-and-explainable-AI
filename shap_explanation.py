import pandas as pd
import shap
import joblib

# Load dataset
df = pd.read_csv("data/Final_Marks_Data.csv")

# Features used by our final model
features = [
    "Attendance (%)",
    "Internal Test 1 (out of 40)",
    "Internal Test 2 (out of 40)",
    "Assignment Score (out of 10)",
    "Daily Study Hours"
]

# Load final trained model
model = joblib.load("final_student_performance_model.pkl")

# Select features
X = df[features]

# Select one student to explain
student = X.iloc[[0]]

# Make prediction
prediction = model.predict(student)[0]

print("===== STUDENT PREDICTION =====")
print(f"Predicted Final Marks: {prediction:.2f}")

print("\nStudent Input:")
for feature in features:
    print(f"{feature}: {student.iloc[0][feature]}")

# Create SHAP explainer
explainer = shap.Explainer(
    model.predict,
    shap.maskers.Independent(X, max_samples=2000)
)

# Calculate SHAP values
shap_values = explainer(student)

print("\n===== SHAP EXPLANATION =====")

for feature, value in zip(features, shap_values.values[0]):
    print(f"{feature}: {value:+.4f}")

print("\nBase Value:")
print(f"{shap_values.base_values[0]:.4f}")