import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

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

# Load final model
model = joblib.load("final_student_performance_model.pkl")

# Feature data
X = df[features]

# Select the same student used previously
student = X.iloc[[0]]

# Prediction
prediction = model.predict(student)[0]

# SHAP explainer using all 2,000 background samples
masker = shap.maskers.Independent(
    X,
    max_samples=2000
)

explainer = shap.Explainer(
    model.predict,
    masker
)

# Calculate SHAP values
shap_values = explainer(student)

print("===== SHAP VISUALIZATION =====")
print(f"Predicted Final Marks: {prediction:.2f}")

# Display waterfall plot
shap.plots.waterfall(
    shap_values[0],
    max_display=5,
    show=False
)

plt.title(
    f"Why was the prediction {prediction:.2f} marks?"
)

plt.tight_layout()

# Save the plot
plt.savefig(
    "student_shap_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

# Display the plot
plt.show()

print("\nSHAP waterfall plot saved as:")
print("student_shap_waterfall.png")