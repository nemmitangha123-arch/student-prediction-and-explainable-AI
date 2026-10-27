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

# SHAP background using all 2,000 records
masker = shap.maskers.Independent(
    X,
    max_samples=2000
)

# Create SHAP explainer
explainer = shap.Explainer(
    model.predict,
    masker
)

print("Calculating SHAP values for all 2,000 students...")

# Calculate SHAP values
shap_values = explainer(X)

print("SHAP calculation completed.")

# Create beeswarm plot
shap.plots.beeswarm(
    shap_values,
    max_display=5,
    show=False
)

plt.title("SHAP Feature Impact Across Students")
plt.tight_layout()

# Save plot
plt.savefig(
    "shap_beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nBeeswarm plot saved as:")
print("shap_beeswarm.png")