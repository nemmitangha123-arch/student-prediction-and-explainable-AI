import pandas as pd
import joblib
import shap

# ============================================================
# 1. LOAD DATASET AND MODEL
# ============================================================

df = pd.read_csv("data/Final_Marks_Data.csv")
model = joblib.load("final_student_performance_model.pkl")

features = [
    "Attendance (%)",
    "Internal Test 1 (out of 40)",
    "Internal Test 2 (out of 40)",
    "Assignment Score (out of 10)",
    "Daily Study Hours"
]

# ============================================================
# 2. STUDENT INPUT
# ============================================================

student = pd.DataFrame([{
    "Attendance (%)": 84,
    "Internal Test 1 (out of 40)": 30,
    "Internal Test 2 (out of 40)": 36,
    "Assignment Score (out of 10)": 7,
    "Daily Study Hours": 3
}])

# ============================================================
# 3. PREDICTION
# ============================================================

prediction = model.predict(student)[0]

print("\n===== STUDENT PERFORMANCE =====")
print(f"Predicted Final Marks: {prediction:.2f}")

# ============================================================
# 4. SHAP EXPLANATION
# ============================================================

masker = shap.maskers.Independent(
    df[features],
    max_samples=2000
)

explainer = shap.Explainer(
    model.predict,
    masker
)

shap_values = explainer(student)

contributions = dict(
    zip(features, shap_values.values[0])
)

# ============================================================
# 5. DATASET STATISTICS
# ============================================================

statistics = {}

for feature in features:
    statistics[feature] = {
        "q25": df[feature].quantile(0.25),
        "median": df[feature].median(),
        "q75": df[feature].quantile(0.75)
    }

# ============================================================
# 6. PERFORMANCE LEVEL
# ============================================================

def get_performance_level(value, stats):

    if value < stats["q25"]:
        return "Below typical range"

    elif value >= stats["q75"]:
        return "Strong range"

    elif value < stats["median"]:
        return "Slightly below average"

    else:
        return "Typical range"


# ============================================================
# 7. ACTIONABLE RECOMMENDATION KNOWLEDGE BASE
# ============================================================

actions = {

    "Attendance (%)": {
        "improve":
            "Try to maintain regular class attendance and avoid unnecessary absences.",
        "maintain":
            "Continue maintaining consistent attendance."
    },

    "Internal Test 1 (out of 40)": {
        "improve":
            "Review the topics where marks were lost and practice similar questions before the next assessment.",
        "maintain":
            "Continue your current preparation strategy for Internal Test 1."
    },

    "Internal Test 2 (out of 40)": {
        "improve":
            "Review weaker topics and practice test-style questions to improve your assessment performance.",
        "maintain":
            "Maintain your current preparation strategy because your Internal Test 2 performance is supporting your prediction."
    },

    "Assignment Score (out of 10)": {
        "improve":
            "Focus on completing assignments accurately and on time. Aim to move toward the dataset's stronger range of 8/10 or above.",
        "maintain":
            "Continue submitting assignments accurately and on time."
    },

    "Daily Study Hours": {
        "improve":
            "Try to build a more consistent daily study routine.",
        "maintain":
            "Maintain your current consistent study routine."
    }
}

# ============================================================
# 8. GENERATE ACTIONABLE RECOMMENDATIONS
# ============================================================

recommendations = []

for feature in features:

    value = student[feature].iloc[0]
    shap_value = contributions[feature]
    stats = statistics[feature]

    level = get_performance_level(value, stats)

    # Strong negative contributor
    if shap_value < -1:

        priority = "HIGH"
        action = actions[feature]["improve"]

        message = (
            f"Your {feature} value is {value:.1f}. "
            f"It is currently your strongest negative contributor "
            f"to the predicted score (SHAP: {shap_value:+.2f}). "
            f"{action}"
        )

    # Moderate negative contributor
    elif shap_value < -0.5:

        priority = "MEDIUM"
        action = actions[feature]["improve"]

        message = (
            f"Your {feature} value is {value:.1f} and it is having "
            f"a moderate negative influence on your prediction "
            f"(SHAP: {shap_value:+.2f}). "
            f"{action}"
        )

    # Positive contributor
    elif shap_value > 0.5:

        priority = "POSITIVE"
        action = actions[feature]["maintain"]

        message = (
            f"Your {feature} value is {value:.1f} and is contributing "
            f"positively to your prediction (SHAP: {shap_value:+.2f}). "
            f"{action}"
        )

    else:
        continue

    recommendations.append({
        "feature": feature,
        "priority": priority,
        "message": message
    })

# ============================================================
# 9. DISPLAY RECOMMENDATIONS
# ============================================================

print("\n===== ACTIONABLE RECOMMENDATIONS =====")

for recommendation in recommendations:

    print(
        f"\n[{recommendation['priority']}] "
        f"{recommendation['feature']}"
    )

    print(recommendation["message"])

print("\n===== COMPLETE =====")