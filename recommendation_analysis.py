import pandas as pd

# Load dataset
df = pd.read_csv("data/Final_Marks_Data.csv")

features = [
    "Attendance (%)",
    "Internal Test 1 (out of 40)",
    "Internal Test 2 (out of 40)",
    "Assignment Score (out of 10)",
    "Daily Study Hours"
]

print("===== DATASET PERFORMANCE ANALYSIS =====\n")

for feature in features:
    print(f"\n{feature}")

    print(f"Mean   : {df[feature].mean():.2f}")
    print(f"Median : {df[feature].median():.2f}")
    print(f"25%    : {df[feature].quantile(0.25):.2f}")
    print(f"75%    : {df[feature].quantile(0.75):.2f}")
    print(f"Min    : {df[feature].min():.2f}")
    print(f"Max    : {df[feature].max():.2f}")

print("\n===== ANALYSIS COMPLETE =====")