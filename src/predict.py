import pandas as pd
import numpy as np
import joblib

rf     = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# 5 realistic test cases
samples = pd.DataFrame({
    "study_hours":      [8.0,  2.0, 5.5, 1.5, 7.0],
    "attendance_hours": [45.0, 15.0, 30.0, 12.0, 40.0],
    "assignments":      [9,    2,   6,   1,    8],
    "past_score":       [85.0, 80.0, 55.0, 90.0, 40.0]  # note: high past_score ≠ guaranteed pass
})

X_scaled = scaler.transform(samples)
predictions = rf.predict(X_scaled)
probabilities = rf.predict_proba(X_scaled)

print("\nSample Predictions")
print("="*70)
header = f"{'Study':>6} {'Attend':>7} {'Assign':>7} {'Past':>6}  {'Pred':>5}  {'P(Pass)':>8}"
print(header)
print("-"*70)

for i, row in samples.iterrows():
    label = "PASS" if predictions[i] == 1 else "FAIL"
    prob  = probabilities[i][1]
    print(f"{row.study_hours:>6.1f} {row.attendance_hours:>7.1f} "
          f"{row.assignments:>7}  {row.past_score:>6.1f}  "
          f"{label:>5}  {prob:>8.3f}")

print("\nKey insight: Student 4 (past_score=90 but low effort) → FAIL")
print("This proves past_score alone does not determine the outcome.")
