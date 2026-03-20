import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 500

study_hours      = np.random.uniform(1, 10, N)          # 1–10 hrs/day
attendance_hours = np.random.uniform(10, 50, N)          # 10–50 hrs/semester
assignments      = np.random.randint(0, 11, N)           # 0–10 submitted
past_score       = np.random.uniform(30, 100, N)         # historical exam score

# --- Label generation logic ---
# Strong features: study_hours, attendance, assignments
# Weak feature: past_score (weight = 0.05 only)
# Normalize each feature to [0,1] range before combining
def norm(x): return (x - x.min()) / (x.max() - x.min())

score = (
    0.40 * norm(study_hours)       +   # strongest predictor
    0.30 * norm(attendance_hours)  +   # second strongest
    0.25 * norm(assignments)       +   # third
    0.05 * norm(past_score)            # weak — only minor influence
)

# Add noise to make the boundary realistic (not perfectly separable)
noise = np.random.normal(0, 0.07, N)
score = score + noise

# Pass if combined score > 0.5
result = (score > 0.50).astype(int)   # 1 = Pass, 0 = Fail

df = pd.DataFrame({
    "study_hours":      np.round(study_hours, 2),
    "attendance_hours": np.round(attendance_hours, 2),
    "assignments":      assignments,
    "past_score":       np.round(past_score, 2),
    "result":           result          # target column
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/students.csv", index=False)
print(f"Dataset saved. Shape: {df.shape}")
print(df["result"].value_counts().rename({1:"Pass",0:"Fail"}))