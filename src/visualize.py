import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/students.csv")
feature_cols = ["study_hours", "attendance_hours", "assignments", "past_score"]

# ── 1. Class distribution ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5,4))
counts = df["result"].value_counts().rename({1:"Pass", 0:"Fail"})
bars = ax.bar(counts.index, counts.values, color=["#2ecc71","#e74c3c"], width=0.5)
ax.bar_label(bars, fmt="%d", padding=4)
ax.set_title("Class Distribution"); ax.set_ylabel("Count")
plt.tight_layout(); plt.savefig("outputs/class_dist.png", dpi=150); plt.close()

# ── 2. Feature distributions by outcome ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10,8))
for ax, col in zip(axes.flatten(), feature_cols):
    for label, color, name in [(0,"#e74c3c","Fail"),(1,"#2ecc71","Pass")]:
        sns.kdeplot(df[df["result"]==label][col], ax=ax,
                    fill=True, alpha=0.4, color=color, label=name)
    ax.set_title(col.replace("_"," ").title())
    ax.legend()
plt.suptitle("Feature Distributions: Pass vs Fail", y=1.01)
plt.tight_layout(); plt.savefig("outputs/feature_dist.png", dpi=150); plt.close()

# ── 3. Correlation heatmap ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6,5))
corr = df[feature_cols + ["result"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout(); plt.savefig("outputs/correlation.png", dpi=150); plt.close()

# ── 4. Pairplot (sampled for speed) ───────────────────────────────────────
sample = df.sample(150, random_state=42)
sample["Outcome"] = sample["result"].map({1:"Pass", 0:"Fail"})
pp = sns.pairplot(sample, vars=feature_cols,
                  hue="Outcome", palette={"Pass":"#2ecc71","Fail":"#e74c3c"},
                  diag_kind="kde", plot_kws={"alpha":0.5})
pp.fig.suptitle("Pairplot of Features", y=1.02)
plt.savefig("outputs/pairplot.png", dpi=150); plt.close()

# ── 5. Feature importance from Random Forest ──────────────────────────────
rf = joblib.load("models/rf_model.pkl")
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
fig, ax = plt.subplots(figsize=(6,4))
colors = ["#e74c3c" if f=="past_score" else "#3498db" for f in importances.index]
importances.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Feature Importance — Random Forest")
ax.set_xlabel("Importance Score")
# Annotate the weak feature
ax.annotate("← weak feature (by design)", xy=(importances["past_score"], 0),
            xytext=(0.1, 0.3), fontsize=9, color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c"))
plt.tight_layout(); plt.savefig("outputs/feature_importance.png", dpi=150); plt.close()

# ── 6. Study hours vs attendance scatter ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7,5))
for label, color, name in [(0,"#e74c3c","Fail"),(1,"#2ecc71","Pass")]:
    subset = df[df["result"]==label]
    ax.scatter(subset["study_hours"], subset["attendance_hours"],
               alpha=0.4, c=color, s=20, label=name)
ax.set_xlabel("Study Hours"); ax.set_ylabel("Attendance Hours")
ax.set_title("Study Hours vs Attendance (coloured by outcome)")
ax.legend(); plt.tight_layout()
plt.savefig("outputs/scatter_main.png", dpi=150); plt.close()

print("All visualizations saved to outputs/")