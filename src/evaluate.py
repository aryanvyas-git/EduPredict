from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from train import train_models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np, os

os.makedirs("outputs", exist_ok=True)

def evaluate_all():
    models, X_tr, X_te, y_tr, y_te, cols = train_models()

    for name, model in models.items():
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        print(f"\n{'='*40}")
        print(f"  {name} — Test Set Evaluation")
        print(f"{'='*40}")
        print(f"  Accuracy : {accuracy_score(y_te, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_te, y_pred):.4f}")
        print(f"  Recall   : {recall_score(y_te, y_pred):.4f}")
        print(f"  F1 Score : {f1_score(y_te, y_pred):.4f}")
        print(f"  ROC-AUC  : {roc_auc_score(y_te, y_prob):.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_te, y_pred, target_names=["Fail","Pass"]))

        # Confusion matrix
        cm = confusion_matrix(y_te, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Fail","Pass"],
                    yticklabels=["Fail","Pass"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        fname = name.lower().replace(" ","_")
        plt.savefig(f"outputs/cm_{fname}.png", dpi=150)
        plt.close()

    # ROC curve comparison
    fig, ax = plt.subplots(figsize=(6,5))
    for name, model in models.items():
        y_prob = model.predict_proba(X_te)[:,1]
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        auc = roc_auc_score(y_te, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
    ax.plot([0,1],[0,1],"k--", alpha=0.4, label="Random baseline")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(); plt.tight_layout()
    plt.savefig("outputs/roc_curves.png", dpi=150)
    plt.close()
    print("\nAll evaluation plots saved to outputs/")

if __name__ == "__main__":
    evaluate_all()