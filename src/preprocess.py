import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib, os

def load_and_preprocess(path="data/students.csv"):
    df = pd.read_csv(path)

    # Feature matrix and target
    X = df[["study_hours", "attendance_hours", "assignments", "past_score"]]
    y = df["result"]

    # 80/20 stratified split — ensures both classes appear in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # StandardScaler: zero mean, unit variance
    # Important for Logistic Regression; harmless for Random Forest
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)   # NEVER fit on test data

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    return X_train_sc, X_test_sc, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, cols = load_and_preprocess()
    print("Train size:", X_tr.shape, "| Test size:", X_te.shape)