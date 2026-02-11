from __future__ import annotations

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = Path("data/raw/Training.csv")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {DATA_PATH}\n"
            "Place Training.csv inside data/raw/ (it must remain untracked)."
        )
    df = pd.read_csv(DATA_PATH)

    # Remove empty 'Unnamed' columns caused by trailing commas in some CSVs
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    return df


def main() -> None:
    df = load_data()

    target_col = "prognosis"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print("=== Baseline Symptom Classifier (Logistic Regression) ===")
    print("Dataset shape:", df.shape)
    print("X shape:", X.shape)
    print("Number of classes:", y.nunique())
    print("Top classes:\n", y.value_counts().head(10), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline = scaling + classifier
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=None, solver="lbfgs")),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
