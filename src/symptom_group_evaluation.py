from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = Path("data/raw/Training.csv")


def load_training_csv() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {DATA_PATH}\n"
            "Place Training.csv inside data/raw/ (it must remain untracked)."
        )

    df = pd.read_csv(DATA_PATH)

    # Remove any unnamed columns caused by trailing commas
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    return df


def make_row_signature(X: pd.DataFrame) -> np.ndarray:
    """
    Create a stable signature (hash) for each feature row.
    Any identical symptom vectors will have the same signature.
    """
    # Ensure consistent types
    X_values = X.astype(int).to_numpy()

    sigs: list[str] = []
    for row in X_values:
        # Convert row to bytes and hash it
        row_bytes = row.tobytes()
        sigs.append(hashlib.md5(row_bytes).hexdigest())
    return np.array(sigs)


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )


def main() -> None:
    df = load_training_csv()
    target_col = "prognosis"

    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    groups = make_row_signature(X)

    print("=== Group-based Evaluation (no identical rows across splits) ===")
    print(f"Dataset shape: {df.shape} | X: {X.shape} | Classes: {y.nunique()}")

    # How many unique symptom vectors exist?
    unique_groups = np.unique(groups)
    print(f"Unique symptom vectors (groups): {len(unique_groups)} / {len(groups)}")

    # ---- Hold-out split with grouping ----
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    # Verify: no overlap in groups
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = len(train_groups.intersection(test_groups))

    print(f"\n[Check] Group overlap between train/test: {overlap} (should be 0)")

    model = build_model()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = model.predict(X.iloc[test_idx])

    acc = accuracy_score(y.iloc[test_idx], y_pred)
    print(f"\nHold-out Accuracy (group split): {acc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y.iloc[test_idx], y_pred))
    print("\nClassification Report:")
    print(classification_report(y.iloc[test_idx], y_pred))

    # ---- Cross-validation with grouping ----
    # Use GroupKFold: ensures each group appears in only one fold
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    cv_scores = cross_val_score(
        build_model(),
        X,
        y,
        cv=gkf,
        groups=groups,
        scoring="accuracy",
        n_jobs=None,
    )

    print(f"\n{n_splits}-fold GroupKFold Accuracy: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")
    print("Per-fold:", np.round(cv_scores, 4))


if __name__ == "__main__":
    main()