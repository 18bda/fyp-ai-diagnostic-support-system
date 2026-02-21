from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_PATH = Path("data/raw/Training.csv")
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    return df


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )


def main() -> None:
    df = load_data()
    target = "prognosis"

    X = df.drop(columns=[target])
    y = df[target]

    print("=== Verification Pack: Symptom Baseline ===")
    print(f"Dataset shape: {df.shape} | X: {X.shape} | Classes: {y.nunique()}")

    # 1) Duplicate feature rows (can make task trivial)
    dup_mask = X.duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    print(f"\n[Check 1] Duplicate feature rows (same symptom vector): {dup_count} / {len(X)}")

    # 2) Train/test overlap check using hashes of feature rows
    X_hash = pd.util.hash_pandas_object(X, index=False)
    X_train, X_test, y_train, y_test, h_train, h_test = train_test_split(
        X, y, X_hash, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    overlap = len(set(h_train).intersection(set(h_test)))
    print(f"[Check 2] Train/Test overlap of identical feature rows: {overlap}")

    # 3) Baseline split accuracy (recompute)
    model = make_model()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n[Check 3] Hold-out accuracy (stratified split): {acc:.4f}")

    # 4) Cross-validation accuracy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(make_model(), X, y, cv=cv, scoring="accuracy")
    print(f"[Check 4] 5-fold CV accuracy: mean={scores.mean():.4f}, std={scores.std():.4f}")
    print("Per-fold:", np.round(scores, 4))

    # 5) Label shuffle sanity check (should drop to near-chance)
    y_shuffled = y.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    X_reset = X.reset_index(drop=True)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_reset, y_shuffled, test_size=0.2, random_state=RANDOM_STATE, stratify=y_shuffled
    )
    model_s = make_model()
    model_s.fit(X_train_s, y_train_s)
    acc_shuffled = accuracy_score(y_test_s, model_s.predict(X_test_s))
    chance = 1.0 / y.nunique()
    print(f"\n[Check 5] Label-shuffled accuracy: {acc_shuffled:.4f} (chance ≈ {chance:.4f})")

    print("\n✅ If Check 2 overlap is near 0 and Check 5 is near chance, leakage is unlikely.")


if __name__ == "__main__":
    main()