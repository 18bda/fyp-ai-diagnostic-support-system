"""
ml_sanity_check.py

Purpose (FYP Feasibility Evidence):
- Confirm Python ML environment works (NumPy/Pandas/scikit-learn).
- Validate end-to-end classification pipeline (train -> predict -> evaluate).
- This pipeline will later be reused for symptom-based disease prediction.

Output:
- Library versions
- Accuracy score
- Confusion matrix
- Classification report
"""

from __future__ import annotations
# Prints Python version
import sys

import numpy as np  # Common data handling libraries in Ml pipelines
import pandas as pd  # Common data handling libraries in Ml pipelines
from sklearn.datasets import load_breast_cancer  # Provides a ready-made, trusted classification dataset
from sklearn.linear_model import LogisticRegression  # A standard baseline classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Evaluation metrics
from sklearn.model_selection import train_test_split  # Splits data into training/testing portions
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    # Print environment info
    print("Python:", sys.version.split()[0])
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)

    # Load a simple benchmark classification dataset
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    print("\nDataset loaded for feasibility test:")
    print("X shape:", X.shape)
    print("y distribution:\n", y.value_counts())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline model
    # Logistic Regression is a standard baseline for medical/tabular prediction tasks.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nBaseline Model: Logistic Regression")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Entry point
if __name__ == "__main__":
    main()
