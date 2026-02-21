from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

DATA_PATH = Path("data/raw/Training.csv")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    return df


def main() -> None:
    print("=== Multi-Algorithm Symptom Benchmark ===")

    df = load_data()
    X = df.drop(columns=["prognosis"])
    y = df["prognosis"]

    print("Dataset:", X.shape)
    print("Classes:", y.nunique())

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Extra Trees": ExtraTreesClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "SVM": SVC(),
        "MLP (Neural Net)": MLPClassifier(max_iter=1000, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    for name, model in models.items():
        print(f"\nTraining: {name}")

        pipeline = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", model)
        ])

        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

        results.append({
            "Model": name,
            "Mean Accuracy": scores.mean(),
            "Std Accuracy": scores.std()
        })

        print(f"Mean CV Accuracy: {scores.mean():.4f}")
        print(f"Std CV Accuracy: {scores.std():.4f}")

    results_df = pd.DataFrame(results).sort_values(
        by="Mean Accuracy", ascending=False
    )

    print("\n=== Final Comparison ===")
    print(results_df)

    results_df.to_csv("experiments/symptom_model_comparison.csv", index=False)
    print("\nResults saved to experiments/symptom_model_comparison.csv")


if __name__ == "__main__":
    main()