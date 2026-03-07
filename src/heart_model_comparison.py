from pathlib import Path
import pandas as pd

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
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

DATA_PATH = Path("data/raw/heart_disease.csv")


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]

    print("=== Heart Disease Dataset ===")
    print("Shape:", df.shape)
    print("Features:", X.shape)
    print("Classes:", y.value_counts())

    return X, y


def main():

    X, y = load_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Extra Trees": ExtraTreesClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "SVM": SVC(),
        "MLP Neural Net": MLPClassifier(max_iter=2000)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    print("\n=== Model Benchmark ===")

    for name, model in models.items():

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )

        mean_score = scores.mean()
        std_score = scores.std()

        print(f"{name}: {mean_score:.4f} ± {std_score:.4f}")

        results.append({
            "Model": name,
            "Mean Accuracy": mean_score,
            "Std": std_score
        })

    results_df = pd.DataFrame(results).sort_values(
        by="Mean Accuracy",
        ascending=False
    )

    print("\n=== Final Ranking ===")
    print(results_df)

    results_df.to_csv(
        "experiments/heart_model_comparison.csv",
        index=False
    )

    print("\nResults saved to experiments/heart_model_comparison.csv")


if __name__ == "__main__":
    main()