from pathlib import Path
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


DATA_PATH = Path("data/raw/heart_disease.csv")


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def main():

    print("=== Repeated Stratified Cross Validation Benchmark ===")

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
        "MLP Neural Net": MLPClassifier(max_iter=2000),
        "HistGradientBoosting": HistGradientBoostingClassifier()
    }

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=42
    )

    results = []

    for name, model in models.items():

        print(f"\nTraining: {name}")

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=1
        )

        mean = scores.mean()
        std = scores.std()

        print(f"Mean Accuracy: {mean:.4f}")
        print(f"Std Accuracy: {std:.4f}")

        results.append([name, mean, std])

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Mean Accuracy", "Std Accuracy"]
    )

    results_df = results_df.sort_values(
        by="Mean Accuracy",
        ascending=False
    )

    print("\n=== Final Ranking ===")
    print(results_df)

    results_df.to_csv(
        "experiments/repeated_cv_results.csv",
        index=False
    )

    print("\nResults saved to experiments/repeated_cv_results.csv")


if __name__ == "__main__":
    main()