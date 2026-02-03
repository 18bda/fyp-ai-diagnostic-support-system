import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/Training.csv")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}. Put Training.csv in data/raw/")

    df = pd.read_csv(DATA_PATH)

    print("=== Symptom Dataset Quick Check ===")
    print("Shape:", df.shape)
    print("Columns (first 10):", df.columns[:10].tolist())

    target = "prognosis"
    print("Target column present:", target in df.columns)
    if target not in df.columns:
        raise ValueError(f"Expected target column '{target}' not found.")

    print("\nMissing values per column (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print("\nClass distribution (top 10):")
    print(df[target].value_counts().head(10))

    # Feature sanity check: symptoms are usually 0/1
    feature_cols = [c for c in df.columns if c != target]
    unique_vals = pd.Series(pd.unique(df[feature_cols].values.ravel())).dropna()
    sample_vals = sorted(set(unique_vals.tolist()))[:10]
    print("\nUnique values in feature columns (sample):", sample_vals)

    print("\n✅ Dataset loads correctly and is suitable for ML (tabular multi-class classification).")

if __name__ == "__main__":
    main()
