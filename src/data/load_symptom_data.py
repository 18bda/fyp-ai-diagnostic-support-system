from __future__ import annotations

from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")


def load_symptom_dataset(filename: str) -> pd.DataFrame:
    """
    Load a symptom-based disease dataset from data/raw/.

    Note: Raw datasets are excluded from GitHub to avoid large files and licensing issues.
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place your CSV in data/raw/ (not committed), then rerun."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    # Change filename to match your downloaded dataset name
    df = load_symptom_dataset("symptom_dataset.csv")
    print("Loaded:", df.shape)
    print(df.head())
