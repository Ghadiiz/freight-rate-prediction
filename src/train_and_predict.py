from pathlib import Path

import pandas as pd

# Folder paths, so the script works no matter where it is run from
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def clean(df):
    """Fixes applied to every file: train, validation and December."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "weight" in df.columns:
        df["weight"] = df["weight"].abs()  # negative weights are sign errors
    return df


def remove_price_outliers(df):
    """Training data only: drop loads priced below 0.5x or above 3x their own quote."""
    ratio = df["posted_rate"] / (df["distance"] * df["quote_signal"])
    return df[(ratio >= 0.5) & (ratio <= 3)]


if __name__ == "__main__":
    raw = pd.read_csv(DATA / "train_test.csv")
    data = clean(raw)
    filtered = remove_price_outliers(data)

    print("Negative weights before:", (raw["weight"] < 0).sum())
    print("Negative weights after: ", (data["weight"] < 0).sum())
    print("Date type:", data["date"].dtype)
    print(f"Rows before outlier filter: {len(data):,}")
    print(f"Rows after outlier filter:  {len(filtered):,} (removed {len(data) - len(filtered)})")