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

# Columns the model will use, all available in train, validation AND December
CATEGORICAL = ["pickup", "delivery", "equipment"]
NUMERIC = ["distance", "weight", "day_of_week", "day_of_month", "month"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "posted_rate"


def add_features(df):
    """Turn the date into numbers the model can learn patterns from."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek   # 0 = Monday ... 6 = Sunday
    df["day_of_month"] = df["date"].dt.day        # 1 ... 31
    df["month"] = df["date"].dt.month             # 1 ... 12
    return df


if __name__ == "__main__":
    data = add_features(clean(pd.read_csv(DATA / "train_test.csv")))

    print(data[["date"] + FEATURES + [TARGET]].head())
    print()
    print("Missing values in model columns:")
    print(data[FEATURES].isna().sum())