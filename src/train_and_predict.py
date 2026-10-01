from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def clean(df):
    """Fixes applied to every file: train, validation and December."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "weight" in df.columns:
        df["weight"] = df["weight"].abs()  
    return df


def remove_price_outliers(df):
    """Training data only: drop loads priced below 0.5x or above 3x their own quote."""
    ratio = df["posted_rate"] / (df["distance"] * df["quote_signal"])
    return df[(ratio >= 0.5) & (ratio <= 3)]


CATEGORICAL = ["pickup", "delivery", "equipment"]
NUMERIC = ["distance", "weight", "day_of_week", "day_of_month", "month"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "posted_rate"
SPLIT_DATE = "2025-09-01"  


def add_features(df):
    """Turn the date into numbers the model can learn patterns from."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek   
    df["day_of_month"] = df["date"].dt.day        
    df["month"] = df["date"].dt.month             
    return df

def time_split(df):
    """Train on the past, test on the most recent months."""
    train = df[df["date"] < SPLIT_DATE]
    test = df[df["date"] >= SPLIT_DATE]
    return remove_price_outliers(train), test  # clean only the training part


if __name__ == "__main__":
    data = add_features(clean(pd.read_csv(DATA / "train_test.csv")))
    train, test = time_split(data)

    print(f"Train: {len(train):,} rows, {train['date'].min().date()} to {train['date'].max().date()}")
    print(f"Test:  {len(test):,} rows, {test['date'].min().date()} to {test['date'].max().date()}")
    print(f"Test share: {len(test) / (len(train) + len(test)):.1%}")