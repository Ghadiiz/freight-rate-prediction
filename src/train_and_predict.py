from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


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
    return remove_price_outliers(train), test  

def baseline_predict(train, test):
    """Simple benchmark: median rate per mile of the same lane and equipment."""
    rpm = train[TARGET] / train["distance"]
    lane_rpm = rpm.groupby([train["pickup"], train["delivery"], train["equipment"]]).median()
    equip_rpm = rpm.groupby(train["equipment"]).median()

    keys = pd.MultiIndex.from_frame(test[["pickup", "delivery", "equipment"]])
    test_rpm = pd.Series(lane_rpm.reindex(keys).values, index=test.index)
    test_rpm = test_rpm.fillna(test["equipment"].map(equip_rpm))  # unseen lane -> equipment median
    return test_rpm * test["distance"]


def to_matrix(df, categories=None):
    """Select the model columns; text columns become categories."""
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        if categories is None:
            X[col] = X[col].astype("category")
        else:
            # Cities never seen in training become "missing" instead of crashing
            known = X[col].where(X[col].isin(categories[col]))
            X[col] = pd.Categorical(known, categories=categories[col])
    return X


def fit(train):
    X = to_matrix(train)
    categories = {col: X[col].cat.categories for col in CATEGORICAL}
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=500,
        categorical_features="from_dtype",
        random_state=42,
    )
    model.fit(X, train[TARGET])
    return model, categories


def predict(model, categories, df):
    return model.predict(to_matrix(df, categories))


def report(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    print(f"{name:<10} MAE: ${mae:,.2f}   MAPE: {mape:.2%}")


if __name__ == "__main__":
    data = add_features(clean(pd.read_csv(DATA / "train_test.csv")))
    train, test = time_split(data)
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")

    report("Baseline", test[TARGET], baseline_predict(train, test))

    model, categories = fit(train)
    report("Model", test[TARGET], predict(model, categories, test))