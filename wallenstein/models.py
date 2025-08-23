import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Optional


def train_per_stock(df_stock: pd.DataFrame) -> Optional[float]:
    """Train a simple LogisticRegression on lagged close and sentiment data.

    Parameters
    ----------
    df_stock : pd.DataFrame
        DataFrame containing at least ``date``, ``close`` and ``sentiment`` columns.
        The function will sort by date, create lagged features of ``close`` and
        ``sentiment`` and attempt to predict if the next day's close is higher
        than the previous day's.

    Returns
    -------
    Optional[float]
        Accuracy of the model on the hold‑out test set. ``None`` is returned if
        there are insufficient samples or only one class in the training data.
    """

    if df_stock.empty:
        return None

    df = df_stock.sort_values("date").copy()
    df["sentiment"] = df["sentiment"].fillna(0).infer_objects(copy=False)

    # Lagged features
    df["Close_lag1"] = df["close"].shift(1)
    df["Sentiment_lag1"] = df["sentiment"].shift(1)
    df["y"] = (df["close"] > df["Close_lag1"]).astype(int)
    df.dropna(inplace=True)

    if len(df) < 2:
        return None

    train_size = max(int(len(df) * 0.8), 1)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    X_train = df_train[["Close_lag1", "Sentiment_lag1"]]
    y_train = df_train["y"]

    # Need at least two classes to train logistic regression
    if y_train.nunique() < 2:
        return None

    model = LogisticRegression()
    model.fit(X_train, y_train)

    if df_test.empty:
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
    else:
        X_test = df_test[["Close_lag1", "Sentiment_lag1"]]
        y_test = df_test["y"]
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    return float(accuracy)
