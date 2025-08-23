import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

log = logging.getLogger(__name__)


def train_per_stock(
    df_stock: pd.DataFrame,
    use_kfold: bool = False,
    n_splits: int = 5,
) -> Optional[Tuple[float, float]]:
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
    Optional[Tuple[float, float]]
        Tuple of (accuracy, F1-score) of the model on the evaluation set. ``None``
        is returned for both metrics if there are insufficient samples or only
        one class in the training data.
    """

    if df_stock.empty:
        return None, None

    df = df_stock.sort_values("date").copy()
    # Convert sentiment to numeric and replace invalid entries with 0
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0)

    # Lagged features
    df["Close_lag1"] = df["close"].shift(1)
    df["Sentiment_lag1"] = df["sentiment"].shift(1)

    # Rolling features for close and sentiment
    df["Close_MA3"] = df["close"].rolling(window=3).mean().shift(1)
    df["Close_STD3"] = df["close"].rolling(window=3).std().shift(1)
    df["Sentiment_MA3"] = df["sentiment"].rolling(window=3).mean().shift(1)
    df["Sentiment_STD3"] = df["sentiment"].rolling(window=3).std().shift(1)

    df["y"] = (df["close"] > df["Close_lag1"]).astype(int)
    df.dropna(inplace=True)

    if len(df) < 2:
        return None, None

    features = [
        "Close_lag1",
        "Sentiment_lag1",
        "Close_MA3",
        "Close_STD3",
        "Sentiment_MA3",
        "Sentiment_STD3",
    ]

    X = df[features]
    y = df["y"]

    model = LogisticRegression()

    if use_kfold and len(df) >= n_splits:
        kf = KFold(n_splits=n_splits)
        accuracies = []
        f1_scores = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

        if not accuracies:
            return None, None

        accuracy = float(np.mean(accuracies))
        f1 = float(np.mean(f1_scores))
    else:
        train_size = max(int(len(df) * 0.8), 1)
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]

        X_train = df_train[features]
        y_train = df_train["y"]

        # Need at least two classes to train logistic regression
        if y_train.nunique() < 2:
            return None, None

        model.fit(X_train, y_train)

        if df_test.empty:
            y_pred = model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)
            f1 = f1_score(y_train, y_pred, zero_division=0)
        else:
            X_test = df_test[features]
            y_test = df_test["y"]
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)

    log.info(f"Model accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    return float(accuracy), float(f1)
