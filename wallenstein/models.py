import logging

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold

log = logging.getLogger(__name__)


def backtest_strategy(df: pd.DataFrame, signals: pd.Series) -> float:
    """Compute average next-day return when signal is 1 (buy)."""

    bt = df.copy()
    bt["next_close"] = bt["close"].shift(-1)
    bt["return"] = bt["next_close"] / bt["close"] - 1
    returns = bt.loc[signals == 1, "return"].dropna()
    if returns.empty:
        return 0.0
    return float(returns.mean())


def train_per_stock(
    df_stock: pd.DataFrame,
    use_kfold: bool = True,
    n_splits: int = 5,
    model_type: str = "logistic",
) -> tuple[float, float, float | None, float, float] | None:
    """Train a classifier on lagged close and sentiment data.

    Parameters
    ----------
    df_stock : pd.DataFrame
        DataFrame containing at least ``date``, ``close`` and ``sentiment`` columns.
        The function will sort by date, create lagged features of ``close`` and
        ``sentiment`` and attempt to predict if the next day's close is higher
        than the previous day's.

    Returns
    -------
    Optional[Tuple[float, float, float | None, float, float]]
        Tuple of (accuracy, F1-score, ROC-AUC, precision, recall) of the model
        on the evaluation set. ``None`` is returned for all metrics if there are
        insufficient samples or only one class in the training data.

    Notes
    -----
    The function supports several model types (``logistic``, ``random_forest``
    and ``gradient_boosting``) and performs a GridSearchCV for basic
    hyperparameter optimisation. K-fold cross validation is enabled by default
    and falls back to a simple train/test split when there are too few
    samples.
    """

    if df_stock.empty:
        return None, None, None, None, None

    df = df_stock.sort_values("date").copy()
    # Convert sentiment to numeric and replace invalid entries with 0
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0)

    # Lagged features for close and sentiment
    df["Close_lag1"] = df["close"].shift(1)
    df["Close_lag2"] = df["close"].shift(2)
    df["Close_lag3"] = df["close"].shift(3)
    df["Sentiment_lag1"] = df["sentiment"].shift(1)
    df["Sentiment_lag2"] = df["sentiment"].shift(2)
    df["Sentiment_lag3"] = df["sentiment"].shift(3)

    # Rolling features for close and sentiment
    df["Close_MA3"] = df["close"].rolling(window=3).mean().shift(1)
    df["Close_STD3"] = df["close"].rolling(window=3).std().shift(1)
    df["Sentiment_MA3"] = df["sentiment"].rolling(window=3).mean().shift(1)
    df["Sentiment_STD3"] = df["sentiment"].rolling(window=3).std().shift(1)

    df["Close_MA7"] = df["close"].rolling(window=7).mean().shift(1)
    df["Close_STD7"] = df["close"].rolling(window=7).std().shift(1)
    df["Sentiment_MA7"] = df["sentiment"].rolling(window=7).mean().shift(1)
    df["Sentiment_STD7"] = df["sentiment"].rolling(window=7).std().shift(1)

    df["y"] = (df["close"] > df["Close_lag1"]).astype(int)
    df.dropna(inplace=True)

    if len(df) < 2:
        return None, None, None, None, None

    features = [
        "Close_lag1",
        "Close_lag2",
        "Close_lag3",
        "Sentiment_lag1",
        "Sentiment_lag2",
        "Sentiment_lag3",
        "Close_MA3",
        "Close_STD3",
        "Sentiment_MA3",
        "Sentiment_STD3",
        "Close_MA7",
        "Close_STD7",
        "Sentiment_MA7",
        "Sentiment_STD7",
    ]

    X = df[features]
    y = df["y"]

    if y.nunique() < 2:
        return None, None, None, None, None

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
        param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, None],
        }
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5],
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if use_kfold and len(df) >= n_splits:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring={"accuracy": "accuracy", "f1": "f1"},
            refit="accuracy",
        )
        search.fit(X, y)
        best_model = search.best_estimator_
        log.info(f"{model_type} best params: {search.best_params_}")
        X_eval, y_eval, df_eval = X, y, df
    else:
        train_size = max(int(len(df) * 0.8), 1)
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]

        X_train = df_train[features]
        y_train = df_train["y"]

        if y_train.nunique() < 2:
            return None, None, None, None, None

        cv = min(3, len(y_train))
        search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring={"accuracy": "accuracy", "f1": "f1"},
            refit="accuracy",
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        log.info(f"{model_type} best params: {search.best_params_}")

        if df_test.empty:
            X_eval, y_eval, df_eval = X_train, y_train, df_train
        else:
            X_eval, y_eval, df_eval = df_test[features], df_test["y"], df_test

    y_pred = best_model.predict(X_eval)
    y_proba = (
        best_model.predict_proba(X_eval)[:, 1]
        if hasattr(best_model, "predict_proba")
        else None
    )

    accuracy = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred, zero_division=0)
    precision = precision_score(y_eval, y_pred, zero_division=0)
    recall = recall_score(y_eval, y_pred, zero_division=0)

    roc_auc = None
    if y_proba is not None and len(set(y_eval)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_eval, y_proba))
        except ValueError:
            roc_auc = None

    avg_return = backtest_strategy(df_eval, pd.Series(y_pred, index=df_eval.index))

    log.info(
        "Model accuracy: %.4f, F1: %.4f, ROC-AUC: %s, Precision: %.4f, Recall: %.4f",
        accuracy,
        f1,
        f"{roc_auc:.4f}" if roc_auc is not None else "nan",
        precision,
        recall,
    )
    log.info(f"Avg strategy return: {avg_return:.4f}")
    return float(accuracy), float(f1), roc_auc, float(precision), float(recall)
