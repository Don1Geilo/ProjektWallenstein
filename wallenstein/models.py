import logging

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_validate
from scipy.stats import loguniform, randint
import optuna
from optuna.integration import OptunaSearchCV

log = logging.getLogger(__name__)


def train_per_stock(
    df_stock: pd.DataFrame,
    use_kfold: bool = True,
    n_splits: int = 5,
    model_type: str = "logistic",
    search_method: str = "grid",
    ) -> tuple[float, float] | None:
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
    Optional[Tuple[float, float]]
        Tuple of (accuracy, F1-score) of the model on the evaluation set. ``None``
        is returned for both metrics if there are insufficient samples or only
        one class in the training data.

    Notes
    -----
    The function supports several model types (``logistic``, ``random_forest``
    and ``gradient_boosting``) and performs hyperparameter optimisation using
    ``GridSearchCV``, ``RandomizedSearchCV`` or ``Optuna`` depending on the
    ``search_method`` argument. K-fold cross validation is enabled by default
    and falls back to a simple train/test split when there are too few samples.
    """

    if df_stock.empty:
        return None, None

    df = df_stock.sort_values("date").copy()
    # Convert sentiment to numeric and replace invalid entries with 0
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0)

    # Lagged and rolling features for mandatory columns
    df["Close_lag1"] = df["close"].shift(1)
    df["Close_lag2"] = df["close"].shift(2)
    df["Close_lag3"] = df["close"].shift(3)
    df["Sentiment_lag1"] = df["sentiment"].shift(1)
    df["Sentiment_lag2"] = df["sentiment"].shift(2)
    df["Sentiment_lag3"] = df["sentiment"].shift(3)

    df["Close_MA3"] = df["close"].rolling(window=3).mean().shift(1)
    df["Close_STD3"] = df["close"].rolling(window=3).std().shift(1)
    df["Sentiment_MA3"] = df["sentiment"].rolling(window=3).mean().shift(1)
    df["Sentiment_STD3"] = df["sentiment"].rolling(window=3).std().shift(1)

    df["Close_MA7"] = df["close"].rolling(window=7).mean().shift(1)
    df["Close_STD7"] = df["close"].rolling(window=7).std().shift(1)
    df["Sentiment_MA7"] = df["sentiment"].rolling(window=7).mean().shift(1)
    df["Sentiment_STD7"] = df["sentiment"].rolling(window=7).std().shift(1)

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

    # Optional price-related columns
    extra_cols = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
    }
    for col, prefix in extra_cols.items():
        if col in df.columns:
            df[f"{prefix}_lag1"] = df[col].shift(1)
            df[f"{prefix}_lag2"] = df[col].shift(2)
            df[f"{prefix}_lag3"] = df[col].shift(3)
            df[f"{prefix}_MA3"] = df[col].rolling(window=3).mean().shift(1)
            df[f"{prefix}_STD3"] = df[col].rolling(window=3).std().shift(1)
            df[f"{prefix}_MA7"] = df[col].rolling(window=7).mean().shift(1)
            df[f"{prefix}_STD7"] = df[col].rolling(window=7).std().shift(1)
            features.extend(
                [
                    f"{prefix}_lag1",
                    f"{prefix}_lag2",
                    f"{prefix}_lag3",
                    f"{prefix}_MA3",
                    f"{prefix}_STD3",
                    f"{prefix}_MA7",
                    f"{prefix}_STD7",
                ]
            )

    # Technical indicators based on close price
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = (100 - (100 / (1 + rs))).shift(1)
    features.append("RSI")

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd.shift(1)
    df["MACD_Signal"] = signal.shift(1)
    df["MACD_Hist"] = (macd - signal).shift(1)
    features.extend(["MACD", "MACD_Signal", "MACD_Hist"])

    bb_ma = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["BB_Middle"] = bb_ma.shift(1)
    df["BB_Upper"] = (bb_ma + 2 * bb_std).shift(1)
    df["BB_Lower"] = (bb_ma - 2 * bb_std).shift(1)
    features.extend(["BB_Upper", "BB_Lower", "BB_Middle"])

    df["y"] = (df["close"] > df["Close_lag1"]).astype(int)
    df.dropna(inplace=True)

    if len(df) < 2:
        return None, None

    X = df[features]
    y = df["y"]

    # Log class distribution
    class_distribution = y.value_counts().sort_index().to_dict()
    log.info(f"Class distribution: {class_distribution}")

    # Baseline predictions
    majority_class = y.mode()[0]
    majority_pred = pd.Series(majority_class, index=y.index)
    majority_acc = accuracy_score(y, majority_pred)
    majority_f1 = f1_score(y, majority_pred, zero_division=0)
    log.info(
        "Baseline (majority class %s) accuracy: %.4f, F1: %.4f",
        majority_class,
        majority_acc,
        majority_f1,
    )

    buy_hold_pred = pd.Series(1, index=y.index)
    buy_hold_acc = accuracy_score(y, buy_hold_pred)
    buy_hold_f1 = f1_score(y, buy_hold_pred, zero_division=0)
    log.info(
        "Baseline (buy and hold) accuracy: %.4f, F1: %.4f",
        buy_hold_acc,
        buy_hold_f1,
    )

    if y.nunique() < 2:
        return None, None

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
        param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
        param_distributions = {"C": loguniform(1e-4, 1e2)}
        optuna_distributions = {
            "C": optuna.distributions.FloatDistribution(1e-4, 1e2, log=True)
        }
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, None],
        }
        param_distributions = {
            "n_estimators": randint(50, 401),
            "max_depth": [3, 5, 7, None],
            "min_samples_split": randint(2, 11),
        }
        optuna_distributions = {
            "n_estimators": optuna.distributions.IntDistribution(50, 400),
            "max_depth": optuna.distributions.CategoricalDistribution([3, 5, 7, None]),
            "min_samples_split": optuna.distributions.IntDistribution(2, 10),
        }
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5],
        }
        param_distributions = {
            "n_estimators": randint(50, 301),
            "learning_rate": loguniform(0.01, 0.2),
            "max_depth": randint(3, 8),
        }
        optuna_distributions = {
            "n_estimators": optuna.distributions.IntDistribution(50, 300),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.2, log=True),
            "max_depth": optuna.distributions.IntDistribution(3, 7),
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if search_method not in {"grid", "random", "optuna"}:
        raise ValueError(f"Unknown search_method: {search_method}")

    if use_kfold and len(df) >= n_splits:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        if search_method == "grid":
            search = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                scoring={"accuracy": "accuracy", "f1": "f1"},
                refit="accuracy",
            )
        elif search_method == "random":
            search = RandomizedSearchCV(
                model,
                param_distributions,
                n_iter=20,
                cv=cv,
                scoring={"accuracy": "accuracy", "f1": "f1"},
                refit="accuracy",
                random_state=42,
            )
        else:
            search = OptunaSearchCV(
                model,
                optuna_distributions,
                cv=cv,
                n_trials=20,
                scoring="accuracy",
                random_state=42,
            )
        search.fit(X, y)
        if search_method == "optuna":
            best_model = search.best_estimator_
            scores = cross_validate(
                best_model, X, y, cv=cv, scoring={"accuracy": "accuracy", "f1": "f1"}
            )
            accuracy = float(scores["test_accuracy"].mean())
            f1 = float(scores["test_f1"].mean())
        else:
            accuracy = float(
                search.cv_results_["mean_test_accuracy"][search.best_index_]
            )
            f1 = float(search.cv_results_["mean_test_f1"][search.best_index_])
        log.info(f"{model_type} best params: {search.best_params_}")
    else:
        train_size = max(int(len(df) * 0.8), 1)
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]

        X_train = df_train[features]
        y_train = df_train["y"]

        if y_train.nunique() < 2:
            return None, None

        cv = min(3, len(y_train))
        if search_method == "grid":
            search = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                scoring={"accuracy": "accuracy", "f1": "f1"},
                refit="accuracy",
            )
        elif search_method == "random":
            search = RandomizedSearchCV(
                model,
                param_distributions,
                n_iter=20,
                cv=cv,
                scoring={"accuracy": "accuracy", "f1": "f1"},
                refit="accuracy",
                random_state=42,
            )
        else:
            search = OptunaSearchCV(
                model,
                optuna_distributions,
                cv=cv,
                n_trials=20,
                scoring="accuracy",
                random_state=42,
            )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        log.info(f"{model_type} best params: {search.best_params_}")

        if df_test.empty:
            y_pred = best_model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)
            f1 = f1_score(y_train, y_pred, zero_division=0)
        else:
            X_test = df_test[features]
            y_test = df_test["y"]
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)

    log.info(f"Model accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    return float(accuracy), float(f1)
