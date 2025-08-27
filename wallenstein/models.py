import logging

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import loguniform, randint
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    balance_method: str = "class_weight",
    search_method: str = "grid",
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Train a classifier on lagged close and sentiment data.

    Parameters
    ----------
    df_stock : pd.DataFrame
        DataFrame containing at least ``date``, ``close`` and ``sentiment`` columns.
        The function will sort by date, create lagged features of ``close`` and
        ``sentiment`` and attempt to predict if the next day's close is higher
        than the previous day's.
    use_kfold : bool, default True
        If ``True`` and there are enough samples, perform ``TimeSeriesSplit``
        cross-validation on the whole dataset. Otherwise a simple train/test
        split is used.
    n_splits : int, default 5
        Number of folds for cross-validation.
    model_type : str, default "logistic"
        Which classifier to train. Supported values are ``"logistic"``,
        ``"random_forest"`` and ``"gradient_boosting"``.
    balance_method : str, default "class_weight"
        Strategy for handling class imbalance when ``model_type`` is
        ``"logistic"``. Use ``"class_weight"`` for
        ``LogisticRegression(class_weight="balanced")``, ``"smote"`` for SMOTE
        oversampling or ``"undersample"`` for random undersampling.
    search_method : str, default "grid"
        Hyperparameter search strategy: ``"grid"``, ``"random"`` or
        ``"optuna"``.

    Returns
    -------
    tuple[float | None, float | None, float | None, float | None, float | None]
        Tuple of (accuracy, F1-score, ROC-AUC, precision, recall) evaluated on
        a hold-out set. ``None`` is returned for all metrics if there are
        insufficient samples or only one class in the training data.
    """
    if df_stock.empty:
        return None, None, None, None, None

    required_cols = {"date", "close", "sentiment"}
    if not required_cols.issubset(df_stock.columns):
        return None, None, None, None, None

    df = df_stock.sort_values("date").copy()
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

    # Technical indicators
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

    if len(df) >= 21:
        bb_ma = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["BB_Middle"] = bb_ma.shift(1)
        df["BB_Upper"] = (bb_ma + 2 * bb_std).shift(1)
        df["BB_Lower"] = (bb_ma - 2 * bb_std).shift(1)
        features.extend(["BB_Upper", "BB_Lower", "BB_Middle"])

    df["y"] = (df["close"] > df["Close_lag1"]).astype(int)
    df.dropna(inplace=True)

    if len(df) < 2:
        return None, None, None, None, None

    X = df[features]
    y = df["y"]

    class_distribution = y.value_counts().sort_index().to_dict()
    log.info(f"Class distribution: {class_distribution}")

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
        return None, None, None, None, None

    if search_method == "optuna":
        import optuna  # type: ignore
        from optuna.integration import OptunaSearchCV  # type: ignore

    param_distributions = None
    optuna_distributions = None

    if model_type == "logistic":
        if balance_method == "smote":
            steps = [
                ("sampler", SMOTE(random_state=42, k_neighbors=1)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
            model = ImbPipeline(steps)
            param_name = "clf__C"
        elif balance_method == "undersample":
            steps = [
                ("sampler", RandomUnderSampler(random_state=42)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
            model = ImbPipeline(steps)
            param_name = "clf__C"
        else:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            )
            param_name = "clf__C"
        param_grid = {param_name: [0.01, 0.1, 1.0, 10.0]}
        if search_method == "random":
            param_distributions = {param_name: loguniform(1e-4, 1e2)}
        elif search_method == "optuna":
            optuna_distributions = {
                param_name: optuna.distributions.FloatDistribution(1e-4, 1e2, log=True)
            }
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, None],
        }
        if search_method == "random":
            param_distributions = {
                "n_estimators": randint(50, 401),
                "max_depth": [3, 5, 7, None],
                "min_samples_split": randint(2, 11),
            }
        elif search_method == "optuna":
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
        if search_method == "random":
            param_distributions = {
                "n_estimators": randint(50, 201),
                "learning_rate": loguniform(1e-3, 1e-1),
                "max_depth": randint(3, 6),
            }
        elif search_method == "optuna":
            optuna_distributions = {
                "n_estimators": optuna.distributions.IntDistribution(50, 200),
                "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.2, log=True),
                "max_depth": optuna.distributions.IntDistribution(3, 5),
            }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if search_method not in {"grid", "random", "optuna"}:
        raise ValueError(f"Unknown search_method: {search_method}")

    if use_kfold and len(df) >= n_splits:
        if balance_method in {"smote", "undersample"}:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = TimeSeriesSplit(n_splits=n_splits)
        X_train, y_train = X, y
        df_train, df_test = df, pd.DataFrame()
    else:
        train_size = max(int(len(df) * 0.8), 1)
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]
        X_train = df_train[features]
        y_train = df_train["y"]
        if y_train.nunique() < 2:
            return None, None, None, None, None
        cv = KFold(n_splits=min(3, len(y_train)), shuffle=True, random_state=42)

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
    else:  # optuna
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
