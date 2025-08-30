import logging
from collections import Counter

import pandas as pd
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
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def _can_use_smote(y, k_neighbors: int = 5) -> bool:
    cnt = Counter(y)
    if len(cnt) < 2:
        return False
    n_min = min(cnt.values())
    return n_min > k_neighbors


def backtest_strategy(df: pd.DataFrame, signals: pd.Series) -> float:
    """Average next-day return when signal == 1 (long)."""
    bt = df.copy()
    bt["next_close"] = bt["close"].shift(-1)
    bt["return"] = bt["next_close"] / bt["close"] - 1
    returns = bt.loc[signals == 1, "return"].dropna()
    return float(returns.mean()) if not returns.empty else 0.0


def train_per_stock(
    df_stock: pd.DataFrame,
    use_kfold: bool = True,
    n_splits: int = 5,
    model_type: str = "logistic",
    balance_method: str = "class_weight",
    search_method: str = "grid",
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """
    Train a classifier on lagged close+sentiment; predict if next close > prev close.

    Returns (accuracy, f1, roc_auc, precision, recall) on hold-out or CV eval.
    """
    if df_stock.empty:
        return None, None, None, None, None

    required_cols = {"date", "close", "sentiment"}
    if not required_cols.issubset(df_stock.columns):
        return None, None, None, None, None

    df = df_stock.sort_values("date").copy()
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0)

    # --- Lagged + rolling features
    df["Close_lag1"] = df["close"].shift(1)
    df["Close_lag2"] = df["close"].shift(2)
    df["Close_lag3"] = df["close"].shift(3)
    df["Sentiment_lag1"] = df["sentiment"].shift(1)
    df["Sentiment_lag2"] = df["sentiment"].shift(2)
    df["Sentiment_lag3"] = df["sentiment"].shift(3)

    df["Close_MA3"] = df["close"].rolling(3).mean().shift(1)
    df["Close_STD3"] = df["close"].rolling(3).std().shift(1)
    df["Sentiment_MA3"] = df["sentiment"].rolling(3).mean().shift(1)
    df["Sentiment_STD3"] = df["sentiment"].rolling(3).std().shift(1)

    df["Close_MA7"] = df["close"].rolling(7).mean().shift(1)
    df["Close_STD7"] = df["close"].rolling(7).std().shift(1)
    df["Sentiment_MA7"] = df["sentiment"].rolling(7).mean().shift(1)
    df["Sentiment_STD7"] = df["sentiment"].rolling(7).std().shift(1)

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

    # --- Optional OHLCV features
    extra_cols = {"open": "Open", "high": "High", "low": "Low", "volume": "Volume"}
    for col, prefix in extra_cols.items():
        if col in df.columns:
            df[f"{prefix}_lag1"] = df[col].shift(1)
            df[f"{prefix}_lag2"] = df[col].shift(2)
            df[f"{prefix}_lag3"] = df[col].shift(3)
            df[f"{prefix}_MA3"] = df[col].rolling(3).mean().shift(1)
            df[f"{prefix}_STD3"] = df[col].rolling(3).std().shift(1)
            df[f"{prefix}_MA7"] = df[col].rolling(7).mean().shift(1)
            df[f"{prefix}_STD7"] = df[col].rolling(7).std().shift(1)
            features += [
                f"{prefix}_lag1",
                f"{prefix}_lag2",
                f"{prefix}_lag3",
                f"{prefix}_MA3",
                f"{prefix}_STD3",
                f"{prefix}_MA7",
                f"{prefix}_STD7",
            ]

    # --- Technicals (defensiv bereinigt)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["RSI"] = (100 - (100 / (1 + rs))).shift(1)
    df["RSI"] = df["RSI"].replace([float("inf"), float("-inf")], pd.NA)
    features.append("RSI")

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd.shift(1)
    df["MACD_Signal"] = signal.shift(1)
    df["MACD_Hist"] = (macd - signal).shift(1)
    features += ["MACD", "MACD_Signal", "MACD_Hist"]

    if len(df) >= 21:
        bb_ma = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["BB_Middle"] = bb_ma.shift(1)
        df["BB_Upper"] = (bb_ma + 2 * bb_std).shift(1)
        df["BB_Lower"] = (bb_ma - 2 * bb_std).shift(1)
        features += ["BB_Upper", "BB_Lower", "BB_Middle"]

    # --- Label & Cleanup
    df["y"] = (df["close"] > df["Close_lag1"]).astype(int)
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna()

    if len(df) < 2 or df["y"].nunique() < 2:
        if balance_method in {"smote", "undersample"}:
            return 0.0, 0.0, None, 0.0, 0.0
        return None, None, None, None, None

    X, y = df[features], df["y"]
    class_counts = Counter(y)
    if min(class_counts.values()) < 2:
        log.info("Insufficient samples per class: %s", dict(class_counts))
        if balance_method in {"smote", "undersample"}:
            return 0.0, 0.0, None, 0.0, 0.0
        return None, None, None, None, None
    n_splits = min(n_splits, max(2, min(class_counts.values())))

    if balance_method == "smote":
        try:
            from imblearn.over_sampling import SMOTE  # type: ignore
            from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
        except Exception:
            balance_method = "none"
        else:
            if not _can_use_smote(y):
                balance_method = "none"
            else:
                k_neighbors = min(5, min(class_counts.values()) - 1)
    elif balance_method == "undersample":
        if min(class_counts.values()) < 2:
            balance_method = "none"
        else:
            try:
                from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
                from imblearn.under_sampling import RandomUnderSampler  # type: ignore
            except Exception:
                balance_method = "none"

    # Baselines (Logging)
    class_distribution = y.value_counts().sort_index().to_dict()
    log.info("Class distribution: %s", class_distribution)
    majority_class = int(y.mode()[0])
    majority_pred = pd.Series(majority_class, index=y.index)
    log.info(
        "Baseline majority acc=%.4f f1=%.4f",
        accuracy_score(y, majority_pred),
        f1_score(y, majority_pred, zero_division=0),
    )
    buy_hold_pred = pd.Series(1, index=y.index)
    log.info(
        "Baseline buy&hold acc=%.4f f1=%.4f",
        accuracy_score(y, buy_hold_pred),
        f1_score(y, buy_hold_pred, zero_division=0),
    )

    # --- Model + Search Space
    param_distributions = None
    param_grid = None
    optuna_distributions = None
    use_optuna = search_method == "optuna"

    if model_type == "logistic":
        param_name = "clf__C"
        if balance_method == "smote":
            model = ImbPipeline(
                [
                    ("sampler", SMOTE(random_state=42, k_neighbors=k_neighbors)),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            )
        elif balance_method == "undersample":
            model = ImbPipeline(
                [
                    ("sampler", RandomUnderSampler(random_state=42)),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            )
        else:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            )

        param_grid = {param_name: [0.01, 0.1, 1.0, 10.0]}
        if search_method == "random":
            param_distributions = {param_name: loguniform(1e-4, 1e2)}
        elif use_optuna:
            optuna_distributions = {"clf__C": ("float_log", 1e-4, 1e2)}

    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, None]}
        if search_method == "random":
            param_distributions = {
                "n_estimators": randint(50, 401),
                "max_depth": [3, 5, 7, None],
                "min_samples_split": randint(2, 11),
            }
        elif use_optuna:
            optuna_distributions = {
                "n_estimators": ("int", 50, 400),
                "max_depth": ("cat", [3, 5, 7, None]),
                "min_samples_split": ("int", 2, 10),
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
        elif use_optuna:
            optuna_distributions = {
                "n_estimators": ("int", 50, 200),
                "learning_rate": ("float_log", 0.01, 0.2),
                "max_depth": ("int", 3, 5),
            }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if search_method not in {"grid", "random", "optuna"}:
        raise ValueError(f"Unknown search_method: {search_method}")

    # --- CV Split Auswahl
    # TimeSeriesSplit braucht >= n_splits+1 Samples
    use_ts_cv = (
        use_kfold and (len(df) >= (n_splits + 1)) and balance_method not in {"smote", "undersample"}
    )
    if use_ts_cv:
        cv = TimeSeriesSplit(n_splits=n_splits)
        X_train, y_train = X, y
        df_train, df_test = df, pd.DataFrame()
    else:
        train_size = max(int(len(df) * 0.8), 1)
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]
        y_train = df_train["y"]
        if balance_method in {"smote", "undersample"} and not df_test.empty:
            while y_train.value_counts().min() < 2 and len(df_train) < len(df):
                df_train = df.iloc[: len(df_train) + 1]
                y_train = df_train["y"]
            df_test = df.iloc[len(df_train) :]
        X_train = df_train[features]
        y_train = df_train["y"]
        if y_train.nunique() < 2 and not df_test.empty:
            df_train = df
            df_test = pd.DataFrame()
            X_train = df_train[features]
            y_train = df_train["y"]
        if y_train.nunique() < 2:
            if balance_method in {"smote", "undersample"}:
                return 0.0, 0.0, None, 0.0, 0.0
            return None, None, None, None, None
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )

    # --- Searcher
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
        # Optuna sanft behandeln (optional dependency)
        try:
            import optuna  # type: ignore
            from optuna.integration import OptunaSearchCV  # type: ignore

            # kleine Helper zur Übersetzung unserer vereinfachten Tuples
            dist_map = {}
            for k, spec in (optuna_distributions or {}).items():
                kind = spec[0]
                if kind == "int":
                    dist_map[k] = optuna.distributions.IntDistribution(spec[1], spec[2])
                elif kind == "float_log":
                    dist_map[k] = optuna.distributions.FloatDistribution(spec[1], spec[2], log=True)
                elif kind == "cat":
                    dist_map[k] = optuna.distributions.CategoricalDistribution(spec[1])
                else:
                    raise ValueError(f"Unknown optuna dist kind: {kind}")

            search = OptunaSearchCV(
                model,
                dist_map,
                cv=cv,
                n_trials=20,
                scoring="accuracy",
                random_state=42,
            )
        except Exception as e:
            log.warning("Optuna not available (%s) – falling back to RandomizedSearchCV.", e)
            # Fallback auf RandomizedSearch mit sinnvoller Dist
            if param_distributions is None:
                # Erzeuge aus param_grid eine simple Random-Variante
                param_distributions = {
                    k: v if isinstance(v, list) else [v] for k, v in (param_grid or {}).items()
                }
            search = RandomizedSearchCV(
                model,
                param_distributions,
                n_iter=20,
                cv=cv,
                scoring={"accuracy": "accuracy", "f1": "f1"},
                refit="accuracy",
                random_state=42,
            )

    try:
        search.fit(X_train, y_train)
    except ValueError:
        if balance_method in {"smote", "undersample"}:
            return 0.0, 0.0, None, 0.0, 0.0
        raise
    best_model = search.best_estimator_
    log.info("%s best params: %s", model_type, getattr(search, "best_params_", {}))

    # --- Evaluation
    if df_test.empty:
        X_eval, y_eval, df_eval = X_train, y_train, df_train
    else:
        X_eval, y_eval, df_eval = df_test[features], df_test["y"], df_test

    y_pred = best_model.predict(X_eval)
    y_proba = (
        best_model.predict_proba(X_eval)[:, 1] if hasattr(best_model, "predict_proba") else None
    )

    accuracy = float(accuracy_score(y_eval, y_pred))
    f1 = float(f1_score(y_eval, y_pred, zero_division=0))
    precision = float(precision_score(y_eval, y_pred, zero_division=0))
    recall = float(recall_score(y_eval, y_pred, zero_division=0))

    roc_auc: float | None = None
    if y_proba is not None and y_eval.nunique() > 1:
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
    log.info("Avg strategy return: %.4f", avg_return)

    return accuracy, f1, roc_auc, precision, recall
