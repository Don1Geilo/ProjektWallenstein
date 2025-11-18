import logging
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.impute import SimpleImputer
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


def _find_optimal_threshold(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float]:
    """Return (threshold, f1) that maximises F1 on ``y_true``."""

    if y_proba.size == 0:
        return 0.5, 0.0

    thresholds = np.linspace(0.2, 0.8, 25)
    best_thr = 0.5
    best_f1 = -1.0
    y_true_arr = np.asarray(y_true, dtype=int)

    for thr in thresholds:
        preds = (y_proba >= thr).astype(int)
        score = f1_score(y_true_arr, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(thr)

    return best_thr, max(best_f1, 0.0)


def _calibrate_dual_thresholds(
    y_true: pd.Series, y_proba: np.ndarray
) -> tuple[float, float]:
    """Learn asymmetric buy/sell thresholds based on validation metrics."""

    if y_proba.size == 0 or y_true.empty:
        return 0.55, 0.45

    thresholds = np.linspace(0.25, 0.75, 21)
    y_true_arr = np.asarray(y_true, dtype=int)

    best_buy_thr = 0.55
    best_buy_prec = -1.0
    best_sell_thr = 0.45
    best_sell_recall0 = -1.0

    for thr in thresholds:
        buy_preds = (y_proba >= thr).astype(int)
        prec = precision_score(y_true_arr, buy_preds, zero_division=0)
        if prec > best_buy_prec:
            best_buy_prec = float(prec)
            best_buy_thr = float(thr)

        neg_mask = y_true_arr == 0
        if not np.any(neg_mask):
            continue
        sell_preds = y_proba <= thr
        tp_neg = np.sum(sell_preds & neg_mask)
        fn_neg = np.sum(~sell_preds & neg_mask)
        recall0 = tp_neg / (tp_neg + fn_neg) if (tp_neg + fn_neg) else 0.0
        if recall0 > best_sell_recall0:
            best_sell_recall0 = float(recall0)
            best_sell_thr = float(thr)

    return best_buy_thr, best_sell_thr


def derive_signal_from_proba(
    next_day_proba: float | None,
    buy_threshold: float,
    sell_threshold: float,
    expected_return: float | None = None,
) -> tuple[str | None, float | None, float | None]:
    """Translate an up-move probability into a trading signal."""

    if next_day_proba is None:
        return None, None, None

    try:
        proba = float(next_day_proba)
    except (TypeError, ValueError):
        return None, None, None

    if np.isnan(proba):
        return None, None, None

    buy_thr = min(max(float(buy_threshold), 0.0), 1.0)
    sell_thr = min(max(float(sell_threshold), 0.0), buy_thr)

    if expected_return is not None and expected_return < 0:
        down_proba = 1.0 - proba
        return "sell", down_proba, abs(expected_return)

    if proba >= buy_thr:
        return "buy", proba, proba - buy_thr

    if proba <= sell_thr:
        down_proba = 1.0 - proba
        return "sell", down_proba, sell_thr - proba

    down_proba = 1.0 - proba
    confidence = max(proba, down_proba)
    margin = min(buy_thr - proba, proba - sell_thr)
    return "hold", confidence, max(margin, 0.0)


def train_per_stock(
    df_stock: pd.DataFrame,
    use_kfold: bool = True,
    n_splits: int = 5,
    model_type: str = "logistic",
    balance_method: str = "class_weight",
    search_method: str = "grid",
) -> tuple[float | None, float | None, float | None, float | None, float | None, dict | None]:
    """
    Train a classifier on lagged close+sentiment; predict if next close > prev close.

    Returns (accuracy, f1, roc_auc, precision, recall, metadata) on hold-out or CV
    evaluation. ``metadata`` contains the next-day probability and derived signal
    when enough history is present.
    """
    if df_stock.empty:
        return None, None, None, None, None, None

    required_cols = {"date", "close", "sentiment"}
    if not required_cols.issubset(df_stock.columns):
        return None, None, None, None, None, None

    df_stock = df_stock.copy()
    df_stock["date"] = pd.to_datetime(df_stock["date"], errors="coerce")
    df_stock = df_stock.sort_values("date")
    hist_len = len(df_stock)
    df_stock["sentiment"] = pd.to_numeric(df_stock["sentiment"], errors="coerce").fillna(0)
    df_stock["close"] = pd.to_numeric(df_stock["close"], errors="coerce")

    valid_dates = df_stock["date"].dropna()
    if valid_dates.empty:
        return None, None, None, None, None, None

    latest_date = valid_dates.max()
    future_date = latest_date + pd.Timedelta(days=1)

    numeric_like = {"close", "open", "high", "low", "volume", "adj_close", "sentiment"}
    future_row: dict[str, object] = {}
    for col in df_stock.columns:
        if col == "date":
            future_row[col] = future_date
        elif col in numeric_like:
            future_row[col] = np.nan
        else:
            future_row[col] = pd.NA
    if "date" not in future_row:
        future_row["date"] = future_date
    df_all = pd.concat([df_stock, pd.DataFrame([future_row])], ignore_index=True, sort=False)

    for col in ("open", "high", "low", "volume", "adj_close"):
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    df = df_all.sort_values("date").copy()
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0)

    # --- Lagged + rolling features
    df["Close_lag1"] = df["close"].shift(1)
    df["Close_lag2"] = df["close"].shift(2)
    df["Close_lag3"] = df["close"].shift(3)
    df["Sentiment_lag1"] = df["sentiment"].shift(1)
    df["Sentiment_lag2"] = df["sentiment"].shift(2)
    df["Sentiment_lag3"] = df["sentiment"].shift(3)

    df["Close_MA3"] = df["close"].rolling(3, min_periods=1).mean().shift(1)
    df["Close_STD3"] = df["close"].rolling(3, min_periods=2).std().shift(1)
    df["Sentiment_MA3"] = df["sentiment"].rolling(3, min_periods=1).mean().shift(1)
    df["Sentiment_STD3"] = df["sentiment"].rolling(3, min_periods=2).std().shift(1)

    df["Close_MA7"] = df["close"].rolling(7, min_periods=3).mean().shift(1)
    df["Close_STD7"] = df["close"].rolling(7, min_periods=3).std().shift(1)
    df["Sentiment_MA7"] = df["sentiment"].rolling(7, min_periods=3).mean().shift(1)
    df["Sentiment_STD7"] = df["sentiment"].rolling(7, min_periods=3).std().shift(1)

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

    # Momentum- und Trend-Features
    df["Return_1d"] = df["close"].pct_change(fill_method=None).shift(1)
    df["Return_3d"] = df["close"].pct_change(3, fill_method=None).shift(1)
    df["Return_5d"] = df["close"].pct_change(5, fill_method=None).shift(1)
    df["Return_10d"] = df["close"].pct_change(10, fill_method=None).shift(1)
    df["Sentiment_Change1"] = df["sentiment"].diff().shift(1)
    df["Sentiment_Momentum3"] = df["sentiment"].diff().rolling(3, min_periods=1).sum().shift(1)
    df["Sentiment_Momentum7"] = df["sentiment"].diff().rolling(7, min_periods=1).sum().shift(1)
    df["Volatility_7d"] = (
        df["close"].pct_change(fill_method=None).rolling(7, min_periods=2).std().shift(1)
    )

    price_returns = df["close"].pct_change(fill_method=None)
    sentiment_series = df["sentiment"]
    df["Price_Sentiment_Corr7"] = (
        price_returns.rolling(7, min_periods=3).corr(sentiment_series).shift(1)
    )
    df["Return_Momentum7"] = price_returns.rolling(7, min_periods=2).sum().shift(1)

    features += [
        "Return_1d",
        "Return_3d",
        "Return_5d",
        "Return_10d",
        "Sentiment_Change1",
        "Sentiment_Momentum3",
        "Sentiment_Momentum7",
        "Volatility_7d",
        "Price_Sentiment_Corr7",
        "Return_Momentum7",
    ]

    # --- Optional OHLCV features
    extra_cols = {"open": "Open", "high": "High", "low": "Low", "volume": "Volume"}
    for col, prefix in extra_cols.items():
        if col in df.columns:
            df[f"{prefix}_lag1"] = df[col].shift(1)
            df[f"{prefix}_lag2"] = df[col].shift(2)
            df[f"{prefix}_lag3"] = df[col].shift(3)
            df[f"{prefix}_MA3"] = df[col].rolling(3, min_periods=1).mean().shift(1)
            df[f"{prefix}_STD3"] = df[col].rolling(3, min_periods=2).std().shift(1)
            df[f"{prefix}_MA7"] = df[col].rolling(7, min_periods=3).mean().shift(1)
            df[f"{prefix}_STD7"] = df[col].rolling(7, min_periods=3).std().shift(1)
            features += [
                f"{prefix}_lag1",
                f"{prefix}_lag2",
                f"{prefix}_lag3",
                f"{prefix}_MA3",
                f"{prefix}_STD3",
                f"{prefix}_MA7",
                f"{prefix}_STD7",
            ]

            if prefix == "Volume":
                df["Volume_Change1"] = df[col].pct_change(fill_method=None).shift(1)
                df["Volume_Momentum3"] = (
                df[col].pct_change(fill_method=None).rolling(3, min_periods=1).mean().shift(1)
                )
                features += ["Volume_Change1", "Volume_Momentum3"]

    # --- Technicals (defensiv bereinigt)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=3).mean()
    avg_loss = loss.rolling(14, min_periods=3).mean()
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

    if hist_len >= 5:
        bb_ma = df["close"].rolling(20, min_periods=5).mean()
        bb_std = df["close"].rolling(20, min_periods=5).std()
        df["BB_Middle"] = bb_ma.shift(1)
        df["BB_Upper"] = (bb_ma + 2 * bb_std).shift(1)
        df["BB_Lower"] = (bb_ma - 2 * bb_std).shift(1)
        features += ["BB_Upper", "BB_Lower", "BB_Middle"]

    # --- Label (predict next-day move) & Cleanup
    df["future_close"] = df["close"].shift(-1)
    df["y"] = (df["future_close"] > df["close"]).astype(float)
    invalid_mask = df["close"].isna() | df["future_close"].isna()
    df.loc[invalid_mask, "y"] = np.nan

    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df[features] = df[features].ffill().fillna(0)

    future_mask = df["date"] == future_date
    inference_features: pd.DataFrame | None = None
    if future_mask.any():
        future_slice = df.loc[future_mask, features]
        if not future_slice.empty:
            inference_features = future_slice.tail(1)

    df = df.loc[~future_mask].copy()

    # Tatsächliche Forward-Rendite (gegen aktuellen Schlusskurs) – Basis für Erwartungswerte
    with np.errstate(divide="ignore", invalid="ignore"):
        close_base = df["close"].replace(0, pd.NA)
        df["actual_return"] = (df["future_close"] / close_base) - 1

    df = df.dropna(subset=["future_close", "y", "actual_return", "close"])

    if len(df) < 2 or df["y"].nunique() < 2:
        if balance_method in {"smote", "undersample"}:
            return 0.0, 0.0, None, 0.0, 0.0, None
        return None, None, None, None, None, None

    df["y"] = df["y"].astype(int)

    X, y = df[features], df["y"]
    pos_returns_all = df.loc[y == 1, "actual_return"].astype(float)
    neg_returns_all = df.loc[y == 0, "actual_return"].astype(float)
    avg_pos_return_all = (
        float(pos_returns_all.mean()) if not pos_returns_all.empty else None
    )
    avg_neg_return_all = (
        float(neg_returns_all.mean()) if not neg_returns_all.empty else None
    )
    class_counts = Counter(y)
    if min(class_counts.values()) < 2:
        log.info("Insufficient samples per class: %s", dict(class_counts))
        if balance_method in {"smote", "undersample"}:
            return 0.0, 0.0, None, 0.0, 0.0, None
        return None, None, None, None, None, None
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
    imputer_step = ("imputer", SimpleImputer(strategy="median"))

    if model_type == "logistic":
        param_name = "clf__C"
        if balance_method == "smote":
            model = ImbPipeline(
                [
                    ("sampler", SMOTE(random_state=42, k_neighbors=k_neighbors)),
                    imputer_step,
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            )
        elif balance_method == "undersample":
            model = ImbPipeline(
                [
                    ("sampler", RandomUnderSampler(random_state=42)),
                    imputer_step,
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            )
        else:
            model = Pipeline(
                [
                    imputer_step,
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
        model = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("clf", RandomForestClassifier(random_state=42))]
        )
        param_grid = {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [3, 5, None]}
        if search_method == "random":
            param_distributions = {
                "clf__n_estimators": randint(50, 401),
                "clf__max_depth": [3, 5, 7, None],
                "clf__min_samples_split": randint(2, 11),
            }
        elif use_optuna:
            optuna_distributions = {
                "clf__n_estimators": ("int", 50, 400),
                "clf__max_depth": ("cat", [3, 5, 7, None]),
                "clf__min_samples_split": ("int", 2, 10),
            }

    elif model_type == "gradient_boosting":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", GradientBoostingClassifier(random_state=42)),
            ]
        )
        param_grid = {
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__max_depth": [3, 5],
        }
        if search_method == "random":
            param_distributions = {
                "clf__n_estimators": randint(50, 201),
                "clf__learning_rate": loguniform(1e-3, 1e-1),
                "clf__max_depth": randint(3, 6),
            }
        elif use_optuna:
            optuna_distributions = {
                "clf__n_estimators": ("int", 50, 200),
                "clf__learning_rate": ("float_log", 0.01, 0.2),
                "clf__max_depth": ("int", 3, 5),
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
                return 0.0, 0.0, None, 0.0, 0.0, None
            return None, None, None, None, None, None
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
            return 0.0, 0.0, None, 0.0, 0.0, None
        raise
    best_model = search.best_estimator_
    log.info("%s best params: %s", model_type, getattr(search, "best_params_", {}))

    # --- Evaluation
    if df_test.empty:
        X_eval, y_eval, df_eval = X_train, y_train, df_train
    else:
        X_eval, y_eval, df_eval = df_test[features], df_test["y"], df_test

    y_pred = best_model.predict(X_eval)
    y_proba = None
    if hasattr(best_model, "predict_proba"):
        try:
            y_proba = np.asarray(best_model.predict_proba(X_eval)[:, 1], dtype=float)
        except Exception:
            y_proba = None

    accuracy = float(accuracy_score(y_eval, y_pred))
    f1 = float(f1_score(y_eval, y_pred, zero_division=0))
    precision = float(precision_score(y_eval, y_pred, zero_division=0))
    recall = float(recall_score(y_eval, y_pred, zero_division=0))
    base_f1 = f1
    final_pred = y_pred

    roc_auc: float | None = None
    if y_proba is not None and y_eval.nunique() > 1:
        try:
            roc_auc = float(roc_auc_score(y_eval, y_proba))
        except ValueError:
            roc_auc = None

    decision_threshold = 0.5
    sell_threshold = 0.5
    if y_proba is not None and len(y_proba) == len(y_eval):
        thr, best_f1 = _find_optimal_threshold(y_eval, y_proba)
        tuned_pred = (y_proba >= thr).astype(int)
        if best_f1 > f1:
            accuracy = float(accuracy_score(y_eval, tuned_pred))
            precision = float(precision_score(y_eval, tuned_pred, zero_division=0))
            recall = float(recall_score(y_eval, tuned_pred, zero_division=0))
            f1 = best_f1
            final_pred = tuned_pred
            decision_threshold = float(thr)
            log.info(
                "Adjusted decision threshold to %.2f (F1 %.4f → %.4f)",
                thr,
                base_f1,
                f1,
            )

        buy_thr, sell_thr = _calibrate_dual_thresholds(y_eval, y_proba)
        decision_threshold = float(buy_thr)
        sell_threshold = float(sell_thr)

    final_series = pd.Series(final_pred, index=df_eval.index)
    avg_return = backtest_strategy(df_eval, final_series)

    # Trefferquote der Strategie (nur Tage mit Long-Signal betrachtet)
    long_returns = df_eval.loc[final_series == 1, "actual_return"].dropna()
    long_win_rate: float | None
    if not long_returns.empty:
        long_win_rate = float((long_returns > 0).mean())
    else:
        long_win_rate = None

    avg_pos_return_eval: float | None = None
    avg_neg_return_eval: float | None = None
    if not df_eval.empty:
        pos_eval = df_eval.loc[y_eval == 1, "actual_return"].dropna()
        neg_eval = df_eval.loc[y_eval == 0, "actual_return"].dropna()
        if not pos_eval.empty:
            avg_pos_return_eval = float(pos_eval.mean())
        if not neg_eval.empty:
            avg_neg_return_eval = float(neg_eval.mean())

    avg_positive_return = (
        avg_pos_return_eval if avg_pos_return_eval is not None else avg_pos_return_all
    )
    avg_negative_return = (
        avg_neg_return_eval if avg_neg_return_eval is not None else avg_neg_return_all
    )
    log.info(
        "Model accuracy: %.4f, F1: %.4f, ROC-AUC: %s, Precision: %.4f, Recall: %.4f",
        accuracy,
        f1,
        f"{roc_auc:.4f}" if roc_auc is not None else "nan",
        precision,
        recall,
    )
    log.info("Avg strategy return: %.4f", avg_return)
    if long_win_rate is not None:
        log.info(
            "Strategy long win-rate: %.2f%% (%d trades)",
            long_win_rate * 100,
            int(long_returns.shape[0]),
        )

    try:
        best_model.fit(X, y)
    except Exception as exc:  # pragma: no cover - best effort refit
        log.debug("Refit on full dataset failed: %s", exc)

    next_day_proba: float | None = None
    next_signal: str | None = None
    confidence_val: float | None = None
    expected_return: float | None = None
    probability_margin: float | None = None
    if inference_features is not None and hasattr(best_model, "predict_proba"):
        try:
            next_proba = best_model.predict_proba(inference_features[features])[:, 1]
            if len(next_proba):
                next_day_proba = float(next_proba[0])
                base_pos = avg_positive_return
                base_neg = avg_negative_return
                if base_pos is not None or base_neg is not None:
                    if base_pos is not None and base_neg is not None:
                        expected_return = float(
                            next_day_proba * base_pos + (1 - next_day_proba) * base_neg
                        )
                    elif base_pos is not None:
                        expected_return = float(next_day_proba * base_pos)
                    elif base_neg is not None:
                        expected_return = float((1 - next_day_proba) * base_neg)
                (
                    next_signal,
                    confidence_val,
                    probability_margin,
                ) = derive_signal_from_proba(
                    next_day_proba,
                    decision_threshold,
                    sell_threshold,
                    expected_return,
                )
                if probability_margin is not None:
                    probability_margin = float(probability_margin)
        except Exception as exc:  # pragma: no cover - prediction optional
            log.debug("Next-day probability estimation failed: %s", exc)

    if next_day_proba is not None:
        down_proba = 1.0 - next_day_proba
        log.info(
            "Next-day up-move probability: %.2f (down %.2f) → %s (buy %.2f / sell %.2f, margin %.2f)",
            next_day_proba,
            down_proba,
            next_signal,
            decision_threshold,
            sell_threshold,
            probability_margin if probability_margin is not None else float("nan"),
        )

    meta: dict | None = {
        "next_day_proba": next_day_proba,
        "decision_threshold": decision_threshold,
        "decision_threshold_sell": sell_threshold,
        "signal": next_signal,
        "confidence": confidence_val,
        "horizon_days": 1,
        "as_of": latest_date.to_pydatetime() if latest_date is not None else None,
        "prediction_target": future_date.to_pydatetime(),
        "avg_strategy_return": avg_return,
        "long_win_rate": long_win_rate,
        "avg_positive_return": avg_positive_return,
        "avg_negative_return": avg_negative_return,
        "long_trades": int(long_returns.shape[0]),
        "expected_return": expected_return,
        "probability_margin": probability_margin,
        "downside_probability": (1.0 - next_day_proba) if next_day_proba is not None else None,
        "sample_size": int(len(df)),
        "evaluation_size": int(len(df_eval)),
        "version": f"ml-v2:{model_type}",
    }

    return accuracy, f1, roc_auc, precision, recall, meta
