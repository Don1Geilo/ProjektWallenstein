import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wallenstein.models import train_per_stock


def test_train_per_stock_basic():
    np.random.seed(0)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": 10 + np.cumsum(np.random.randn(60)),
        "sentiment": np.sin(np.linspace(0, 3, 60)),
    })
    acc, f1 = train_per_stock(df, n_splits=3)
    print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}")
    assert acc is not None
    assert f1 is not None
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_train_per_stock_random_forest():
    np.random.seed(1)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": 20 + np.cumsum(np.random.randn(60)),
        "sentiment": np.cos(np.linspace(0, 4, 60)),
    })
    acc, f1 = train_per_stock(df, n_splits=3, model_type="random_forest")
    assert acc is not None
    assert f1 is not None


def test_train_per_stock_insufficient_classes():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": [1, 1, 1, 1, 1],  # no variation -> only one class
        "sentiment": [0, 0, 0, 0, 0],
    })
    acc, f1 = train_per_stock(df)
    assert acc is None and f1 is None


def test_train_per_stock_with_additional_features(monkeypatch):
    np.random.seed(2)
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": 10 + np.cumsum(np.random.randn(80)),
            "high": 10 + np.cumsum(np.random.randn(80)) + 1,
            "low": 10 + np.cumsum(np.random.randn(80)) - 1,
            "close": 10 + np.cumsum(np.random.randn(80)),
            "volume": np.random.randint(100, 1000, size=80),
            "sentiment": np.random.randn(80),
        }
    )

    captured: dict[str, pd.DataFrame] = {}

    class DummyGrid:
        def __init__(self, *args, **kwargs):
            self.best_index_ = 0
            self.cv_results_ = {"mean_test_accuracy": [0.5], "mean_test_f1": [0.5]}
            self.best_params_ = {}

        def fit(self, X, y):
            captured["X"] = X
            return self

    monkeypatch.setattr("wallenstein.models.GridSearchCV", DummyGrid)
    acc, f1 = train_per_stock(df, n_splits=3)
    assert acc is not None and f1 is not None

    X = captured["X"]
    expected = {
        "Open_lag1",
        "High_MA3",
        "Low_STD7",
        "Volume_lag2",
        "RSI",
        "MACD",
        "BB_Upper",
    }
    assert expected.issubset(set(X.columns))
