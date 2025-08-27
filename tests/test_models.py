import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wallenstein.models import backtest_strategy, train_per_stock


def test_train_per_stock_basic():
    np.random.seed(0)
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": 10 + np.cumsum(np.random.randn(20)),
        "sentiment": np.sin(np.linspace(0, 3, 20)),
    })
    acc, f1, roc_auc, precision, recall = train_per_stock(df, n_splits=3)
    print(
        f"Accuracy: {acc:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f},"
        f" Precision: {precision:.3f}, Recall: {recall:.3f}"
    )
    assert acc is not None and f1 is not None
    assert roc_auc is not None and 0.0 <= roc_auc <= 1.0
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0


def test_train_per_stock_random_forest():
    np.random.seed(1)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": 20 + np.cumsum(np.random.randn(30)),
        "sentiment": np.cos(np.linspace(0, 4, 30)),
    })
    acc, f1, roc_auc, precision, recall = train_per_stock(
        df, n_splits=3, model_type="random_forest"
    )
    assert acc is not None
    assert f1 is not None
    assert roc_auc is not None


def test_train_per_stock_insufficient_classes():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": [1, 1, 1, 1, 1],  # no variation -> only one class
        "sentiment": [0, 0, 0, 0, 0],
    })
    acc, f1, roc_auc, precision, recall = train_per_stock(df)
    assert all(m is None for m in [acc, f1, roc_auc, precision, recall])


def test_backtest_strategy():
    df = pd.DataFrame(
        {
            "close": [10, 11, 12, 11],
        }
    )
    signals = pd.Series([1, 0, 1, 0])
    avg = backtest_strategy(df, signals)
    # Returns: 1/10 and -1/12 -> average is 1/120
    expected = (1 / 10 - 1 / 12) / 2
    assert abs(avg - expected) < 1e-6
