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
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": 10 + np.cumsum(np.random.randn(20)),
        "sentiment": np.sin(np.linspace(0, 3, 20)),
    })
    acc, f1 = train_per_stock(df, n_splits=3)
    print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}")
    assert acc is not None
    assert f1 is not None
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_train_per_stock_random_forest():
    np.random.seed(1)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": 20 + np.cumsum(np.random.randn(30)),
        "sentiment": np.cos(np.linspace(0, 4, 30)),
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
