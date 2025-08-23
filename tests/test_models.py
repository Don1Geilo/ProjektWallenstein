import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wallenstein.models import train_per_stock


def test_train_per_stock_basic():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": [1, 2, 1.5, 1.8, 2.2, 2.1, 2.3, 2.5, 2.4, 2.6],
        "sentiment": [0.1, -0.2, 0.0, 0.3, 0.5, -0.1, 0.2, 0.1, -0.2, 0.3],
    })
    acc, f1 = train_per_stock(df, use_kfold=True, n_splits=3)
    print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}")
    assert acc is not None
    assert f1 is not None
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_train_per_stock_insufficient_classes():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": [1, 1, 1, 1, 1],  # no variation -> only one class
        "sentiment": [0, 0, 0, 0, 0],
    })
    acc, f1 = train_per_stock(df)
    assert acc is None and f1 is None
