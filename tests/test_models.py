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
    acc = train_per_stock(df)
    assert acc is not None
    assert 0.0 <= acc <= 1.0


def test_train_per_stock_insufficient_classes():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": [1, 1, 1, 1, 1],  # no variation -> only one class
        "sentiment": [0, 0, 0, 0, 0],
    })
    acc = train_per_stock(df)
    assert acc is None
