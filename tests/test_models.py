import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import wallenstein.models as models
from wallenstein.models import (
    backtest_strategy,
    derive_signal_from_proba,
    train_per_stock,
)


def test_train_per_stock_basic():
    np.random.seed(0)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": 10 + np.cumsum(np.random.randn(60)),
        "sentiment": np.sin(np.linspace(0, 3, 60)),
    })
    acc, f1, roc_auc, precision, recall, info = train_per_stock(df, n_splits=3)
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
    assert info is not None
    assert info.get("horizon_days") == 1
    proba = info.get("next_day_proba")
    if proba is not None:
        assert 0.0 <= proba <= 1.0
    assert "avg_positive_return" in info
    assert "avg_negative_return" in info
    assert "expected_return" in info
    assert "long_win_rate" in info
    assert "probability_margin" in info
    expected = info.get("expected_return")
    if expected is not None:
        assert -1.0 < expected < 1.0
    win_rate = info.get("long_win_rate")
    if win_rate is not None:
        assert 0.0 <= win_rate <= 1.0


def test_train_per_stock_smote():
    np.random.seed(2)
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    increments = np.ones(n)
    increments[-5:] = -1
    close = 10 + np.cumsum(increments)
    df = pd.DataFrame({
        "date": dates,
        "close": close,
        "sentiment": np.random.randn(n),
    })
    acc, f1, *rest = train_per_stock(df, n_splits=3, balance_method="smote")
    assert acc is not None
    assert f1 is not None
    assert isinstance(rest[-1], (dict, type(None)))


def test_train_per_stock_undersample():
    np.random.seed(3)
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    increments = np.ones(n)
    increments[-5:] = -1
    close = 5 + np.cumsum(increments)
    df = pd.DataFrame({
        "date": dates,
        "close": close,
        "sentiment": np.random.randn(n),
    })
    acc, f1, *rest = train_per_stock(df, n_splits=3, balance_method="undersample")
    assert acc is not None
    assert f1 is not None
    assert isinstance(rest[-1], (dict, type(None)))


def test_train_per_stock_random_forest():
    np.random.seed(1)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": 20 + np.cumsum(np.random.randn(60)),
        "sentiment": np.cos(np.linspace(0, 4, 60)),
    })
    acc, f1, roc_auc, precision, recall, info = train_per_stock(
        df, n_splits=3, model_type="random_forest"
    )
    assert acc is not None
    assert f1 is not None
    assert roc_auc is not None
    assert isinstance(info, (dict, type(None)))



def test_train_per_stock_insufficient_classes():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": [1, 1, 1, 1, 1],  # no variation -> only one class
        "sentiment": [0, 0, 0, 0, 0],
    })

    acc, f1, roc_auc, precision, recall, info = train_per_stock(df)
    assert all(m is None for m in [acc, f1, roc_auc, precision, recall])
    assert info is None


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

    acc, f1, *rest = train_per_stock(df)
    assert acc is None and f1 is None
    assert rest[-1] is None



def test_train_per_stock_uses_timeseries_split(monkeypatch):
    from sklearn.model_selection import TimeSeriesSplit as _TimeSeriesSplit

    class RecordingTS(_TimeSeriesSplit):
        called = False

        def __init__(self, *args, **kwargs):
            RecordingTS.called = True
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(models, "TimeSeriesSplit", RecordingTS)

    np.random.seed(0)
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": 10 + np.cumsum(np.random.randn(20)),
        "sentiment": np.sin(np.linspace(0, 3, 20)),
    })

    train_per_stock(df, n_splits=3, use_kfold=True)
    assert RecordingTS.called


def test_derive_signal_from_proba_three_way():
    signal, confidence, margin = derive_signal_from_proba(0.8, 0.6)
    assert signal == "buy"
    assert confidence == pytest.approx(0.8)
    assert margin == pytest.approx(0.2)

    signal, confidence, margin = derive_signal_from_proba(0.2, 0.6)
    assert signal == "sell"
    assert confidence == pytest.approx(0.8)
    assert margin == pytest.approx(0.2)

    signal, confidence, margin = derive_signal_from_proba(0.5, 0.6)
    assert signal == "hold"
    assert confidence == pytest.approx(0.5)
    assert margin == pytest.approx(0.1)

    signal, confidence, margin = derive_signal_from_proba(None, 0.6)
    assert signal is None and confidence is None and margin is None

