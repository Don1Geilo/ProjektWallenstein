from datetime import date, datetime

import duckdb
import pytest

from wallenstein.db_schema import ensure_tables
from wallenstein.model_state import (
    TrainingSnapshot,
    load_training_state,
    should_skip_training,
    upsert_training_state,
)


def test_training_state_roundtrip(tmp_path):
    con = duckdb.connect(str(tmp_path / "state.duckdb"))
    ensure_tables(con)

    snapshot = TrainingSnapshot(
        latest_price_date=date(2024, 1, 1),
        price_row_count=10,
        latest_sentiment_date=date(2024, 1, 2),
        sentiment_row_count=5,
        latest_post_utc=datetime(2024, 1, 3, 12, 0, 0),
    )

    upsert_training_state(
        con,
        "AAA",
        snapshot,
        accuracy=0.8,
        f1=0.7,
        roc_auc=0.75,
        precision=0.65,
        recall=0.6,
    )

    state = load_training_state(con)
    assert "AAA" in state
    loaded = state["AAA"]
    assert loaded.snapshot.matches(snapshot)
    assert loaded.accuracy == pytest.approx(0.8)
    assert loaded.f1 == pytest.approx(0.7)
    assert loaded.roc_auc == pytest.approx(0.75)
    assert loaded.precision == pytest.approx(0.65)
    assert loaded.recall == pytest.approx(0.6)
    assert loaded.trained_at is not None

    con.close()


def test_should_skip_training(tmp_path):
    con = duckdb.connect(str(tmp_path / "skip.duckdb"))
    ensure_tables(con)

    snapshot = TrainingSnapshot(
        latest_price_date=date(2024, 2, 1),
        price_row_count=12,
        latest_sentiment_date=date(2024, 2, 2),
        sentiment_row_count=6,
        latest_post_utc=datetime(2024, 2, 3, 15, 30, 0),
    )
    upsert_training_state(
        con,
        "BBB",
        snapshot,
        accuracy=None,
        f1=None,
        roc_auc=None,
        precision=None,
        recall=None,
    )

    state = load_training_state(con)
    existing = state["BBB"]
    assert should_skip_training(existing, snapshot)

    changed = TrainingSnapshot(
        latest_price_date=date(2024, 2, 2),
        price_row_count=13,
        latest_sentiment_date=date(2024, 2, 2),
        sentiment_row_count=6,
        latest_post_utc=datetime(2024, 2, 3, 15, 30, 0),
    )
    assert not should_skip_training(existing, changed)

    con.close()
