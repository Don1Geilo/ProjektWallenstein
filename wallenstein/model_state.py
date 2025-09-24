"""Utilities for persisting per-ticker model training state."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Optional

import duckdb


@dataclass(frozen=True)
class TrainingSnapshot:
    """Aggregated information about the data used for training a ticker."""

    latest_price_date: Optional[date]
    price_row_count: int
    latest_sentiment_date: Optional[date]
    sentiment_row_count: int
    latest_post_utc: Optional[datetime]

    def matches(self, other: "TrainingSnapshot") -> bool:
        """Return ``True`` if both snapshots describe the same dataset."""

        return (
            self.latest_price_date == other.latest_price_date
            and self.price_row_count == other.price_row_count
            and self.latest_sentiment_date == other.latest_sentiment_date
            and self.sentiment_row_count == other.sentiment_row_count
            and self.latest_post_utc == other.latest_post_utc
        )


@dataclass
class TrainingState:
    """Persisted model state including metrics from the last training run."""

    snapshot: TrainingSnapshot
    trained_at: Optional[datetime]
    accuracy: Optional[float]
    f1: Optional[float]
    roc_auc: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    avg_strategy_return: Optional[float] = None
    long_win_rate: Optional[float] = None


def load_training_state(con: duckdb.DuckDBPyConnection) -> Dict[str, TrainingState]:
    """Return the stored training state for all tickers."""

    query_extended = """
        SELECT
            ticker,
            latest_price_date,
            price_row_count,
            latest_sentiment_date,
            sentiment_row_count,
            latest_post_utc,
            trained_at,
            accuracy,
            f1,
            roc_auc,
            precision_score,
            recall_score,
            avg_strategy_return,
            long_win_rate
        FROM model_training_state
    """

    try:
        rows = con.execute(query_extended).fetchall()
        extended = True
    except duckdb.Error:
        try:
            rows = con.execute(
                """
                SELECT
                    ticker,
                    latest_price_date,
                    price_row_count,
                    latest_sentiment_date,
                    sentiment_row_count,
                    latest_post_utc,
                    trained_at,
                    accuracy,
                    f1,
                    roc_auc,
                    precision_score,
                    recall_score
                FROM model_training_state
                """
            ).fetchall()
        except duckdb.Error:
            return {}
        extended = False

    state: Dict[str, TrainingState] = {}
    for row in rows:
        snapshot = TrainingSnapshot(
            latest_price_date=row[1],
            price_row_count=int(row[2]) if row[2] is not None else 0,
            latest_sentiment_date=row[3],
            sentiment_row_count=int(row[4]) if row[4] is not None else 0,
            latest_post_utc=row[5],
        )
        avg_return = row[12] if extended and len(row) > 12 else None
        win_rate = row[13] if extended and len(row) > 13 else None
        state[row[0]] = TrainingState(
            snapshot=snapshot,
            trained_at=row[6],
            accuracy=row[7],
            f1=row[8],
            roc_auc=row[9],
            precision=row[10],
            recall=row[11],
            avg_strategy_return=avg_return,
            long_win_rate=win_rate,
        )
    return state


def should_skip_training(
    existing: Optional[TrainingState], snapshot: TrainingSnapshot
) -> bool:
    """Return ``True`` when ``snapshot`` matches ``existing``."""

    if existing is None:
        return False
    return existing.snapshot.matches(snapshot)


def upsert_training_state(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    snapshot: TrainingSnapshot,
    *,
    accuracy: Optional[float],
    f1: Optional[float],
    roc_auc: Optional[float],
    precision: Optional[float],
    recall: Optional[float],
    avg_strategy_return: Optional[float],
    long_win_rate: Optional[float],
) -> None:
    """Insert or update the stored training state for ``ticker``."""

    con.execute(
        """
        MERGE INTO model_training_state AS target
        USING (
            SELECT
                ? AS ticker,
                ? AS latest_price_date,
                ? AS price_row_count,
                ? AS latest_sentiment_date,
                ? AS sentiment_row_count,
                ? AS latest_post_utc,
                CURRENT_TIMESTAMP AS trained_at,
                ? AS accuracy,
                ? AS f1,
                ? AS roc_auc,
                ? AS precision_score,
                ? AS recall_score,
                ? AS avg_strategy_return,
                ? AS long_win_rate
        ) AS src
        ON target.ticker = src.ticker
        WHEN MATCHED THEN UPDATE SET
            latest_price_date = src.latest_price_date,
            price_row_count = src.price_row_count,
            latest_sentiment_date = src.latest_sentiment_date,
            sentiment_row_count = src.sentiment_row_count,
            latest_post_utc = src.latest_post_utc,
            trained_at = src.trained_at,
            accuracy = src.accuracy,
            f1 = src.f1,
            roc_auc = src.roc_auc,
            precision_score = src.precision_score,
            recall_score = src.recall_score,
            avg_strategy_return = src.avg_strategy_return,
            long_win_rate = src.long_win_rate
        WHEN NOT MATCHED THEN INSERT (
            ticker,
            latest_price_date,
            price_row_count,
            latest_sentiment_date,
            sentiment_row_count,
            latest_post_utc,
            trained_at,
            accuracy,
            f1,
            roc_auc,
            precision_score,
            recall_score,
            avg_strategy_return,
            long_win_rate
        ) VALUES (
            src.ticker,
            src.latest_price_date,
            src.price_row_count,
            src.latest_sentiment_date,
            src.sentiment_row_count,
            src.latest_post_utc,
            src.trained_at,
            src.accuracy,
            src.f1,
            src.roc_auc,
            src.precision_score,
            src.recall_score,
            src.avg_strategy_return,
            src.long_win_rate
        )
        """,
        [
            ticker,
            snapshot.latest_price_date,
            snapshot.price_row_count,
            snapshot.latest_sentiment_date,
            snapshot.sentiment_row_count,
            snapshot.latest_post_utc,
            accuracy,
            f1,
            roc_auc,
            precision,
            recall,
            avg_strategy_return,
            long_win_rate,
        ],
    )
