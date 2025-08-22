# wallenstein/export_targets.py
import duckdb
import pandas as pd
from datetime import datetime, timezone

def export_latest_targets(db_path: str, csv_path: str, tickers: list[str]):
    con = duckdb.connect(db_path)
    # Für jeden Ticker den jüngsten Snapshot
    q = """
    WITH ranked AS (
      SELECT *,
             ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fetched_at_utc DESC) AS rn
      FROM broker_targets
      WHERE ticker IN ({})
    )
    SELECT ticker,
           to_timestamp(fetched_at_utc) AS fetched_at,
           target_mean, target_high, target_low,
           rec_mean, rec_text,
           strong_buy, buy, hold, sell, strong_sell,
           source
    FROM ranked
    WHERE rn = 1
    ORDER BY ticker;
    """.format(", ".join(["'{}'".format(t) for t in tickers]))
    df = con.execute(q).fetch_df()
    con.close()
    # Export
    df.to_csv(csv_path, index=False)
    return df


def export_latest_recs(db_path: str, csv_path: str, tickers: list[str]):
    """Exportiert die jüngsten Empfehlungen inkl. einfacher Bewertung."""
    con = duckdb.connect(db_path)
    q = """
    WITH ranked AS (
      SELECT *,
             ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY fetched_at_utc DESC) AS rn
      FROM broker_targets
      WHERE ticker IN ({})
    )
    SELECT ticker,
           to_timestamp(fetched_at_utc) AS fetched_at,
           target_mean, target_high, target_low,
           rec_mean, rec_text,
           strong_buy, buy, hold, sell, strong_sell,
           source
    FROM ranked
    WHERE rn = 1
    ORDER BY ticker;
    """.format(", ".join(["'{}'".format(t) for t in tickers]))
    df = con.execute(q).fetch_df()
    con.close()

    # Zusätzliche Kennzahlen (Broker-Signal, Platzhalter für Reddit etc.)
    def _broker_sig(v):
        try:
            return 0.0 if pd.isna(v) else 3.0 - float(v)
        except Exception:
            return 0.0

    df["broker_sig"] = df["rec_mean"].apply(_broker_sig)
    df["reddit_score"] = 0.0
    df["combined_score"] = df["broker_sig"] + df["reddit_score"]

    def _reco(score):
        if score > 0.5:
            return "Buy"
        if score < -0.5:
            return "Sell"
        return "Hold"

    df["recommendation"] = df["combined_score"].apply(_reco)
    df.to_csv(csv_path, index=False)
    return df

