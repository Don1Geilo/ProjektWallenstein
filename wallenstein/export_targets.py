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
