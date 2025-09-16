from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wallenstein.db_schema import ensure_tables
from wallenstein.trending import scan_reddit_for_candidates


STOPWORD_SYMBOLS = ["TO", "SO", "TR", "U"]


def test_scan_candidates_ignores_common_cashtags():
    con = duckdb.connect(database=":memory:")
    ensure_tables(con)

    now = datetime.now(timezone.utc)

    rows = [
        (
            f"new-{i}",
            now - timedelta(hours=i),
            f"New ticker post {i}",
            "$NEW heading higher",
            10,
        )
        for i in range(5)
    ]

    for idx, sym in enumerate(STOPWORD_SYMBOLS):
        for j in range(3):
            rows.append(
                (
                    f"{sym.lower()}-{idx}-{j}",
                    now - timedelta(hours=j),
                    f"Discussing {sym}",
                    f"🚀 ${sym} to the moon",
                    5,
                )
            )

    con.executemany(
        "INSERT INTO reddit_posts (id, created_utc, title, text, upvotes) VALUES (?, ?, ?, ?, ?)",
        rows,
    )

    candidates = scan_reddit_for_candidates(
        con,
        lookback_days=2,
        window_hours=24,
        min_mentions=1,
        min_lift=1.0,
    )

    symbols = {c.symbol for c in candidates}
    assert "NEW" in symbols
    for sym in STOPWORD_SYMBOLS:
        assert sym not in symbols

    con.close()
