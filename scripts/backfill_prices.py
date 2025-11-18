"""CLI helper to backfill historical price data into DuckDB.

The script reuses :func:`wallenstein.stock_data.update_prices` but forces a
custom ``start_date`` so that existing rows are refreshed/extended far beyond
the regular 30-day retention window of the pipeline run.  Typical usage:

```
python scripts/backfill_prices.py --days 365 --tickers NVDA,AMD,TSLA
```
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wallenstein.config import settings
from wallenstein.stock_data import update_prices
from wallenstein.watchlist import all_unique_symbols


def _read_ticker_file(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8").splitlines()
    return [line.strip().upper() for line in content if line.strip()]


def _ensure_tickers(
    *,
    tickers_arg: str | None,
    tickers_file: str | None,
    include_watchlist: bool,
    db_path: str,
) -> list[str]:
    tickers: list[str] = []
    if tickers_arg:
        tickers.extend(t.strip().upper() for t in tickers_arg.split(",") if t.strip())
    if tickers_file:
        tickers.extend(_read_ticker_file(Path(tickers_file)))
    if include_watchlist:
        with duckdb.connect(db_path) as con:
            tickers.extend(all_unique_symbols(con))

    if not tickers:
        env = (settings.WALLENSTEIN_TICKERS or "").strip()
        tickers = [t.strip().upper() for t in env.split(",") if t.strip()]

    deduped = sorted({t for t in tickers if t})
    if not deduped:
        raise SystemExit("Keine Ticker angegeben und auch keine Defaults gefunden.")
    return deduped


def _resolve_start(args: argparse.Namespace) -> str:
    if args.start_date:
        return args.start_date
    if args.days is not None:
        start = date.today() - timedelta(days=int(args.days))
        return start.isoformat()
    # Default: 180 Tage zurück (entspricht neuer Retention)
    start = date.today() - timedelta(days=180)
    return start.isoformat()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill historical price data")
    parser.add_argument(
        "--tickers",
        help="Kommagetrennte Liste an Symbolen (überschreibt Defaults)",
    )
    parser.add_argument(
        "--tickers-file",
        help="Pfad zu einer Datei mit einem Ticker pro Zeile",
    )
    parser.add_argument(
        "--include-watchlist",
        action="store_true",
        help="Fügt alle Symbol der DuckDB-Watchlist hinzu",
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Wie viele Tage rückwirkend geladen werden sollen",
    )
    parser.add_argument(
        "--start-date",
        help="Alternatives Startdatum (YYYY-MM-DD) – überschreibt --days",
    )
    parser.add_argument(
        "--db-path",
        default=settings.WALLENSTEIN_DB_PATH,
        help="Pfad zur DuckDB-Datei",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    db_path = args.db_path or settings.WALLENSTEIN_DB_PATH
    tickers = _ensure_tickers(
        tickers_arg=args.tickers,
        tickers_file=args.tickers_file,
        include_watchlist=args.include_watchlist,
        db_path=db_path,
    )
    start_date = _resolve_start(args)

    print(f"Backfill starte für {len(tickers)} Ticker ab {start_date} …")
    rows = update_prices(db_path, tickers, start_date=start_date)
    print(f"Fertig. Neue/aktualisierte Kurszeilen: {rows}")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
