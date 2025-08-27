import duckdb
import pandas as pd


# ---------- Helpers ----------
def _view_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    q = "SELECT COUNT(*) FROM information_schema.views WHERE lower(table_name) = ?"
    return bool(con.execute(q, [name.lower()]).fetchone()[0])


def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    q = "SELECT COUNT(*) FROM information_schema.tables WHERE lower(table_name) = ?"
    return bool(con.execute(q, [name.lower()]).fetchone()[0])


def _ensure_fx_table(con: duckdb.DuckDBPyConnection) -> None:
    # Falls FX-Tabelle noch nicht existiert, eine leere Hülle anlegen,
    # damit die View-Erstellung nicht scheitert.
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fx_rates (
            date DATE,
            pair VARCHAR,
            rate_usd_per_eur DOUBLE,
            UNIQUE(date, pair)
        )
    """
    )


def _resolve_prices_view_name(con: duckdb.DuckDBPyConnection) -> str | None:
    """Gibt 'stocks' oder 'stocks_view' zurück, wenn View existiert; sonst None."""
    for candidate in ("stocks", "stocks_view"):
        if _view_exists(con, candidate):
            return candidate
    return None


# ---------- View: stocks (USD + EUR) ----------
def ensure_prices_view(db_path: str, view_name: str = "stocks", table_name: str = "prices") -> str:
    """
    Erstellt eine View mit USD- und EUR-Spalten.
    Wenn unter view_name bereits eine TABLE existiert, weicht auf <view_name>_view aus.
    Gibt den tatsächlich verwendeten View-Namen zurück.
    """
    con = duckdb.connect(db_path)

    # Kollisionen prüfen
    is_table = _table_exists(con, view_name)
    actual_view = view_name if not is_table else f"{view_name}_view"

    # Nur droppen, wenn unter dem Zielnamen wirklich eine VIEW existiert
    if _view_exists(con, actual_view):
        con.execute(f"DROP VIEW {actual_view}")

    # sicherstellen, dass fx_rates existiert (leere Hülle reicht)
    _ensure_fx_table(con)

    # View erstellen (EUR via jüngstem EURUSD <= p.date)
    cols = [row[1].lower() for row in con.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
    has_adj = "adj_close" in cols

    adj_close_select = (
        "p.adj_close::DOUBLE  AS adj_close," if has_adj else "p.close::DOUBLE      AS adj_close,"
    )
    adj_close_eur_select = (
        "p.adj_close / fx.rate_usd_per_eur AS adj_close_eur"
        if has_adj
        else "p.close / fx.rate_usd_per_eur AS adj_close_eur"
    )

    con.execute(
        f"""
        CREATE VIEW {actual_view} AS
        SELECT
            p.date::DATE         AS date,
            p.ticker::VARCHAR    AS ticker,
            p.open::DOUBLE       AS open,
            p.high::DOUBLE       AS high,
            p.low::DOUBLE        AS low,
            p.close::DOUBLE      AS close,
            {adj_close_select}
            p.volume::BIGINT     AS volume,
            p.open      / fx.rate_usd_per_eur AS open_eur,
            p.high      / fx.rate_usd_per_eur AS high_eur,
            p.low       / fx.rate_usd_per_eur AS low_eur,
            p.close     / fx.rate_usd_per_eur AS close_eur,
            {adj_close_eur_select}
        FROM {table_name} p
        LEFT JOIN LATERAL (
            SELECT rate_usd_per_eur
            FROM fx_rates
            WHERE pair = 'EURUSD' AND date <= p.date
            ORDER BY date DESC
            LIMIT 1
        ) fx ON TRUE
    """
    )
    con.close()
    return actual_view


# ---------- Latest prices (USD/EUR) ----------
def get_latest_prices(
    db_path: str, tickers: list[str], use_eur: bool = False
) -> dict[str, float | None]:
    """
    Liefert je Ticker den jüngsten Schlusskurs in USD (use_eur=False) oder EUR (use_eur=True).
    - Nutzt 'stocks' oder 'stocks_view', wenn vorhanden.
    - Wenn keine View vorhanden ist und use_eur=True, rechnet direkt aus 'prices' + 'fx_rates'.
    """
    con = duckdb.connect(db_path)

    values_sql = ", ".join(["(?)"] * len(tickers))
    params = tickers

    view_name = _resolve_prices_view_name(con)

    try:
        if use_eur:
            if view_name:
                # EUR direkt aus der View
                q = f"""
                    WITH t(ticker) AS (VALUES {values_sql}),
                    ranked AS (
                        SELECT s.ticker, s.close_eur AS px, s.date,
                               ROW_NUMBER() OVER (PARTITION BY s.ticker ORDER BY s.date DESC) rn
                        FROM {view_name} s
                        JOIN t ON s.ticker = t.ticker
                    )
                    SELECT ticker, px FROM ranked WHERE rn=1 ORDER BY ticker
                """
                df = con.execute(q, params).fetchdf()
            else:
                # Kein View? EUR on-the-fly aus prices + fx_rates (jüngster FX ≤ date)
                q = f"""
                    WITH t(ticker) AS (VALUES {values_sql}),
                    base AS (
                        SELECT p.ticker, p.date, p.close,
                               ROW_NUMBER() OVER (PARTITION BY p.ticker ORDER BY p.date DESC) rn
                        FROM prices p
                        JOIN t ON p.ticker = t.ticker
                    ),
                    with_fx AS (
                        SELECT b.ticker,
                               b.close / fx.rate_usd_per_eur AS px
                        FROM base b
                        LEFT JOIN LATERAL (
                            SELECT rate_usd_per_eur
                            FROM fx_rates
                            WHERE pair='EURUSD' AND date <= b.date
                            ORDER BY date DESC
                            LIMIT 1
                        ) fx ON TRUE
                        WHERE b.rn = 1
                    )
                    SELECT ticker, px FROM with_fx ORDER BY ticker
                """
                df = con.execute(q, params).fetchdf()
        else:
            # USD: bevorzuge View (falls vorhanden), sonst direkt prices
            from_rel = view_name or "prices"
            q = f"""
                WITH t(ticker) AS (VALUES {values_sql}),
                ranked AS (
                    SELECT s.ticker, s.close AS px, s.date,
                           ROW_NUMBER() OVER (PARTITION BY s.ticker ORDER BY s.date DESC) rn
                    FROM {from_rel} s
                    JOIN t ON s.ticker = t.ticker
                )
                SELECT ticker, px FROM ranked WHERE rn=1 ORDER BY ticker
            """
            df = con.execute(q, params).fetchdf()

        return {
            row["ticker"]: (float(row["px"]) if pd.notna(row["px"]) else None)
            for _, row in df.iterrows()
        }
    finally:
        con.close()
