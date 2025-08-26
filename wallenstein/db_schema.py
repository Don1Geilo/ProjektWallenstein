import duckdb

SCHEMAS = {
    "prices": {
        "date": "DATE",
        "ticker": "TEXT",
        "open": "DOUBLE",
        "high": "DOUBLE",
        "low": "DOUBLE",
        "close": "DOUBLE",
        "volume": "BIGINT",
    },
    "reddit_posts": {
        "id": "TEXT",
        "created_utc": "TIMESTAMP",
        "title": "TEXT",
        "text": "TEXT",
    },
    "sentiments": {
        "post_id": "TEXT",
        "model": "TEXT",
        "score": "DOUBLE",
        "label": "TEXT",
        "created_utc": "TIMESTAMP",
    },
    "predictions": {
        "as_of": "TIMESTAMP",
        "ticker": "TEXT",
        "horizon_days": "INT",
        "signal": "TEXT",
        "confidence": "DOUBLE",
        "version": "TEXT",
    },
}


def ensure_tables(con: duckdb.DuckDBPyConnection):
    for table, cols in SCHEMAS.items():
        coldefs = ", ".join(f"{c} {t}" for c, t in cols.items())
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} ({coldefs});")


def validate_df(df, table_name: str):
    expected = set(SCHEMAS[table_name].keys())
    got = set(df.columns)
    if expected - got:
        raise ValueError(f"Missing columns {expected - got} in {table_name}")
    return True
