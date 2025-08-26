import duckdb

SCHEMAS = {
    "prices": {
        "date": "DATE",
        "ticker": "TEXT",
        "open": "DOUBLE",
        "high": "DOUBLE",
        "low": "DOUBLE",
        "close": "DOUBLE",
        "adj_close": "DOUBLE",
        "volume": "BIGINT",
    },
    "reddit_posts": {
        "id": "TEXT PRIMARY KEY",
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
    "fx_rates": {
        "date": "DATE",
        "pair": "TEXT",
        "rate_usd_per_eur": "DOUBLE",
    },
}


def ensure_tables(con: duckdb.DuckDBPyConnection):
    for table, cols in SCHEMAS.items():
        coldefs = ", ".join(f"{c} {t}" for c, t in cols.items())
        if table == "prices":
            coldefs += ", PRIMARY KEY (date, ticker)"
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} ({coldefs});")
        if table == "reddit_posts":
            info = con.execute("PRAGMA table_info('reddit_posts')").fetchall()
            has_pk = any(row[1] == "id" and row[5] for row in info)
            if not has_pk:
                try:
                    con.execute("ALTER TABLE reddit_posts ADD PRIMARY KEY (id)")
                    has_pk = True
                except duckdb.Error:
                    pass
            if not has_pk:
                con.execute(
                    "CREATE TABLE reddit_posts_tmp (id TEXT PRIMARY KEY, created_utc TIMESTAMP, title TEXT, text TEXT)"
                )
                con.execute(
                    "INSERT INTO reddit_posts_tmp SELECT id, created_utc, title, text FROM reddit_posts"
                )
                con.execute("DROP TABLE reddit_posts")
                con.execute("ALTER TABLE reddit_posts_tmp RENAME TO reddit_posts")

            try:
                con.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS reddit_posts_id_idx ON reddit_posts(id)"
                )
            except duckdb.Error:  # pragma: no cover - index may exist already
                pass

            con.execute("CREATE UNIQUE INDEX IF NOT EXISTS reddit_posts_id_idx ON reddit_posts(id)")
        if table == "fx_rates":
            try:
                con.execute(
                    "ALTER TABLE fx_rates ADD CONSTRAINT fx_rates_date_pair_unique UNIQUE(date, pair)"
                )
            except duckdb.Error:
                con.execute(
                    """
                    CREATE TABLE fx_rates_tmp AS
                    SELECT DISTINCT * FROM fx_rates
                    """
                )
                con.execute("DROP TABLE fx_rates")
                con.execute("ALTER TABLE fx_rates_tmp RENAME TO fx_rates")
                try:
                    con.execute(
                        "ALTER TABLE fx_rates ADD CONSTRAINT fx_rates_date_pair_unique UNIQUE(date, pair)"
                    )
                except duckdb.Error:
                    pass
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS fx_rates_date_pair_idx ON fx_rates(date, pair)"
            )



def validate_df(df, table_name: str):
    expected = set(SCHEMAS[table_name].keys())
    got = set(df.columns)
    if expected - got:
        raise ValueError(f"Missing columns {expected - got} in {table_name}")
    return True
