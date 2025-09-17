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
        "upvotes": "INTEGER",
    },
    "reddit_enriched": {
        "id": "VARCHAR",
        "ticker": "TEXT",
        "created_utc": "TIMESTAMP",
        "text": "TEXT",
        "upvotes": "INTEGER",
        "sentiment_dict": "DOUBLE",
        "sentiment_weighted": "DOUBLE",
        "sentiment_ml": "DOUBLE",
        "return_1d": "DOUBLE",
        "return_3d": "DOUBLE",
        "return_7d": "DOUBLE",
    },
    "reddit_sentiment_hourly": {
        "created_utc": "TIMESTAMP",
        "ticker": "TEXT",
        "sentiment_dict": "DOUBLE",
        "sentiment_weighted": "DOUBLE",
        "posts": "INTEGER",
    },
    "reddit_sentiment_daily": {
        "date": "DATE",
        "ticker": "TEXT",
        "sentiment_dict": "DOUBLE",
        "sentiment_weighted": "DOUBLE",
        "posts": "INTEGER",
    },
    "reddit_trends": {
        "date": "DATE",
        "ticker": "TEXT",
        "mentions": "INTEGER",
        "avg_upvotes": "DOUBLE",
        "hotness": "DOUBLE",
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
    "alerts": {
        "id": "INTEGER",
        "ticker": "TEXT",
        "op": "TEXT",
        "price": "DOUBLE",
        "active": "BOOLEAN",
    },
    "model_training_state": {
        "ticker": "TEXT",
        "latest_price_date": "DATE",
        "price_row_count": "INTEGER",
        "latest_sentiment_date": "DATE",
        "sentiment_row_count": "INTEGER",
        "latest_post_utc": "TIMESTAMP",
        "trained_at": "TIMESTAMP",
        "accuracy": "DOUBLE",
        "f1": "DOUBLE",
        "roc_auc": "DOUBLE",
        "precision_score": "DOUBLE",
        "recall_score": "DOUBLE",
    },
}


def ensure_tables(con: duckdb.DuckDBPyConnection):
    for table, cols in SCHEMAS.items():
        coldefs = ", ".join(f"{c} {t}" for c, t in cols.items())
        if table == "prices":
            coldefs += ", PRIMARY KEY (date, ticker)"
        if table == "reddit_trends":
            coldefs += ", PRIMARY KEY (date, ticker)"
        if table == "alerts":
            coldefs += ", PRIMARY KEY (id)"
        if table == "model_training_state":
            coldefs += ", PRIMARY KEY (ticker)"
        if table == "reddit_sentiment_hourly":
            coldefs += ", PRIMARY KEY (ticker, created_utc)"
        if table == "reddit_sentiment_daily":
            coldefs += ", PRIMARY KEY (date, ticker)"
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} ({coldefs});")
        if table == "reddit_posts":
            info = con.execute("PRAGMA table_info('reddit_posts')").fetchall()
            cols_existing = {row[1] for row in info}
            if "upvotes" not in cols_existing:
                try:
                    con.execute("ALTER TABLE reddit_posts ADD COLUMN upvotes INTEGER")
                except duckdb.Error:
                    pass
            has_pk = any(row[1] == "id" and row[5] for row in info)
            if not has_pk:
                try:
                    con.execute("ALTER TABLE reddit_posts ADD PRIMARY KEY (id)")
                    has_pk = True
                except duckdb.Error:
                    pass
            if not has_pk:
                con.execute(
                    "CREATE TABLE reddit_posts_tmp (id TEXT PRIMARY KEY, created_utc TIMESTAMP, title TEXT, text TEXT, upvotes INTEGER)"
                )
                con.execute(
                    "INSERT INTO reddit_posts_tmp SELECT id, created_utc, title, text, COALESCE(upvotes,0) FROM reddit_posts"
                )
                con.execute("DROP TABLE reddit_posts")
                con.execute("ALTER TABLE reddit_posts_tmp RENAME TO reddit_posts")

            try:
                con.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS reddit_posts_id_idx ON reddit_posts(id)"
                )
            except duckdb.Error:  # pragma: no cover - index may exist already
                pass

        if table == "reddit_enriched":
            info = con.execute("PRAGMA table_info('reddit_enriched')").fetchall()
            col_types = {row[1]: row[2].upper() for row in info}
            if col_types.get("id") != "VARCHAR":
                try:
                    con.execute("ALTER TABLE reddit_enriched ALTER COLUMN id SET DATA TYPE VARCHAR")
                except duckdb.Error:
                    pass
            try:
                con.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_reddit_enriched_id_ticker ON reddit_enriched(id, ticker)"
                )
            except duckdb.Error:
                pass
            try:
                con.execute(
                    "CREATE INDEX IF NOT EXISTS idx_reddit_enriched_tkr_date ON reddit_enriched(ticker, created_utc)"
                )
            except duckdb.Error:
                pass
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
