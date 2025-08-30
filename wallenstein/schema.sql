-- SQL schema for Wallenstein watchlists and alerts

CREATE TABLE IF NOT EXISTS watchlists (
    chat_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chat_id, ticker)
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    op TEXT NOT NULL,
    price DOUBLE NOT NULL,
    active BOOLEAN DEFAULT TRUE
);
