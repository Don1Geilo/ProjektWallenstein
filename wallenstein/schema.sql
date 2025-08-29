-- SQL schema for Wallenstein watchlists and alerts

CREATE TABLE IF NOT EXISTS watchlists (
    chat_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chat_id, ticker)
);

CREATE TABLE IF NOT EXISTS alerts (
    chat_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    target_price DOUBLE NOT NULL,
    direction TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    triggered_at TIMESTAMP,
    PRIMARY KEY (chat_id, ticker, target_price, direction)
);
