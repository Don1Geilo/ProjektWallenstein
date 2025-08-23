# Projekt Wallenstein — Final Pack

This is a ready-to-push bundle. It includes:
- Robust price updater with DuckDB storage
- Reddit sentiment integration (PRAW) with basic English/German keyword analysis
- Telegram alerts (optional)

Broker-target support is temporarily disabled pending a new data provider.

## Quickstart
```bash
pip install -r requirements.txt
cp .env.example .env   # and fill values
python main.py
```

### ENV (via `.env` or system env)
- `WALLENSTEIN_DB_PATH` = path to DuckDB (default `data/wallenstein.duckdb`)
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (optional alerts)
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` (optional for true Reddit scraping)

### Structure
```
.
├─ main.py
├─ requirements.txt
├─ .env.example
├─ data/                  # DuckDB lives here
└─ wallenstein/
   ├─ __init__.py
   ├─ stock_data.py
   ├─ db_utils.py
   ├─ reddit_scraper.py
   └─ sentiment.py
```

## Sentiment evaluation

The repository ships with a tiny labelled dataset in
`data/sentiment_labels.csv`. To compare the keyword based sentiment analysis
with a transformer model, run:

```bash
python scripts/evaluate_sentiment.py
```

The script prints accuracy, precision and recall for both approaches. Use the
`SENTIMENT_BACKEND` environment variable to select the BERT model (default
`finbert`).

### Notes
- If you already have your own `stock_data.py`, you can keep it. This repo’s version is robust and compatible with `update_prices(TICKERS)` signature.
- GitHub push safe: no secrets, data folder is empty (with `.gitkeep`).

Happy hacking!
