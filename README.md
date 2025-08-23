# Projekt Wallenstein — Final Pack

This is a ready-to-push bundle. It includes:
- Robust price updater with DuckDB storage
- Broker targets & recommendations (Yahoo QuoteSummary)
- Reddit sentiment integration (PRAW)
- Telegram alerts (optional)
- CSV exports in `stockOverview/`

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
├─ stockOverview/         # CSV exports
└─ wallenstein/
   ├─ __init__.py
   ├─ stock_data.py
   ├─ db_utils.py
   ├─ broker_targets.py
   ├─ reddit_scraper.py
   └─ sentiment.py
```

### Notes
- If you already have your own `stock_data.py`, you can keep it. This repo’s version is robust and compatible with `update_prices(TICKERS)` signature.
- GitHub push safe: no secrets, data folder is empty (with `.gitkeep`).

Happy hacking!
