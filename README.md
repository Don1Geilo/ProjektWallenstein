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

### Ticker aliases

Extra company name variants can be maintained in `data/ticker_aliases.json`
or a YAML equivalent.  The file contains a mapping of ticker symbols to a
list of additional names that should be recognised in Reddit posts:

```json
{
  "NVDA": ["nvidia"],
  "MSFT": ["microsoft"]
}
```

`reddit_scraper` loads this file on startup and merges it with its built-in
defaults. If the file is missing, the internal map is used unchanged.

### Notes
- If you already have your own `stock_data.py`, you can keep it. This repo’s version is robust and compatible with `update_prices(TICKERS)` signature.
- GitHub push safe: no secrets, data folder is empty (with `.gitkeep`).

Happy hacking!
