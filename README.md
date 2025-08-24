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
python telegram_bot.py  # optional: interactive Telegram bot
```

### ENV (via `.env` or system env)
- `WALLENSTEIN_DB_PATH` = path to DuckDB (default `data/wallenstein.duckdb`)
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (optional alerts)
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` (optional for true Reddit scraping)
- `USE_BERT_SENTIMENT` = choose sentiment backend. When unset the app uses a
  BERT model if the `transformers` package is available, otherwise a lightweight
  keyword approach. Set to `1`/`true` to force BERT or `0`/`false` to force the
  keyword method.

### Structure
```
.
├─ main.py
├─ requirements.txt
├─ .env.example
├─ data/                  # DuckDB lives here
├─ telegram_bot.py       # listens for `!TICKER` messages on Telegram
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

Aliases can also be supplied dynamically when calling
``update_reddit_data``. Pass either a path to a JSON/YAML file via
``aliases_path`` or an in-memory mapping via ``aliases``:

```python
from wallenstein import reddit_scraper

# reload aliases from a custom file before each update
reddit_scraper.update_reddit_data(["NVDA"], aliases_path="my_aliases.json")

# or merge a dict directly
reddit_scraper.update_reddit_data(["NVDA"], aliases={"NVDA": ["nvidia corp"]})
```

The file specified by ``aliases_path`` is read on every call so changes are
picked up immediately.

## Telegram Bot

Start a small bot that reacts to messages like ``!NVDA`` and returns the number
of matching Reddit posts:

```bash
python telegram_bot.py
```

The bot uses ``reddit_scraper.update_reddit_data`` internally.  Set
``TELEGRAM_BOT_TOKEN`` (and optionally ``TELEGRAM_CHAT_ID`` for the broadcast
helper ``notify_telegram``) in your environment or ``.env`` file.

## Sentiment evaluation

The repository ships with a tiny labelled dataset in
`data/sentiment_labels.csv`. To compare the keyword based sentiment analysis
with a transformer model, run:

```bash
python scripts/evaluate_sentiment.py
```

The script prints accuracy, precision and recall for both approaches. Use the
`SENTIMENT_BACKEND` environment variable to select the BERT model (default
`finbert`). Sentiment analysis uses a BERT model automatically when
`transformers` is installed. Set `USE_BERT_SENTIMENT=0` to force the lightweight
keyword approach or `USE_BERT_SENTIMENT=1` to explicitly enable the BERT path.


## Model training

The price movement model now uses a richer feature set including multiple
lags for ``close`` and ``sentiment`` as well as 3/7-day moving averages and
volatility measures. K-fold cross validation (5-fold) is enabled by default and
GridSearchCV tunes hyperparameters for each supported model.

Supported models:

- Logistic Regression
- Random Forest
- Gradient Boosting

Example cross-validated scores on a synthetic 80-day dataset:

| Model              | Accuracy | F1   |
|--------------------|---------:|-----:|
| Logistic Regression | 0.65     | 0.64 |
| Random Forest       | 0.60     | 0.54 |
| Gradient Boosting   | 0.60     | 0.57 |


## Saving Reddit snapshots

To store the raw Reddit data for offline inspection or to feed the model
without network access, run:

```bash
python scripts/fetch_reddit_posts.py --subreddit wallstreetbets --limit 50
```

The command writes a JSON file to `data/reddit_posts.json` containing the
queried posts and comments.

### Notes
- If you already have your own `stock_data.py`, you can keep it. This repo’s version is robust and compatible with `update_prices(TICKERS)` signature.
- GitHub push safe: no secrets, data folder is empty (with `.gitkeep`).

Happy hacking!
