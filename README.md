# Projekt Wallenstein — Final Pack

This is a ready-to-push bundle. It includes:
- Robust price updater with DuckDB storage (Stooq default with Yahoo/Stooq fallback)
- Reddit sentiment integration (PRAW) with basic English/German keyword analysis
- Telegram alerts (optional)

Broker-target support is temporarily disabled pending a new data provider.

## Quickstart
```bash
pip install -r requirements.txt
cp .env.example .env   # and fill values
python main.py
python -m bot.telegram_bot  # optional: manage watchlist via Telegram
```

The main pipeline reads symbols from the watchlist and exits with a warning if none exist.

## Install (reproducible)

Pinned dependencies live in `requirements.txt`.

```bash
pip install -r requirements.txt
```

An optional `requirements.in` is provided for use with
[`pip-tools`](https://github.com/jazzband/pip-tools). Regenerate the pinned
file via:

```bash
pip install pip-tools
pip-compile requirements.in
```

## Dev setup

Install git hooks for consistent formatting and linting:

```bash
pre-commit install
```

### ENV (via `.env` or system env)
- `WALLENSTEIN_DB_PATH` = path to DuckDB (default `data/wallenstein.duckdb`)
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (optional alerts)
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` (optional for true Reddit scraping)
- `WALLENSTEIN_DATA_SOURCE` = `stooq` (default) uses Stooq with Yahoo fallback; set to `yahoo` to use Yahoo with a Stooq fallback for missing tickers
- `USE_BERT_SENTIMENT` = choose sentiment backend. When unset the app uses a
  BERT model if the `transformers` package is available, otherwise a lightweight
  keyword approach. Set to `1`/`true` to force BERT or `0`/`false` to force the
  keyword method.
- `PIPELINE_MAX_WORKERS` = number of parallel threads for price, Reddit and FX updates (default `4`)

### Structure
```
.
├─ main.py
├─ requirements.txt
├─ .env.example
├─ data/                  # DuckDB lives here
├─ bot/telegram_bot.py   # dynamic watchlist via Telegram
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

### Sentiment keywords

The keyword based sentiment analyser ships with a small built‑in dictionary.
Additional terms can be supplied via ``data/sentiment_keywords.json`` or a
YAML equivalent.  The file should contain a mapping of words to sentiment
scores (``1`` for positive, ``-1`` for negative):

```json
{
  "moonshot": 1,
  "schrott": -1
}
```

If the file exists it is loaded on import and merged with the default
mapping. Custom keywords therefore influence all subsequent calls to
``analyze_sentiment``.

## Dynamic Watchlist + Telegram Bot

Expose ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` (override the database path with ``WALLENSTEIN_DB_PATH`` if needed) and start the bot with:

```bash
python -m bot.telegram_bot
```

Example commands:

```text
/add NVDA, AMZN
/list
/alerts add NVDA < 150
```

The main pipeline pulls tickers from this watchlist and exits with a warning when it is empty.

## Sentiment evaluation

The repository ships with a labelled dataset in
`data/sentiment_labels.csv`. This file now contains 500 synthetic sentences
balanced between positive and negative labels. To compare the keyword based
sentiment analysis with transformer models, run:

```bash
python scripts/evaluate_sentiment.py
```

The script prints accuracy, precision and recall for the keyword baseline, the
pretrained FinBERT model and, if available, a locally fine‑tuned variant. Use
the `SENTIMENT_BACKEND` environment variable to select the BERT model (default
`finbert`). Sentiment analysis uses a BERT model automatically when
`transformers` is installed. Set `USE_BERT_SENTIMENT=0` to force the lightweight
keyword approach or `USE_BERT_SENTIMENT=1` to explicitly enable the BERT path.

To fine‑tune FinBERT on the extended dataset run:

```bash
python scripts/train_finbert.py
```

The resulting model and tokenizer are saved under
`models/finetuned-finbert`. Evaluation of this model requires that the directory
contains trained weights.


## Model training

The price movement model now uses a richer feature set including multiple
lags for ``close`` and ``sentiment`` as well as 3/7-day moving averages and
volatility measures. K-fold cross validation (5-fold) is enabled by default and
hyperparameters can be tuned via ``GridSearchCV``, ``RandomizedSearchCV`` or
``Optuna`` using the ``search_method`` parameter.

Supported models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine
- XGBoost (optional, requires ``pip install xgboost``)

Select a model via ``model_type`` when calling ``train_per_stock``:

```python
from wallenstein.models import train_per_stock

acc, f1 = train_per_stock(df, model_type="svm")
```

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
