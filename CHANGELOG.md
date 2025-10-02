# Changelog

## [Unreleased]
- Expanded `data/sentiment_labels.csv` to 500 synthetic, balanced examples.
- Added `scripts/train_finbert.py` for fine-tuning `ProsusAI/finbert`.
- `wallenstein.sentiment` now supports `SENTIMENT_BACKEND=finetuned-finbert`.
- `scripts/evaluate_sentiment.py` evaluates keyword, baseline FinBERT and fine-tuned FinBERT models.
- Added placeholder directory `models/finetuned-finbert` for trained weights.
- Removed deprecated functions `_select_latest_prices` and `get_latest_prices_auto` from `wallenstein.db_utils`.
- Stored ML-based buy signals with probabilities/backtests and surfaced them in the overview output.
- Enriched ML buy-signal metadata with expected returns, win rates and calibration details; overview now highlights expected vs. backtested performance.
- Added probability margins plus a composite signal-strength score and surfaced both in the overview summaries.
