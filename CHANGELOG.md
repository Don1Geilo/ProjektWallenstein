# Changelog

## [Unreleased]
- Expanded `data/sentiment_labels.csv` to 500 synthetic, balanced examples.
- Added `scripts/train_finbert.py` for fine-tuning `ProsusAI/finbert`.
- `wallenstein.sentiment` now supports `SENTIMENT_BACKEND=finetuned-finbert`.
- `scripts/evaluate_sentiment.py` evaluates keyword, baseline FinBERT and fine-tuned FinBERT models.
- Added placeholder directory `models/finetuned-finbert` for trained weights.
