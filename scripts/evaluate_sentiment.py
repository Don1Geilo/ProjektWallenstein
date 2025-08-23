"""Evaluate keyword and BERT sentiment analysers on a labelled dataset."""

import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wallenstein.sentiment import analyze_sentiment, BertSentiment


def keyword_prediction(texts):
    """Return ``positive``/``negative`` predictions using keyword analysis."""
    predictions = []
    for text in texts:
        score = analyze_sentiment(text)
        predictions.append("positive" if score > 0 else "negative")
    return predictions


def bert_prediction(texts):  # pragma: no cover - downloads model
    """Return ``positive``/``negative`` predictions using a BERT model."""
    analyzer = BertSentiment()
    predictions = []
    for text in texts:
        result = analyzer(text)[0]
        label = result["label"].lower()
        predictions.append("positive" if label.startswith("pos") else "negative")
    return predictions


def evaluate(true, pred):
    return {
        "accuracy": accuracy_score(true, pred),
        "precision": precision_score(true, pred, pos_label="positive"),
        "recall": recall_score(true, pred, pos_label="positive"),
    }


def main():
    data = pd.read_csv("data/sentiment_labels.csv")
    texts = data["text"].tolist()
    labels = data["label"].tolist()

    kw_pred = keyword_prediction(texts)
    bert_pred = bert_prediction(texts)

    kw_metrics = evaluate(labels, kw_pred)
    bert_metrics = evaluate(labels, bert_pred)

    print("Keyword-based approach:")
    for k, v in kw_metrics.items():
        print(f"{k.capitalize()}: {v:.2f}")

    print("\nBERT-based approach:")
    for k, v in bert_metrics.items():
        print(f"{k.capitalize()}: {v:.2f}")


if __name__ == "__main__":  # pragma: no cover - simple script
    main()
