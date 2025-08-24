"""Evaluate keyword and BERT sentiment analysers on a labelled dataset."""

import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wallenstein import sentiment as ws
from wallenstein.sentiment import analyze_sentiment, BertSentiment


def keyword_prediction(texts):
    """Return ``positive``/``negative`` predictions using keyword analysis."""
    predictions = []
    for text in texts:
        score = analyze_sentiment(text)
        predictions.append("positive" if score > 0 else "negative")
    return predictions


def bert_prediction(texts, backend):  # pragma: no cover - downloads model
    """Return ``positive``/``negative`` predictions using a BERT model."""
    os.environ["SENTIMENT_BACKEND"] = backend
    ws.BertSentiment._pipe = None  # reset cached pipeline
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
    kw_metrics = evaluate(labels, kw_pred)

    try:
        bert_pred = bert_prediction(texts, "finbert")
        bert_metrics = evaluate(labels, bert_pred)
    except Exception as exc:  # pragma: no cover - depends on network
        bert_metrics = None
        print(f"FinBERT model unavailable: {exc}")

    try:
        fine_pred = bert_prediction(texts, "finetuned-finbert")
        fine_metrics = evaluate(labels, fine_pred)
    except Exception as exc:  # pragma: no cover - depends on model
        fine_metrics = None
        print(f"Finetuned model unavailable: {exc}")

    print("Keyword-based approach:")
    for k, v in kw_metrics.items():
        print(f"{k.capitalize()}: {v:.2f}")

    if bert_metrics:
        print("\nFinBERT baseline:")
        for k, v in bert_metrics.items():
            print(f"{k.capitalize()}: {v:.2f}")

    if fine_metrics:
        print("\nFinetuned FinBERT:")
        for k, v in fine_metrics.items():
            print(f"{k.capitalize()}: {v:.2f}")


if __name__ == "__main__":  # pragma: no cover - simple script
    main()
