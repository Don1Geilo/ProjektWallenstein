import argparse

import numpy as np
import pandas as pd


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT on sentiment data")
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=[
            "data/sentiment_labels.csv",
            "data/financial_phrasebank.csv",
        ],
        help="CSV files with 'text' and 'label' columns",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Downsample classes to achieve class balance",
    )
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        print(
            "Missing optional dependencies for FinBERT training: datasets, transformers, scikit-learn"
        )
        print(exc)
        return

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
        }

    # Load and merge datasets
    frames = [pd.read_csv(path) for path in args.data_files]
    data = pd.concat(frames, ignore_index=True)
    data = data[data["label"].isin(["negative", "positive"])]

    if args.balance:
        min_count = data["label"].value_counts().min()
        data = (
            data.groupby("label", group_keys=False)
            .sample(min_count, random_state=42)
            .reset_index(drop=True)
        )

    dataset = Dataset.from_pandas(data, preserve_index=False)
    dataset = dataset.train_test_split(
        test_size=0.2, seed=42, stratify_by_column="label"
    )

    label2id = {"negative": 0, "positive": 1}
    dataset = dataset.map(
        lambda x: {"labels": [label2id[label] for label in x["label"]]},
        batched=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["text", "label"])
    tokenized_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert", num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="models/finetuned-finbert",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("models/finetuned-finbert")
    tokenizer.save_pretrained("models/finetuned-finbert")
    eval_results = trainer.evaluate()
    print(eval_results)


if __name__ == "__main__":
    main()
