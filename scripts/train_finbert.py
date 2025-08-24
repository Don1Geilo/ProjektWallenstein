import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score


def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
    }


def main():
    dataset = load_dataset('csv', data_files={'data': 'data/sentiment_labels.csv'})['data']
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    label2id = {'negative': 0, 'positive': 1}
    dataset = dataset.map(lambda x: {'labels': [label2id[l] for l in x['label']]}, batched=True)

    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label'])
    tokenized_dataset.set_format('torch')

    model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=2)

    training_args = TrainingArguments(
        output_dir='models/finetuned-finbert',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model('models/finetuned-finbert')
    tokenizer.save_pretrained('models/finetuned-finbert')
    eval_results = trainer.evaluate()
    print(eval_results)


if __name__ == '__main__':
    main()
