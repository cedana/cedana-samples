#!/usr/bin/env python3

"""
Transformers training

Finetune any hugging face model
"""

import os
import sys

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


def train(model):
    os.environ['WANDB_DISABLED'] = 'true'  # Disable wandb
    ds = load_dataset('yelp_review_full')
    try:
        # this step needs to be pre-done asnyc from the benchmark and stored somewhere? - takes forever
        tokenized_datasets = ds.map(tokenize_function, batched=True)
        small_train_dataset = (
            tokenized_datasets['train'].shuffle(seed=42).select(range(1000))
        )
        small_eval_dataset = (
            tokenized_datasets['test'].shuffle(seed=42).select(range(1000))
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model,
            num_labels=5,
            torch_dtype='auto',
        )

        training_args = TrainingArguments(
            output_dir='/tmp', eval_strategy='epoch'
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

    except Exception as e:
        raise e


def compute_metrics(eval_pred):
    metric = evaluate.load('accuracy')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(
        model,
    )
    return tokenizer(examples['text'], padding='max_length', truncation=True)


def usage():
    print(f'Usage: {sys.argv[0]} <model>')
    sys.exit(1)


def handle_exit(signum, frame):
    sys.exit(1)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    if len(sys.argv) < 2:
        usage()
    model = None
    for arg in sys.argv[1:]:
        if not arg.startswith('--'):
            model = arg
    if model is None:
        usage()
    train(model)
