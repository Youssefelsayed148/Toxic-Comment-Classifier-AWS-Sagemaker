import os
import argparse
import json
import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
if not hasattr(np, "object"):
    np.object = object  # alias to restore compatibility

def prepare_dataset(df, tokenizer):
    # Keep only the relevant columns
    df = df[["comment_text", "toxic"]].copy() 
    # Ensure proper types
    df["comment_text"] = df["comment_text"].astype(str)
    df["toxic"] = df["toxic"].astype(int)
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    # Tokenize correctly in batches
    dataset = dataset.map(
        lambda batch: tokenizer(batch["comment_text"], padding="max_length", truncation=True),
        batched=True)
    # Map labels
    dataset = dataset.map(lambda batch: {"labels": batch["toxic"]})
    # Set format for PyTorch
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset

# ---- Metrics ----
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)  # pick class with highest score
    labels = labels.astype(int)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0)
    }

# ---- Main function ----
def main(args):
    # ---- Load data ----
    train_df = pd.read_csv(os.path.join(args.data_dir,"train11.csv"))
    valid_df = pd.read_csv(os.path.join(args.data_dir,"valid.csv"))
    test_df  = pd.read_csv(os.path.join(args.data_dir,"test.csv"))

    # ---- Tokenizer ----
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # ---- Prepare datasets ----
    train_ds = prepare_dataset(train_df, tokenizer)
    val_ds   = prepare_dataset(valid_df, tokenizer)
    test_ds  = prepare_dataset(test_df, tokenizer)

    # ---- Model ----
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,  # binary classification
        problem_type="single_label_classification",
    )

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(args.model_dir, "logs"),
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ---- Train ----
    trainer.train()

    # ---- Evaluate on test set ----
    metrics = trainer.evaluate(test_ds)
    print("Test metrics:", metrics)

    # ---- Save model & tokenizer ----
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument("--data-dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model-dir", type=str, default="./model")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)

    args = parser.parse_args()
    main(args)
