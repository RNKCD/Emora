
from pathlib import Path
import json
import inspect

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "distilroberta"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilroberta-base"

print("GPU:", torch.cuda.get_device_name(0))
print("Loading dataset...")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
val_df = pd.read_parquet(DATA_DIR / "validation.parquet")

with open(DATA_DIR / "label_names.json", encoding="utf-8") as f:
    label_names = json.load(f)

num_labels = len(label_names)

mlb = MultiLabelBinarizer(classes=list(range(num_labels)))
train_labels = mlb.fit_transform(train_df["labels"]).astype(np.float32)
val_labels = mlb.transform(val_df["labels"]).astype(np.float32)

def make_dataset(df, labels):
    return Dataset.from_dict({
        "text": df["text"].tolist(),
        "labels": labels.tolist()
    })

train_dataset = make_dataset(train_df, train_labels)
val_dataset = make_dataset(val_df, val_labels)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

print("Loading pretrained model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification",
    id2label={i: name for i, name in enumerate(label_names)},
    label2id={name: i for i, name in enumerate(label_names)}
)

model.gradient_checkpointing_enable()

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probabilities = 1 / (1 + np.exp(
        -np.clip(logits, -50, 50)
    ))

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "micro_f1": f1_score(
            labels,
            predictions,
            average="micro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    }

# Handle Transformers version differences
args_kwargs = {
    "output_dir": str(MODEL_DIR / "checkpoints"),
    "num_train_epochs": 2,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "save_strategy": "epoch",
    "fp16": True,
    "gradient_checkpointing": True,
    "optim": "adamw_torch",
    "report_to": "none",
    "save_total_limit": 1
}

parameters = inspect.signature(TrainingArguments).parameters

if "eval_strategy" in parameters:
    args_kwargs["eval_strategy"] = "epoch"
else:
    args_kwargs["evaluation_strategy"] = "epoch"

training_args = TrainingArguments(**args_kwargs)

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_dataset,
    "eval_dataset": val_dataset,
    "data_collator": DataCollatorWithPadding(tokenizer),
    "compute_metrics": compute_metrics
}

trainer_parameters = inspect.signature(Trainer).parameters

if "processing_class" in trainer_parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Trainer(**trainer_kwargs)

print("Starting DistilRoBERTa training...")
trainer.train()

print("Evaluating on validation data...")
metrics = trainer.evaluate()

print("\n=== TRANSFORMER RESULTS ===")
print("Micro F1:", round(metrics["eval_micro_f1"], 4))
print("Macro F1:", round(metrics["eval_macro_f1"], 4))

trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

with open(MODEL_DIR / "validation_metrics.json", "w") as f:
    json.dump(
        {
            "micro_f1": metrics["eval_micro_f1"],
            "macro_f1": metrics["eval_macro_f1"],
            "threshold": 0.5
        },
        f,
        indent=2
    )

print("\nTransformer training completed!")

