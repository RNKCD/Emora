
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "distilroberta"
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "label_names.json", encoding="utf-8") as f:
    label_names = json.load(f)

if isinstance(label_names, dict):
    label_names = (
        label_names.get("label_names")
        or label_names.get("names")
        or label_names.get("labels")
    )

with open(
    MODEL_DIR / "per_emotion_thresholds.json",
    encoding="utf-8",
) as f:
    settings = json.load(f)

thresholds = np.array(settings["thresholds"])

df = pd.read_parquet(DATA_DIR / "test.parquet")

y_true = np.zeros(
    (len(df), len(label_names)),
    dtype=np.int32,
)

for i, labels in enumerate(df["labels"]):
    for label_id in labels:
        y_true[i, int(label_id)] = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model.to(device)
model.eval()

print("Device:", device)
print("Test examples:", len(df))

batch_size = 8 if device.type == "cuda" else 16
probability_batches = []

start_time = time.perf_counter()

with torch.inference_mode():
    for start in range(0, len(df), batch_size):
        texts = df["text"].iloc[
            start:start + batch_size
        ].tolist()

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(device) for k, v in inputs.items()
        }

        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        probability_batches.append(probs)

        if start % (batch_size * 50) == 0:
            print(
                f"Processed "
                f"{min(start + batch_size, len(df))}"
                f"/{len(df)}"
            )

if device.type == "cuda":
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start_time

probabilities = np.concatenate(
    probability_batches, axis=0
)

strategies = {
    "shared_0.30": np.full(len(label_names), 0.30),
    "per_emotion": thresholds,
}

summary = []

for name, strategy_thresholds in strategies.items():
    predictions = (
        probabilities >= strategy_thresholds[None, :]
    ).astype(int)

    metrics = {
        "strategy": name,
        "micro_f1": f1_score(
            y_true, predictions,
            average="micro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, predictions,
            average="macro", zero_division=0
        ),
        "precision": precision_score(
            y_true, predictions,
            average="micro", zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions,
            average="micro", zero_division=0
        ),
    }

    summary.append(metrics)

    report = classification_report(
        y_true,
        predictions,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIR / f"test_{name}_per_emotion.csv"
    )

results = pd.DataFrame(summary)
results.to_csv(
    REPORT_DIR / "transformer_test_results.csv",
    index=False,
)

print("\n=== FINAL TRANSFORMER TEST RESULTS ===")
print(results.to_string(index=False))

print("\n=== INFERENCE SPEED ===")
print(f"Total seconds: {elapsed:.2f}")
print(f"Examples per second: {len(df) / elapsed:.2f}")

print("\nTest evaluation completed!")
