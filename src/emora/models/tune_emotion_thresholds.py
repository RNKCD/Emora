
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "distilroberta"
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
CACHE_FILE = REPORT_DIR / "transformer_validation_probabilities.npz"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "label_names.json", encoding="utf-8") as f:
    label_names = json.load(f)

if isinstance(label_names, dict):
    label_names = (
        label_names.get("label_names")
        or label_names.get("names")
        or label_names.get("labels")
    )

if not isinstance(label_names, list):
    raise ValueError("Label names must be a list.")

df = pd.read_parquet(DATA_DIR / "validation.parquet")
num_labels = len(label_names)

# Convert emotion IDs into multi-label binary targets.
y_true = np.zeros((len(df), num_labels), dtype=np.int32)

for row_index, labels in enumerate(df["labels"]):
    for label_id in labels:
        y_true[row_index, int(label_id)] = 1

# Reuse predictions if this script has already run.
if CACHE_FILE.exists():
    print("Loading cached validation probabilities...")
    with np.load(CACHE_FILE) as cache:
        probabilities = cache["probabilities"]
        cached_labels = cache["y_true"]

    if (
        probabilities.shape != y_true.shape
        or not np.array_equal(cached_labels, y_true)
    ):
        raise ValueError(
            "Cached predictions do not match validation data. "
            "Delete the cache and rerun."
        )
else:
    print("Generating validation predictions...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print("Using:", device)

    batch_size = 8 if device.type == "cuda" else 16
    batches = []

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
                key: value.to(device)
                for key, value in inputs.items()
            }

            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            batches.append(probs)

            if start % (batch_size * 50) == 0:
                print(
                    f"Processed "
                    f"{min(start + batch_size, len(df))}"
                    f"/{len(df)}"
                )

    probabilities = np.concatenate(batches, axis=0)

    np.savez_compressed(
        CACHE_FILE,
        probabilities=probabilities,
        y_true=y_true,
    )

    print("Validation probabilities saved!")

# Current baseline: shared threshold of 0.30
baseline_predictions = (probabilities >= 0.30).astype(int)

baseline_micro = f1_score(
    y_true, baseline_predictions,
    average="micro", zero_division=0
)
baseline_macro = f1_score(
    y_true, baseline_predictions,
    average="macro", zero_division=0
)

# Find the best F1 threshold for each emotion.
threshold_candidates = np.arange(0.10, 0.81, 0.05)
best_thresholds = []
emotion_results = []

for index, emotion in enumerate(label_names):
    actual = y_true[:, index]
    scores = probabilities[:, index]

    best_threshold = 0.30
    best_f1 = -1.0

    for threshold in threshold_candidates:
        predicted = (scores >= threshold).astype(int)

        score = f1_score(
            actual, predicted, zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    best_thresholds.append(best_threshold)

    emotion_results.append({
        "emotion": emotion,
        "threshold": round(best_threshold, 2),
        "f1": round(float(best_f1), 4),
        "validation_examples": int(actual.sum()),
    })

# Evaluate all individually tuned thresholds together.
best_thresholds = np.array(best_thresholds)
tuned_predictions = (
    probabilities >= best_thresholds[None, :]
).astype(int)

tuned_micro = f1_score(
    y_true, tuned_predictions,
    average="micro", zero_division=0
)
tuned_macro = f1_score(
    y_true, tuned_predictions,
    average="macro", zero_division=0
)

results_df = pd.DataFrame(emotion_results)
results_df.to_csv(
    REPORT_DIR / "per_emotion_thresholds.csv",
    index=False
)

settings = {
    "label_names": label_names,
    "thresholds": best_thresholds.tolist(),
    "validation_micro_f1": float(tuned_micro),
    "validation_macro_f1": float(tuned_macro),
}

with open(
    MODEL_DIR / "per_emotion_thresholds.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(settings, f, indent=2)

print("\n=== BEFORE: SHARED THRESHOLD ===")
print(f"Micro F1: {baseline_micro:.4f}")
print(f"Macro F1: {baseline_macro:.4f}")

print("\n=== AFTER: PER-EMOTION THRESHOLDS ===")
print(f"Micro F1: {tuned_micro:.4f}")
print(f"Macro F1: {tuned_macro:.4f}")

print("\n=== EMOTION THRESHOLDS ===")
print(results_df.to_string(index=False))

print("\nResults saved successfully!")
