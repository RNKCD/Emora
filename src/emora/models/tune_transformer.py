
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "distilroberta"
VALIDATION_FILE = ROOT / "data" / "processed" / "validation.parquet"
LABEL_FILE = ROOT / "data" / "processed" / "label_names.json"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(LABEL_FILE, encoding="utf-8") as f:
    label_names = json.load(f)

# Handle either a list or a dictionary of label names.
if isinstance(label_names, dict):
    label_names = (
        label_names.get("label_names")
        or label_names.get("names")
        or label_names.get("labels")
    )

if not isinstance(label_names, list):
    raise ValueError("Could not read label names from JSON.")

df = pd.read_parquet(VALIDATION_FILE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Device:", device)
print("Validation examples:", len(df))

# Use the same label encoding as training.
y_true = np.zeros((len(df), len(label_names)), dtype=np.int32)

for i, labels in enumerate(df["labels"]):
    for label_id in labels:
        y_true[i, int(label_id)] = 1

probabilities = []

batch_size = 8 if device.type == "cuda" else 16

with torch.inference_mode():
    for start in range(0, len(df), batch_size):
        texts = df["text"].iloc[start:start + batch_size].tolist()

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        probabilities.append(probs)

        if start % (batch_size * 50) == 0:
            print(f"Processed {min(start + batch_size, len(df))}/{len(df)}")

probabilities = np.concatenate(probabilities, axis=0)

results = []

for threshold in np.arange(0.10, 0.81, 0.05):
    predictions = (probabilities >= threshold).astype(int)

    micro = f1_score(y_true, predictions, average="micro", zero_division=0)
    macro = f1_score(y_true, predictions, average="macro", zero_division=0)

    results.append({
        "threshold": round(float(threshold), 2),
        "micro_f1": round(float(micro), 4),
        "macro_f1": round(float(macro), 4),
    })

results_df = pd.DataFrame(results)
results_df.to_csv(REPORT_DIR / "transformer_threshold_results.csv", index=False)

best = results_df.loc[results_df["micro_f1"].idxmax()]

settings = {
    "threshold": float(best["threshold"]),
    "micro_f1": float(best["micro_f1"]),
    "macro_f1": float(best["macro_f1"]),
}

with open(MODEL_DIR / "best_threshold.json", "w", encoding="utf-8") as f:
    json.dump(settings, f, indent=2)

print("\n=== TRANSFORMER THRESHOLD RESULTS ===")
print(results_df.to_string(index=False))

print("\n=== BEST THRESHOLD (BY MICRO F1) ===")
print(json.dumps(settings, indent=2))
