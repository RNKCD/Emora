
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load the previously trained model
model = joblib.load(MODEL_DIR / "baseline_model.joblib")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
mlb = joblib.load(MODEL_DIR / "label_binarizer.joblib")

# Load validation data only
validation = pd.read_parquet(
    DATA_DIR / "validation.parquet"
)

X_val = vectorizer.transform(validation["text"])
y_val = mlb.transform(validation["labels"])

# Get predicted probabilities
probabilities = model.predict_proba(X_val)

results = []

print("Testing different probability thresholds...")

for threshold in np.arange(0.20, 0.81, 0.05):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    micro_f1 = f1_score(
        y_val, predictions,
        average="micro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_val, predictions,
        average="macro",
        zero_division=0
    )

    precision = precision_score(
        y_val, predictions,
        average="micro",
        zero_division=0
    )

    recall = recall_score(
        y_val, predictions,
        average="micro",
        zero_division=0
    )

    results.append({
        "threshold": round(float(threshold), 2),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall
    })

results_df = pd.DataFrame(results)

print("\n=== THRESHOLD COMPARISON ===")
print(results_df.round(4).to_string(index=False))

# Select threshold using validation Micro F1
best_row = results_df.loc[
    results_df["micro_f1"].idxmax()
]

best_threshold = float(best_row["threshold"])

print("\n=== BEST THRESHOLD ===")
print(f"Threshold: {best_threshold:.2f}")
print(f"Micro F1: {best_row['micro_f1']:.4f}")
print(f"Macro F1: {best_row['macro_f1']:.4f}")
print(f"Precision: {best_row['precision']:.4f}")
print(f"Recall: {best_row['recall']:.4f}")

# Save threshold for future predictions
with open(
    MODEL_DIR / "best_threshold.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {"threshold": best_threshold},
        f,
        indent=2
    )

# Save results for later Power BI work
results_df.to_csv(
    REPORT_DIR / "threshold_comparison.csv",
    index=False
)

# Plot threshold comparison
plt.figure(figsize=(9, 5))

plt.plot(
    results_df["threshold"],
    results_df["micro_f1"],
    marker="o",
    label="Micro F1"
)

plt.plot(
    results_df["threshold"],
    results_df["macro_f1"],
    marker="s",
    label="Macro F1"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best threshold: {best_threshold:.2f}"
)

plt.xlabel("Probability Threshold")
plt.ylabel("F1 Score")
plt.title("Emora Baseline: Threshold Tuning")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "figures" / "threshold_tuning.png",
    dpi=300
)

plt.close()

print("\nThreshold tuning completed!")

