
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading datasets...")

train = pd.read_parquet(DATA_DIR / "train.parquet")
val = pd.read_parquet(DATA_DIR / "validation.parquet")

# Reuse the same TF-IDF vectorizer and label mapping
# for a fair comparison with Logistic Regression.
vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.joblib"
)
mlb = joblib.load(
    MODEL_DIR / "label_binarizer.joblib"
)

X_train = vectorizer.transform(train["text"])
X_val = vectorizer.transform(val["text"])

y_train = mlb.transform(train["labels"])
y_val = mlb.transform(val["labels"])

print("Training Linear SVM...")

svm = OneVsRestClassifier(
    LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )
)

svm.fit(X_train, y_train)

print("Training completed!")

# LinearSVC produces decision scores, not probabilities.
scores = svm.decision_function(X_val)

# Tune a shared decision-score threshold
thresholds = np.arange(-1.0, 1.01, 0.1)
results = []

for threshold in thresholds:
    predictions = (scores >= threshold).astype(int)

    results.append({
        "threshold": round(float(threshold), 2),
        "micro_f1": f1_score(
            y_val, predictions,
            average="micro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_val, predictions,
            average="macro",
            zero_division=0
        ),
        "precision": precision_score(
            y_val, predictions,
            average="micro",
            zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions,
            average="micro",
            zero_division=0
        )
    })

results_df = pd.DataFrame(results)

best = results_df.loc[
    results_df["micro_f1"].idxmax()
]

print("\n=== BEST LINEAR SVM RESULTS ===")
print(f"Decision threshold: {best['threshold']:.2f}")
print(f"Micro F1: {best['micro_f1']:.4f}")
print(f"Macro F1: {best['macro_f1']:.4f}")
print(f"Precision: {best['precision']:.4f}")
print(f"Recall: {best['recall']:.4f}")

# Save model
joblib.dump(
    svm,
    MODEL_DIR / "svm_model.joblib"
)

# Save SVM threshold
with open(
    MODEL_DIR / "svm_threshold.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {"decision_threshold": float(best["threshold"])},
        f,
        indent=2
    )

# Save threshold comparison
results_df.to_csv(
    REPORT_DIR / "svm_threshold_results.csv",
    index=False
)

# Compare with our optimized Logistic Regression
comparison = pd.DataFrame([
    {
        "model": "Logistic Regression",
        "micro_f1": 0.5088,
        "macro_f1": 0.4614
    },
    {
        "model": "Linear SVM",
        "micro_f1": float(best["micro_f1"]),
        "macro_f1": float(best["macro_f1"])
    }
])

print("\n=== MODEL COMPARISON ===")
print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\nLinear SVM experiment completed!")
