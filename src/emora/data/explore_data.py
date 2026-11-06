
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
POWERBI_DIR = ROOT / "data" / "powerbi"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
POWERBI_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
train = pd.read_parquet(DATA_DIR / "train.parquet")
validation = pd.read_parquet(DATA_DIR / "validation.parquet")
test = pd.read_parquet(DATA_DIR / "test.parquet")

with open(DATA_DIR / "label_names.json", encoding="utf-8") as f:
    label_names = json.load(f)

splits = {
    "Train": train,
    "Validation": validation,
    "Test": test
}

print("\n=== DATASET SUMMARY ===")

summary = []

for name, df in splits.items():
    summary.append({
        "split": name,
        "samples": len(df),
        "average_words": round(df["word_count"].mean(), 2),
        "average_characters": round(df["text_length"].mean(), 2),
        "multi_label_pct": round(
            (df["num_emotions"] > 1).mean() * 100, 2
        ),
        "duplicate_texts": int(df["text"].duplicated().sum())
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

summary_df.to_csv(
    POWERBI_DIR / "dataset_summary.csv",
    index=False
)

# Emotion frequency for all splits
emotion_records = []

for name, df in splits.items():
    counts = df["emotion_names"].explode().value_counts()

    for emotion, count in counts.items():
        emotion_records.append({
            "split": name,
            "emotion": emotion,
            "count": int(count)
        })

emotion_df = pd.DataFrame(emotion_records)

emotion_df.to_csv(
    POWERBI_DIR / "emotion_distribution.csv",
    index=False
)

# Chart 1: Emotion distribution
train_emotions = (
    emotion_df[emotion_df["split"] == "Train"]
    .sort_values("count", ascending=False)
)

plt.figure(figsize=(12, 10))
sns.barplot(
    data=train_emotions,
    x="count",
    y="emotion"
)
plt.title("GoEmotions: Training Emotion Distribution")
plt.xlabel("Number of Label Assignments")
plt.ylabel("Emotion")
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "emotion_distribution.png",
    dpi=300
)
plt.close()

# Chart 2: Word count distribution
plt.figure(figsize=(10, 6))
sns.histplot(
    train["word_count"],
    bins=50
)
plt.title("Text Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "word_count_distribution.png",
    dpi=300
)
plt.close()

# Chart 3: Number of emotions per message
plt.figure(figsize=(8, 5))
sns.countplot(
    data=train,
    x="num_emotions"
)
plt.title("Number of Emotion Labels per Message")
plt.xlabel("Number of Labels")
plt.ylabel("Number of Messages")
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "multi_label_distribution.png",
    dpi=300
)
plt.close()

# Chart 4: Split size comparison
plt.figure(figsize=(8, 5))
sns.barplot(
    data=summary_df,
    x="split",
    y="samples"
)
plt.title("Dataset Split Sizes")
plt.xlabel("Split")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "dataset_splits.png",
    dpi=300
)
plt.close()

# Check text overlap between splits
print("\n=== CROSS-SPLIT TEXT OVERLAP ===")
for a, b in [
    ("Train", "Validation"),
    ("Train", "Test"),
    ("Validation", "Test")
]:
    overlap = len(
        set(splits[a]["text"]) &
        set(splits[b]["text"])
    )
    print(f"{a} vs {b}: {overlap} matching texts")

print("\nEDA completed successfully!")
print(f"Charts saved to: {FIGURES_DIR}")
print(f"Power BI files saved to: {POWERBI_DIR}")

