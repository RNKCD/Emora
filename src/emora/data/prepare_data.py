
from pathlib import Path
import json

import pandas as pd
from datasets import load_dataset

# Find the Emora project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def prepare_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading GoEmotions dataset...")

    dataset = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified"
    )

    # Retrieve all emotion label names
    label_names = dataset["train"].features[
        "labels"
    ].feature.names

    # Save label names for future ML models
    with open(
        OUTPUT_DIR / "label_names.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(label_names, file, indent=2)

    for split in ["train", "validation", "test"]:
        df = dataset[split].to_pandas()

        # Remove empty text
        df = df.dropna(subset=["text"]).copy()
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 0].copy()

        # Preserve all labels for multi-label ML
        df["labels"] = df["labels"].apply(
            lambda ids: [int(i) for i in ids]
        )

        # Convert label IDs into emotion names
        df["emotion_names"] = df["labels"].apply(
            lambda ids: [label_names[i] for i in ids]
        )

        # Useful features for exploratory analysis
        df["word_count"] = (
            df["text"].str.split().str.len()
        )
        df["text_length"] = df["text"].str.len()
        df["num_emotions"] = df["labels"].str.len()

        # Save prepared dataset
        output_file = OUTPUT_DIR / f"{split}.parquet"
        df.to_parquet(output_file, index=False)

        print(f"{split}: {len(df):,} samples saved")

    print("\nDataset preparation completed!")


if __name__ == "__main__":
    prepare_dataset()
