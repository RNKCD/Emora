
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "distilroberta"
LABEL_FILE = ROOT / "data" / "processed" / "label_names.json"


def _get_hf_token():
    """Read a private model token from environment or Streamlit secrets."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    try:
        import streamlit as st
        return st.secrets.get("HF_TOKEN")
    except (ImportError, FileNotFoundError, KeyError):
        return None


class EmotionPredictor:
    def __init__(self, threshold=0.30, device=None):
        with open(LABEL_FILE, encoding="utf-8") as f:
            self.label_names = json.load(f)

        if isinstance(self.label_names, dict):
            self.label_names = (
                self.label_names.get("label_names")
                or self.label_names.get("names")
                or self.label_names.get("labels")
            )

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Local by default; set EMORA_MODEL_REPO to load from Hugging Face.
        model_source = os.getenv("EMORA_MODEL_REPO") or str(MODEL_DIR)
        token = _get_hf_token() if os.getenv("EMORA_MODEL_REPO") else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source, token=token
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source, token=token
        ).to(self.device)

        self.model.eval()
        self.threshold = threshold

    def predict(self, text, top_k=3):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()

        ranked = np.argsort(probabilities)[::-1]

        detected = [
            {
                "emotion": self.label_names[i],
                "confidence": round(float(probabilities[i]), 4),
            }
            for i in ranked
            if probabilities[i] >= self.threshold
        ]

        return {
            "detected_emotions": detected[:top_k],
            "top_emotion": self.label_names[ranked[0]],
            "top_confidence": round(float(probabilities[ranked[0]]), 4),
        }


if __name__ == "__main__":
    predictor = EmotionPredictor()

    examples = [
        "I am so happy that I passed my exam!",
        "I feel anxious about tomorrow.",
        "I am frustrated and exhausted.",
    ]

    for example in examples:
        print("\nMessage:", example)
        print(json.dumps(predictor.predict(example), indent=2))
