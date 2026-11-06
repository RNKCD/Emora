
from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score
import joblib

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading GoEmotions dataset...")

train = pd.read_parquet(DATA_DIR / "train.parquet")
validation = pd.read_parquet(DATA_DIR / "validation.parquet")

with open(DATA_DIR / "label_names.json", encoding="utf-8") as f:
    emotion_names = json.load(f)

# Convert multi-label annotations to binary columns
mlb = MultiLabelBinarizer(classes=list(range(len(emotion_names))))

y_train = mlb.fit_transform(train["labels"])
y_val = mlb.transform(validation["labels"])

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

print("Creating TF-IDF features...")

X_train = vectorizer.fit_transform(train["text"])
X_val = vectorizer.transform(validation["text"])

# Train one classifier for each emotion
model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
)

print("Training baseline model...")
model.fit(X_train, y_train)

# Predict emotions on validation data
y_pred = model.predict(X_val)

print("\n=== BASELINE MODEL RESULTS ===")
print("Micro F1:", round(f1_score(y_val, y_pred, average="micro", zero_division=0), 4))
print("Macro F1:", round(f1_score(y_val, y_pred, average="macro", zero_division=0), 4))

print(
    classification_report(
        y_val,
        y_pred,
        target_names=emotion_names,
        zero_division=0
    )
)

# Save the trained model and vectorizer
joblib.dump(model, MODEL_DIR / "baseline_model.joblib")
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
joblib.dump(mlb, MODEL_DIR / "label_binarizer.joblib")

print("\nBaseline model saved successfully!")

