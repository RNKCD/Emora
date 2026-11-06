from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

model_path = "models/emora_inference"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)

sample = tokenizer(
    "I am feeling happy today!",
    return_tensors="pt",
    truncation=True,
)

output = model(**sample)

print("Model loaded successfully!")
print("Output shape:", output.logits.shape)