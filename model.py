import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Select GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to your trained models
BINARY_MODEL_PATH = "saved_models/binary_distilbert"
MULTI_MODEL_PATH = "saved_models/multiclass_distilbert"

print("🔄 Loading DistilBERT models...")
tokenizer_bin = DistilBertTokenizer.from_pretrained(BINARY_MODEL_PATH)
model_bin = DistilBertForSequenceClassification.from_pretrained(BINARY_MODEL_PATH).to(device)
model_bin.eval()

tokenizer_multi = DistilBertTokenizer.from_pretrained(MULTI_MODEL_PATH)
model_multi = DistilBertForSequenceClassification.from_pretrained(MULTI_MODEL_PATH).to(device)
model_multi.eval()
print("✅ Models loaded successfully.")

def predict_sentiment(text: str):
    """Run sentiment analysis on a single text string."""

    # ---- Binary Prediction ----
    inputs_bin = tokenizer_bin(text, return_tensors="pt", truncation=True, padding=True, max_length=160).to(device)
    with torch.no_grad():
        outputs_bin = model_bin(**inputs_bin)
        probs_bin = torch.softmax(outputs_bin.logits, dim=1)
        pred_bin = torch.argmax(probs_bin, dim=1).item()
    binary_sentiment = "Negative" if pred_bin == 0 else "Positive"
    binary_conf = probs_bin[0][pred_bin].item()

    # ---- Multiclass Prediction ----
    inputs_multi = tokenizer_multi(text, return_tensors="pt", truncation=True, padding=True, max_length=160).to(device)
    with torch.no_grad():
        outputs_multi = model_multi(**inputs_multi)
        probs_multi = torch.softmax(outputs_multi.logits, dim=1)
        pred_multi = torch.argmax(probs_multi, dim=1).item()
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    multi_sentiment = label_map[pred_multi]
    multi_conf = probs_multi[0][pred_multi].item()

    return {
        "binary_sentiment": binary_sentiment,
        "binary_confidence": round(binary_conf, 4),
        "multiclass_sentiment": multi_sentiment,
        "multiclass_confidence": round(multi_conf, 4),
    }