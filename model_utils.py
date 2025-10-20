import torch
import numpy as np
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Use HF_TOKEN from environment for private repos
HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face model repo paths
BINARY_PATH = "localhost2002/binary-distilbert"
MULTI_PATH = "localhost2002/multiclass-distilbert"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load binary model
binary_tokenizer = DistilBertTokenizer.from_pretrained(BINARY_PATH, use_auth_token=HF_TOKEN)
binary_model = DistilBertForSequenceClassification.from_pretrained(BINARY_PATH, use_auth_token=HF_TOKEN)
binary_model.to(device)
binary_model.eval()

# Load multiclass model
multi_tokenizer = DistilBertTokenizer.from_pretrained(MULTI_PATH, use_auth_token=HF_TOKEN)
multi_model = DistilBertForSequenceClassification.from_pretrained(MULTI_PATH, use_auth_token=HF_TOKEN)
multi_model.to(device)
multi_model.eval()

label_map_multi = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_binary(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = binary_tokenizer(batch, truncation=True, padding=True, max_length=160, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = binary_model(**enc)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        for text, pred, p in zip(batch, preds, probs):
            results.append({
                "text": text,
                "label": "Positive" if pred == 1 else "Negative",
                "confidence": float(np.max(p))
            })
    return results

def predict_multiclass(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = multi_tokenizer(batch, truncation=True, padding=True, max_length=160, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = multi_model(**enc)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        for text, pred, p in zip(batch, preds, probs):
            results.append({
                "text": text,
                "label": label_map_multi[pred],
                "confidence": float(np.max(p))
            })
    return results
