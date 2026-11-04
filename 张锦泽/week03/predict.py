import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MODEL_PATH = "./language_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
        confidence, predicted_id = probabilities.max(dim=-1)

    language = model.config.id2label[predicted_id.item()]

    return language, confidence.item()


texts = [
    "Hello, how are you?",
    "你好，今天天气不错",
    "こんにちは、お元気ですか",
    "Bonjour tout le monde",
]

for text in texts:
    language, confidence = predict(text)
    print(f"{text} -> {language} ({confidence:.2%})")
