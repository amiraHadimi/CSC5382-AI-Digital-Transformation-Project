import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

MODEL_ID = "DEVCamiloSepulveda/2-LLAMA3SP-talendesb"

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise Exception("HF_TOKEN missing")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    peft_config = PeftConfig.from_pretrained(MODEL_ID, token=hf_token)
    base_model_name = peft_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        token=hf_token,
        torch_dtype=torch.float32,
    )

    model = PeftModel.from_pretrained(base_model, MODEL_ID, token=hf_token)

    model.eval()
    model.to(device)

    _model = model
    _tokenizer = tokenizer

    return model, tokenizer


# 🔴 fallback (VERY IMPORTANT for HF)
def fallback_prediction(text):
    words = len(text.split())
    return float(max(1, min(13, words // 8 + 1)))


def predict_story_points(title, description):
    text = f"{title} {description}"

    try:
        model, tokenizer = load_model()
        device = next(model.parameters()).device

        inputs = tokenizer(
            title,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=20,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.logits.squeeze().item()

        return round(float(pred), 2), "LLAMA3SP"

    except Exception:
        # fallback if model crashes (likely on HF)
        return fallback_prediction(text), "fallback"