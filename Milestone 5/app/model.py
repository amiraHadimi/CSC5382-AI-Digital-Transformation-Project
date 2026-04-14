import os
from typing import Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftConfig, PeftModel
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "DEVCamiloSepulveda/2-LLAMA3SP-talendesb"

_model = None
_tokenizer = None
_model_name = None


def _load_llama3sp():
    global _model, _tokenizer, _model_name

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer, _model_name

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is missing.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    peft_config = PeftConfig.from_pretrained(MODEL_ID, token=hf_token)
    base_model_name = peft_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        token=hf_token,
        num_labels=1,
        dtype=torch.float32,
    )

    model = PeftModel.from_pretrained(
        base_model,
        MODEL_ID,
        token=hf_token,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.to(device)

    _model = model
    _tokenizer = tokenizer
    _model_name = MODEL_ID

    return _model, _tokenizer, _model_name


def _fallback_predict(title: str, description: str) -> float:
    text = f"{title} {description}".strip()
    words = len(text.split())
    return float(max(1, min(13, words // 8 + 1)))


def predict_story_points(title: str, description: str) -> Tuple[float, str]:
    try:
        model, tokenizer, model_name = _load_llama3sp()
        device = next(model.parameters()).device

        text = title.strip()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=20,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = outputs.logits.squeeze().item()

        return round(float(prediction), 2), model_name

    except Exception:
        prediction = _fallback_predict(title, description)
        return prediction, "fallback"