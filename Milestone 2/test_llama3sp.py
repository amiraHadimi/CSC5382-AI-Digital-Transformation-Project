import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftConfig, PeftModel

LLAMA3SP_ID = "DEVCamiloSepulveda/0-LLAMA3SP-usergrid"

def load_llama3sp_cpu():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set.\n"
            "Windows CMD:  set HF_TOKEN=YOUR_TOKEN\n"
            "PowerShell:   $env:HF_TOKEN='YOUR_TOKEN'\n"
        )

    # Adapter config tells us which base model to load
    peft_cfg = PeftConfig.from_pretrained(LLAMA3SP_ID, token=hf_token)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA3SP_ID, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Regression head
    base_config = AutoConfig.from_pretrained(peft_cfg.base_model_name_or_path, token=hf_token)
    base_config.num_labels = 1
    base_config.problem_type = "regression"

    base = AutoModelForSequenceClassification.from_pretrained(
        peft_cfg.base_model_name_or_path,
        config=base_config,
        torch_dtype=torch.float32,
        device_map=None,            # CPU
        low_cpu_mem_usage=False,    # avoids accelerate disk offload issues
        ignore_mismatched_sizes=True
    )

    model = PeftModel.from_pretrained(base, LLAMA3SP_ID, token=hf_token)
    model.eval()

    # Proof adapter is attached
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    return tokenizer, model

@torch.no_grad()
def predict(tokenizer, model, title: str, desc: str = "", clamp_nonnegative: bool = True) -> float:
    # Llama3SP usergrid is documented as title-focused + max_length=20
    text = title if not desc else (title + "\n\n" + desc)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=20,
        padding="max_length"
    )

    # Move inputs to model device (robust even if you later use GPU)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    yhat = float(outputs.logits.squeeze().item())

    if clamp_nonnegative:
        yhat = max(0.0, yhat)

    return yhat

def round_to_half(x: float) -> float:
    return round(x * 2) / 2.0

if __name__ == "__main__":
    print("Loading Llama3SP (CPU mode)...")
    tok, mdl = load_llama3sp_cpu()

    title = "Add export feature to CSV"
    raw = predict(tok, mdl, title, clamp_nonnegative=False)
    clamped = max(0.0, raw)
    rounded = round_to_half(clamped)

    print(f"Input: {title}")
    print(f"Raw prediction: {raw:.3f}")
    print(f"Clamped (>=0): {clamped:.3f}")
    print(f"Rounded (0.5): {rounded:.1f}")