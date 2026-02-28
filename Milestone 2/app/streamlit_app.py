import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftConfig, PeftModel

LLAMA3SP_ID = "DEVCamiloSepulveda/0-LLAMA3SP-usergrid"

def round_to_half(x: float) -> float:
    return round(x * 2) / 2.0

@st.cache_resource
def load_llama3sp_cpu_cached():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not set in your environment.")

    peft_cfg = PeftConfig.from_pretrained(LLAMA3SP_ID, token=hf_token)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA3SP_ID, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_config = AutoConfig.from_pretrained(peft_cfg.base_model_name_or_path, token=hf_token)
    base_config.num_labels = 1
    base_config.problem_type = "regression"

    base = AutoModelForSequenceClassification.from_pretrained(
        peft_cfg.base_model_name_or_path,
        config=base_config,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True
    )

    model = PeftModel.from_pretrained(base, LLAMA3SP_ID, token=hf_token)
    model.eval()
    return tokenizer, model

@torch.no_grad()
def predict(tokenizer, model, title: str, desc: str = "") -> float:
    text = title if not desc else (title + "\n\n" + desc)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=20,
        padding="max_length"
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model(**inputs)
    return float(out.logits.squeeze().item())

st.set_page_config(page_title="Story Points PoC", layout="centered")
st.title("Milestone 2 PoC — Story Point Estimation (Llama3SP)")

st.caption(
    "Decision-support demo: the model suggests story points, but a human still validates the final estimate."
)

title = st.text_input("Issue title", value="Add export feature to CSV")
desc = st.text_area("Description (optional)", value="", height=120)

col1, col2 = st.columns(2)
with col1:
    clamp_nonnegative = st.checkbox("Clamp negative to 0", value=True)
with col2:
    round_half = st.checkbox("Round to nearest 0.5", value=True)

if st.button("Predict"):
    try:
        tok, mdl = load_llama3sp_cpu_cached()
        raw = predict(tok, mdl, title, desc)

        final = max(0.0, raw) if clamp_nonnegative else raw
        final2 = round_to_half(final) if round_half else final

        st.success(f"Raw prediction: {raw:.3f}")
        st.info(f"Final suggestion: {final2:.1f}" if round_half else f"Final suggestion: {final2:.3f}")
        st.caption("Note: Llama3SP usergrid uses max_length=20 and is title-focused; long text is truncated.")
    except Exception as e:
        st.error(str(e))
