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


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Story Points PoC", layout="centered")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,60,255,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(0,200,160,0.10) 0%, transparent 55%);
    min-height: 100vh;
}

/* ── Main container ── */
.block-container {
    max-width: 680px !important;
    padding-top: 3rem !important;
    padding-bottom: 4rem !important;
}

/* ── Header badge ── */
.header-badge {
    display: inline-block;
    background: rgba(99,60,255,0.18);
    border: 1px solid rgba(99,60,255,0.4);
    color: #a78bfa;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 100px;
    margin-bottom: 1rem;
}

/* ── Title ── */
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.1;
    color: #f0eeff;
    letter-spacing: -0.03em;
    margin-bottom: 0.4rem;
}
.main-title span {
    color: #7c5cfc;
}

/* ── Subtitle ── */
.subtitle {
    color: #6b6880;
    font-size: 0.88rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.02em;
    margin-bottom: 2.5rem;
    line-height: 1.6;
}

/* ── Divider ── */
.styled-divider {
    height: 1px;
    background: linear-gradient(to right, rgba(99,60,255,0.5), rgba(0,200,160,0.3), transparent);
    margin: 2rem 0;
    border: none;
}

/* ── Card wrapper ── */
.input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}

/* ── Section labels ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #7c5cfc;
    margin-bottom: 0.5rem;
}

/* ── Streamlit input overrides ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e4ff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(99,60,255,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,60,255,0.12) !important;
}
.stTextInput > label,
.stTextArea > label {
    color: #9490aa !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
}

/* ── Checkbox overrides ── */
.stCheckbox > label {
    color: #9490aa !important;
    font-size: 0.85rem !important;
    font-family: 'DM Mono', monospace !important;
}
.stCheckbox > label > span {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 5px !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #6330ff 0%, #4f27d4 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.65rem 2.2rem !important;
    width: 100% !important;
    transition: opacity 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(99,60,255,0.35) !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(99,60,255,0.5) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Result cards ── */
.result-raw {
    background: rgba(99,60,255,0.08);
    border: 1px solid rgba(99,60,255,0.25);
    border-left: 3px solid #7c5cfc;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.8rem;
}
.result-final {
    background: rgba(0,200,160,0.07);
    border: 1px solid rgba(0,200,160,0.25);
    border-left: 3px solid #00c8a0;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.8rem;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b6880;
    margin-bottom: 0.25rem;
}
.result-value {
    font-size: 2rem;
    font-weight: 800;
    color: #f0eeff;
    letter-spacing: -0.02em;
}
.result-value.accent { color: #00c8a0; }

.result-note {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #5a5670;
    margin-top: 1.2rem;
    line-height: 1.6;
    padding: 0.8rem 1rem;
    background: rgba(255,255,255,0.02);
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* ── Error ── */
.stAlert {
    border-radius: 12px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="header-badge">Milestone 2 · PoC</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Story Point<br><span>Estimator</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Llama3SP · Decision-support demo — model suggests story points,<br>a human validates the final estimate.</div>',
    unsafe_allow_html=True
)
st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Issue Details</div>', unsafe_allow_html=True)
title = st.text_input("Issue title", value="Add export feature to CSV")
desc  = st.text_area("Description (optional)", value="", height=120)

st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Post-processing</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    clamp_nonnegative = st.checkbox("Clamp negative → 0", value=True)
with col2:
    round_half = st.checkbox("Round to nearest 0.5", value=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict button ─────────────────────────────────────────────────────────────
if st.button("⚡  Run Estimation"):
    try:
        with st.spinner("Loading model & running inference…"):
            tok, mdl = load_llama3sp_cpu_cached()
            raw = predict(tok, mdl, title, desc)

        final = max(0.0, raw) if clamp_nonnegative else raw
        final2 = round_to_half(final) if round_half else final

        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="result-raw">
                <div class="result-label">Raw prediction</div>
                <div class="result-value">{raw:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            fmt = f"{final2:.1f}" if round_half else f"{final2:.3f}"
            st.markdown(f"""
            <div class="result-final">
                <div class="result-label">Final suggestion</div>
                <div class="result-value accent">{fmt}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            '<div class="result-note">⚠ Llama3SP usergrid uses max_length=20 and is title-focused — long descriptions are truncated at inference time.</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(str(e))
