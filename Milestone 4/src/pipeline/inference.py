"""
pipeline/inference.py
=====================
Batched inference for the Llama3SP story point regression model.
"""
import numpy as np
import torch


@torch.no_grad()
def predict_batch(tokenizer, model, titles, descriptions=None,
                  max_len=20, use_description=False):
    """Run a single inference batch; return numpy array of predictions."""
    if use_description and descriptions is not None:
        texts = [
            f"{t}\n\n{d}" if (isinstance(d, str) and d.strip()) else t
            for t, d in zip(titles, descriptions)
        ]
    else:
        texts = list(titles)

    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=max_len, padding="max_length",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs.logits.squeeze(-1).float().cpu().numpy()


def run_inference_on_dataframe(df, tokenizer, peft_model,
                                batch_size=16, max_len=20, use_description=False):
    """Run full inference over a DataFrame; return numpy array of predictions."""
    titles = df["title"].fillna("").astype(str).tolist()
    descriptions = (
        df["description"].fillna("").astype(str).tolist()
        if use_description and "description" in df.columns else None
    )
    preds = []
    for i in range(0, len(df), batch_size):
        batch_preds = predict_batch(
            tokenizer, peft_model,
            titles[i:i+batch_size],
            descriptions[i:i+batch_size] if descriptions else None,
            max_len, use_description,
        )
        preds.extend(batch_preds.tolist())
    return np.array(preds, dtype=float)
