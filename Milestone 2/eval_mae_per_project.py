# =========================
# Windows stability settings (MUST be first)
# =========================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import glob
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel, PeftConfig

HF_AUTHOR = "DEVCamiloSepulveda"
DATA_GLOB = r"data\llama3sp_dataset\*.csv"

MAX_LEN = 20             # Llama3SP artifacts mention title-focused + short max length
BATCH_SIZE = 16
USE_DESCRIPTION = False  # Baseline-aligned evaluation = title only (recommended)

# For quick debug runs; set to None to evaluate full test split.
LIMIT_TEST_ROWS = 100


def get_token() -> str:
    tok = os.getenv("HF_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN is not set. (Windows) set HF_TOKEN=YOUR_TOKEN")
    return tok


def get_base_model_id(hf_token: str) -> str:
    """Read adapter config to discover the correct base model id."""
    any_adapter = f"{HF_AUTHOR}/0-LLAMA3SP-usergrid"
    peft_cfg = PeftConfig.from_pretrained(any_adapter, token=hf_token)
    return peft_cfg.base_model_name_or_path


def load_tokenizer(hf_token: str):
    """Load tokenizer from an adapter repo and ensure padding is defined."""
    any_adapter = f"{HF_AUTHOR}/0-LLAMA3SP-usergrid"
    tok = AutoTokenizer.from_pretrained(any_adapter, token=hf_token)

    # Llama tokenizers often have no PAD token; use EOS as PAD for batching
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

    return tok


def load_base_once(hf_token: str, base_model_id: str, pad_token_id: int):
    """Load the base model once on CPU configured for regression."""
    cfg = AutoConfig.from_pretrained(base_model_id, token=hf_token)
    cfg.num_labels = 1
    cfg.problem_type = "regression"
    cfg.pad_token_id = pad_token_id

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        config=cfg,
        torch_dtype=torch.float32,
        device_map=None,          # CPU
        low_cpu_mem_usage=False,
        token=hf_token,
        ignore_mismatched_sizes=True,
    )
    base.config.pad_token_id = pad_token_id
    base.eval()
    return base


@torch.no_grad()
def predict_batch(tokenizer, model, titles, descs=None):
    """Predict a batch of story points."""
    if USE_DESCRIPTION and descs is not None:
        texts = []
        for t, d in zip(titles, descs):
            if isinstance(d, str) and d.strip():
                texts.append(t + "\n\n" + d)
            else:
                texts.append(t)
    else:
        texts = titles

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model(**inputs)
    return out.logits.squeeze(-1).float().cpu().numpy()


def get_project_name(csv_path: str) -> str:
    """Project name is the CSV filename without extension (lowercased)."""
    return os.path.splitext(os.path.basename(csv_path))[0].lower()


def load_adapter_if_needed(peft_model: PeftModel, adapter_id: str, adapter_name: str, hf_token: str):
    """
    Load a LoRA adapter into an existing PEFT model only if it's not already loaded.
    Then you can activate it with peft_model.set_adapter(adapter_name).
    """
    loaded = []
    try:
        loaded = list(peft_model.peft_config.keys())
    except Exception:
        pass

    if adapter_name not in loaded:
        peft_model.load_adapter(adapter_id, adapter_name=adapter_name, token=hf_token)

    peft_model.set_adapter(adapter_name)


def eval_one_project_csv(csv_path, tokenizer, peft_model, hf_token):
    """
    Evaluate ONE project:
    - Filter test split
    - Activate corresponding adapter in the shared PEFT model
    - Predict and compute MAE
    - Save per-project predictions CSV for evidence
    """
    project = get_project_name(csv_path)
    adapter_id = f"{HF_AUTHOR}/0-LLAMA3SP-{project}"

    df = pd.read_csv(csv_path)

    if "split_mark" not in df.columns:
        raise ValueError(f"{csv_path} missing split_mark column.")

    test_df = df[df["split_mark"].astype(str).str.lower().str.contains("test")].copy()
    if test_df.empty:
        raise ValueError(f"No test rows found in {csv_path} (split_mark).")

    # Optional quick debugging limit
    if LIMIT_TEST_ROWS is not None:
        test_df = test_df.head(LIMIT_TEST_ROWS).copy()

    # Activate adapter for this project (no stacking bug)
    load_adapter_if_needed(peft_model, adapter_id, adapter_name=project, hf_token=hf_token)
    peft_model.eval()

    if "storypoint" not in test_df.columns:
        raise ValueError(f"{csv_path} missing storypoint column.")
    if "title" not in test_df.columns:
        raise ValueError(f"{csv_path} missing title column.")

    y_true = test_df["storypoint"].astype(float).to_numpy()
    titles = test_df["title"].fillna("").astype(str).tolist()

    descs = None
    if USE_DESCRIPTION:
        descs = test_df["description"].fillna("").astype(str).tolist() if "description" in test_df.columns else [""] * len(test_df)

    preds = []
    for i in range(0, len(test_df), BATCH_SIZE):
        bt = titles[i:i + BATCH_SIZE]
        bd = descs[i:i + BATCH_SIZE] if descs is not None else None
        p = predict_batch(tokenizer, peft_model, bt, bd)
        preds.extend(p.tolist())

    y_pred = np.array(preds, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)

    # Save evidence
    os.makedirs("results", exist_ok=True)

    id_col = None
    for cand in ["issuekey", "issue_id", "id", "key"]:
        if cand in test_df.columns:
            id_col = cand
            break

    out_df = pd.DataFrame({
        "project": project,
        "title": titles,
        "y_true": y_true,
        "y_pred": y_pred,
        "abs_error": np.abs(y_true - y_pred),
    })

    if id_col:
        out_df.insert(0, id_col, test_df[id_col].astype(str).values)

    out_df.to_csv(f"results/predictions_{project}.csv", index=False)

    return project, len(test_df), mae


def main():
    print(">>> eval_mae_per_project.py started")
    torch.set_num_threads(1)

    hf_token = get_token()

    csv_files = sorted(glob.glob(DATA_GLOB))
    print(f"Found CSV files: {len(csv_files)}")
    if not csv_files:
        raise RuntimeError("No CSV files found. Check your DATA_GLOB path.")

    base_model_id = get_base_model_id(hf_token)
    print(f"Base model from adapter config: {base_model_id}")

    tokenizer = load_tokenizer(hf_token)
    print("Tokenizer loaded.")

    base_model = load_base_once(hf_token, base_model_id, tokenizer.pad_token_id)
    print("Base model loaded successfully.")

    # Build ONE PEFT model instance using a known adapter (first project in list)
    first_project = get_project_name(csv_files[0])
    first_adapter_id = f"{HF_AUTHOR}/0-LLAMA3SP-{first_project}"
    peft_model = PeftModel.from_pretrained(base_model, first_adapter_id, token=hf_token)
    peft_model.eval()
    print(f"PEFT model created with initial adapter: {first_project}")

    results = []
    for csv_path in csv_files:
        project, n, mae = eval_one_project_csv(csv_path, tokenizer, peft_model, hf_token)
        results.append((project, n, mae))
        print(f"{project:18s} | test_n={n:4d} | MAE={mae:.4f}")

        # Light cleanup
        gc.collect()

    os.makedirs("results", exist_ok=True)

    out = pd.DataFrame(results, columns=["project", "test_size", "mae"])
    out.to_csv("results/mae_per_project.csv", index=False)

    summary = {
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "use_description": USE_DESCRIPTION,
        "limit_test_rows": LIMIT_TEST_ROWS,
        "projects_evaluated": len(results),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model_id": base_model_id,
    }
    with open("results/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved -> results/mae_per_project.csv")
    print("Saved -> results/summary.json")
    print("Saved -> results/predictions_<project>.csv (one per project)")
    print("Done.")


if __name__ == "__main__":
    main()