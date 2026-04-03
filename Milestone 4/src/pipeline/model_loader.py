"""
pipeline/model_loader.py
========================
Loads the Llama3SP base model and per-project LoRA adapters from HF Hub.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel, PeftConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def resolve_base_model_id(hf_author, hf_token):
    """Discover base model ID from an adapter PeftConfig."""
    cfg = PeftConfig.from_pretrained(f"{hf_author}/0-LLAMA3SP-usergrid", token=hf_token)
    return cfg.base_model_name_or_path


def load_tokenizer(hf_author, hf_token):
    """Load Llama tokenizer; set pad token to eos token."""
    tokenizer = AutoTokenizer.from_pretrained(
        f"{hf_author}/0-LLAMA3SP-usergrid", token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_base_model(base_model_id, pad_token_id, hf_token):
    """Load base Llama model configured for regression on CPU."""
    cfg = AutoConfig.from_pretrained(base_model_id, token=hf_token)
    cfg.num_labels = 1
    cfg.problem_type = "regression"
    cfg.pad_token_id = pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        config=cfg,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=False,
        token=hf_token,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = pad_token_id
    model.eval()
    return model


def build_peft_model(base_model, hf_author, first_project, hf_token):
    """Wrap base model in PeftModel with first project adapter."""
    peft_model = PeftModel.from_pretrained(
        base_model, f"{hf_author}/0-LLAMA3SP-{first_project}", token=hf_token
    )
    peft_model.eval()
    return peft_model


def load_adapter_for_project(peft_model, hf_author, project, hf_token):
    """Dynamically load and activate a project-specific LoRA adapter."""
    loaded = list(peft_model.peft_config.keys()) if hasattr(peft_model, "peft_config") else []
    if project not in loaded:
        peft_model.load_adapter(
            f"{hf_author}/0-LLAMA3SP-{project}", adapter_name=project, token=hf_token
        )
    peft_model.set_adapter(project)
