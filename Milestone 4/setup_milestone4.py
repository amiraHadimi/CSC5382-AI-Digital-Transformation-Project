"""
setup_milestone4.py
===================
Run this script from inside your 'Milestone 4/' folder to create
all missing source files with the correct content.

Usage:
    cd "Milestone 4"
    python setup_milestone4.py
"""

import os

def w(path, content):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  created: {path}")


# ── requirements.txt ──────────────────────────────────────────────────────────
w("requirements.txt", """\
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0
accelerate>=0.27.0
sentencepiece>=0.1.99
protobuf>=3.20.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0
scikit-learn>=1.3.0
zenml>=0.56.0
mlflow>=2.11.0
codecarbon>=2.3.4
pyyaml>=6.0
pytest>=7.4.0
""")

# ── .gitignore ────────────────────────────────────────────────────────────────
w(".gitignore", """\
results/
mlruns/
__pycache__/
*.py[cod]
.env
.venv/
venv/
*.safetensors
*.bin
*.pt
.DS_Store
""")

# ── src/utils/config.py ───────────────────────────────────────────────────────
w("src/utils/config.py", '''\
"""
utils/config.py
===============
Loads the central params.yaml configuration file.
"""
import os
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "params.yaml"


def load_config(config_path=None):
    """Load and return the central params.yaml configuration."""
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_hf_token():
    """Retrieve the Hugging Face API token from the environment."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set.\\n"
            "  Windows:    set HF_TOKEN=your_token_here\\n"
            "  Linux/Mac:  export HF_TOKEN=your_token_here"
        )
    return token
''')

# ── src/pipeline/model_loader.py ─────────────────────────────────────────────
w("src/pipeline/model_loader.py", '''\
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
''')

# ── src/pipeline/inference.py ────────────────────────────────────────────────
w("src/pipeline/inference.py", '''\
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
            f"{t}\\n\\n{d}" if (isinstance(d, str) and d.strip()) else t
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
''')

# ── src/pipeline/zenml_pipeline.py ───────────────────────────────────────────
w("src/pipeline/zenml_pipeline.py", '''\
"""
pipeline/zenml_pipeline.py
==========================
Milestone 4 ZenML pipeline: load_model_step -> evaluate_step -> report_step
"""
from zenml import pipeline
from zenml.logger import get_logger
from src.pipeline.zenml_steps import load_model_step, evaluate_step, report_step

logger = get_logger(__name__)


@pipeline(name="milestone4_training_pipeline", enable_cache=False)
def training_pipeline():
    """
    Milestone 4 full training and evaluation pipeline.
    Steps: load_model -> evaluate (MLflow + CodeCarbon) -> report
    """
    model_info = load_model_step()
    metrics_df = evaluate_step(model_info=model_info)
    summary    = report_step(metrics_df=metrics_df)


if __name__ == "__main__":
    training_pipeline()
''')

# ── src/pipeline/__init__.py (overwrite with proper content) ─────────────────
w("src/pipeline/__init__.py", '''\
from .zenml_pipeline import training_pipeline
__all__ = ["training_pipeline"]
''')

# ── src/evaluation/__init__.py ───────────────────────────────────────────────
w("src/evaluation/__init__.py", '''\
from .metrics import compute_metrics, aggregate_metrics, EvalMetrics
__all__ = ["compute_metrics", "aggregate_metrics", "EvalMetrics"]
''')

# ── src/tracking/__init__.py ─────────────────────────────────────────────────
w("src/tracking/__init__.py", '''\
from .mlflow_tracker import MLflowTracker
from .carbon_tracker import CarbonTracker
__all__ = ["MLflowTracker", "CarbonTracker"]
''')

# ── src/utils/__init__.py ────────────────────────────────────────────────────
w("src/utils/__init__.py", '''\
from .config import load_config, get_hf_token
__all__ = ["load_config", "get_hf_token"]
''')

# ── tests/test_metrics.py ────────────────────────────────────────────────────
w("tests/test_metrics.py", '''\
"""Unit tests for src/evaluation/metrics.py"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.metrics import compute_metrics, aggregate_metrics, EvalMetrics


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
        m = compute_metrics(y, y, project="test")
        assert m.mae == pytest.approx(0.0)
        assert m.rmse == pytest.approx(0.0)
        assert m.accuracy_at_1 == pytest.approx(1.0)

    def test_known_mae(self):
        y_true = np.array([2.0, 4.0, 6.0])
        y_pred = np.array([1.0, 3.0, 5.0])
        m = compute_metrics(y_true, y_pred, project="test")
        assert m.mae == pytest.approx(1.0)
        assert m.accuracy_at_1 == pytest.approx(1.0)

    def test_accuracy_at_1_half(self):
        y_true = np.array([1.0, 1.0])
        y_pred = np.array([2.0, 5.0])
        m = compute_metrics(y_true, y_pred, project="test")
        assert m.accuracy_at_1 == pytest.approx(0.5)

    def test_project_name_stored(self):
        y = np.array([3.0, 5.0])
        m = compute_metrics(y, y, project="moodle")
        assert m.project == "moodle"

    def test_test_size_correct(self):
        y = np.ones(42)
        m = compute_metrics(y, y, project="x")
        assert m.test_size == 42

    def test_to_dict_keys(self):
        y = np.array([1.0, 2.0])
        m = compute_metrics(y, y, project="springxd")
        assert set(m.to_dict().keys()) == {"project", "test_size", "mae", "rmse", "accuracy_at_1"}


class TestAggregateMetrics:
    def _make(self, maes, rmses, accs):
        return [EvalMetrics(f"p{i}", 100, m, r, a)
                for i, (m, r, a) in enumerate(zip(maes, rmses, accs))]

    def test_mean_mae(self):
        agg = aggregate_metrics(self._make([1.0, 3.0], [1.0, 3.0], [0.8, 0.6]))
        assert agg["mean_mae"] == pytest.approx(2.0)

    def test_num_projects(self):
        agg = aggregate_metrics(self._make([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.9, 0.8, 0.7]))
        assert agg["num_projects"] == 3

    def test_std_mae_zero(self):
        agg = aggregate_metrics(self._make([2.0, 2.0], [2.0, 2.0], [0.5, 0.5]))
        assert agg["std_mae"] == pytest.approx(0.0)

    def test_mean_accuracy(self):
        agg = aggregate_metrics(self._make([1.0, 1.0], [1.0, 1.0], [0.4, 0.6]))
        assert agg["mean_accuracy_at_1"] == pytest.approx(0.5)
''')

# ── tests/test_config.py ─────────────────────────────────────────────────────
w("tests/test_config.py", '''\
"""Unit tests for src/utils/config.py"""
import os, sys, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import load_config


class TestLoadConfig:
    def test_loads_without_error(self):
        assert isinstance(load_config(), dict)

    def test_has_required_sections(self):
        cfg = load_config()
        for s in ["model", "inference", "data", "mlflow", "zenml", "codecarbon"]:
            assert s in cfg

    def test_model_section(self):
        cfg = load_config()
        assert "hf_author" in cfg["model"]
        assert cfg["model"]["num_labels"] == 1

    def test_inference_section(self):
        cfg = load_config()
        assert cfg["inference"]["max_len"] > 0

    def test_mlflow_section(self):
        cfg = load_config()
        assert "experiment_name" in cfg["mlflow"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/params.yaml")
''')

print("\nAll files created successfully!")
print("\nNext steps:")
print("  1. pip install -r requirements.txt")
print("  2. python -m pytest tests/ -v")
print("  3. set HF_TOKEN=your_token_here")
print("  4. python run_pipeline.py")
