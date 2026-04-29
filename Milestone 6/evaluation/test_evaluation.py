"""
Milestone 6 – Requirement 6.1
Test Set Evaluation on Unseen Data

Evaluates:
- Llama3SP real model inference
- TF-IDF + Ridge baseline

To make CPU inference feasible, this script evaluates a balanced subset
of MAX_SAMPLES across all projects.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_SAMPLES = 500
RANDOM_STATE = 42


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def accuracy_at_1(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred)) <= 1))


def balanced_sample_by_project(df: pd.DataFrame, max_samples: int = MAX_SAMPLES) -> pd.DataFrame:
    """
    Create a balanced subset across all projects while preserving columns.
    """
    projects = df["project"].dropna().unique()
    samples_per_project = max(1, max_samples // len(projects))

    sampled_parts = []

    for project in projects:
        group = df[df["project"] == project]
        n = min(len(group), samples_per_project)
        sampled_parts.append(group.sample(n=n, random_state=RANDOM_STATE))

    sampled = pd.concat(sampled_parts, ignore_index=True)

    return sampled.reset_index(drop=True)


class Llama3SPModel:
    MODEL_ID = "DEVCamiloSepulveda/2-LLAMA3SP-talendesb"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._load()

    def _load(self):
        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            raise RuntimeError("HF_TOKEN is not set. Refusing to use fallback.")

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from peft import PeftConfig, PeftModel

            log.info("Loading Llama3SP model …")

            config = PeftConfig.from_pretrained(self.MODEL_ID, token=hf_token)

            self.tokenizer = AutoTokenizer.from_pretrained(
                config.base_model_name_or_path,
                token=hf_token,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            base = AutoModelForSequenceClassification.from_pretrained(
                config.base_model_name_or_path,
                num_labels=1,
                token=hf_token,
            )

            self.model = PeftModel.from_pretrained(
                base,
                self.MODEL_ID,
                token=hf_token,
            )

            self.model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

            log.info(f"Model loaded on {self.device}.")

        except Exception as exc:
            raise RuntimeError(f"Real Llama3SP model failed to load: {exc}")

    def predict(self, text: str) -> float:
        import torch

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**enc)

        return float(out.logits.squeeze().item())


class BaselineModel:
    def __init__(self, train_path: Optional[Path] = None):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge

        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))),
            ("lr", Ridge(alpha=1.0)),
        ])

        self._trained = False

        if train_path and train_path.exists():
            self._train(train_path)

    def _train(self, path: Path):
        log.info(f"Training baseline on {path} …")

        df = pd.read_parquet(path)
        X = df["input_text"].fillna("").tolist()
        y = df["storypoints"].tolist()

        self.pipe.fit(X, y)
        self._trained = True

        log.info("Baseline trained.")

    def predict_batch(self, texts):
        if not self._trained:
            return [float(len(t.split()) // 8 + 1) for t in texts]

        return self.pipe.predict(texts).tolist()


def run_evaluation(
    test_path: Path,
    train_path: Optional[Path] = None,
    use_llama: bool = True,
) -> dict:
    log.info(f"Loading test data from {test_path} …")
    test_df = pd.read_parquet(test_path)

    assert "storypoints" in test_df.columns, "Missing 'storypoints' column."
    assert "input_text" in test_df.columns, "Missing 'input_text' column."
    assert "project" in test_df.columns, "Missing 'project' column."

    test_df = balanced_sample_by_project(test_df, MAX_SAMPLES)

    y_true = test_df["storypoints"].tolist()
    texts = test_df["input_text"].fillna("").tolist()

    log.info(
        f"Using balanced subset: {len(test_df)} samples "
        f"across {test_df['project'].nunique()} projects."
    )

    baseline = BaselineModel(train_path)
    log.info("Running baseline predictions …")
    y_pred_baseline = baseline.predict_batch(texts)

    if not use_llama:
        raise RuntimeError("use_llama=False is disabled for final real evaluation.")

    llama = Llama3SPModel()
    log.info("Running Llama3SP predictions …")

    y_pred_llama = []
    start_time = None

    for i, text in enumerate(texts):
        if i == 0:
            import time
            start_time = time.time()

        y_pred_llama.append(llama.predict(text))

        if (i + 1) % 100 == 0:
            log.info(f"  {i + 1}/{len(texts)} done …")

    results_df = test_df[["issuekey", "project", "storypoints", "input_text"]].copy()
    results_df["y_pred_llama"] = y_pred_llama
    results_df["y_pred_baseline"] = y_pred_baseline
    results_df["abs_err_llama"] = (
        results_df["storypoints"] - results_df["y_pred_llama"]
    ).abs()
    results_df["abs_err_baseline"] = (
        results_df["storypoints"] - results_df["y_pred_baseline"]
    ).abs()

    results_csv = RESULTS_DIR / "test_evaluation_results.csv"
    results_df.to_csv(results_csv, index=False)
    log.info(f"Saved per-row results → {results_csv}")

    global_metrics = {
        "llama3sp": {
            "mae": mae(y_true, y_pred_llama),
            "rmse": rmse(y_true, y_pred_llama),
            "acc_at_1": accuracy_at_1(y_true, y_pred_llama),
        },
        "baseline": {
            "mae": mae(y_true, y_pred_baseline),
            "rmse": rmse(y_true, y_pred_baseline),
            "acc_at_1": accuracy_at_1(y_true, y_pred_baseline),
        },
    }

    per_project = {}
    for proj, grp in results_df.groupby("project"):
        per_project[proj] = {
            "n": len(grp),
            "llama3sp": {
                "mae": mae(grp["storypoints"], grp["y_pred_llama"]),
                "rmse": rmse(grp["storypoints"], grp["y_pred_llama"]),
                "acc_at_1": accuracy_at_1(grp["storypoints"], grp["y_pred_llama"]),
            },
            "baseline": {
                "mae": mae(grp["storypoints"], grp["y_pred_baseline"]),
                "rmse": rmse(grp["storypoints"], grp["y_pred_baseline"]),
                "acc_at_1": accuracy_at_1(grp["storypoints"], grp["y_pred_baseline"]),
            },
        }

    summary = {
        "n_test": len(test_df),
        "n_projects": test_df["project"].nunique(),
        "max_samples": MAX_SAMPLES,
        "sampling_strategy": "balanced_sample_by_project",
        "global_metrics": global_metrics,
        "per_project_metrics": per_project,
    }

    summary_json = RESULTS_DIR / "test_evaluation_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Saved summary → {summary_json}")

    print("\n" + "=" * 68)
    print("  TEST SET EVALUATION RESULTS")
    print("=" * 68)
    print(f"  Total evaluated issues : {summary['n_test']}")
    print(f"  Projects               : {summary['n_projects']}")
    print(f"  Sampling strategy      : {summary['sampling_strategy']}")
    print()

    for model_name in ("llama3sp", "baseline"):
        m = global_metrics[model_name]
        print(f"  [{model_name.upper()}]")
        print(f"    MAE      : {m['mae']:.4f}")
        print(f"    RMSE     : {m['rmse']:.4f}")
        print(f"    Acc@±1   : {m['acc_at_1']:.4f}")
        print()

    print("=" * 68)

    return summary


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent

    test_path = base_dir / "Milestone 3" / "data" / "processed" / "test.parquet"
    train_path = base_dir / "Milestone 3" / "data" / "processed" / "train.parquet"

    if not test_path.exists():
        log.error(f"test.parquet not found at {test_path}.")
        sys.exit(1)

    run_evaluation(
        test_path=test_path,
        train_path=train_path,
        use_llama=True,
    )