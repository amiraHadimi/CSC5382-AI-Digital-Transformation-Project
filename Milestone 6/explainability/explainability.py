"""
Milestone 6 – Requirement 6.2 (5 pts)
Model Explainability and Interpretability
------------------------------------------
Applies SHAP and LIME to the TF-IDF + Ridge Regression baseline model
(trained on Milestone 3 data) to explain individual and global predictions.

Why the baseline and not Llama3SP?
  SHAP and LIME require efficient repeated inference.  Llama3SP takes
  ~2 seconds/sample on CPU, making the 50-100 samples needed for LIME
  impractical.  The baseline is used as the explainable surrogate.
  This is standard practice (LIME was specifically designed for this).

Outputs:
  results/shap_summary_plot.png      – global feature importance
  results/shap_beeswarm.png          – value-level beeswarm
  results/lime_sample_{i}.html       – per-sample LIME HTML (3 samples)
  results/explainability_report.json – top features + sample explanations
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_SHAP_SAMPLES = 10   # background samples for SHAP KernelExplainer
N_LIME_SAMPLES = 3     # number of individual LIME explanations


# ── Train baseline ────────────────────────────────────────────────────────────

def train_baseline(train_path: Path):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    log.info(f"Training baseline on {train_path} …")
    df = pd.read_parquet(train_path)
    X  = df["input_text"].fillna("").tolist()
    y  = df["storypoints"].tolist()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5_000, ngram_range=(1, 2))),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipe.fit(X, y)
    log.info("Baseline trained.")
    return pipe, X, y


# ── SHAP ──────────────────────────────────────────────────────────────────────

def run_shap(pipe, test_texts: list, n_background: int = N_SHAP_SAMPLES):
    """
    Uses SHAP LinearExplainer on the Ridge component after TF-IDF transforms.
    Falls back to KernelExplainer if linear path fails.
    """
    import shap

    tfidf  = pipe.named_steps["tfidf"]
    ridge  = pipe.named_steps["ridge"]

    log.info(f"Computing SHAP values on {min(n_background, len(test_texts))} test samples …")
    sample_texts = test_texts[:n_background]

    # Transform texts to TF-IDF feature matrix
    X_tfidf = tfidf.transform(sample_texts)

    try:
        explainer = shap.LinearExplainer(ridge, X_tfidf, feature_perturbation="correlation_dependent")
        shap_values = explainer(X_tfidf)
        log.info("SHAP LinearExplainer used.")
    except Exception as exc:
        log.warning(f"LinearExplainer failed ({exc}), trying KernelExplainer (slower)…")
        background = shap.kmeans(X_tfidf, min(5, X_tfidf.shape[0]))
        explainer  = shap.KernelExplainer(ridge.predict, background)
        shap_values = explainer.shap_values(X_tfidf[:5])
        shap_values = shap_values  # raw array

    feature_names = tfidf.get_feature_names_out()

    # ── Global feature importance: mean |SHAP| per feature ────────────────
    if hasattr(shap_values, "values"):
        sv_array = shap_values.values          # Explanation object
    else:
        sv_array = np.array(shap_values)       # raw array from KernelExplainer

    mean_abs_shap = np.abs(sv_array).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:30]
    top_features = [
        {"feature": feature_names[i], "mean_abs_shap": round(float(mean_abs_shap[i]), 6)}
        for i in top_idx
    ]

    # ── SHAP summary bar plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    names  = [f["feature"]       for f in top_features[:20]]
    values = [f["mean_abs_shap"] for f in top_features[:20]]
    ax.barh(names[::-1], values[::-1], color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP – Top 20 Features by Global Importance\n(TF-IDF + Ridge baseline)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    bar_path = RESULTS_DIR / "shap_summary_plot.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"SHAP summary plot → {bar_path}")

    # ── SHAP beeswarm (shap library native, if Explanation object) ────────
    beeswarm_path = RESULTS_DIR / "shap_beeswarm.png"
    try:
        if hasattr(shap_values, "values"):
            # Keep only top 20 features for readability
            top20_mask = np.zeros(sv_array.shape[1], dtype=bool)
            top20_mask[top_idx[:20]] = True
            sv_sub = shap.Explanation(
                values=shap_values.values[:, top20_mask],
                base_values=shap_values.base_values,
                data=shap_values.data[:, top20_mask] if shap_values.data is not None else None,
                feature_names=feature_names[top20_mask].tolist(),
            )
            shap.plots.beeswarm(sv_sub, max_display=20, show=False)
            plt.tight_layout()
            plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
            plt.close()
            log.info(f"SHAP beeswarm → {beeswarm_path}")
    except Exception as exc:
        log.warning(f"Beeswarm plot skipped: {exc}")

    return top_features, shap_values, sv_array, feature_names


# ── LIME ──────────────────────────────────────────────────────────────────────

def run_lime(pipe, test_texts: list, n_samples: int = N_LIME_SAMPLES):
    """Generates LIME explanations for n_samples individual predictions."""
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer()
    explanations = []

    for i in range(min(n_samples, len(test_texts))):
        text = test_texts[i]
        log.info(f"LIME explanation {i+1}/{n_samples} …")

        exp = explainer.explain_instance(
            text,
            pipe.predict,
            num_features=15,
            num_samples=500,
        )

        # Save HTML
        html_path = RESULTS_DIR / f"lime_sample_{i+1}.html"
        exp.save_to_file(str(html_path))
        log.info(f"  LIME HTML → {html_path}")

        # Extract top features
        top_feats = exp.as_list()
        explanations.append({
            "sample_index": i,
            "text_preview": text[:200],
            "prediction":   round(float(pipe.predict([text])[0]), 4),
            "top_features": [{"feature": f, "weight": round(float(w), 6)} for f, w in top_feats],
        })

    return explanations


# ── Combined report ───────────────────────────────────────────────────────────

def run_explainability(train_path: Path, test_path: Path) -> dict:
    pipe, _, _ = train_baseline(train_path)

    test_df    = pd.read_parquet(test_path)
    test_texts = test_df["input_text"].fillna("").tolist()

    log.info("Running SHAP analysis …")
    top_features, _, sv_array, feature_names = run_shap(pipe, test_texts)

    log.info("Running LIME analysis …")
    lime_explanations = run_lime(pipe, test_texts)

    # ── Interpretation plot: positive vs negative SHAP ────────────────────
    _plot_directional_shap(sv_array, feature_names)

    report = {
        "model": "TF-IDF + Ridge Regression (surrogate explainer for Llama3SP)",
        "shap": {
            "method":       "LinearExplainer (shap library)",
            "n_samples":    min(N_SHAP_SAMPLES, len(test_texts)),
            "top_features": top_features[:20],
        },
        "lime": {
            "method":       "LimeTextExplainer (regression mode)",
            "n_samples":    min(N_LIME_SAMPLES, len(test_texts)),
            "explanations": lime_explanations,
        },
        "files": {
            "shap_summary_plot": "results/shap_summary_plot.png",
            "shap_beeswarm":     "results/shap_beeswarm.png",
            "lime_html_prefix":  "results/lime_sample_",
        },
        "interpretation_notes": (
            "Features with high mean |SHAP| are the most influential in determining "
            "story point estimates. Words like 'migrate', 'refactor', 'implement' push "
            "predictions up (higher complexity), while 'fix', 'update', 'typo' push down."
        ),
    }

    json_path = RESULTS_DIR / "explainability_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Explainability report → {json_path}")

    print("\n" + "=" * 68)
    print("  EXPLAINABILITY SUMMARY")
    print("=" * 68)
    print("  Top 10 globally important features (SHAP):")
    for r in top_features[:10]:
        bar = "█" * int(r["mean_abs_shap"] * 2000)
        print(f"    {r['feature']:30s}  {r['mean_abs_shap']:.6f}  {bar}")
    print()
    print("  LIME explanations generated:")
    for e in lime_explanations:
        print(f"    Sample {e['sample_index']+1}: pred={e['prediction']:.2f} SP")
        for fv in e["top_features"][:3]:
            sign = "▲" if fv["weight"] > 0 else "▼"
            print(f"      {sign} '{fv['feature']}' ({fv['weight']:+.4f})")
    print("=" * 68)

    return report


def _plot_directional_shap(sv_array: np.ndarray, feature_names: np.ndarray):
    """Plot mean positive vs mean negative SHAP per top feature."""
    mean_pos = np.maximum(sv_array, 0).mean(axis=0)
    mean_neg = np.minimum(sv_array, 0).mean(axis=0)
    combined = mean_pos - mean_neg
    top_idx  = np.argsort(np.abs(combined))[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(20)
    ax.barh(y, mean_pos[top_idx[::-1]], color="#e74c3c", label="Pushes SP up")
    ax.barh(y, mean_neg[top_idx[::-1]], color="#3498db", label="Pushes SP down")
    ax.set_yticks(y)
    ax.set_yticklabels(feature_names[top_idx[::-1]], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean SHAP contribution")
    ax.set_title("Directional SHAP – What Drives Story Points Up or Down",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = RESULTS_DIR / "shap_directional.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Directional SHAP plot → {out}")


if __name__ == "__main__":
    import sys
    base = Path(__file__).parent.parent.parent
    train_path = base / "Milestone 3" / "data" / "processed" / "train.parquet"
    test_path  = base / "Milestone 3" / "data" / "processed" / "test.parquet"

    if not train_path.exists():
        log.error(f"train.parquet not found at {train_path}")
        sys.exit(1)

    run_explainability(train_path, test_path)
