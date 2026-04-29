"""
Milestone 6 – Requirement 6.3 (3 pts)
Data Distribution Drift Monitoring with Evidently
--------------------------------------------------
Compares feature distributions between:
  - Reference dataset: training split (Milestone 3 train.parquet)
  - Current dataset:   test split  (Milestone 3 test.parquet)

Detects drift in:
  - storypoints      (regression target)
  - text_length      (input complexity proxy)
  - word_count
  - has_description
  - log_storypoints  (engineered feature)

Uses Evidently DataDriftPreset to generate a visual HTML report
and a structured JSON summary.

Outputs:
  results/evidently_drift_report.html  – visual report (open in browser)
  results/evidently_drift_summary.json – drift metrics per feature
  results/evidently_drift_plot.png     – static plot of distributions
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Features to monitor
NUMERIC_FEATURES = ["storypoints", "text_length", "word_count",
                    "has_description", "log_storypoints", "title_word_count"]


# ── Evidently report ──────────────────────────────────────────────────────────

def run_evidently_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """
    Generate an Evidently drift report.
    Falls back to manual KS-test drift detection if Evidently is not installed.
    """
    available_cols = [c for c in NUMERIC_FEATURES if c in reference.columns and c in current.columns]

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

        log.info("Running Evidently drift report …")
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        report.run(
            reference_data=reference[available_cols],
            current_data=current[available_cols],
        )

        html_path = RESULTS_DIR / "evidently_drift_report.html"
        report.save_html(str(html_path))
        log.info(f"Evidently HTML report → {html_path}")

        result_dict = report.as_dict()
        return _extract_evidently_summary(result_dict, available_cols)

    except ImportError:
        log.warning("evidently not installed – using manual KS-test drift detection.")
        return _manual_drift(reference, current, available_cols)
    except Exception as exc:
        log.warning(f"Evidently failed ({exc}) – using manual KS-test.")
        return _manual_drift(reference, current, available_cols)


def _extract_evidently_summary(result_dict: dict, cols: list) -> dict:
    """Extract per-column drift metrics from Evidently result dict."""
    summary = {"method": "evidently", "columns": {}}
    try:
        for metric in result_dict.get("metrics", []):
            if metric.get("metric") == "DataDriftMetric":
                r = metric.get("result", {})
                col = r.get("column_name")
                if col:
                    summary["columns"][col] = {
                        "drift_detected": r.get("drift_detected", False),
                        "stattest":       r.get("stattest_name", "unknown"),
                        "p_value":        r.get("p_value"),
                        "drift_score":    r.get("drift_score"),
                    }
        # overall
        for metric in result_dict.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                r = metric.get("result", {})
                summary["dataset_drift_detected"] = r.get("dataset_drift", False)
                summary["n_drifted_columns"]       = r.get("n_drifted_features", 0)
    except Exception:
        pass
    return summary


def _manual_drift(reference: pd.DataFrame, current: pd.DataFrame, cols: list) -> dict:
    """Manual drift detection using Kolmogorov-Smirnov and Wasserstein tests."""
    from scipy.stats import ks_2samp, wasserstein_distance

    log.info("Manual drift detection via KS test …")
    result = {"method": "manual_ks_test", "columns": {}, "dataset_drift_detected": False}
    n_drifted = 0

    for col in cols:
        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        ks_stat, p_val = ks_2samp(ref_vals, cur_vals)
        wd = float(wasserstein_distance(ref_vals, cur_vals))
        drifted = p_val < 0.05

        if drifted:
            n_drifted += 1

        result["columns"][col] = {
            "drift_detected":    drifted,
            "stattest":          "kolmogorov-smirnov",
            "p_value":           round(float(p_val), 6),
            "ks_statistic":      round(float(ks_stat), 6),
            "wasserstein_dist":  round(wd, 6),
            "ref_mean":          round(float(ref_vals.mean()), 4),
            "cur_mean":          round(float(cur_vals.mean()), 4),
            "mean_shift":        round(float(cur_vals.mean() - ref_vals.mean()), 4),
            "ref_std":           round(float(ref_vals.std()), 4),
            "cur_std":           round(float(cur_vals.std()), 4),
        }

    result["n_drifted_columns"]      = n_drifted
    result["n_monitored_columns"]    = len(cols)
    result["dataset_drift_detected"] = n_drifted > len(cols) // 2

    # Generate manual HTML
    _generate_manual_html(result, reference, current, cols)

    return result


def _generate_manual_html(result: dict, reference: pd.DataFrame,
                           current: pd.DataFrame, cols: list):
    rows = ""
    for col, stats in result["columns"].items():
        drift_style = 'style="background:#ffe0e0"' if stats["drift_detected"] else 'style="background:#d4edda"'
        icon = "🔴 YES" if stats["drift_detected"] else "✅ No"
        rows += (
            f'<tr {drift_style}>'
            f'<td>{col}</td>'
            f'<td>{icon}</td>'
            f'<td>{stats.get("p_value","N/A")}</td>'
            f'<td>{stats.get("ks_statistic","N/A")}</td>'
            f'<td>{stats.get("wasserstein_dist","N/A")}</td>'
            f'<td>{stats.get("ref_mean","N/A")}</td>'
            f'<td>{stats.get("cur_mean","N/A")}</td>'
            f'<td>{stats.get("mean_shift","N/A")}</td>'
            f'</tr>'
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Drift Report</title>
<style>
  body {{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;color:#333}}
  h1 {{color:#2c3e50;border-bottom:3px solid #2c3e50;padding-bottom:8px}}
  table {{border-collapse:collapse;width:100%;font-size:13px}}
  th,td {{border:1px solid #ccc;padding:6px 10px}}
  th {{background:#2c3e50;color:#fff}}
  .overall {{padding:12px;border-radius:6px;margin:16px 0;font-size:15px;font-weight:bold}}
  .drift {{'background:#ffe0e0;color:#721c24'}}
  .nodrift {{'background:#d4edda;color:#155724'}}
  img {{max-width:100%;margin:20px 0}}
</style>
</head>
<body>
<h1>Milestone 6 – Data Drift Report (Evidently / KS-test)</h1>
<p><strong>Reference:</strong> Training split (M3 train.parquet) &nbsp;|&nbsp;
   <strong>Current:</strong> Test split (M3 test.parquet)</p>
<div class="overall {'drift' if result['dataset_drift_detected'] else 'nodrift'}">
  Dataset drift detected: {"🔴 YES" if result["dataset_drift_detected"] else "✅ No"} &nbsp;|&nbsp;
  Drifted columns: {result["n_drifted_columns"]} / {result["n_monitored_columns"]}
</div>
<table>
<tr><th>Feature</th><th>Drift Detected</th><th>p-value</th><th>KS Statistic</th>
    <th>Wasserstein Dist</th><th>Ref Mean</th><th>Current Mean</th><th>Mean Shift</th></tr>
{rows}
</table>
<h2>Distribution Plots</h2>
<img src="evidently_drift_plot.png" alt="Drift distribution plots">
</body>
</html>"""
    html_path = RESULTS_DIR / "evidently_drift_report.html"
    html_path.write_text(html, encoding="utf-8")
    log.info(f"Manual drift HTML → {html_path}")


# ── Distribution plots ────────────────────────────────────────────────────────

def _plot_drift(reference: pd.DataFrame, current: pd.DataFrame,
                drift_result: dict, cols: list):
    n = min(len(cols), 6)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(cols[:n]):
        ax = axes[i]
        ref_vals = reference[col].dropna()
        cur_vals = current[col].dropna()
        ax.hist(ref_vals, bins=30, alpha=0.6, color="#3498db", label="Train (reference)", density=True)
        ax.hist(cur_vals, bins=30, alpha=0.6, color="#e74c3c", label="Test (current)",    density=True)

        # Drift annotation
        col_stats = drift_result.get("columns", {}).get(col, {})
        drifted   = col_stats.get("drift_detected", False)
        p_val     = col_stats.get("p_value", "N/A")
        title_color = "#c0392b" if drifted else "#27ae60"
        drift_label = "DRIFT ⚠" if drifted else "No drift ✅"
        ax.set_title(f"{col}\n{drift_label}  (p={p_val})", fontsize=9,
                     fontweight="bold", color=title_color)
        ax.legend(fontsize=7)
        ax.set_ylabel("Density")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distribution: Train vs Test (Drift Detection)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "evidently_drift_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Drift plot → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_drift_monitoring(train_path: Path, test_path: Path) -> dict:
    log.info(f"Loading reference (train): {train_path}")
    reference = pd.read_parquet(train_path)
    log.info(f"Loading current (test):    {test_path}")
    current   = pd.read_parquet(test_path)

    # Add text_length / word_count if not already there
    for df in (reference, current):
        if "text_length" not in df.columns and "input_text" in df.columns:
            df["text_length"] = df["input_text"].fillna("").str.len()
        if "word_count" not in df.columns and "input_text" in df.columns:
            df["word_count"] = df["input_text"].fillna("").str.split().str.len()

    available_cols = [c for c in NUMERIC_FEATURES
                      if c in reference.columns and c in current.columns]

    log.info(f"Monitoring {len(available_cols)} features: {available_cols}")

    drift_result = run_evidently_report(reference, current)
    _plot_drift(reference, current, drift_result, available_cols)

    json_path = RESULTS_DIR / "evidently_drift_summary.json"
    with open(json_path, "w") as f:
        json.dump(drift_result, f, indent=2, default=str)
    log.info(f"Drift summary → {json_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  DATA DRIFT MONITORING SUMMARY")
    print("=" * 68)
    print(f"  Method          : {drift_result.get('method', 'unknown')}")
    print(f"  Dataset drift   : {'🔴 YES' if drift_result.get('dataset_drift_detected') else '✅ No'}")
    print(f"  Drifted columns : {drift_result.get('n_drifted_columns', 0)} / "
          f"{drift_result.get('n_monitored_columns', len(available_cols))}")
    print()
    for col, stats in drift_result.get("columns", {}).items():
        icon = "🔴" if stats["drift_detected"] else "✅"
        pv   = stats.get("p_value", "N/A")
        shift = stats.get("mean_shift", "N/A")
        print(f"  {icon} {col:25s}  p={pv}  mean_shift={shift}")
    print("=" * 68)

    return drift_result


if __name__ == "__main__":
    import sys
    base = Path(__file__).parent.parent.parent
    train_path = base / "Milestone 3" / "data" / "processed" / "train.parquet"
    test_path  = base / "Milestone 3" / "data" / "processed" / "test.parquet"

    if not train_path.exists():
        log.error(f"train.parquet not found at {train_path}")
        sys.exit(1)

    run_drift_monitoring(train_path, test_path)
