"""
Milestone 6 – Requirement 6.3 (3 pts)
Model Performance Monitoring with WhyLogs
------------------------------------------
Profiles the model's input features and prediction distributions using
whylogs. Simulates what a production monitoring system would do:
  - Log input statistics (text_length, word_count, etc.)
  - Log prediction distribution (story_points)
  - Generate a WhyLogs DatasetProfileView
  - Produce summary statistics and alerts for out-of-range values

Outputs:
  results/whylogs_profile.bin         – serialized whylogs profile
  results/whylogs_monitoring_report.json
  results/whylogs_monitoring_plot.png
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

# Thresholds for alerting
ALERT_THRESHOLDS = {
    "prediction_mean_max": 8.0,    # mean story point > 8 is unusual
    "prediction_mean_min": 1.5,    # mean story point < 1.5 is suspicious
    "null_rate_max": 0.05,         # >5% nulls triggers alert
    "text_length_min_mean": 20,    # very short texts on average = suspicious
}


# ── WhyLogs profiling ─────────────────────────────────────────────────────────

def profile_with_whylogs(df: pd.DataFrame, dataset_name: str = "test_predictions"):
    """
    Profiles the dataframe using whylogs and returns the profile view.
    Handles both whylogs v1 and gracefully degrades if not installed.
    """
    try:
        import whylogs as why
        from whylogs.core.datatypes import DataType

        log.info("Profiling with whylogs …")
        result = why.log(df)
        profile = result.profile()

        # Save binary profile
        bin_path = RESULTS_DIR / "whylogs_profile.bin"
        profile.write(path=str(bin_path))
        log.info(f"WhyLogs profile saved → {bin_path}")

        # Get summary view
        profile_view = profile.view()
        summary = profile_view.to_pandas()

        return profile, summary

    except ImportError:
        log.warning("whylogs not installed – computing manual statistics.")
        return None, _manual_stats(df)
    except Exception as exc:
        log.warning(f"whylogs profiling failed ({exc}) – computing manual statistics.")
        return None, _manual_stats(df)


def _manual_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: compute pandas describe() in a format resembling whylogs output."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        rows.append({
            "column":  col,
            "count":   len(s),
            "null_pct": round(df[col].isna().mean(), 4),
            "mean":    round(float(s.mean()), 4),
            "std":     round(float(s.std()), 4),
            "min":     round(float(s.min()), 4),
            "p25":     round(float(s.quantile(0.25)), 4),
            "p50":     round(float(s.quantile(0.50)), 4),
            "p75":     round(float(s.quantile(0.75)), 4),
            "max":     round(float(s.max()), 4),
        })
    return pd.DataFrame(rows).set_index("column")


# ── Alert detection ───────────────────────────────────────────────────────────

def detect_alerts(stats_df: pd.DataFrame, pred_col: str = "y_pred_llama") -> list[dict]:
    alerts = []

    def _check(col, stat, threshold, direction, msg):
        try:
            val = float(stats_df.loc[col, stat]) if col in stats_df.index else None
            if val is None:
                return
            triggered = (direction == ">" and val > threshold) or \
                        (direction == "<" and val < threshold)
            if triggered:
                alerts.append({"level": "WARNING", "column": col, "stat": stat,
                                "value": val, "threshold": threshold,
                                "direction": direction, "message": msg})
        except Exception:
            pass

    _check(pred_col, "mean", ALERT_THRESHOLDS["prediction_mean_max"], ">",
           "Mean predicted SP is unusually high — possible distribution shift")
    _check(pred_col, "mean", ALERT_THRESHOLDS["prediction_mean_min"], "<",
           "Mean predicted SP is unusually low — possible model degradation")
    _check("text_length", "mean", ALERT_THRESHOLDS["text_length_min_mean"], "<",
           "Average input text_length very short — possible data quality issue")

    # Null rate check from raw df
    return alerts


# ── Temporal simulation ───────────────────────────────────────────────────────

def simulate_temporal_monitoring(df: pd.DataFrame, n_windows: int = 5):
    """
    Simulates monitoring across time windows (batches) to show how metrics
    would drift if the model were deployed in production.
    Splits the test set into n_windows batches and computes rolling stats.
    """
    windows = []
    step = len(df) // n_windows
    for i in range(n_windows):
        batch = df.iloc[i * step: (i + 1) * step]
        windows.append({
            "window": i + 1,
            "n":      len(batch),
            "mean_pred_llama":    round(float(batch["y_pred_llama"].mean()), 4),
            "mean_pred_baseline": round(float(batch["y_pred_baseline"].mean()), 4),
            "mean_actual":        round(float(batch["storypoints"].mean()), 4),
            "mae_llama":          round(float((batch["storypoints"] - batch["y_pred_llama"]).abs().mean()), 4),
            "mae_baseline":       round(float((batch["storypoints"] - batch["y_pred_baseline"]).abs().mean()), 4),
        })
    return windows


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_monitoring(df: pd.DataFrame, windows: list, alerts: list):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Prediction distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df["y_pred_llama"],    bins=30, alpha=0.7, color="#3498db", label="Llama3SP")
    ax1.hist(df["y_pred_baseline"], bins=30, alpha=0.7, color="#e67e22", label="Baseline")
    ax1.axvline(df["storypoints"].mean(), color="red", ls="--", lw=1.5, label="Actual mean")
    ax1.set_title("Prediction Distribution", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Predicted story points")
    ax1.legend(fontsize=8)

    # 2. Text length distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df["text_length"].dropna(), bins=40, color="#2ecc71", alpha=0.8, edgecolor="white")
    ax2.set_title("Input Text Length Distribution", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Characters")
    ax2.set_ylabel("Count")

    # 3. Actual vs predicted scatter
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(df["storypoints"], df["y_pred_llama"], alpha=0.15, s=8, color="#3498db", label="Llama3SP")
    ax3.scatter(df["storypoints"], df["y_pred_baseline"], alpha=0.15, s=8, color="#e67e22", label="Baseline")
    lim = max(df["storypoints"].max(), df[["y_pred_llama","y_pred_baseline"]].max().max())
    ax3.plot([0, lim], [0, lim], "k--", lw=1, label="Perfect")
    ax3.set_xlabel("Actual SP")
    ax3.set_ylabel("Predicted SP")
    ax3.set_title("Actual vs Predicted", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=8)

    # 4. Rolling MAE over time windows
    ax4 = fig.add_subplot(gs[1, :2])
    w_nums    = [w["window"] for w in windows]
    mae_llama = [w["mae_llama"] for w in windows]
    mae_base  = [w["mae_baseline"] for w in windows]
    ax4.plot(w_nums, mae_llama, "o-", color="#3498db", label="Llama3SP", lw=2)
    ax4.plot(w_nums, mae_base,  "s-", color="#e67e22", label="Baseline", lw=2)
    ax4.set_xlabel("Time Window (batch)")
    ax4.set_ylabel("MAE")
    ax4.set_title("Rolling MAE Across Time Windows (simulated production)", fontsize=10, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.set_xticks(w_nums)

    # 5. Alerts panel
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    if alerts:
        alert_text = "\n\n".join(
            f"⚠ {a['column']}.{a['stat']}\n  {a['value']:.3f} {a['direction']} {a['threshold']}\n  {a['message']}"
            for a in alerts
        )
        ax5.text(0.05, 0.95, alert_text, transform=ax5.transAxes,
                 fontsize=9, verticalalignment="top", color="#c0392b",
                 bbox=dict(boxstyle="round", facecolor="#ffe0e0", alpha=0.8))
    else:
        ax5.text(0.5, 0.5, "✅ No alerts triggered",
                 transform=ax5.transAxes, ha="center", va="center",
                 fontsize=12, color="#27ae60",
                 bbox=dict(boxstyle="round", facecolor="#d4edda", alpha=0.8))
    ax5.set_title("Monitoring Alerts", fontsize=10, fontweight="bold")

    fig.suptitle("WhyLogs Model Performance Monitoring Dashboard", fontsize=13, fontweight="bold")
    out = RESULTS_DIR / "whylogs_monitoring_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Monitoring plot → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_monitoring(results_csv: Path) -> dict:
    if not results_csv.exists():
        raise FileNotFoundError(f"{results_csv} not found – run test_evaluation.py first.")

    df = pd.read_csv(results_csv)

    # Add text_length if not present
    if "text_length" not in df.columns:
        df["text_length"] = df["input_text"].fillna("").str.len()

    monitor_cols = ["storypoints", "y_pred_llama", "y_pred_baseline",
                    "text_length", "abs_err_llama", "abs_err_baseline"]
    monitor_df   = df[[c for c in monitor_cols if c in df.columns]].copy()

    # WhyLogs profiling
    _, stats_df = profile_with_whylogs(monitor_df, "test_predictions")

    # Alerts
    alerts = detect_alerts(stats_df)

    # Temporal simulation
    windows = simulate_temporal_monitoring(df)

    # Plots
    _plot_monitoring(df, windows, alerts)

    # Summary statistics
    stats_dict = {}
    for col in monitor_cols:
        if col in df.columns:
            s = df[col].dropna()
            stats_dict[col] = {
                "count":    int(len(s)),
                "null_pct": round(float(df[col].isna().mean()), 4),
                "mean":     round(float(s.mean()), 4),
                "std":      round(float(s.std()), 4),
                "min":      round(float(s.min()), 4),
                "p25":      round(float(s.quantile(0.25)), 4),
                "p50":      round(float(s.quantile(0.50)), 4),
                "p75":      round(float(s.quantile(0.75)), 4),
                "max":      round(float(s.max()), 4),
            }

    report = {
        "dataset":          "test_predictions",
        "n_samples":        len(df),
        "statistics":       stats_dict,
        "alerts":           alerts,
        "temporal_windows": windows,
        "thresholds_used":  ALERT_THRESHOLDS,
    }

    json_path = RESULTS_DIR / "whylogs_monitoring_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Monitoring report → {json_path}")

    print("\n" + "=" * 68)
    print("  WHYLOGS MONITORING SUMMARY")
    print("=" * 68)
    print(f"  Samples profiled : {report['n_samples']}")
    print(f"  Alerts triggered : {len(alerts)}")
    for a in alerts:
        print(f"    ⚠  {a['message']}")
    if not alerts:
        print("    ✅ All metrics within normal bounds.")
    print()
    print(f"  Mean predicted SP (Llama3SP) : {stats_dict.get('y_pred_llama', {}).get('mean', 'N/A')}")
    print(f"  Mean actual SP               : {stats_dict.get('storypoints', {}).get('mean', 'N/A')}")
    print("=" * 68)

    return report


if __name__ == "__main__":
    run_monitoring(RESULTS_DIR / "test_evaluation_results.csv")
