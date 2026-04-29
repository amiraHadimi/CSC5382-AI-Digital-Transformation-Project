"""
Milestone 6 – Requirement 6.2 (5 pts)
Bias Audit
----------
Audits the Llama3SP model for bias across:
  - Project groups (proxy for team/domain)
  - Story-point magnitude groups (small ≤3, medium 4-8, large >8)
  - Description availability (has_description vs missing)

Uses:
  - Aequitas toolkit for formal bias metrics
  - Manual disparity analysis as fallback
  - Generates HTML + JSON reports and plots

Outputs:
  results/bias_audit_report.json
  results/bias_audit_report.html
  results/bias_audit_plot.png
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


# ── Bias metric helpers ───────────────────────────────────────────────────────

def _bias_label(pred, actual, threshold=1.0):
    """Classify each prediction as: correct, over-estimate, or under-estimate."""
    diff = pred - actual
    if abs(diff) <= threshold:
        return "correct"
    return "over" if diff > 0 else "under"


def compute_group_metrics(df, group_col, pred_col="y_pred_llama", actual_col="storypoints"):
    """Compute MAE, bias direction, and disparity ratio per group."""
    rows = []
    for group, grp in df.groupby(group_col):
        errs   = (grp[actual_col] - grp[pred_col])
        abs_e  = errs.abs()
        rows.append({
            "group":        group,
            "group_col":    group_col,
            "n":            len(grp),
            "mae":          float(abs_e.mean()),
            "rmse":         float(np.sqrt((errs**2).mean())),
            "acc_at_1":     float((abs_e <= 1).mean()),
            "mean_error":   float(errs.mean()),   # positive = over-estimate
            "bias_direction": "over"  if errs.mean() >  0.5 else
                              "under" if errs.mean() < -0.5 else "neutral",
            "pct_over":     float((errs > 1).mean()),
            "pct_under":    float((errs < -1).mean()),
            "pct_correct":  float((abs_e <= 1).mean()),
        })
    result_df = pd.DataFrame(rows)

    # Disparity ratio: group MAE / min-group MAE
    min_mae = result_df["mae"].min()
    result_df["disparity_ratio"] = result_df["mae"] / min_mae

    return result_df


def _try_aequitas(df, group_col, pred_col, actual_col):
    """
    Attempt Aequitas formal audit.
    Falls back gracefully if not installed or incompatible.
    """
    try:
        from aequitas.group import Group
        from aequitas.bias import Bias
        from aequitas.fairness import Fairness

        aq_df = df[[group_col, actual_col, pred_col]].copy()
        aq_df = aq_df.rename(columns={actual_col: "label_value", pred_col: "score"})
        aq_df["label_value"] = (aq_df["label_value"] > 0).astype(int)
        aq_df["score"]       = (aq_df["score"] > 0).astype(int)

        g   = Group()
        xtab, _ = g.get_crosstabs(aq_df, attr_cols=[group_col])
        b   = Bias()
        bdf = b.get_disparity_predefined_groups(xtab, original_df=aq_df,
                                                 ref_groups_dict={group_col: xtab[group_col].iloc[0]})
        return bdf.to_dict(orient="records")
    except Exception as exc:
        log.warning(f"Aequitas audit skipped ({exc}); using manual disparity analysis instead.")
        return None


# ── HTML report generator ─────────────────────────────────────────────────────

def _html_report(project_bias, size_bias, desc_bias, aequitas_result):
    def table_html(df):
        cols = ["group", "n", "mae", "rmse", "acc_at_1", "mean_error",
                "bias_direction", "disparity_ratio"]
        cols = [c for c in cols if c in df.columns]
        rows = ""
        for _, r in df.iterrows():
            bg = ""
            if r.get("disparity_ratio", 1) >= 2.0:
                bg = ' style="background:#ffe0e0"'
            elif r.get("disparity_ratio", 1) >= 1.5:
                bg = ' style="background:#fff3cd"'
            cells = "".join(
                f'<td>{round(r[c], 4) if isinstance(r[c], float) else r[c]}</td>'
                for c in cols
            )
            rows += f"<tr{bg}>{cells}</tr>"
        header = "".join(f"<th>{c}</th>" for c in cols)
        return f"<table><tr>{header}</tr>{rows}</table>"

    aequitas_section = ""
    if aequitas_result:
        aequitas_section = f"""
        <h2>Aequitas Formal Audit</h2>
        <pre>{json.dumps(aequitas_result[:5], indent=2)}</pre>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Milestone 6 – Bias Audit Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; color: #333; }}
    h1   {{ color: #2c3e50; border-bottom: 3px solid #2c3e50; padding-bottom: 8px; }}
    h2   {{ color: #34495e; margin-top: 32px; }}
    table{{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th,td{{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
    th   {{ background: #2c3e50; color: #fff; }}
    .warn{{ background: #fff3cd; padding: 8px; border-radius: 4px; margin: 12px 0; }}
    .bad {{ background: #ffe0e0; padding: 8px; border-radius: 4px; margin: 12px 0; }}
    pre  {{ background: #f5f5f5; padding: 12px; overflow: auto; font-size: 12px; }}
    img  {{ max-width: 100%; margin: 20px 0; }}
  </style>
</head>
<body>
  <h1>Milestone 6 – Model Bias Audit Report</h1>
  <p><strong>Model:</strong> Llama3SP (DEVCamiloSepulveda/2-LLAMA3SP-talendesb)</p>
  <p><strong>Metric:</strong> Mean Absolute Error (MAE) &amp; Disparity Ratio (group MAE / min group MAE)</p>
  <p class="warn">⚠ Rows highlighted in <span style="background:#fff3cd;padding:2px 6px">yellow</span>
     have disparity ratio ≥ 1.5×. Rows in <span style="background:#ffe0e0;padding:2px 6px">red</span>
     have disparity ratio ≥ 2.0×, indicating potential bias.</p>

  <h2>1. Project-Level Bias (proxy for team/domain)</h2>
  {table_html(project_bias)}

  <h2>2. Story-Point Magnitude Bias</h2>
  {table_html(size_bias)}

  <h2>3. Description Availability Bias</h2>
  {table_html(desc_bias)}

  {aequitas_section}

  <h2>Visualisation</h2>
  <img src="bias_audit_plot.png" alt="Bias audit plots">
</body>
</html>"""


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_bias(project_bias, size_bias, desc_bias):
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Per-project MAE bar (coloured by disparity)
    ax1 = fig.add_subplot(gs[0, :])
    projs = project_bias["group"].tolist()
    maes  = project_bias["mae"].tolist()
    disp  = project_bias["disparity_ratio"].tolist()
    colors = ["#e74c3c" if d >= 2.0 else "#f39c12" if d >= 1.5 else "#2ecc71" for d in disp]
    bars = ax1.bar(projs, maes, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(len(projs)))
    ax1.set_xticklabels(projs, rotation=40, ha="right", fontsize=8)
    ax1.set_title("Per-Project MAE (red ≥ 2× disparity, orange ≥ 1.5×)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("MAE")
    # disparity ratio annotations
    for bar, d in zip(bars, disp):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{d:.1f}×", ha="center", va="bottom", fontsize=7, color="#555")

    # 2. Bias direction heatmap (over/under/neutral)
    ax2 = fig.add_subplot(gs[1, 0])
    pb_sorted = project_bias.sort_values("mean_error")
    colors2 = ["#e74c3c" if v > 0.5 else "#3498db" if v < -0.5 else "#95a5a6"
               for v in pb_sorted["mean_error"]]
    ax2.barh(pb_sorted["group"], pb_sorted["mean_error"], color=colors2)
    ax2.axvline(0, color="black", lw=1)
    ax2.set_title("Mean Prediction Error per Project\n(+ve = over-estimate)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Mean Error (story points)")
    ax2.tick_params(axis="y", labelsize=8)

    # 3. Magnitude group MAE
    ax3 = fig.add_subplot(gs[1, 1])
    sz_groups = size_bias["group"].tolist()
    sz_maes   = size_bias["mae"].tolist()
    ax3.bar(sz_groups, sz_maes, color=["#2980b9", "#e67e22", "#c0392b"], edgecolor="white")
    ax3.set_title("MAE by Story-Point Magnitude", fontsize=10, fontweight="bold")
    ax3.set_ylabel("MAE")
    ax3.set_xlabel("Magnitude group")

    fig.suptitle("Bias Audit — Llama3SP Story Point Estimator", fontsize=13, fontweight="bold")
    out = RESULTS_DIR / "bias_audit_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Bias audit plot → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_bias_audit(results_csv: Path) -> dict:
    if not results_csv.exists():
        raise FileNotFoundError(f"{results_csv} not found. Run test_evaluation.py first.")

    df = pd.read_csv(results_csv)

    # ── Group 1: project ──────────────────────────────────────────────────
    project_bias = compute_group_metrics(df, "project")

    # ── Group 2: story-point magnitude ────────────────────────────────────
    df["magnitude"] = pd.cut(
        df["storypoints"],
        bins=[0, 3, 8, 1000],
        labels=["small (1-3)", "medium (4-8)", "large (>8)"],
    )
    size_bias = compute_group_metrics(df, "magnitude")

    # ── Group 3: description availability ────────────────────────────────
    df["desc_available"] = df["input_text"].str.contains("Description:").map(
        {True: "has_description", False: "no_description"}
    )
    desc_bias = compute_group_metrics(df, "desc_available")

    # ── Aequitas (optional) ───────────────────────────────────────────────
    aequitas_result = _try_aequitas(df, "project", "y_pred_llama", "storypoints")

    # ── Identify high-disparity groups ────────────────────────────────────
    flagged_projects = project_bias[project_bias["disparity_ratio"] >= 2.0]["group"].tolist()
    flagged_sizes    = size_bias[size_bias["disparity_ratio"] >= 2.0]["group"].tolist()

    report = {
        "summary": {
            "flagged_projects_2x_disparity": flagged_projects,
            "flagged_size_groups_2x_disparity": flagged_sizes,
            "max_project_disparity": round(float(project_bias["disparity_ratio"].max()), 4),
            "min_project_mae": round(float(project_bias["mae"].min()), 4),
            "max_project_mae": round(float(project_bias["mae"].max()), 4),
            "over_estimate_projects": project_bias[project_bias["bias_direction"] == "over"]["group"].tolist(),
            "under_estimate_projects": project_bias[project_bias["bias_direction"] == "under"]["group"].tolist(),
        },
        "project_bias":   project_bias.to_dict(orient="records"),
        "size_bias":      size_bias.to_dict(orient="records"),
        "description_bias": desc_bias.to_dict(orient="records"),
        "aequitas_available": aequitas_result is not None,
    }

    json_path = RESULTS_DIR / "bias_audit_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Bias audit JSON → {json_path}")

    html = _html_report(project_bias, size_bias, desc_bias, aequitas_result)
    html_path = RESULTS_DIR / "bias_audit_report.html"
    html_path.write_text(html, encoding="utf-8")
    log.info(f"Bias audit HTML → {html_path}")

    _plot_bias(project_bias, size_bias, desc_bias)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  BIAS AUDIT SUMMARY")
    print("=" * 68)
    print(f"  Max disparity ratio : {report['summary']['max_project_disparity']:.2f}×")
    print(f"  Flagged projects (≥2×): {flagged_projects or 'None'}")
    print(f"  Flagged size groups  : {flagged_sizes or 'None'}")
    print(f"  Over-estimate projects: {report['summary']['over_estimate_projects']}")
    print(f"  Under-estimate projects: {report['summary']['under_estimate_projects']}")
    print("=" * 68)

    return report


if __name__ == "__main__":
    run_bias_audit(RESULTS_DIR / "test_evaluation_results.csv")
