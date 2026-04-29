"""
Milestone 6 – Requirement 6.1 (2 pts)
Online Testing: A/B Test — Llama3SP vs TF-IDF+LR Baseline
----------------------------------------------------------
Reads the per-row evaluation results produced by test_evaluation.py,
then performs a rigorous statistical comparison:

  - Wilcoxon signed-rank test (non-parametric, paired per-issue)
  - Paired t-test (parametric sanity check)
  - Effect size: Cohen's d
  - Per-project win-rate analysis
  - Multi-armed bandit simulation (epsilon-greedy)

Outputs:
  results/ab_test_report.json
  results/ab_test_plot.png
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
from scipy import stats

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Statistical helpers ───────────────────────────────────────────────────────

def cohen_d(a, b):
    """Cohen's d effect size for two paired samples."""
    diff = np.array(a) - np.array(b)
    return float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0


def effect_size_label(d):
    ad = abs(d)
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    return "large"


# ── Multi-armed bandit simulation (epsilon-greedy) ────────────────────────────

def epsilon_greedy_bandit(errors_a, errors_b, epsilon=0.1, seed=42):
    """
    Simulate an epsilon-greedy bandit choosing between model A and B.
    At each step the bandit either explores (random) or exploits (best so far).
    Returns the fraction of traffic sent to each model by the end.
    """
    rng = np.random.default_rng(seed)
    n = len(errors_a)
    counts  = [0, 0]
    rewards = [0.0, 0.0]   # reward = -abs_error (higher = better)
    chosen  = []

    for i in range(n):
        if rng.random() < epsilon or counts[0] == 0 or counts[1] == 0:
            arm = int(rng.integers(0, 2))
        else:
            avg_a = rewards[0] / counts[0]
            avg_b = rewards[1] / counts[1]
            arm = 0 if avg_a >= avg_b else 1
        chosen.append(arm)
        err = errors_a[i] if arm == 0 else errors_b[i]
        counts[arm]  += 1
        rewards[arm] += -err   # negate so lower error = higher reward

    traffic_a = counts[0] / n
    traffic_b = counts[1] / n
    return {
        "traffic_model_a_pct": round(traffic_a * 100, 2),
        "traffic_model_b_pct": round(traffic_b * 100, 2),
        "chosen_counts": {"model_a": counts[0], "model_b": counts[1]},
        "avg_reward_a": round(rewards[0] / counts[0], 4) if counts[0] else 0,
        "avg_reward_b": round(rewards[1] / counts[1], 4) if counts[1] else 0,
    }


# ── Main A/B test routine ─────────────────────────────────────────────────────

def run_ab_test(results_csv: Path) -> dict:
    if not results_csv.exists():
        raise FileNotFoundError(
            f"{results_csv} not found. Run evaluation/test_evaluation.py first."
        )

    df = pd.read_csv(results_csv)
    errors_llama = np.array(df["abs_err_llama"].tolist())
    errors_baseline = np.array(df["abs_err_baseline"].tolist())

    diff = errors_llama - errors_baseline
    identical_errors = np.allclose(diff, 0)

    # ── Global statistical tests ─────────────────────────────────────────
    if identical_errors:
        log.warning("Llama3SP and baseline have identical errors. Skipping Wilcoxon test.")

        wilcoxon_statistic = 0.0
        wilcoxon_pvalue = 1.0
        wilcoxon_note = "Skipped because both models produced identical absolute errors."
    else:
        wilcoxon = stats.wilcoxon(errors_llama, errors_baseline, alternative="two-sided")
        wilcoxon_statistic = float(wilcoxon.statistic)
        wilcoxon_pvalue = float(wilcoxon.pvalue)
        wilcoxon_note = "Wilcoxon signed-rank test completed."

    ttest = stats.ttest_rel(errors_llama, errors_baseline)
    d = cohen_d(errors_baseline, errors_llama)  # positive d = llama better

    global_stats = {
        "n": len(df),
        "mae_llama3sp": float(np.mean(errors_llama)),
        "mae_baseline": float(np.mean(errors_baseline)),
        "wilcoxon": {
            "statistic": wilcoxon_statistic,
            "p_value": wilcoxon_pvalue,
            "significant_at_0.05": bool(wilcoxon_pvalue < 0.05),
            "note": wilcoxon_note,
        },
        "paired_ttest": {
            "statistic": float(ttest.statistic) if not np.isnan(ttest.statistic) else 0.0,
            "p_value": float(ttest.pvalue) if not np.isnan(ttest.pvalue) else 1.0,
            "significant_at_0.05": bool(ttest.pvalue < 0.05) if not np.isnan(ttest.pvalue) else False,
        },
        "cohens_d": {
            "value": round(d, 4),
            "interpretation": effect_size_label(d),
            "direction": "llama3sp_better" if d > 0 else ("baseline_better" if d < 0 else "tie"),
        },
    }

    # ── Per-project win-rate ──────────────────────────────────────────────
    per_project = {}
    for proj, grp in df.groupby("project"):
        wins_llama = int((grp["abs_err_llama"] < grp["abs_err_baseline"]).sum())
        wins_baseline = int((grp["abs_err_baseline"] < grp["abs_err_llama"]).sum())
        ties = int((grp["abs_err_llama"] == grp["abs_err_baseline"]).sum())
        n_proj = len(grp)

        per_project[proj] = {
            "n": n_proj,
            "mae_llama3sp": round(float(grp["abs_err_llama"].mean()), 4),
            "mae_baseline": round(float(grp["abs_err_baseline"].mean()), 4),
            "wins_llama3sp": wins_llama,
            "wins_baseline": wins_baseline,
            "ties": ties,
            "winner": "llama3sp" if wins_llama > wins_baseline else
                      ("baseline" if wins_baseline > wins_llama else "tie"),
        }

    # ── Multi-armed bandit ────────────────────────────────────────────────
    bandit_result = epsilon_greedy_bandit(errors_llama, errors_baseline)

    if identical_errors:
        conclusion = "Models produced identical absolute errors; no statistically significant difference detected"
    elif wilcoxon_pvalue < 0.05 and global_stats["mae_llama3sp"] < global_stats["mae_baseline"]:
        conclusion = "Llama3SP significantly outperforms baseline"
    elif wilcoxon_pvalue < 0.05 and global_stats["mae_baseline"] < global_stats["mae_llama3sp"]:
        conclusion = "Baseline significantly outperforms Llama3SP"
    else:
        conclusion = "No statistically significant difference detected"

    report = {
        "experiment": {
            "model_a": "Llama3SP (DEVCamiloSepulveda/2-LLAMA3SP-talendesb)",
            "model_b": "TF-IDF + Ridge Regression baseline",
            "metric": "Absolute Error per issue",
            "test_type": "Paired (same test issues evaluated by both models)",
        },
        "global_statistics": global_stats,
        "per_project_win_rate": per_project,
        "bandit_simulation": bandit_result,
        "conclusion": conclusion,
    }

    report_path = RESULTS_DIR / "ab_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"A/B test report saved → {report_path}")

    _plot_ab_results(df, per_project, global_stats, bandit_result)

    print("\n" + "=" * 68)
    print("  A/B TEST RESULTS")
    print("=" * 68)
    print(f"  Model A : {report['experiment']['model_a']}")
    print(f"  Model B : {report['experiment']['model_b']}")
    print()
    print(f"  Global MAE — Llama3SP : {global_stats['mae_llama3sp']:.4f}")
    print(f"  Global MAE — Baseline : {global_stats['mae_baseline']:.4f}")
    print()
    print(f"  Wilcoxon p-value      : {wilcoxon_pvalue:.4e}  "
          f"({'significant' if wilcoxon_pvalue < 0.05 else 'not significant'} @ α=0.05)")
    print(f"  Note                  : {wilcoxon_note}")
    print(f"  Cohen's d             : {d:.4f} ({effect_size_label(d)} effect, "
          f"{report['global_statistics']['cohens_d']['direction']})")
    print()
    print(f"  CONCLUSION: {report['conclusion']}")
    print()
    print(f"  Bandit traffic → Llama3SP: {bandit_result['traffic_model_a_pct']}%  "
          f"| Baseline: {bandit_result['traffic_model_b_pct']}%")
    print("=" * 68)

    return report

    report_path = RESULTS_DIR / "ab_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"A/B test report saved → {report_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_ab_results(df, per_project, global_stats, bandit_result)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  A/B TEST RESULTS")
    print("=" * 68)
    print(f"  Model A : {report['experiment']['model_a']}")
    print(f"  Model B : {report['experiment']['model_b']}")
    print()
    print(f"  Global MAE — Llama3SP : {global_stats['mae_llama3sp']:.4f}")
    print(f"  Global MAE — Baseline : {global_stats['mae_baseline']:.4f}")
    print()
    print(f"  Wilcoxon p-value      : {wilcoxon.pvalue:.4e}  "
          f"({'significant' if wilcoxon.pvalue < 0.05 else 'not significant'} @ α=0.05)")
    print(f"  Cohen's d             : {d:.4f} ({effect_size_label(d)} effect, "
          f"{report['global_statistics']['cohens_d']['direction']})")
    print()
    print(f"  CONCLUSION: {report['conclusion']}")
    print()
    print(f"  Bandit traffic → Llama3SP: {bandit_result['traffic_model_a_pct']}%  "
          f"| Baseline: {bandit_result['traffic_model_b_pct']}%")
    print("=" * 68)

    return report


def _plot_ab_results(df, per_project, global_stats, bandit_result):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Error distribution comparison
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(df["abs_err_llama"],    bins=40, alpha=0.6, color="#4C72B0", label="Llama3SP")
    ax1.hist(df["abs_err_baseline"], bins=40, alpha=0.6, color="#DD8452", label="Baseline")
    ax1.axvline(global_stats["mae_llama3sp"],  color="#4C72B0", lw=2, ls="--",
                label=f"Llama3SP MAE={global_stats['mae_llama3sp']:.2f}")
    ax1.axvline(global_stats["mae_baseline"],  color="#DD8452", lw=2, ls="--",
                label=f"Baseline MAE={global_stats['mae_baseline']:.2f}")
    ax1.set_title("Absolute Error Distribution: Llama3SP vs Baseline", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Absolute Error (story points)")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=9)

    # 2. Per-project MAE comparison
    ax2 = fig.add_subplot(gs[1, :])
    projs    = list(per_project.keys())
    mae_l    = [per_project[p]["mae_llama3sp"]  for p in projs]
    mae_b    = [per_project[p]["mae_baseline"]   for p in projs]
    x        = np.arange(len(projs))
    w        = 0.35
    bars_l   = ax2.bar(x - w/2, mae_l, w, color="#4C72B0", alpha=0.85, label="Llama3SP")
    bars_b   = ax2.bar(x + w/2, mae_b, w, color="#DD8452", alpha=0.85, label="Baseline")
    ax2.set_xticks(x)
    ax2.set_xticklabels(projs, rotation=45, ha="right", fontsize=8)
    ax2.set_title("Per-Project MAE: Llama3SP vs Baseline", fontsize=11, fontweight="bold")
    ax2.set_ylabel("MAE (story points)")
    ax2.legend(fontsize=9)

    # 3. Bandit pie chart
    ax3 = fig.add_subplot(gs[0, 2])
    sizes  = [bandit_result["traffic_model_a_pct"], bandit_result["traffic_model_b_pct"]]
    labels = [f"Llama3SP\n{sizes[0]:.1f}%", f"Baseline\n{sizes[1]:.1f}%"]
    ax3.pie(sizes, labels=labels, colors=["#4C72B0", "#DD8452"], autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 9})
    ax3.set_title("ε-Greedy Bandit\nTraffic Allocation", fontsize=10, fontweight="bold")

    fig.suptitle("A/B Test: Llama3SP vs Baseline — Story Point Estimation", fontsize=13, fontweight="bold")

    out = RESULTS_DIR / "ab_test_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"A/B test plot saved → {out}")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results_csv = RESULTS_DIR / "test_evaluation_results.csv"
    run_ab_test(results_csv)
