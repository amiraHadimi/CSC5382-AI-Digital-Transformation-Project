"""
tracking/carbon_tracker.py
==========================
Disabled CodeCarbon wrapper for Milestone 4.

This version intentionally disables CO₂ tracking to avoid repeated GPU / pynvml
warnings on Windows systems where GPU power queries are not supported.

It keeps the same interface so the rest of the pipeline works unchanged.
"""

import os
import json
from pathlib import Path
import mlflow


class CarbonTracker:
    """
    No-op CarbonTracker used to keep the pipeline stable on Windows.
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: Full config dict (load_config() output).
        """
        cc_cfg = cfg.get("codecarbon", {})
        self.project_name = cc_cfg.get("project_name", "story_point_estimation_m4")
        self.country_iso = cc_cfg.get("country_iso_code", "MAR")

        output_dir = cc_cfg.get("output_dir", "results/carbon")
        m4_root = Path(__file__).resolve().parents[2]
        self.output_dir = str(m4_root / output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def start(self) -> None:
        """
        Start tracking.

        In this disabled version, we skip CodeCarbon entirely.
        """
        print("[CarbonTracker] CodeCarbon disabled for this local Windows run.")

    def stop(self) -> dict | None:
        """
        Stop tracking and return a placeholder summary.
        """
        summary = {
            "emissions_kg_co2eq": 0.0,
            "energy_kwh": 0.0,
            "duration_seconds": 0.0,
            "country": self.country_iso,
        }

        summary_path = Path(self.output_dir) / "carbon_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("[CarbonTracker] Carbon tracking skipped.")
        return summary

    def log_to_mlflow(self, summary: dict | None) -> None:
        """
        Log placeholder carbon data to the currently active MLflow run.
        """
        if summary is None:
            return

        mlflow.log_metrics({
            "co2_kg": summary["emissions_kg_co2eq"],
            "energy_kwh": summary["energy_kwh"],
            "inference_time_s": summary["duration_seconds"],
        })

        summary_json = Path(self.output_dir) / "carbon_summary.json"
        if summary_json.exists():
            mlflow.log_artifact(str(summary_json), artifact_path="carbon")