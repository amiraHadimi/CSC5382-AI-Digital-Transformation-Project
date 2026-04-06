from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union
import csv
import json

from codecarbon import OfflineEmissionsTracker
import mlflow


@dataclass
class CarbonSummary:
    duration: float
    emissions_kg: float
    energy_kwh: float
    cpu_energy_kwh: float
    gpu_energy_kwh: float
    ram_energy_kwh: float
    country_name: str


class CarbonTracker:
    def __init__(self, cfg_or_output_dir: Union[dict, str, Path]) -> None:
        if isinstance(cfg_or_output_dir, dict):
            data_cfg = cfg_or_output_dir.get("data", {})
            carbon_cfg = cfg_or_output_dir.get("codecarbon", {})

            milestone_root = Path(__file__).resolve().parents[2]
            results_dir = milestone_root / data_cfg.get("results_dir", "results")
            self.output_dir = results_dir / "carbon"

            self.country_iso_code = carbon_cfg.get("country_iso_code", "MAR")
            self.project_name = carbon_cfg.get("project_name", "milestone4_evaluation")
        else:
            self.output_dir = Path(cfg_or_output_dir)
            self.country_iso_code = "MAR"
            self.project_name = "milestone4_evaluation"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "emissions.csv"
        self.summary_path = self.output_dir / "carbon_summary.json"

        self.tracker = OfflineEmissionsTracker(
            country_iso_code=self.country_iso_code,
            project_name=self.project_name,
            output_dir=str(self.output_dir),
            output_file="emissions.csv",
            save_to_file=True,
        )

    def start(self) -> None:
        self.tracker.start()
        

    def stop(self) -> CarbonSummary:
        returned_emissions = self.tracker.stop()

        rows = []
        if self.csv_path.exists():
            with self.csv_path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        last = rows[-1] if rows else {}

        csv_emissions = float(last.get("emissions", 0.0) or 0.0)
        emissions_kg = float(
            returned_emissions if returned_emissions is not None else csv_emissions
        )

        summary = CarbonSummary(
            duration=float(last.get("duration", 0.0) or 0.0),
            emissions_kg=emissions_kg,
            energy_kwh=float(last.get("energy_consumed", 0.0) or 0.0),
            cpu_energy_kwh=float(last.get("cpu_energy", 0.0) or 0.0),
            gpu_energy_kwh=float(last.get("gpu_energy", 0.0) or 0.0),
            ram_energy_kwh=float(last.get("ram_energy", 0.0) or 0.0),
            country_name=str(last.get("country_name", "")),
        )

        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)

        return summary

    def log_to_mlflow(self, summary: CarbonSummary) -> None:
        mlflow.log_metrics(
            {
                "co2_kg": summary.emissions_kg,
                "energy_kwh": summary.energy_kwh,
                "inference_time_s": summary.duration,
                "cpu_energy_kwh": summary.cpu_energy_kwh,
                "gpu_energy_kwh": summary.gpu_energy_kwh,
                "ram_energy_kwh": summary.ram_energy_kwh,
            }
        )

        if self.summary_path.exists():
            mlflow.log_artifact(str(self.summary_path), artifact_path="carbon")

        if self.csv_path.exists():
            mlflow.log_artifact(str(self.csv_path), artifact_path="carbon")