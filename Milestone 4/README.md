# Milestone 4 – Model Development and Evaluation
> **Project:** AI-Based Story Point Estimation for Agile Software Development  
> **Course:** CSC5382 – AI for Digital Transformation  
> **Repository:** github.com/amiraHadimi/CSC5382-AI-Digital-Transformation-Project

---

## Table of Contents
1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Requirement 1 · Project Structure & Modularity](#requirement-1--project-structure--modularity)
4. [Requirement 2 · Code Versioning (GitHub Flow)](#requirement-2--code-versioning-github-flow)
5. [Requirement 3 · Experiment Tracking & Model Versioning (MLflow)](#requirement-3--experiment-tracking--model-versioning-mlflow)
6. [Requirement 4 · MLOps Platform Integration (ZenML)](#requirement-4--mlops-platform-integration-zenml)
7. [Optional · Energy Efficiency Measurement (CodeCarbon)](#optional--energy-efficiency-measurement-codecarbon)
8. [How to Run](#how-to-run)
9. [Results](#results)
10. [References](#references)

---

## Overview

Milestone 4 adds the **Model Development and Evaluation** layer to the MLOps platform built in Milestones 1–3. It implements:

- A **modular project structure** following the Cookiecutter Data Science standard
- **GitHub Flow** for code versioning
- **MLflow** experiment tracking and model versioning (nested per-project runs, model registry)
- A **ZenML pipeline** that wires model loading → inference → evaluation → reporting into a single reproducible workflow
- **CodeCarbon** for CO₂ emissions measurement during inference *(+2 pts optional)*

The model itself is the pre-trained **Llama3SP** (Llama 3.2 + per-project LoRA adapters) from Milestone 2 — no GPU fine-tuning is required. This milestone focuses on the **MLOps infrastructure** around the model.

### Pipeline Summary

| Step | Name | Tool | Output |
|---|---|---|---|
| Step 1 | `load_model_step` | ZenML + HF Transformers | Model metadata dict |
| Step 2 | `evaluate_step` | ZenML + MLflow + CodeCarbon | Per-project metrics DataFrame |
| Step 3 | `report_step` | ZenML | Leaderboard + aggregate summary |

---

## Repository Structure

```
Milestone 4/
├── src/                            ← All source code (modular packages)
│   ├── pipeline/
│   │   ├── model_loader.py         ← Base model + LoRA adapter loading
│   │   ├── inference.py            ← Batched inference logic
│   │   ├── zenml_steps.py          ← ZenML @step definitions (Steps 1–3)
│   │   └── zenml_pipeline.py       ← ZenML @pipeline wiring all steps
│   ├── evaluation/
│   │   └── metrics.py              ← MAE, RMSE, Accuracy@±1 + EvalMetrics dataclass
│   ├── tracking/
│   │   ├── mlflow_tracker.py       ← MLflow experiment / registry wrapper
│   │   └── carbon_tracker.py       ← CodeCarbon emissions wrapper
│   └── utils/
│       └── config.py               ← params.yaml loader + HF_TOKEN helper
├── configs/
│   └── params.yaml                 ← Central parameter file (all hyperparameters)
├── models/
│   └── README.md                   ← Model storage strategy (HF Hub + MLflow registry)
├── notebooks/
│   └── milestone4_demo.ipynb       ← Interactive demo notebook
├── tests/
│   ├── test_metrics.py             ← Unit tests for metrics module
│   └── test_config.py              ← Unit tests for config loader
├── results/                        ← Generated at runtime (gitignored)
│   ├── mae_per_project.csv
│   ├── summary.json
│   ├── predictions_<project>.csv
│   └── carbon/
│       ├── emissions.csv
│       └── carbon_summary.json
├── mlruns/                         ← MLflow tracking artefacts (gitignored)
├── run_pipeline.py                 ← Entry point: python run_pipeline.py
├── requirements.txt
└── README.md                       ← This file
```

---

## Requirement 1 · Project Structure & Modularity
**Tool:** Cookiecutter Data Science standard | **Points: 2**

The `Milestone 4/` directory follows the **Cookiecutter Data Science** project template conventions, adapted for an MLOps context:

| Cookiecutter Convention | Implementation |
|---|---|
| `src/` — all source code as importable packages | `src/pipeline/`, `src/evaluation/`, `src/tracking/`, `src/utils/` |
| `configs/` — centralised configuration | `configs/params.yaml` (single source of truth for all hyperparameters) |
| `models/` — model artefacts directory | `models/README.md` (weights on HF Hub, registry entry in MLflow) |
| `notebooks/` — exploratory and demo notebooks | `notebooks/milestone4_demo.ipynb` |
| `tests/` — unit tests | `tests/test_metrics.py`, `tests/test_config.py` |
| `results/` — generated outputs (gitignored) | `results/mae_per_project.csv`, `results/summary.json`, etc. |

**Key modularity principles applied:**

- Every concern lives in its own module: model loading (`model_loader.py`), inference (`inference.py`), metrics (`metrics.py`), MLflow tracking (`mlflow_tracker.py`), CodeCarbon (`carbon_tracker.py`), config (`config.py`).
- ZenML steps (`zenml_steps.py`) are kept thin — they orchestrate calls to the above modules, not implement logic.
- `configs/params.yaml` is the **single source of truth**: no magic numbers appear anywhere in the codebase.
- All packages expose clean `__init__.py` interfaces.

---

## Requirement 2 · Code Versioning (GitHub Flow)
**Tool:** Git with GitHub Flow | **Points: 2**

This milestone follows **GitHub Flow**:

```
main
 └── feature/milestone4-project-structure    (Req 1: folder layout, configs)
 └── feature/milestone4-mlflow-tracking      (Req 3: MLflow integration)
 └── feature/milestone4-zenml-pipeline       (Req 4: ZenML steps + pipeline)
 └── feature/milestone4-codecarbon           (Optional: energy tracking)
```

**Branch workflow:**
1. Each requirement was developed on a dedicated feature branch.
2. A pull request was opened for each branch with a descriptive title and summary.
3. PRs were reviewed and merged into `main` via squash merges to keep history clean.
4. Commit messages follow the **Conventional Commits** standard:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `refactor:` for code restructuring
   - `test:` for test additions
   - `docs:` for documentation updates

**Example commit history:**
```
feat: add Cookiecutter project structure for Milestone 4
feat: add MLflow experiment tracker with nested per-project runs
feat: add ZenML pipeline (load_model → evaluate → report)
feat: add CodeCarbon CO₂ tracking around inference loop
test: add unit tests for metrics and config modules
docs: write Milestone 4 README
```

---

## Requirement 3 · Experiment Tracking & Model Versioning (MLflow)
**Tool:** MLflow (local) | **Points: 5**
**Code:** [`src/tracking/mlflow_tracker.py`](src/tracking/mlflow_tracker.py)

### 3.1 Experiment Structure

MLflow is configured with a **nested run hierarchy** that maps naturally to the project structure:

```
Experiment: llama3sp_story_point_estimation
└── Parent run: m4_eval_<timestamp>           ← full sweep
    ├── Params logged: base_model_id, max_len, batch_size, ...
    ├── Metrics logged: mean_mae, std_mae, mean_rmse, mean_accuracy_at_1
    ├── Artefacts: mae_per_project.csv, summary.json, model_card.json
    ├── Child run: appceleratorstudio         ← per-project
    │     └── Metrics: mae, rmse, accuracy_at_1, test_size
    ├── Child run: bamboo
    ├── Child run: moodle
    └── ... (16 projects total)
```

### 3.2 What is Logged

| Category | What | Where |
|---|---|---|
| **Hyperparameters** | `base_model_id`, `hf_author`, `max_len`, `batch_size`, `use_description`, `limit_test_rows` | Parent run params |
| **Per-project metrics** | `mae`, `rmse`, `accuracy_at_1`, `test_size` | Nested child runs |
| **Aggregate metrics** | `mean_mae`, `std_mae`, `mean_rmse`, `mean_accuracy_at_1`, `num_projects` | Parent run metrics |
| **Artefacts** | `mae_per_project.csv`, `summary.json`, `carbon_summary.json` | Parent run artefacts |
| **Model info** | `model_card.json` (framework, adapter source, task) | `model_info/` artefact path |
| **Energy** | `co2_kg`, `energy_kwh`, `inference_time_s` | Parent run metrics |

### 3.3 Model Registry

The model is registered in the **MLflow Model Registry** under the name `Llama3SP-StoryPoints`. Since the actual weights live on Hugging Face Hub (not binary-logged to MLflow due to size), a lightweight `PythonModel` reference entry is created that stores the model card and points to the HF Hub location.

```python
mlflow.register_model(model_uri=model_uri, name="Llama3SP-StoryPoints")
# → models:/Llama3SP-StoryPoints/1
```

### 3.4 Viewing Results

```bash
cd "Milestone 4"
mlflow ui --port 5000
# Open: http://localhost:5000
```

The MLflow UI shows:
- All runs with timestamps and aggregate metrics
- Nested child runs per project for drill-down
- Artefact browser for CSV/JSON outputs
- Model registry entry under **Models** tab

---

## Requirement 4 · MLOps Platform Integration (ZenML)
**Tool:** ZenML | **Points: 5**
**Code:** [`src/pipeline/zenml_pipeline.py`](src/pipeline/zenml_pipeline.py) | [`src/pipeline/zenml_steps.py`](src/pipeline/zenml_steps.py)

### 4.1 Pipeline Architecture

The Milestone 4 ZenML pipeline **extends** the Milestone 3 data pipeline by adding three new steps for model development and evaluation:

```
[Milestone 3 Pipeline]              [Milestone 4 Pipeline]
ingest_data                         load_model_step
    │                                   │
validate_data                       evaluate_step  ←─ MLflow + CodeCarbon
    │                                   │
preprocess_and_engineer             report_step
    │
setup_feature_store
```

### 4.2 Step Definitions

**Step 1 — `load_model_step`**  
Resolves the base model ID from the HF adapter config and returns a metadata dict. Isolates all HF Hub probing from the heavy inference work.

**Step 2 — `evaluate_step`**  
The main step. For each of the 16 projects:
1. Loads and activates the project-specific LoRA adapter (dynamic adapter switching, no model reload).
2. Runs batched CPU inference on the test split (up to 100 rows).
3. Computes MAE, RMSE, and Accuracy@±1.
4. Logs a nested MLflow child run for the project.
5. Saves per-project predictions CSV.

CodeCarbon wraps the entire evaluation loop to measure total inference emissions.

**Step 3 — `report_step`**  
Aggregates metrics, prints a sorted leaderboard to logs, and returns the summary dict.

### 4.3 Pipeline DAG

```
load_model_step
       │
       ▼
evaluate_step  ─── MLflow (nested runs) ─── CodeCarbon (CO₂ tracking)
       │
       ▼
report_step    ─── Leaderboard output
```

### 4.4 Running the Pipeline

```bash
cd "Milestone 4"

# Set HF token
export HF_TOKEN=your_token_here

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python run_pipeline.py

# Or via ZenML CLI (tracked)
zenml pipeline run src/pipeline/zenml_pipeline.py:training_pipeline
```

### 4.5 ZenML Dashboard

```bash
zenml login --local --blocking
# Opens: http://127.0.0.1:8237
```

---

## Optional · Energy Efficiency Measurement (CodeCarbon)
**Tool:** CodeCarbon | **Points: 2**
**Code:** [`src/tracking/carbon_tracker.py`](src/tracking/carbon_tracker.py)

CodeCarbon is integrated to measure the environmental cost of running the inference pipeline.

### What is measured

The `CarbonTracker` wraps the full evaluation loop in `evaluate_step`:

```python
carbon.start()
# ← 16-project inference loop runs here →
emissions = carbon.stop()   # returns kg CO₂eq
carbon.log_to_mlflow(emissions)
```

### Output

| File | Contents |
|---|---|
| `results/carbon/emissions.csv` | Full CodeCarbon output (CPU, RAM, energy, emissions) |
| `results/carbon/carbon_summary.json` | Compact summary: `emissions_kg_co2eq`, `energy_kwh`, `duration_seconds` |

Carbon metrics are also logged to MLflow (`co2_kg`, `energy_kwh`, `inference_time_s`) so they appear alongside accuracy metrics in the experiment UI.

### Configuration

Country is set to **Morocco (MAR)** in `configs/params.yaml` to use the correct electricity carbon intensity for the AUI campus location.

```yaml
codecarbon:
  project_name: "story_point_estimation_m4"
  country_iso_code: "MAR"
  output_dir: "results/carbon"
```

---

## How to Run

### Prerequisites

```bash
# 1. Install dependencies
cd "Milestone 4"
pip install -r requirements.txt

# 2. Set Hugging Face token (required to download Llama3SP adapters)
export HF_TOKEN=your_huggingface_token   # Linux/Mac
set HF_TOKEN=your_huggingface_token      # Windows
```

### Run the pipeline

```bash
python run_pipeline.py
```

### Run unit tests

```bash
python -m pytest tests/ -v
```

### View MLflow results

```bash
mlflow ui --port 5000
# Open: http://localhost:5000
```

### Interactive notebook

```bash
jupyter notebook notebooks/milestone4_demo.ipynb
```

---

## Results

Results from Milestone 2 (Llama3SP baseline, reproduced):

| Project | n | MAE | RMSE | Acc@±1 |
|---|---|---|---|---|
| duracloud | 100 | 1.06 | — | — |
| bamboo | 100 | 1.10 | — | — |
| talendesb | 100 | 1.10 | — | — |
| springxd | 100 | 1.67 | — | — |
| usergrid | 97 | 1.51 | — | — |
| mesos | 100 | 1.38 | — | — |
| appceleratorstudio | 100 | 1.85 | — | — |
| jirasoftware | 71 | 2.05 | — | — |
| titanium | 100 | 2.83 | — | — |
| mule | 100 | 2.43 | — | — |
| mulestudio | 100 | 3.49 | — | — |
| talenddataquality | 100 | 3.60 | — | — |
| aptanastudio | 100 | 3.84 | — | — |
| clover | 77 | 4.08 | — | — |
| datamanagement | 100 | 6.19 | — | — |
| moodle | 100 | 11.30 | — | — |

> RMSE and Acc@±1 are computed freshly in the Milestone 4 pipeline and logged to MLflow.

---

## References

1. Choetkiertikul et al. (2018). *A deep learning model for estimating story points.* IEEE TSE.
2. Fu & Tantithamthavorn (2022). *GPT2SP: A Transformer-based Agile story point estimation approach.* IEEE TSE.
3. Sepúlveda Montoya et al. (2025). *Llama3SP: Resource-efficient LLM for Agile story point estimation.* https://github.com/DEVCamiloSepulveda/llama3sp
4. MLflow Documentation. https://mlflow.org/docs/latest/index.html
5. ZenML Documentation. https://docs.zenml.io
6. CodeCarbon Documentation. https://mlco2.github.io/codecarbon/
7. Cookiecutter Data Science. https://drivendata.github.io/cookiecutter-data-science/
