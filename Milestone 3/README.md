# Milestone 3 – Data Acquisition, Validation & Preparation
> **Project:** AI-Based Story Point Estimation for Agile Software Development
> **Course:** CSC5382 – AI for Digital Transformation
> **Repository:** github.com/amiraHadimi/CSC5382-AI-Digital-Transformation-Project
> 📄 **[Download Full Report (PDF)](Milestone3_Report.pdf)**
> 🎥 **[Presentation Recording](https://alakhawayn365-my.sharepoint.com/:v:/g/personal/a_hadimi_aui_ma/IQCQGWgS0sEdSp7g-y1OoEtwASADfJ_gxJG6QKUMuRlwODw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D&e=HgFOCt)**

---
## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Step 1 · Data Ingestion & Raw Storage](#step-1--data-ingestion--raw-storage)
4. [Step 2 · Schema Definition & Data Validation](#step-2--schema-definition--data-validation)
5. [Step 3 · Preprocessing & Feature Engineering](#step-3--preprocessing--feature-engineering)
6. [Step 4 · Feature Store (Feast)](#step-4--feature-store-feast)
7. [Step 5 · Data Versioning (DVC)](#step-5--data-versioning-dvc)
8. [Step 6 · ML Pipeline Integration (ZenML)](#step-6--ml-pipeline-integration-zenml)
9. [References](#references)

---

## Overview

Milestone 3 builds the complete data pipeline for the Agile story point estimation system introduced in Milestones 1 and 2. The goal is to ingest, validate, preprocess, and version the raw Agile user story dataset (23,313 issues from 16 open-source JIRA projects), register engineered features in a Feast feature store, and wire everything into a ZenML pipeline as part of the larger MLOps platform — all version-controlled via DVC.

The pipeline connects directly to the Llama3SP model used in Milestone 2 (`meta-llama/Llama-3.2-1B` + per-project LoRA adapters).

**To reproduce the full pipeline with one command:**
```bash
python run_pipeline.py
```

### Pipeline Summary

| Step | Name | Tool | Output |
|---|---|---|---|
| Step 1 | Data Ingestion & Raw Storage | Python + ZenML | `data/raw/raw_data.csv` (23,313 rows) |
| Step 2 | Schema Definition & Data Validation | Great Expectations | `schema.json`, statistics, anomaly report |
| Step 3 | Preprocessing & Feature Engineering | pandas + ZenML Transform | `train/val/test .parquet` files (15 features) |
| Step 4 | Feature Store Registration | Feast | Feature store with 6 registered features |
| Step 5 | Data Versioning | DVC + Git | `.dvc` pointer files + tag `v1.0-milestone3` |
| Step 6 | ML Pipeline Integration | ZenML | Registered pipeline wired into MLOps platform |

---

## Repository Structure

```
Milestone 3/
├── data/
│   ├── raw/
│   │   ├── raw_data.csv.dvc           ← DVC pointer (23,313 rows, all 16 projects merged)
│   │   └── .gitignore
│   └── processed/
│       ├── train.parquet              ← 13,981 rows (60%) — model training
│       ├── val.parquet                ← 4,661  rows (20%) — validation
│       ├── test.parquet               ← 4,671  rows (20%) — final evaluation only
│       ├── processed_data.parquet.dvc ← DVC pointer (full 23,313 rows, 15 features)
│       └── .gitignore
├── schema/
│   └── schema.json                    ← inferred + revised schema (6 features)
├── tfdv_output/
│   ├── data_statistics.json           ← statistics for train, val, test splits
│   ├── split_info.json                ← split sizes and source
│   ├── anomalies_report.json          ← anomaly detection report (0 anomalies found)
│   └── feature_distributions.png     ← feature distribution plots (4 charts)
├── feast_repo/feature_repo/
│   ├── feature_store.yaml             ← Feast project config
│   └── features.py                    ← entity + feature view definitions
├── pipeline/
│   ├── ingestion.py                   ← ZenML @step: raw data ingestion
│   ├── validation.py                  ← ZenML @step: schema + validation
│   ├── transform.py                   ← ZenML @step: preprocessing + features
│   ├── feature_store.py               ← ZenML @step: Feast registration
│   └── zenml_pipeline.py              ← full ZenML @pipeline wiring all steps
├── notebooks/
│   └── milestone3_pipeline.ipynb      ← demonstration notebook
├── configs/
│   └── params.yaml                    ← central parameter file (DVC params)
├── run_pipeline.py                    ← standalone runner (all steps in one script)
├── merge_data.py                      ← merges 16 project CSVs into one file
├── requirements.txt
└── README.md                          ← this file (milestone report)
```

---

## Step 1 · Data Ingestion & Raw Storage

**Tool:** Python + ZenML `@step` (TFX ExampleGen style) | **Points: 5**
**Code:** [`pipeline/ingestion.py`](pipeline/ingestion.py) | [`merge_data.py`](merge_data.py)

### The Challenge
The Llama3SP benchmark dataset is distributed as **16 separate CSV files** — one per JIRA project (`appceleratorstudio.csv`, `moodle.csv`, `springxd.csv`, etc.). These cannot be fed directly to a model and use inconsistent column names across files.

### What Was Done
1. **Merged** all 16 CSV files into one unified file using [`merge_data.py`](merge_data.py)
2. **Normalized column names** — different files used different names for the same field:

| Raw Column Name | Canonical Name |
|---|---|
| `storypoint`, `story_points`, `point` | `storypoints` |
| `concat`, `body` | `description` |
| `summary` | `title` |

3. **Filtered** rows with null or zero story points (invalid training labels)
4. **Saved** the result to `data/raw/raw_data.csv` — tracked by DVC

### Result

| Metric | Value |
|---|---|
| Total rows | 23,313 user stories |
| Projects covered | 16 open-source JIRA projects |
| Columns | `issuekey`, `title`, `description`, `storypoints`, `split_mark`, `project` |
| Rows dropped | 0 (all data valid after normalization) |
| Output | [`data/raw/raw_data.csv.dvc`](data/raw/raw_data.csv.dvc) |

---

## Step 2 · Schema Definition & Data Validation

**Tool:** Great Expectations + custom statistics | **Points: 2 (schema) + 3 (validation)**
**Code:** [`pipeline/validation.py`](pipeline/validation.py)
**Outputs:** [`schema/schema.json`](schema/schema.json) | [`tfdv_output/`](tfdv_output/)

### 2.1 Data Split

The dataset contains a pre-defined `split_mark` column assigned by the Llama3SP authors, ensuring consistent and reproducible splits across all research using this benchmark:

| Split | Rows | Percentage | Purpose |
|---|---|---|---|
| **Train** | 13,981 | 60% | Schema inference + model training |
| **Validation (val)** | 4,661 | 20% | Anomaly detection + hyperparameter tuning |
| **Test** | 4,671 | 20% | Final evaluation only — **never touched during validation** |


Split details: [`tfdv_output/split_info.json`](tfdv_output/split_info.json)

### 2.2 Data Statistics

Statistics were computed for all three splits across all 6 columns. Key findings from the training set:

| Column | Type | Nulls | Key Insight |
|---|---|---|---|
| `storypoints` | numeric | 0% | Mean: 4.87, Max: 100 (right-skewed distribution) |
| `title` | string | 0% | All issues have a title |
| `description` | string | 12.3% | 12% of JIRA issues have no description |
| `project` | string | 0% | 16 unique project values |
| `split_mark` | string | 0% | Values: `train`, `val`, `test` |

Full statistics: [`tfdv_output/data_statistics.json`](tfdv_output/data_statistics.json)

### 2.3 Schema Definition

The schema is inferred **from the training set only** — preventing data leakage. For each column it captures type, min/max (for numeric), unique values (for strings), and a maximum null percentage (training null % + 10% tolerance buffer).

```json
{
  "storypoints": { "type": "numeric", "min": 0.5,  "max": 100.0, "max_null_pct": 10.0 },
  "title":       { "type": "string",  "unique_values": 13891,     "max_null_pct": 10.0 },
  "description": { "type": "string",  "unique_values": 13102,     "max_null_pct": 22.3 },
  "project":     { "type": "string",  "unique_values": 16,        "max_null_pct": 10.0 }
}
```

Saved schema: [`schema/schema.json`](schema/schema.json)

### 2.4 Anomaly Detection & Schema Revision

The **validation set** is checked against the training schema using Great Expectations:
- Null percentage violations (is the missing data rate too high?)
- Numeric range violations (are values outside the expected min/max range?)

**Result: 0 anomalies detected.** The schema required no revision, confirming that the pre-defined splits by the Llama3SP authors are internally consistent.

If anomalies had been found, the schema would have been automatically revised by:
- Widening numeric ranges to accommodate the validation set distribution
- Relaxing null thresholds for text fields

Anomaly report: [`tfdv_output/anomalies_report.json`](tfdv_output/anomalies_report.json)

---

## Step 3 · Preprocessing & Feature Engineering

**Tool:** Python (pandas + regex) wrapped in ZenML Transform step | **Points: 5**
**Code:** [`pipeline/transform.py`](pipeline/transform.py)
**Outputs:** [`data/processed/`](data/processed/) | [`tfdv_output/feature_distributions.png`](tfdv_output/feature_distributions.png)

### 3.1 Text Cleaning

Raw JIRA text undergoes a 5-stage cleaning pipeline:

| Stage | Operation | Example |
|---|---|---|
| Lowercasing | `text.lower()` | `"Add Feature"` → `"add feature"` |
| URL removal | regex `https?://\S+` | `"see https://jira.com"` → `"see"` |
| JIRA markup removal | `{code}`, `[~user]`, `!image!` | `"{code}x=1{code}"` → `" "` |
| Special char removal | keep `[a-z0-9 .,!?-]` | `"fix: issue #42"` → `"fix issue 42"` |
| Whitespace collapse | `\s+` → single space | `"fix  bug"` → `"fix bug"` |

### 3.2 Input Text Construction

Title and description are combined into the exact prompt format expected by the Llama3SP tokenizer:

```python
input_text = f"Title: {cleaned_title} Description: {cleaned_description}"
# Example:
# "Title: add ca against object literals Description: div class p style..."
```

### 3.3 Engineered Features

| Feature | Type | How Computed | Why Useful |
|---|---|---|---|
| `input_text` | string | Title + Description combined | Primary LLM input |
| `text_length` | int | `len(input_text)` | Longer story = more complex task |
| `word_count` | int | `len(input_text.split())` | Complexity proxy |
| `has_description` | 0/1 | 1 if description non-empty | Issues with no description harder to estimate |
| `title_word_count` | int | `len(cleaned_title.split())` | Title length signal |
| `log_storypoints` | float | `log1p(storypoints)` | Reduces right skew of target variable |
| `is_fibonacci` | 0/1 | 1 if value in {1,2,3,5,8,13,20,40,100} | Planning poker alignment flag |

> **Why `log_storypoints`?** Story points are heavily right-skewed — most issues are 1-5 points but a few reach 40-100. The log transformation makes the distribution more balanced, which significantly improves regression model performance.

### 3.4 Output Files

| File | Rows | Columns | Description |
|---|---|---|---|
| `data/processed/train.parquet` | 13,981 | 15 | Training set — model learns from this |
| `data/processed/val.parquet` | 4,661 | 15 | Validation set — hyperparameter tuning |
| `data/processed/test.parquet` | 4,671 | 15 | Test set — untouched until Milestone 4 |
| `data/processed/processed_data.parquet` | 23,313 | 15 | Full dataset with all features |

Files are stored in **Parquet format** for 5-10x faster loading compared to CSV, with exact data types preserved.

---

## Step 4 · Feature Store (Feast)

**Tool:** Feast (local FileSource, offline mode) | **Points: 1**
**Code:** [`pipeline/feature_store.py`](pipeline/feature_store.py) | [`feast_repo/feature_repo/features.py`](feast_repo/feature_repo/features.py)

A Feast feature store was set up to register the engineered features. This solves a critical production problem: without a feature store, the training and inference pipelines might compute features differently, causing **training-serving skew** that silently degrades model performance.

### Registered Components

**Entity:**
```python
issue_entity = Entity(name="issue_id")
# Unique identifier for each JIRA issue / user story
```

**Feature View:**
```python
user_story_features = FeatureView(
    name="user_story_features",
    entities=[issue_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="text_length",      dtype=Int64),
        Field(name="word_count",       dtype=Int64),
        Field(name="has_description",  dtype=Int64),
        Field(name="title_word_count", dtype=Int64),
        Field(name="log_storypoints",  dtype=Float32),
        Field(name="is_fibonacci",     dtype=Int64),
    ],
    source=user_story_source,
)
```

### Feast Apply Output
```
Applying changes for project agile_story_points
Created entity issue_id
Created feature view user_story_features
Created sqlite table agile_story_points_user_story_features
```

---

## Step 5 · Data Versioning (DVC)

**Tool:** DVC (Data Version Control) | **Points: 3**
**Config:** [`dvc.yaml`](dvc.yaml) (removed — DVC initialized at repo root)

DVC does for data what Git does for code. Instead of committing large data files to GitHub (which rejects files over 100MB), DVC stores a tiny pointer file in Git while keeping the actual data separately — enabling full versioning without bloating the repository.

### Versioned Artifacts

| Artifact | DVC Pointer File | Git Tag |
|---|---|---|
| `data/raw/raw_data.csv` | `data/raw/raw_data.csv.dvc` | `v1.0-milestone3` |
| `data/processed/processed_data.parquet` | `data/processed/processed_data.parquet.dvc` | `v1.0-milestone3` |

### Commands Used
```bash
dvc init
dvc add "Milestone 3/data/raw/raw_data.csv"
dvc add "Milestone 3/data/processed/processed_data.parquet"
git add .
git commit -m "feat: Milestone 3 - data pipeline complete"
git tag -a v1.0-milestone3 -m "Milestone 3 complete"
git push origin main
git push origin --tags
```

### Reproducibility
Anyone can reproduce the exact results by running:
```bash
git checkout v1.0-milestone3
dvc checkout
python run_pipeline.py
```

---

## Step 6 · ML Pipeline Integration (ZenML)

**Tool:** ZenML | **Requirement:** Setup the data pipeline as part of the larger ML pipeline
**Code:** [`pipeline/zenml_pipeline.py`](pipeline/zenml_pipeline.py)

This requirement asks that the data pipeline not be a standalone script, but instead be wired into a larger MLOps platform that will eventually include training, evaluation, and deployment. We satisfied this using ZenML.

### Every Step is a ZenML Step

```python
@step
def ingest_data(csv_path: str) -> pd.DataFrame: ...

@step
def validate_data(df: pd.DataFrame) -> dict: ...

@step
def preprocess_and_engineer(df: pd.DataFrame) -> pd.DataFrame: ...

@step
def setup_feature_store(df: pd.DataFrame) -> dict: ...
```

### All Steps Wired Into One Pipeline

```python
@pipeline(name="milestone3_data_pipeline", enable_cache=True)
def data_pipeline(csv_path: str) -> None:
    raw_df       = ingest_data(csv_path=csv_path)
    val_report   = validate_data(df=raw_df)
    processed_df = preprocess_and_engineer(df=raw_df)
    store_info   = setup_feature_store(df=processed_df)
```

### What ZenML Provides

| Feature | Benefit |
|---|---|
| Run tracking | Every pipeline run has a unique ID and is logged automatically |
| Artifact store | DataFrames, reports, and schemas are stored and versioned between steps |
| Caching (`enable_cache=True`) | If raw data has not changed, ZenML skips unchanged steps |
| Dashboard | All runs, artifacts, and step outputs visible in ZenML UI |
| Extensibility | Milestone 4 training steps can be added directly to this same pipeline |

### Pipeline DAG

```
ingest_data
    │
    ▼
validate_data          → schema/schema.json
    │                  → tfdv_output/anomalies_report.json
    │                  → tfdv_output/data_statistics.json
    ▼
preprocess_and_engineer → data/processed/train.parquet
    │                   → data/processed/val.parquet
    │                   → data/processed/test.parquet
    ▼
setup_feature_store    → feast_repo/feature_repo/
```


### Running the Pipeline

```bash
# Standalone (no ZenML server needed)
python run_pipeline.py

# Via ZenML CLI
zenml pipeline run pipeline/zenml_pipeline.py:data_pipeline
```

---


## References

1. Choetkiertikul, M., Dam, H. K., Tran, T., Treude, C., & Ghose, A. (2018). *A deep learning model for estimating story points.* IEEE TSE. https://doi.org/10.1109/TSE.2018.2792247
2. Fu, M., & Tantithamthavorn, C. (2022). *GPT2SP: A Transformer-based Agile story point estimation approach.* IEEE TSE. https://doi.org/10.1109/TSE.2022.3160829
3. Sepúlveda Montoya, C., Ríos Gómez, J., & Jaramillo Villegas, J. A. (2025). *Llama3SP: Resource-efficient LLM for Agile story point estimation.* https://github.com/DEVCamiloSepulveda/llama3sp
4. Great Expectations Documentation. https://docs.greatexpectations.io
5. ZenML Documentation. https://docs.zenml.io
6. Feast Feature Store Documentation. https://docs.feast.dev
7. DVC Documentation. https://dvc.org/doc
