# Milestone 3 – Data Acquisition, Validation & Preparation

> **Project:** AI-Based Story Point Estimation for Agile Software Development
> **Course:** CSC5382 – AI for Digital Transformation
> **Milestone due:** March 22, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [1 · Data Ingestion & Raw Storage](#1--data-ingestion--raw-storage)
4. [2 · Schema Definition](#2--schema-definition)
5. [3 · Data Validation & Anomaly Detection](#3--data-validation--anomaly-detection)
6. [4 · Preprocessing & Feature Engineering](#4--preprocessing--feature-engineering)
7. [5 · Feature Store (Feast)](#5--feature-store-feast)
8. [6 · Data Versioning (DVC)](#6--data-versioning-dvc)
9. [7 · ML Pipeline Integration (ZenML)](#7--ml-pipeline-integration-zenml)
10. [Grading Checklist](#grading-checklist)
11. [References](#references)

---

## Overview

Milestone 3 operationalises the data layer of the Agile story point estimation system introduced in Milestones 1 and 2. The goal is to build a **production-ready data pipeline** that ingests the raw Agile user story dataset (23,313 issues from 16 open-source JIRA projects), validates it against an inferred schema, engineers features suitable for the Llama3SP fine-tuning workflow, and registers those features in a Feast feature store — all version-controlled via DVC and orchestrated through a ZenML pipeline.

The pipeline connects directly to the Llama3SP model used in Milestone 2 (`meta-llama/Llama-3.2-1B` + per-project LoRA adapters): the `input_text` feature produced here is the exact field consumed by the tokenizer in `eval_mae_per_project.py`.

---

## Repository Structure

```
Milestone 3/
├── data/
│   ├── raw/
│   │   ├── raw_data.csv               ← raw ingested snapshot (DVC tracked)
│   │   └── raw_data.csv.dvc           ← DVC pointer file
│   └── processed/
│       ├── processed_data.parquet     ← engineered features (DVC tracked)
│       └── processed_data.parquet.dvc
├── notebooks/
│   └── milestone3_pipeline.ipynb      ← main demonstration notebook
├── pipeline/
│   ├── ingestion.py                   ← ZenML step: raw data ingestion
│   ├── validation.py                  ← ZenML step: TFDV validation
│   ├── transform.py                   ← ZenML step: preprocessing & features
│   ├── feature_store.py               ← ZenML step: Feast registration
│   └── zenml_pipeline.py              ← full ZenML pipeline wiring
├── schema/
│   └── schema.pbtxt                   ← TFDV inferred + revised schema
├── tfdv_output/
│   ├── train_stats.pb                 ← TFDV training statistics
│   ├── eval_stats.pb                  ← TFDV evaluation statistics
│   ├── anomalies_report.json          ← detected anomalies
│   ├── storypoints_distribution.png
│   └── feature_distributions.png
├── feast_repo/feature_repo/
│   ├── feature_store.yaml             ← Feast project config
│   └── features.py                    ← entity + feature view definitions
├── configs/
│   └── params.yaml                    ← central parameter file (DVC params)
├── dvc.yaml                           ← DVC stage pipeline
├── requirements.txt
└── README.md                          ← this file
```

---

## 1 · Data Ingestion & Raw Storage

**Tool:** ZenML step wrapping TFX ExampleGen logic | **Points: 5**

Raw data is ingested from the Agile user story dataset CSV (23,313 issues, 16 OSS projects). The ingestion step normalises heterogeneous column names across project repositories (e.g., `point` → `storypoints`, `concat` → `description`) to a canonical schema, drops rows with missing or non-positive story points, and saves a versioned raw snapshot to disk.

**Implementation:** [`pipeline/ingestion.py`](pipeline/ingestion.py)

```python
@step
def ingest_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)        # handle column aliases across projects
    df = df.dropna(subset=["storypoints"])
    df = df[df["storypoints"] > 0]
    df.to_csv("data/raw/raw_data.csv", index=False)
    return df
```

The raw snapshot is tracked by DVC — see [`data/raw/raw_data.csv.dvc`](data/raw/raw_data.csv.dvc).

**Column normalisation map:**

| Raw column name | Canonical name |
|---|---|
| `point`, `story_points`, `storyPoint` | `storypoints` |
| `concat`, `body` | `description` |
| `summary` | `title` |

---

## 2 · Schema Definition

**Tool:** TensorFlow Data Validation (TFDV) | **Points: 2**

The data schema is automatically inferred from training-split statistics using TFDV's `infer_schema()`. The schema captures:

- **Feature types** for each column (string, float, int)
- **Value domain** bounds for the `storypoints` regression target
- **Presence constraints** (min/max fraction of non-missing values) for text fields
- **Cardinality** of categorical fields (e.g., project name)

**Saved schema:** [`schema/schema.pbtxt`](schema/schema.pbtxt)

```python
train_stats = tfdv.generate_statistics_from_dataframe(train_df)
schema      = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema)
tfdv.write_schema_text(schema, "schema/schema.pbtxt")
```

The schema is the single source of truth for data contracts across training, evaluation, and inference. It is also tracked via DVC so any schema revision is versioned alongside the data.

---

## 3 · Data Validation & Anomaly Detection

**Tool:** TensorFlow Data Validation (TFDV) | **Points: 3**

TFDV validates the evaluation split against the training schema and detects anomalies. The full validation workflow is:

### 3.1 Statistics Visualisation

Training and evaluation statistics are compared side by side using `tfdv.visualize_statistics()`. This surfaces distribution skew between splits — particularly important for `storypoints`, whose distribution varies across the 16 JIRA projects (see [`tfdv_output/storypoints_distribution.png`](tfdv_output/storypoints_distribution.png)).

### 3.2 Anomaly Detection

```python
anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)
tfdv.display_anomalies(anomalies)
```

Typical anomalies found in this dataset include:

| Anomaly type | Affected feature | Cause |
|---|---|---|
| Domain value out of range | `storypoints` | Eval projects use larger point scales |
| Missing value ratio exceeded | `description` | Some JIRA issues have no body text |
| String value not in domain | `project` | Eval projects not seen in training schema |

The full anomaly report is saved to [`tfdv_output/anomalies_report.json`](tfdv_output/anomalies_report.json).

### 3.3 Schema Revision

Anomalies are resolved by relaxing domain constraints rather than discarding valid data:

```python
# Relax storypoints range to accommodate all 16 projects
for feature in schema.feature:
    if feature.name == "storypoints":
        feature.float_domain.min = 0.0
        feature.float_domain.max = 200.0
    if feature.name in ("title", "description"):
        feature.presence.min_fraction = 0.5  # allow partial absence
```

The revised schema is saved back to [`schema/schema.pbtxt`](schema/schema.pbtxt).

**Full notebook walkthrough:** [`notebooks/milestone3_pipeline.ipynb`](notebooks/milestone3_pipeline.ipynb) — Section 2.

---

## 4 · Preprocessing & Feature Engineering

**Tool:** ZenML Transform step (pandas + custom NLP) | **Points: 5**

The preprocessing step applies a multi-stage cleaning and feature engineering pipeline adapted to the characteristics of JIRA user story text.

**Implementation:** [`pipeline/transform.py`](pipeline/transform.py)

### 4.1 Text Cleaning

Raw text undergoes four cleaning passes before being fed to the Llama tokenizer:

| Step | Operation | Rationale |
|---|---|---|
| Lowercasing | `text.lower()` | Normalise case for LLM tokenizer |
| URL removal | regex `https?://\S+` | URLs carry no semantic effort signal |
| JIRA markup removal | `{code}`, `[~user]`, `!image!` | Artefacts of JIRA export format |
| Special character removal | keep `[a-z0-9\s.,!?-]` | Reduce vocabulary noise |
| Whitespace collapse | `\s+` → single space | Canonical token spacing |

### 4.2 Engineered Features

| Feature | Type | Description |
|---|---|---|
| `input_text` | string | `"Title: <t> Description: <d>"` — primary LLM input |
| `cleaned_title` | string | Cleaned title only |
| `cleaned_description` | string | Cleaned description only |
| `text_length` | int | Character count of `input_text` |
| `word_count` | int | Word count of `input_text` |
| `title_word_count` | int | Word count of cleaned title |
| `has_description` | int (0/1) | Flag: non-empty description present |
| `log_storypoints` | float | `log1p(storypoints)` — reduces right skew for regression |
| `is_fibonacci` | int (0/1) | Flag: story point is a standard Fibonacci planning value |

The `input_text` format (`"Title: … Description: …"`) is consistent with the prompt structure used in the Llama3SP paper, ensuring continuity with the Milestone 2 inference pipeline.

**Distribution plots:** [`tfdv_output/feature_distributions.png`](tfdv_output/feature_distributions.png)

### 4.3 Output

Processed data is exported as Parquet for efficient downstream loading:

```
data/processed/processed_data.parquet   (DVC tracked)
```

---

## 5 · Feature Store (Feast)

**Tool:** Feast (local FileSource, offline mode) | **Points: 1**

Engineered features are registered in a Feast feature store, enabling consistent feature retrieval across training, evaluation, and future inference requests — decoupling feature computation from model training.

**Implementation:** [`pipeline/feature_store.py`](pipeline/feature_store.py)

**Feature view definition:** [`feast_repo/feature_repo/features.py`](feast_repo/feature_repo/features.py)

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

To apply the feature store:
```bash
cd feast_repo/feature_repo
feast apply
```

For production deployments, the `FileSource` can be replaced with a BigQuery or Redis online store with no changes to the feature view definitions.

---

## 6 · Data Versioning (DVC)

**Tool:** DVC (Data Version Control) | **Points: 3**

All data artifacts produced by the pipeline are tracked using DVC. This ensures that every experiment can be reproduced by checking out the corresponding DVC pointer files alongside the code.

**DVC pipeline config:** [`dvc.yaml`](dvc.yaml)
**Parameter file:** [`configs/params.yaml`](configs/params.yaml)

### Setup & Commands

```bash
# Initialise DVC (first time only)
dvc init

# Track raw data
dvc add data/raw/raw_data.csv
git add data/raw/raw_data.csv.dvc data/raw/.gitignore
git commit -m "feat: track raw data with DVC [milestone3]"
git tag -a v1.0-raw -m "Milestone 3: initial raw data ingestion"

# Track processed data
dvc add data/processed/processed_data.parquet
git add data/processed/processed_data.parquet.dvc
git commit -m "feat: add processed feature-engineered data [milestone3]"
git tag -a v1.0-processed -m "Milestone 3: processed Parquet after feature engineering"

# Track schema
dvc add schema/schema.pbtxt
git commit -m "feat: add TFDV inferred+revised schema [milestone3]"

# Reproduce the full pipeline
dvc repro

# Push data to remote (configure remote first)
dvc remote add -d myremote gdrive://<folder-id>
dvc push
```

### Versioned Artifacts

| Artifact | DVC file | Git tag |
|---|---|---|
| `data/raw/raw_data.csv` | `data/raw/raw_data.csv.dvc` | `v1.0-raw` |
| `data/processed/processed_data.parquet` | `data/processed/processed_data.parquet.dvc` | `v1.0-processed` |
| `schema/schema.pbtxt` | `schema/schema.pbtxt.dvc` | — |

---

## 7 · ML Pipeline Integration (ZenML)

**Tool:** ZenML | **Points: 5 (ingestion) + pipeline**

All four steps are wired into a single ZenML pipeline, integrating this milestone's data work into the larger MLOps platform that will support model training, experiment tracking, and deployment in subsequent milestones.

**Implementation:** [`pipeline/zenml_pipeline.py`](pipeline/zenml_pipeline.py)

```python
@pipeline(name="milestone3_data_pipeline", enable_cache=True)
def data_pipeline(csv_path: str = "data/raw/raw_data.csv") -> None:
    raw_df       = ingest_data(csv_path=csv_path)
    val_report   = validate_data(df=raw_df)
    processed_df = preprocess_and_engineer(df=raw_df)
    store_info   = setup_feature_store(df=processed_df)
```

### Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run via Python
python pipeline/zenml_pipeline.py --csv-path data/raw/raw_data.csv

# Or via ZenML CLI
zenml pipeline run pipeline/zenml_pipeline.py:data_pipeline

# Launch ZenML dashboard to inspect artifacts
zenml up
```

ZenML's caching (`enable_cache=True`) ensures that unchanged steps are not re-executed, reducing runtime when iterating on downstream steps. Each step produces a tracked artifact (DataFrames, dicts) stored in the ZenML artifact store, providing full lineage from raw CSV to Feast-registered features.

### Pipeline DAG

```
ingest_data
    │
    ▼
validate_data          (TFDV schema + anomaly detection)
    │
    ▼
preprocess_and_engineer (text cleaning + feature engineering)
    │
    ▼
setup_feature_store    (Feast feature view registration)
```

---

## Grading Checklist

| Requirement | Tool used | Points | Deliverable |
|---|---|---|---|
| Data management – Schema definition | TFDV `infer_schema` | 2 | [`schema/schema.pbtxt`](schema/schema.pbtxt) |
| Data Validation & Verification (stats, anomaly detection, anomaly fix) | TFDV | 3 | [`tfdv_output/`](tfdv_output/) · [`notebooks/milestone3_pipeline.ipynb`](notebooks/milestone3_pipeline.ipynb) |
| Data versioning | DVC | 3 | [`dvc.yaml`](dvc.yaml) · `.dvc` pointer files · git tags |
| Feature store | Feast | 1 | [`feast_repo/feature_repo/features.py`](feast_repo/feature_repo/features.py) |
| Ingestion of raw data & storage into repository | ZenML + TFX ExampleGen | 5 | [`pipeline/ingestion.py`](pipeline/ingestion.py) · [`data/raw/raw_data.csv.dvc`](data/raw/raw_data.csv.dvc) |
| Preprocessing & Feature Engineering | ZenML Transform | 5 | [`pipeline/transform.py`](pipeline/transform.py) · [`data/processed/processed_data.parquet`](data/processed/processed_data.parquet) |
| Setup data pipeline within larger ML pipeline | ZenML | (above) | [`pipeline/zenml_pipeline.py`](pipeline/zenml_pipeline.py) |
| **Total** | | **19** | |

---

## References

1. Choetkiertikul, M., Dam, H. K., Tran, T., Treude, C., & Ghose, A. (2018). *A deep learning model for estimating story points.* IEEE TSE. https://doi.org/10.1109/TSE.2018.2792247
2. Mittal, H., Arsalan, M., & Garg, P. (2024). *Story point estimation using deep learning: an empirical re-evaluation.*
3. Fu, M., & Tantithamthavorn, C. (2022). *GPT2SP: A Transformer-based Agile story point estimation approach.* IEEE TSE. https://doi.org/10.1109/TSE.2022.3160829
4. Sepúlveda Montoya, C., Ríos Gómez, J., & Jaramillo Villegas, J. A. (2025). *Llama3SP: Resource-efficient LLM for Agile story point estimation.* https://github.com/DEVCamiloSepulveda/llama3sp
5. TensorFlow Data Validation. https://www.tensorflow.org/tfx/data_validation/get_started
6. ZenML Documentation. https://docs.zenml.io
7. Feast Feature Store. https://docs.feast.dev
8. DVC Documentation. https://dvc.org/doc
