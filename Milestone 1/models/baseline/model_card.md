# Model Card: TF-IDF + Ridge Regression Baseline (Story Point Estimation)

## Model Details
- **Model name:** storypoint-tfidf-ridge-baseline
- **Version:** 1.0 (Milestone 1 baseline)
- **Model type:** Regression (Ridge Regression)
- **Input:** User story text (`title + description`)
- **Output:** Predicted story points (continuous numeric value)
- **Intended use:** Decision-support for Agile planning (estimation + optimization pipeline input)

## Motivation
This baseline provides a **transparent, lightweight, reproducible** starting point for story point estimation. It establishes a reference for comparison with later deep learning and LLM-based models, and supports the project’s longer-term objective of using predictions as inputs to solver-backed optimization.

## Training Data
- **Dataset:** Agile User Story Point Estimation dataset (public)
- **Source:** https://github.com/mrthlinh/Agile-User-Story-Point-Estimation/blob/master/data_csv/data
- **Size:** 23,313 user stories/issues
- **Fields used:**
  - `title` (text)
  - `description` (text)
  - `story_points` (numeric target)

## Preprocessing
- Concatenate `title` and `description`
- TF-IDF vectorization:
  - Recommended settings (baseline-friendly):
    - `ngram_range=(1,2)`
    - `min_df` (e.g., 2–5)
    - `max_features` (optional, e.g., 50k)
- Basic cleanup:
  - lowercasing
  - whitespace normalization

*(Exact hyperparameters are recorded in the training notebook for full reproducibility.)*

## Model Architecture
- **Vectorizer:** TF-IDF (sparse bag-of-words representation)
- **Regressor:** Ridge Regression (L2 regularization)

Why Ridge?
- More stable than unregularized linear regression for high-dimensional sparse text features.

## Training Procedure
- **Notebook:** `notebooks/baseline_retrain.ipynb`
- **Train/validation/test split:** documented and executed in the notebook  
- **Artifact export:** model is saved after training (e.g., joblib)

## Evaluation
### Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Accuracy@±1 story point

### Intended interpretation
- Lower MAE/RMSE means better estimation accuracy
- Accuracy@±1 reflects “planning tolerance” (near-miss estimates that may still be usable in sprint planning)

## Intended Use
### Primary use cases
- Quick and reproducible baseline for story point estimation
- Benchmark for comparison against improved models (DL / LLM)
- Input signal for later milestones’ optimization formulation (e.g., sprint backlog selection)

### Out-of-scope uses
- Replacing team judgment entirely
- High-stakes commitments without human review
- Estimating story points for domains/projects with drastically different conventions without recalibration

## Limitations
- TF-IDF ignores deep semantics and long-range context in text
- Story point scales vary across teams/projects (cross-project generalization may be limited)
- Continuous regression may output non-integer values (rounding strategies should be evaluated carefully)
- Sensitive to dataset bias (open-source JIRA projects may not represent all Agile environments)

## Ethical / Responsible AI Considerations
- This model may encode biases present in historical estimation behavior
- Should be used as a **support tool**, not an automatic authority
- Human-in-the-loop review is recommended, especially for high-uncertainty stories

## Reproducibility
- Training and evaluation are fully reproducible using:
  - `notebooks/baseline_retrain.ipynb`
- Model artifact produced by notebook:
  - `models/baseline/baseline.joblib` (generated)

## Citation
If you use this baseline in academic work, cite:
- Dataset/origin: Choetkiertikul et al. (2016)
- Baseline implementation: this repository’s Milestone 1
