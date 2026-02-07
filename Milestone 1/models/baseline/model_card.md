<!-- models/baseline/model_card.md -->

# Model Card — TF-IDF + Ridge Baseline (Story Point Estimation)

## Model Details
- **Model name:** tfidf_ridge_storypoints_baseline
- **Model type:** Text regression (TF-IDF + Ridge Regression)
- **Intended use:** Decision-support baseline for predicting story points from user story text (title + description)
- **Primary users:** Agile teams / researchers building estimation and planning decision-support tools
- **Version:** 1.0 (Milestone 1 baseline)

## Training Data
- **Dataset:** Agile User Story Point Estimation (public)
- **Source:** https://github.com/mrthlinh/Agile-User-Story-Point-Estimation/blob/master/data_csv/data
- **Size:** 23,313 issues/user stories (16 open-source projects collected via JIRA)
- **Fields used:** `title`, `description` → input text; `story_points` → numeric target

## Input / Output
- **Input:** concatenated text = `title + "\n" + description`
- **Output:** predicted story points (continuous numeric value)

## Training Procedure
- **Text preprocessing:** basic cleaning (as implemented in `notebooks/baseline_retrain.ipynb`)
- **Vectorization:** TF-IDF (word n-grams as configured in the notebook)
- **Regressor:** Ridge Regression (L2 regularization)
- **Train/validation/test split:** documented and implemented in the notebook (seeded for reproducibility)

## Evaluation
Metrics reported by the notebook:
- **MAE** (primary)
- **RMSE**
- **Accuracy@±1 story point**

## Intended Use
This baseline is meant to:
- provide a **reproducible reference point** for later deep learning / LLM approaches
- support later milestones where predicted story points become inputs to optimization (e.g., sprint backlog selection)

## Limitations
- Predicts story points purely from text; does not incorporate additional signals (assignee, team, priority, history).
- Story point scales can differ by project/team; no normalization is applied in this milestone.
- Linear model may underfit complex semantic patterns.

## Ethical / Responsible Use Notes
- Predictions should **assist**, not replace, human planning decisions.
- Results may reflect biases or inconsistencies present in historical labeling practices.

## How to Reproduce
1. Open: `notebooks/baseline_retrain.ipynb`
2. Install dependencies from `requirements.txt` (if provided)
3. Run all cells to:
   - train the model
   - export `models/baseline/baseline.joblib`
   - generate evaluation outputs

## Model Artifact
- **Binary:** `models/baseline/baseline.joblib`
- **Training notebook:** `notebooks/baseline_retrain.ipynb`
