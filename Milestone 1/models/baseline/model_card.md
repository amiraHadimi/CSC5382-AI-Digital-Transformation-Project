# Model Card — TF-IDF + Ridge Baseline (Story Point Estimation)

## Model Details
- **Model name:** tfidf_ridge_storypoints_baseline
- **Model type:** Text regression (TF-IDF + Ridge Regression)
- **Intended use:** Decision-support baseline for estimating story points from user story text (title and description)
- **Primary users:** Agile teams and researchers building estimation and planning decision-support tools
- **Version:** 1.0 (baseline specification – Milestone 1)

## Training Data
- **Dataset:** Agile User Story Point Estimation (public)
- **Source:** https://github.com/mrthlinh/Agile-User-Story-Point-Estimation/blob/master/data_csv/data
- **Size:** 23,313 user stories/issues from 16 open-source Agile projects collected via JIRA
- **Fields used:**  
  - Input: `title`, `description`  
  - Target: `story_points` (numeric)

## Input / Output
- **Input:** Concatenated text formed as `title + description`
- **Output:** Predicted story points as a continuous numeric value

## Training Specification
This baseline model is specified to be trained using the following procedure, which is fully documented for reproducibility:
- **Text preprocessing:** Basic text cleaning and normalization
- **Vectorization:** TF-IDF over word n-grams (configuration specified in the retraining notebook)
- **Regressor:** Ridge Regression with L2 regularization
- **Data split:** Train/validation/test split with fixed random seed

The complete retraining procedure is provided in:
- `notebooks/baseline_retrain.ipynb`

## Evaluation Metrics
When trained, the model is intended to be evaluated using:
- **MAE (Mean Absolute Error)** – primary metric
- **RMSE (Root Mean Squared Error)**
- **Accuracy@±1 story point** – tolerance-based metric aligned with Agile planning practice

## Intended Use
This baseline serves as a **transparent and reproducible reference model** for story point estimation. It establishes feasibility and provides a comparison point for more advanced deep learning and large language model–based approaches explored in later milestones.

Predicted story points from this model are intended to be used as **inputs to downstream optimization tasks**, such as sprint backlog selection under capacity constraints.

## Limitations
- Uses only textual information and does not incorporate additional contextual signals (e.g., team, priority, historical velocity).
- Story point scales may vary across projects; no normalization is applied at this stage.
- Linear regression may underfit complex semantic relationships present in natural-language descriptions.

## Ethical and Responsible Use
- Predictions are intended to **support**, not replace, human decision-making during Agile planning.
- The model may reflect biases and inconsistencies present in historical estimation practices.

## Reproducibility
The baseline is designed to be fully reproducible:
1. Open `notebooks/baseline_retrain.ipynb`
2. Install required dependencies
3. Run the notebook to train the model, evaluate performance, and export the trained model artifact

## Artifacts
- **Training notebook:** `notebooks/baseline_retrain.ipynb`
- **Model artifact location:** `models/baseline/` (generated in later milestones)
