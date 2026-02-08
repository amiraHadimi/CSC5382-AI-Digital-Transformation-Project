# Model Card — GPT2SP Baseline (Story Point Estimation)

## Model Details
- Model name: gpt2sp_storypoints_baseline
- Baseline family: GPT2SP (Transformer / GPT-2–based story point estimator)
- Model type: Text regression (story point prediction from issue text)
- Intended use: Decision-support baseline for estimating story points from user story / issue text (title + description)
- Primary users: Agile teams and researchers building estimation and planning decision-support tools
- Version: 1.0 (baseline selection – Milestone 1)

## Upstream Artifact Locations (Binary + Retraining Notebook)
This baseline is selected specifically because both required artifacts are publicly available:
- Trained model binaries: available via Hugging Face Model Hub (and mirrored via Google Drive) from the GPT2SP replication package.
  - Example model: https://huggingface.co/MickyMike/GPT2SP
- Retraining notebook(s): provided in the GPT2SP replication package:
  - model training: model_training_notebook.ipynb
  - tokenizer training: tokenizer_training_notebook.ipynb
  - repository: https://github.com/awsm-research/gpt2sp

## Training Data
- Dataset: Agile story point estimation benchmark (DeepSE lineage)
- Size & scope: 23,313 issues/user stories across 16 open-source projects collected via JIRA (as used in the benchmark literature)
- Input fields: title, description
- Target: story points (continuous numeric value)

## Input / Output
- Input: concatenated text formed as title + description
- Output: predicted story points (continuous numeric value)

## Training Specification (High-level)
The full end-to-end retraining procedure is documented in the upstream GPT2SP training notebook(s). At a high level:
- Tokenization: GPT-2 BPE (and experiment-specific tokenizers in GPT2SP)
- Model: GPT-2–based Transformer architecture fine-tuned for story point regression
- Splits: train/validation/test based on the dataset’s provided split markers (benchmark convention)
- Output artifacts: trained weights and tokenizer artifacts published for reuse

## Evaluation Metrics
When evaluated, this baseline is intended to be assessed using:
- MAE (Mean Absolute Error) – primary metric
- RMSE (Root Mean Squared Error)
- Accuracy@±1 story point – tolerance-based metric aligned with Agile planning practice

## Intended Use
This baseline serves as a reproducible reference model to:
- establish feasibility for automated story point estimation, and
- provide a comparison point for subsequent baselines (e.g., Llama3SP) and for later milestones integrating optimization (e.g., sprint backlog selection).

Predicted story points are intended to be used as inputs to downstream optimization tasks.

## Limitations
- Predictions are based on text only; does not incorporate team context (priority, assignee, velocity, sprint capacity).
- Story point scales can vary across projects; cross-project generalization may be limited.
- As a Transformer-based model, inference/training is heavier than classical ML baselines (TF-IDF + linear models).

## Ethical and Responsible Use
- Predictions should support, not replace, human planning decisions.
- Story points should not be used to evaluate individual developer performance.
- Historical labeling may encode team-specific biases or inconsistent estimation practices.

## Reproducibility Checklist
To verify baseline reproducibility for grading:
1. Confirm pretrained weights can be accessed from the Hugging Face model page above.
2. Confirm retraining notebooks exist in the GPT2SP replication package.
3. (Optional) Run the upstream training notebook to reproduce training artifacts.

## Local Repository Notes
This repository documents the baseline selection in Milestone 1. In later milestones, we will add:
- wrapper notebook to load GPT2SP weights and run inference on the chosen dataset split
- training/evaluation runs and recorded metrics
- integration of estimates into a formal sprint-planning optimization model
