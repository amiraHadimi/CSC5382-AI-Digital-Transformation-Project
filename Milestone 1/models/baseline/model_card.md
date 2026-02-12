# Model Card — Llama3SP Baseline (Latest Reproducible Model)

## Model Details
- **Model name:** `llama3sp_storypoints_baseline`
- **Model family:** Llama3SP (Llama 3.2 + PEFT / LoRA)
- **Task:** Story point estimation (text regression)
- **Baseline status:** Latest reproducible baseline satisfying assignment constraints
- **Intended use:** Decision-support tool for Agile backlog refinement and sprint planning
- **Version:** v1.0 (Milestone 1 baseline)

This model represents the **most recent publicly available and fully reproducible baseline** for automated story point estimation at the time of this project.

---

## Upstream Artifacts (Binary + Retraining Notebook)

This baseline is selected specifically because **both required artifacts are publicly available**:

- **Trained model binary:**  
  Example checkpoint published on Hugging Face, including:
  - `adapter_model.safetensors`
  - `pytorch_model.bin`  
  https://huggingface.co/DEVCamiloSepulveda/0-LLAMA3SP-usergrid

- **Retraining notebooks:**  
  Provided in the replication package:
  - `model_training.ipynb`
  - `tokenizer_training_notebook.ipynb`
  - `model_upload_notebook.ipynb`  
  https://github.com/DEVCamiloSepulveda/llama3sp


---

## Training Data
- **Dataset:** Agile Story Point Estimation benchmark (Deep-SE lineage)
- **Size:** 23,313 user stories / issues
- **Projects:** 16 open-source Agile projects (JIRA)
- **Input fields:** Issue title and description
- **Target:** Story points (continuous numeric value)
- **Splits:** Train / validation / test splits provided via a `split_mark` column in the replication package

---

## Input and Output
- **Input:** Natural-language issue text (title + description)
- **Output:** Continuous numeric prediction (story points)

---

## Training Setup (High-Level)
- **Base model:** Llama 3.2
- **Adaptation:** Parameter-Efficient Fine-Tuning (LoRA / PEFT)
- **Objective:** Regression with scalar output (`num_labels = 1`)
- **Tokenization:** Llama tokenizer (as provided in replication notebooks)

---

## Evaluation Metrics
The baseline is evaluated using metrics commonly reported in the literature:

- **MAE (Mean Absolute Error)** – primary metric
- **RMSE (Root Mean Squared Error)**
- **Tolerance-based accuracy (±1 story point)**

Example checkpoint reports (from model card):
- MAE ≈ 1.765  
- MdAE ≈ 1.583  

> For coursework evaluation, metrics should be recomputed on the chosen dataset split to ensure consistency.

---

## Intended Use
This model is intended to:
- Support Agile teams during backlog refinement
- Provide consistent, data-driven initial estimates
- Serve as an input to downstream planning and optimization tasks

Predictions are **not** intended to replace human judgment.

---

## Limitations
- Story points are **team-specific** and may not generalize perfectly across projects
- Does not incorporate contextual features (velocity, team size, sprint capacity)


---

## Ethical Considerations
- Must not be used to evaluate individual developer performance
- Should be treated as a **decision-support tool**, not an automated decision-maker

---

## Reproducibility Checklist
- [x] Trained model binary publicly available  
- [x] Retraining notebook publicly available  
- [x] Dataset and splits documented  
- [x] License requirements documented
