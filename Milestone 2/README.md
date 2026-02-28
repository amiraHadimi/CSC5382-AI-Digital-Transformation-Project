# Milestone 2 – Model Integration & Proof of Concept (PoC)

## Overview

This milestone implements a Proof of Concept (PoC) for automated story point estimation using a pre-trained Large Language Model (LLM) combined with project-specific LoRA adapters (Llama3SP framework).

The objectives of this milestone are:

- Integrate a pre-trained LLM regression model
- Evaluate performance per project using Mean Absolute Error (MAE)
- Develop a Streamlit-based interactive application
- Demonstrate a complete end-to-end inference and evaluation pipeline

---

## 1. Model Integration

### 1.1 Base Model

The system uses:

- **meta-llama/Llama-3.2-1B** as the base model  
- HuggingFace Transformers for model loading  
- PEFT (LoRA) for lightweight adapter integration  

The model is configured for regression:

- `num_labels = 1`
- `problem_type = "regression"`

This enables the model to output continuous story point predictions.

---

### 1.2 Adapter-Based Specialization

Each open-source project has a dedicated LoRA adapter.

During evaluation, the system:

1. Loads the base Llama model  
2. Dynamically attaches the project-specific LoRA adapter  
3. Runs regression inference on issue titles  

This allows efficient specialization without retraining the full base model.

Implementation files:

- `eval_mae_per_project.py`
- `test_llama3sp.py`

---

## 2. Dataset

Evaluation was conducted on **16 open-source projects** from the Llama3SP dataset:
Milestone 2/data/llama3sp_dataset/

Each dataset contains a `split_mark` column specifying:

- `train`
- `validation`
- `test`

Only rows marked as **test** were used for evaluation to ensure proper separation between training and testing data.

For computational efficiency, evaluation was limited to **a maximum of 100 test samples per project**.  
If fewer than 100 test samples were available, all test samples were used.

---

## 3. Evaluation

### 3.1 Metric

Performance is measured using **Mean Absolute Error (MAE)**:

MAE = (1/N) × Σ |y_true − y_pred|

This metric quantifies the average absolute difference between predicted and true story points.

---

### 3.2 Results

Per-project MAE results are stored in:
Milestone 2/results/mae_per_project.csv

A summary file is available at:
Milestone 2/results/summary.json

Individual prediction files are stored as:
Milestone 2/results/summary.json


Each prediction file contains:

- `y_true`
- `y_pred`
- `abs_error`

These files provide full transparency of model performance.

---

## 4. Streamlit Proof of Concept

A Streamlit web application was developed to demonstrate real-time inference.

Application file:
Milestone 2/app/streamlit_app.py

The application allows the user to:

- Select a project
- Enter an issue title
- Generate a predicted story point

### Run locally:

```bash
cd "Milestone 2"
streamlit run app/streamlit_app.py

## 5. End-to-End Scenario

The complete pipeline works as follows:

1. User inputs an issue title.
2. Text is tokenized using the Llama tokenizer.
3. The base model processes the tokenized input.
4. The project-specific LoRA adapter refines the prediction.
5. The regression head outputs a continuous story point value.
6. The evaluation script computes MAE on the test data.
7. Results are saved in the `/results` directory.

This confirms a fully functional end-to-end machine learning workflow.

---

## 6. Limitations

- Evaluation limited to 100 test samples per project (runtime constraint).
- CPU-only inference.
- No additional fine-tuning performed in this milestone.
- Performance varies across projects (e.g., higher MAE observed for Moodle).

---

## 7. Deliverables

This milestone includes:

- Model integration scripts  
- Evaluation pipeline  
- Streamlit application  
- Per-project MAE results  
- Word report (PDF)  
- Recorded presentation  
