# Milestone 1 – Project Inception  
## AI-Based Story Point Estimation for Optimization-Driven Agile Planning

> **Repository goal (Milestone 1):** frame the business idea as an ML problem and demonstrate feasibility with a **reproducible baseline** (retrain notebook + trained model artifact + model card).

---

## 1) Business Case Description (0.5)
Accurate effort estimation is central to Agile development because it directly affects sprint planning, resource allocation, and delivery predictability. In practice, teams estimate effort using **story points** (e.g., planning poker). While effective, this process is often **subjective**, **time-consuming**, and **inconsistent** across teams and sprints.

This project aims to **automatically estimate story points from user story text** (title + description) using machine learning. The system is designed as a **decision-support tool** that assists Agile teams during backlog refinement and sprint planning by providing consistent, data-driven estimates.

---

## 2) Business Value of Using ML (Impact) (0.5)
Story point estimation depends on variable natural-language descriptions and historical estimation behavior. Rule-based approaches cannot robustly capture this semantic variability.

Machine learning enables:
- **Consistency:** reduces estimation variability across time and team members  
- **Efficiency:** speeds up grooming and planning by reducing manual estimation effort  
- **Scalability:** supports estimation across multiple projects/teams  
- **Decision support:** helps less-experienced teams calibrate estimates using historical data

---

## 3) Dataset Overview (0.5)
**Source (public):**  
- Agile User Story Point Estimation dataset (CSV):  
  https://github.com/mrthlinh/Agile-User-Story-Point-Estimation/blob/master/data_csv/data  

**Origin:**  
- Introduced by **Choetkiertikul et al.** (*A Deep Learning Model for Estimating Story Points*), collected from JIRA projects.

**Size & coverage:**  
- **23,313** user stories/issues from **16** open-source Agile projects (across multiple repositories).

**Main fields used (this milestone):**
- `title` (text)  
- `description` (text)  
- `story_points` (numeric label)

**Label strategy (this milestone):**  
Story points are modeled as a **continuous numeric target**. Because projects may use different internal scales, labels are **not normalized** in Milestone 1 (normalization is treated as a later experiment).

**Repository assets:**
- Sample for quick inspection: [`data/sample.csv`](data/sample.csv)  
- Full dataset is referenced via the public source link above.

---

## 4) Project Archetype (0.5)
This project is framed as a **decision-support ML system embedded within an optimization-driven Agile planning workflow**.

### 4.1 Predictive component (ML)
Given a user story’s `title + description`, the model predicts its **story points**.

### 4.2 Optimization component (the “real” business decision)
In Agile planning, teams repeatedly solve constrained decision problems such as **sprint backlog selection**:

- **Decision variables:** select which stories to include in the next sprint (and optionally assignment/order)
- **Constraints:** capacity/velocity, dependencies, deadlines, WIP limits, priorities, team availability
- **Objective:** maximize delivered value (or minimize risk/penalties) under constraints

A simple formalization is a **knapsack-style** problem:
- choose stories \(x_i \in \{0,1\}\)
- capacity constraint \(\sum x_i \cdot \hat{sp}_i \le \text{capacity}\)
- maximize value \(\sum x_i \cdot value_i\)

Here, **estimated story points** \(\hat{sp}_i\) are inputs to the optimization, not the final output.

### 4.3 Why “code generation” matters
The optimization constraints typically **do not live inside the story-point dataset**. They come from planning context (capacity notes, dependencies, policies, sprint goals, etc.). In later milestones, an LLM will be used to:
1) extract variables / constraints / objective from natural-language planning context, and  
2) generate **solver-ready symbolic code** (e.g., OR-Tools / Pyomo / MiniZinc / CP-SAT model).

**Milestone 1 scope:** validate feasibility using a reproducible story point estimation baseline only.  
**Later milestones:** integrate solver-based optimization and LLM-generated symbolic formulations.

---

## 5) Feasibility Analysis – Literature Review (2.0)
Prior research shows that predicting story points from Agile text is feasible on public benchmarks and that more advanced architectures (deep learning and LLM-based approaches) can improve accuracy. The works below are selected to (1) establish the benchmark dataset and task feasibility, (2) show a deep-learning refinement on the same benchmark, and (3) show an LLM-based estimator that reports competitive results while providing code/models for reuse.

### Table 1 — Representative studies on automated story point estimation

| ID | Reference | Model / Technique | Dataset | Evaluation Metrics | Key Contribution |
|---|---|---|---|---|---|
| 1 | Choetkiertikul et al. | **LD-RNN** (LSTM + Recurrent Highway Network) | 23,313 stories, 16 OSS projects (JIRA) | MAE, SA | Introduced the benchmark dataset and showed deep learning outperforms common baselines |
| 2 | Mittal, Arsalan, Garg | Novel deep learning model (LD-RNN-style) | 23,313 issues, 16 OSS projects (JIRA) | MAE, SA (also reports comparisons vs Deep-SE) | Confirms feasibility and reports improved performance comparisons on the same benchmark family |
| 3 | Sepúlveda Montoya, Ríos Gómez, Jaramillo Villegas | **Llama3SP** (fine-tuned LLaMA-based model) | Public DeepSE datasets / story-point benchmark lineage | MAE, RMSE, tolerance-based accuracy (reported) | Shows a resource-efficient LLM estimator with released code/models for reuse |

> Note: The optimization/code-generation framework paper is **not** listed here because this table is restricted to **story point estimation** literature (estimation feasibility + progression).

---

## 6) Feasibility + Baseline Specification (0.5)

### 6.1 Baseline choice (what & why)
To validate feasibility with a transparent and reproducible starting point, this milestone uses a **classical NLP regression baseline**:

- **Text representation:** TF-IDF over concatenated `title + description`  
- **Regressor:** **Ridge Regression** (L2-regularized linear regression)

**Why Ridge (instead of plain Linear Regression):**
- Stable on high-dimensional sparse TF-IDF features
- Standard, widely accepted baseline for text regression
- Fast to train, easy to reproduce and interpret

### 6.2 Reproducibility (required by the milestone)
This repository includes:
- Retraining notebook: [`notebooks/baseline_retrain.ipynb`](notebooks/baseline_retrain.ipynb)  
- Trained model artifact (generated by the notebook): [`models/baseline/baseline.joblib`](models/baseline/baseline.joblib)  
- Model card (HuggingFace-inspired): [`models/baseline/model_card.md`](models/baseline/model_card.md)

---

## 7) Metrics for Business Goal Evaluation (0.5)
The baseline is evaluated using:
- **MAE (Mean Absolute Error):** primary metric (interpretable in story points)
- **RMSE (Root Mean Squared Error):** penalizes large errors more heavily
- **Accuracy@±1 story point:** tolerance-based metric aligned with planning practicality

---

## 8) Repository Structure / Assets
- `data/`  
  - `sample.csv` (small sample for quick inspection)  
- `notebooks/`  
  - `baseline_retrain.ipynb`  
- `models/baseline/`  
  - `baseline.joblib` (generated after training)  
  - `model_card.md`  
- `references/`  
  - `references.md`  
- `presentation/`  
  - recorded presentation (≤ 7 minutes)

---

## 9) References
See: [`references/references.md`](references/references.md)
