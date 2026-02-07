# Milestone 1 – Project Inception  
## AI-Based Story Point Estimation for Optimization-Driven Agile Planning

---

## 1. Business Case Description
Accurate effort estimation is central to Agile software development, as it directly affects sprint planning, resource allocation, and delivery predictability. In practice, teams estimate effort using story points (e.g., planning poker). While effective, these processes are subjective, time-consuming, and often inconsistent across teams and sprints.

This project aims to **automatically estimate story points from user story text** (title and description) using machine learning. The system is designed as a **decision-support tool** that assists Agile teams during backlog refinement and sprint planning by providing consistent, data-driven estimates.

---

## 2. Business Value of Using Machine Learning
Story point estimation depends on complex and variable natural-language descriptions and team-specific historical patterns. Rule-based approaches cannot robustly capture this semantic variability.

Machine learning enables:
- **Consistency** by reducing estimation variability across time and team members  
- **Efficiency** by reducing manual effort during planning and grooming  
- **Scalability** across multiple projects and teams  
- **Decision support** for less-experienced teams using historical data  

---

## 3. Dataset Overview
**Dataset:** Agile User Story Point Estimation  
**Source:**  
https://github.com/mrthlinh/Agile-User-Story-Point-Estimation/blob/master/data_csv/data  

**Origin:**  
Introduced by Choetkiertikul et al. (2016), collected from JIRA-based Agile projects.

**Size:**  
23,313 user stories/issues.

**Coverage:**  
16 large open-source Agile projects across 9 repositories (e.g., Apache, Moodle, Spring, Atlassian).

**Fields:**
- `title` (text)  
- `description` (text)  
- `story_points` (numeric label)

**Label strategy:**  
Story points are treated as a continuous numeric target. Due to heterogeneous estimation scales across projects, labels are not normalized in this milestone.

A small dataset sample is included in `data/sample.csv` for inspection. The full dataset is referenced via the public source.

---

## 4. Project Archetype
This project is framed as a **decision-support machine learning system embedded within an optimization-driven Agile planning workflow**. The predictive component estimates story points from natural-language user stories. These estimates are **not final outputs**, but serve as **quantitative inputs to downstream planning and optimization problems** in Agile software development.

Sprint planning and backlog refinement can be formulated as **constrained optimization problems**, where:
- **Decision variables** represent the selection and scheduling of user stories,
- **Constraints** encode sprint capacity, team velocity, dependencies, deadlines, and business rules,
- **Objectives** aim to maximize delivered business value or minimize delivery risk.

In alignment with the course requirement that the project address an optimization problem requiring an **intermediate symbolic representation through code generation**, this project adopts the methodological perspective of LLM-based planning frameworks (e.g., LLM-Based Formalized Programming). In later milestones, large language models will be used to extract decision variables, constraints, and objectives from Agile planning context and natural-language descriptions, and to generate solver-ready symbolic code.

The optimization constraints are **not derived from the estimation dataset itself**, but from Agile planning context (e.g., sprint descriptions, capacity limits, dependencies, and priorities).

**Milestone 1** focuses exclusively on feasibility validation through a reproducible story point estimation baseline. Optimization formulation and symbolic code generation are deferred to subsequent milestones.

---

## 5. Feasibility Analysis

## 5. Feasibility Analysis

### 5.1 Literature Review

Prior research demonstrates that automated story point estimation from Agile artifacts is a feasible and reproducible machine learning task. Early work established the foundations of the problem by introducing large benchmark datasets and showing that learning-based approaches can effectively model the relationship between natural-language user stories and effort estimates. Subsequent studies proposed refined deep learning architectures to improve estimation accuracy on the same datasets, confirming the robustness of the task across different modeling choices.

More recent research explores the use of large language models (LLMs) for story point estimation, indicating that task-specific and resource-efficient LLMs can achieve competitive performance compared to traditional deep learning models. Together, these studies provide strong evidence that story point estimation is a solvable problem and justify the use of a simple, reproducible baseline model in this milestone, which can later be compared against more advanced approaches.

---

### 5.2 Summary of Related Work

Table 1 summarizes representative studies on automated story point estimation. The selected works are chosen to demonstrate feasibility, methodological progression, and reproducibility on publicly available Agile datasets.

| ID | Reference | Model / Technique | Dataset | Evaluation Metrics | Key Contribution |
|---|---|---|---|---|---|
| Choetkiertikul et al. (2016) | *A Deep Learning Model for Estimating Story Points* | LD-RNN (LSTM + Recurrent Highway Network) | 23,313 user stories from 16 open-source Agile projects (JIRA) | MAE, Standardized Accuracy (SA) | Introduced a large public benchmark dataset and demonstrated that deep learning models significantly outperform traditional regression-based baselines |
| Mittal et al. (2023) | *A Novel Deep Learning Model for Effective Story Point Estimation in Agile Software Development* | Deep learning–based estimation model | Public Agile project data | MAE, RMSE | Proposed an alternative deep learning architecture and further validated the feasibility of learning-based story point estimation |
| Llama3SP (2025) | *Llama3SP: A Resource-Efficient Large Language Model for Story Point Estimation* | Fine-tuned LLaMA-based model | Agile user story datasets | MAE, RMSE, Accuracy@Tolerance | Demonstrated that task-specific and resource-efficient LLMs can achieve competitive performance for story point estimation |


---

### 5.3 Baseline Model Specification and Justification
To establish a transparent and reproducible reference point, this milestone adopts a **classical text-based regression baseline**. While deep learning and LLM-based models achieve higher accuracy, they introduce additional complexity that is unnecessary for feasibility validation.

**Baseline model:**
- **Text representation:** TF-IDF applied to concatenated title and description  
- **Prediction model:** Ridge Regression  

This baseline is widely used in NLP regression tasks, offers stability for high-dimensional sparse features, and is easy to retrain and interpret.

Reproducibility is ensured through:
- Retraining notebook: `notebooks/baseline_retrain.ipynb`  
- Trained model artifact: `models/baseline/baseline.joblib`  
- Model card: `models/baseline/model_card.md`

---

## 6. Metrics for Business Goal Evaluation
The baseline is evaluated using:
- **MAE (Mean Absolute Error)**  
- **RMSE (Root Mean Squared Error)**  
- **Accuracy@±1 story point**, reflecting planning tolerance  

---

## 7. Repository Structure
- `data/` – dataset sample and source reference  
- `notebooks/` – baseline retraining notebook  
- `models/baseline/` – trained model and model card  
- `references/` – bibliography  
- `presentation/` – recorded presentation  

---

## 8. References
See `references/references.md`.
