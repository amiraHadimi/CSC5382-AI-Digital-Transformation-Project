# Milestone 1 – Project Inception  
## AI-Based Story Point Estimation for Optimization-Driven Agile Planning

### 1) Business Case Description
Accurate effort estimation is central to Agile development because it affects sprint planning, resource allocation, and delivery predictability. In practice, teams estimate effort using story points (e.g., planning poker). While effective, these processes are subjective, time-consuming, and inconsistent across teams and sprints.

This project aims to **automatically estimate story points from user story text** (title + description) using machine learning. The system is designed as a **decision-support tool** that assists Agile teams during backlog refinement and sprint planning by providing consistent, data-driven estimates.

---

### 2) Business Value of Using ML (Impact)
Story point estimation depends on complex and variable natural-language descriptions and team-specific historical patterns. Rule-based approaches cannot robustly capture this semantic variability.

Machine learning enables:
- **Consistency**: reduces variability in estimates across time and team members  
- **Efficiency**: speeds up planning and grooming by reducing manual estimation effort  
- **Scalability**: supports estimation across multiple projects and teams  
- **Decision support**: helps less-experienced teams calibrate estimates using historical data

---

### 3) Dataset Overview
**Source (public):**  
- Agile User Story Point Estimation dataset:  
  https://github.com/mrthlinh/Agile-User-Story-Point-Estimation/blob/master/data_csv/data  

**Origin:**  
- Introduced by **Choetkiertikul et al. (2016)** (JIRA-based Agile project data)

**Size:**  
- **23,313** user stories/issues

**Coverage:**  
- **16** large open-source Agile projects across **9** repositories (e.g., Apache, Moodle, Spring, Atlassian)

**Features / Fields:**
- `title` (text)
- `description` (text)
- `story_points` (numeric label)

**Label strategy:**  
Story points are treated as a **continuous numeric target**. Since projects may use different internal scales, labels are not forced into a single normalized scale in this milestone (normalization is considered later as an experiment).

---

### 4) Project Archetype (Optimization + Symbolic Code Generation)
This project is framed as a **decision-support ML system embedded in an optimization-driven Agile planning workflow**. The predictive component estimates story points from user story text. These estimates are **not the final goal**: they become **inputs to downstream planning optimization** (e.g., sprint backlog selection, release planning, capacity allocation).

In Agile planning, teams often face constrained decision problems that can be formalized as **constrained optimization**:
- **Decision variables:** which user stories to include in the sprint (and optionally assignment/order)
- **Constraints:** sprint capacity/velocity, dependencies, deadlines, and other business rules
- **Objective:** maximize business value, minimize risk, or maximize delivered value under capacity

This aligns with the course requirement that the project addresses an optimization problem requiring an **intermediate symbolic representation (code generation)**. Inspired by LLM-Based Formalized Programming (LLMFP), later milestones will use an LLM to extract variables/constraints/objectives from Agile artifacts and generate solver-ready symbolic code (e.g., constraints + objective encoded for an optimization/constraint solver). :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

**Milestone 1 scope:** feasibility via a reproducible baseline estimator.  
**Later milestones:** formal optimization formulation + LLM-generated symbolic solver code.

---

### 5) Feasibility Analysis – Literature Review (Cleaned)
Research shows that story point estimation from Agile text is feasible and reproducible on public benchmarks. Early work introduced large datasets and demonstrated deep learning improvements over classical baselines. Later work refined architectures and improved accuracy on the same benchmark. Recent studies explore LLM-based estimators with competitive results. In parallel, LLM planning frameworks (like LLMFP) demonstrate how LLMs can generate intermediate symbolic representations and solver code for optimization tasks; while not an estimation method itself, it motivates the project’s optimization-centric direction. :contentReference[oaicite:2]{index=2}

#### Selected Papers (Positioned Correctly)

| ID | Reference | Model / Technique | Dataset | Metrics | Why it matters here |
|---|---|---|---|---|---|
| Choetkiertikul et al. (2016) | Story point estimation from Agile artifacts | LD-RNN (LSTM + RHN) | 23,313 stories (16 projects, JIRA) | MAE, SA | Foundational dataset + strong evidence the task is learnable |
| Mittal et al. (2024) | Improved deep learning for estimation | LSTM-based model | Same benchmark | MAE, RMSE | Confirms feasibility + improves performance on the same dataset |
| Llama3SP (2025) | LLM-based estimation | Fine-tuned LLaMA-based model | Same benchmark | MAE, RMSE, Accuracy@Tolerance | Shows modern LLM estimators can be competitive (not used as baseline unless fully reproducible) |
| Hao et al. (ICLR 2025) | LLMFP (methodology) | LLM → symbolic representation → solver code | Planning/optimization tasks | Optimality rate, success rate | Motivates the symbolic-code-generation approach for optimization in later milestones :contentReference[oaicite:3]{index=3} |

> **Note:** LLMFP is included as **methodological motivation** for solver-backed optimization with intermediate symbolic code generation, not as an estimation baseline.

---

### 6) Baseline Model Choice / Specification (Reproducible)
To set a transparent and reproducible reference point, this milestone uses a classical NLP regression baseline:

- **Text representation:** TF-IDF over concatenated `title + description`
- **Model:** **Ridge Regression** (regularized linear regression)

**Why Ridge (instead of plain Linear Regression)?**
- TF-IDF produces very high-dimensional sparse features; Ridge is typically **more stable** and less sensitive to multicollinearity while remaining simple and interpretable.

**Reproducibility requirement:**
- Training notebook provided (retrain from scratch)
- Trained model artifact saved by the notebook

**Assets:**
- Retraining notebook: `notebooks/baseline_retrain.ipynb`
- Trained model output: `models/baseline/baseline.joblib` (generated)
- Model card: `models/baseline/model_card.md`

---

### 7) Metrics for Business Goal Evaluation
The baseline is evaluated using:
- **MAE (Mean Absolute Error):** primary metric (easy to interpret in story points)
- **RMSE (Root Mean Squared Error):** penalizes large estimation errors
- **Accuracy@±1 story point:** interpretable tolerance-based metric aligned with planning needs

---

### 8) Repository Structure / Assets
- `data/`  
  - `sample.csv` (small sample for quick inspection)  
  - (full dataset referenced via source link above)
- `notebooks/`  
  - `baseline_retrain.ipynb`
- `models/baseline/`  
  - `model_card.md`  
  - `baseline.joblib` (generated after training)
- `references/`  
  - `references.md`
- `presentation/`  
  - recorded presentation (≤ 7 minutes)

---

### 9) References
See: `references/references.md`
