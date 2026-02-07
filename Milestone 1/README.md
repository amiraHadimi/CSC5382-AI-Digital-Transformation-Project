# Milestone 1 – Project Inception  
## AI-Based Story Point Estimation for Agile Software Development

---

## 1. Business Case Description

Accurate estimation of effort is a core activity in Agile software development, as it directly impacts sprint planning, resource allocation, and delivery predictability. In practice, Agile teams estimate effort using story points, often through expert-based techniques such as planning poker. While effective, these approaches are subjective, time-consuming, and prone to inconsistency across teams and sprints.

This project addresses the problem of automatically estimating story points from user story text (title and description) using machine learning. The objective is to support Agile teams during sprint planning by providing data-driven estimates based on historical project data, improving consistency and reducing manual effort.

The system is designed as a decision-support tool, assisting human decision-makers rather than replacing them.

---

## 2. Business Value of Machine Learning

Story point estimation depends on complex patterns in natural-language descriptions of requirements, as well as historical estimation behavior. Rule-based systems are not suitable for this task because they cannot capture semantic variability, contextual meaning, or evolving project characteristics.

Machine learning enables:

- Learning estimation patterns directly from historical Agile data
- Reducing estimation bias and inconsistency
- Supporting less experienced Agile teams
- Scaling estimation support across multiple projects

---

## 3. Dataset Overview

**Source:**  
https://github.com/mrthlinh/Agile-User-Story-Point-Estimation

**Origin:**  
Dataset introduced by Choetkiertikul et al. (2016).

**Projects:**  
16 large open-source Agile projects across 9 repositories, including Apache, Moodle, Spring, Atlassian, and others.

**Size:**  
23,313 user stories/issues.

**Fields:**
- `title`
- `description`
- `story_points`

**Label Strategy:**  
Story points are treated as continuous numeric values. Due to heterogeneous estimation scales across projects, labels are not normalized or rounded.

Dataset sample available at:  
[`data/sample.csv`](./data/sample.csv)

---

## 4. Project Archetype

This project is framed as a decision-support machine learning system for Agile planning activities,
including backlog grooming and sprint planning. At its core, the system predicts story point estimates
from natural-language user stories using machine learning.

Beyond standalone prediction, the project is conceptually aligned with optimization-based planning
systems, where estimates are used as inputs to constrained decision-making processes. Inspired by
the LLM-Based Formalized Programming (LLMFP) framework, the long-term vision of the project is
to leverage large language models to extract goals, constraints, and decision variables from Agile
artifacts, formulate an intermediate symbolic representation, and generate executable code for
formal optimization solvers.

In this milestone, the focus is limited to validating feasibility through a predictive baseline. The
formal optimization and symbolic code generation components are explicitly deferred to later
milestones.


---

## 5. Literature Review

The problem of Agile story point estimation has been extensively studied in the software engineering
The literature on Agile story point estimation demonstrates that machine learning techniques can
effectively support effort estimation by learning from historical project data and natural-language
descriptions of user stories. Early deep learning work by Choetkiertikul et al. (2016) introduced a
large benchmark dataset and showed that recurrent neural architectures significantly outperform
traditional regression-based baselines when evaluated using standard error metrics such as MAE and
standardized accuracy.

Subsequent research, including the work of Mittal et al. (2024), builds on this foundation by refining
deep learning architectures and further reducing estimation error on the same benchmark dataset.
These studies confirm both the feasibility of the task and the effectiveness of deep learning models
for capturing semantic complexity in Agile requirements.

More recent work explores the role of large language models in Agile estimation and planning.
Explainable local LLMs have been shown to generate human-like estimates while offering transparency
through reasoning traces, highlighting their suitability for decision-support scenarios. In parallel,
LLM-based multi-agent frameworks extend estimation beyond prediction by embedding it within
structured planning and coordination workflows.

Together, these studies motivate the use of a simple, transparent baseline model in this project.
Establishing a reproducible baseline enables objective comparison with more advanced deep learning
and LLM-based approaches in subsequent milestones while satisfying feasibility and reproducibility
requirements.



| Authors (Year) | Technique | Dataset | Evaluation Metrics | Key Findings |
|---------------|----------|---------|--------------------|--------------|
| Choetkiertikul et al. (2016) | Deep Learning (LD-RNN combining LSTM and Recurrent Highway Networks) | 23,313 user stories from 16 open-source Agile projects collected via JIRA | MAE, Standardized Accuracy (SA) | Demonstrated that deep learning models significantly outperform traditional bag-of-words and regression-based baselines for story point estimation |
| Mittal et al. (2024) | Deep Learning (LD-RNN variant) | 23,313 user stories from 16 open-source Agile projects (same benchmark dataset) | MAE, RMSE | Proposed an improved deep learning architecture that achieves lower estimation error compared to existing deep learning baselines |
| Yonathan (2024) | Explainable Local Large Language Models (LLMs) | Agile user stories stored in CSV format | Agreement within ±1 story point, qualitative consistency analysis | Showed that local LLMs can produce human-like estimates while providing interpretability through reasoning traces |
| An LLM-Based Multi-Agent Framework (2024) | Multi-agent LLM system with task decomposition and coordination | Agile planning artifacts (user stories, tasks, constraints) | Task-level accuracy, qualitative evaluation | Demonstrated the potential of LLM-based agents to support Agile planning through structured reasoning and decision support rather than standalone prediction |





Full references available in:  
[`references/references.md`](./references/references.md)

---

## 6. Baseline Model

### Feasibility and Baseline Justification

The feasibility of the proposed approach is supported by prior work demonstrating that both classical
machine learning models and deep learning architectures can successfully predict story point estimates
from textual user stories. Foundational studies and recent advances, including deep learning and
large language model–based approaches, establish that effort estimation from Agile artifacts is a
solvable machine learning problem.

While state-of-the-art models such as recurrent neural networks and large language models achieve
higher accuracy, they introduce increased complexity, higher computational cost, and reduced
reproducibility. For this reason, a simpler model is selected for the baseline to establish a clear
and reproducible point of reference.

### Baseline Specification

A TF-IDF–based Linear Regression model is selected as the baseline for this milestone. This choice
provides a transparent and easily reproducible implementation that enables systematic comparison
with more advanced models in subsequent milestones.

- **Text representation:** TF-IDF applied to the concatenated title and description  
- **Prediction model:** Linear Regression  

The baseline can be fully retrained using the provided notebook:  
[`notebooks/baseline_retrain.ipynb`](./notebooks/baseline_retrain.ipynb)

The trained model artifact is generated as part of the retraining process, and the model is documented
using a structured model card inspired by HuggingFace model cards:  
[`models/baseline/model_card.md`](./models/baseline/model_card.md)


---

## 7. Evaluation Metrics

The following metrics are used:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Accuracy@±1 story point**

---

## 8. Repository Structure and Assets

- Dataset sample: [`data/sample.csv`](./data/sample.csv)
- Baseline notebook: [`notebooks/baseline_retrain.ipynb`](./notebooks/baseline_retrain.ipynb)
- Model card: [`models/baseline/model_card.md`](./models/baseline/model_card.md)
- References: [`references/references.md`](./references/references.md)
- Presentation materials: [`presentation/`](./presentation/)

---

## 9. Recorded Presentation

A short recorded presentation summarizing this milestone is provided in the `presentation/` folder.

---
