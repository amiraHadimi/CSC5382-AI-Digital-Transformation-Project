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
The problem of Agile story point estimation has been widely studied in the software engineering and
machine learning literature. Early work established that learning-based approaches can outperform
traditional expert-driven estimation techniques by leveraging historical Agile project data and
natural-language descriptions of user stories.

Subsequent research focused on deep learning architectures capable of capturing semantic and
contextual information directly from user story text, achieving improved estimation accuracy.
However, these gains often come at the cost of increased model complexity and reduced transparency.
As a result, more recent studies emphasize the importance of interpretability and decision support,
particularly in Agile settings where estimates are used collaboratively during backlog grooming
and sprint planning.

More recent LLM-based approaches extend this line of work by exploring how large language models
and agent-based systems can support higher-level Agile planning tasks. These systems move beyond
standalone prediction and motivate the integration of estimation within structured planning and
optimization pipelines, aligning with formalized programming and symbolic reasoning frameworks.


| ID | Authors (Year) | Technique | Dataset | Evaluation Metrics | Key Findings |
|----|----------------|----------|---------|--------------------|--------------|
| P1 | Choetkiertikul et al. (2016) | Deep Learning (LSTM-based LD-RNN) | 23,313 user stories from 16 open-source Agile projects | MAE, standardized accuracy | Demonstrated that deep learning models outperform traditional effort estimation baselines |
| P2 | A Novel Deep Learning Model for Story Point Estimation (Year) | Deep Learning | Agile user story datasets | MAE, RMSE | Proposed an enhanced deep learning architecture achieving improved estimation accuracy |
| P3 | Explainable AI for Story Point Estimation (Year) | Explainable Machine Learning Models | Scrum project datasets | MAE, interpretability metrics | Highlighted the trade-off between estimation accuracy and model interpretability in Agile settings |
| P4 | LLM-Based Framework for Agile Planning (Year) | Large Language Models / Multi-Agent Systems | Agile artifacts | Task-level accuracy, qualitative evaluation | Demonstrated the potential of LLM-based systems for supporting Agile planning and decision-making |




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
