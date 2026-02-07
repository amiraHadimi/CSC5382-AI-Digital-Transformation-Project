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

This project follows a **decision-support machine learning system** archetype, where predictions assist Agile teams during sprint planning rather than replacing human judgment.

---

## 5. Literature Review

The problem of Agile story point estimation has been extensively studied in the software engineering
and machine learning literature. Early work demonstrated that machine learning models trained on
historical Agile project data can outperform expert-based estimation techniques, motivating the
automation of effort estimation tasks.

Subsequent research explored increasingly expressive models, including deep learning architectures
that operate directly on natural-language user story descriptions. These models leverage distributed
representations of text to capture semantic complexity and contextual information that cannot be
handled by rule-based systems. More recent studies investigate the use of pre-trained language models
and large language models (LLMs), highlighting their ability to generalize across projects and reduce
feature engineering requirements.

Collectively, these works establish the feasibility of predicting story point estimates from textual
user stories and motivate the use of a simple and reproducible baseline model as a reference point for
future, more advanced approaches.

| ID | Authors (Year) | Technique | Dataset | Key Findings |
|----|----------------|----------|---------|--------------|
| P1 | Choetkiertikul et al. (2016) | LSTM + Recurrent Highway Network (LD-RNN) | 23,313 user stories from 16 open-source Agile projects | Deep learning model significantly outperformed traditional baselines in MAE and standardized accuracy |
| P2 | Panda et al. (2015) | Neural Networks | Agile software project datasets | Demonstrated that ML-based effort estimation improves accuracy over expert judgment |
| P3 | Yalçıner et al. (2024) | SBERT embeddings + Gradient Boosted Trees | Industrial Agile datasets | Hybrid embedding-based models achieved better performance than classical ML approaches |
| P4 | Pavlič & Saklamaeva (2024) | Large Language Models (LLMs) | Agile effort estimation datasets | LLMs achieved performance comparable to or exceeding human estimators |
| P5 | Multimodal Generative AI (2024) | Multimodal Generative Models | Agile user stories with auxiliary signals | Showed the potential of generative AI for more robust and context-aware story point estimation |



Full references available in:  
[`references/references.md`](./references/references.md)

---

## 6. Baseline Model
### Feasibility and Baseline Justification

The feasibility of the proposed approach is supported by prior work demonstrating that both classical
machine learning models and deep learning architectures can successfully predict story point estimates
from textual user stories. While state-of-the-art models such as recurrent neural networks and large
language models achieve higher accuracy, they introduce significant complexity and reproducibility
challenges.

For this reason, a TF-IDF–based Linear Regression model is selected as the baseline. This choice provides
a transparent, reproducible reference point that allows the impact of more advanced models to be
measured in subsequent milestones. The baseline can be fully retrained using the provided notebook,
and the trained model artifact is generated as part of the retraining process, satisfying the
reproducibility requirements of this milestone.


A simple and reproducible baseline is used to validate feasibility.

**Baseline:**
- TF-IDF text representation
- Linear Regression model

The baseline can be retrained using:  
[`notebooks/baseline_retrain.ipynb`](./notebooks/baseline_retrain.ipynb)

The model is documented in:  
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
