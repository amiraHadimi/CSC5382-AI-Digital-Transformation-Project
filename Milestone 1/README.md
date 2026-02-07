Milestone 1 – Project Inception
AI-Based Story Point Estimation for Agile Software Development
1. Business Case Description

Accurate estimation of effort is a core activity in Agile software development, as it directly impacts sprint planning, resource allocation, and delivery predictability. In practice, Agile teams estimate effort using story points, often through expert-based techniques such as planning poker. While effective, these approaches are subjective, time-consuming, and prone to inconsistency across teams and sprints.

This project addresses the problem of automatically estimating story points from user story text (title and description) using machine learning. The objective is to support Agile teams during sprint planning by providing data-driven estimates based on historical project data, improving consistency and reducing manual effort.

The system is designed as a decision-support tool, assisting human decision-makers rather than replacing them.

2. Business Value of Machine Learning

Story point estimation depends on complex patterns in natural-language descriptions of requirements, as well as historical estimation behavior. Rule-based systems are not suitable for this task because they cannot capture semantic variability, contextual meaning, or evolving project characteristics.

Machine learning enables:

Learning estimation patterns directly from historical Agile data

Reducing estimation bias and inconsistency

Supporting less experienced teams

Scaling estimation support across multiple projects

In later stages, this project is intended to evolve toward an optimization-oriented setting, where estimates are derived through intermediate structured representations that enable constraint-aware reasoning and planning.

3. Dataset Overview

Source
Public dataset from:
https://github.com/mrthlinh/Agile-User-Story-Point-Estimation

Originally introduced by Choetkiertikul et al. (2016).

Projects
The dataset aggregates Agile user stories from 16 large open-source projects across 9 repositories, including Apache, Moodle, Spring, Atlassian, and others.

Size

23,313 user stories/issues

Data Type
Supervised learning dataset with textual inputs and numeric regression targets.

Format
CSV / Excel (one issue per row).

Fields

title: short summary of the user story or issue

description: detailed natural-language description

story_points: numeric effort estimate assigned by the development team

Label Strategy
Story points are treated as continuous numeric values. Among the 16 projects, 7 use Fibonacci scales while others use different scales. For this reason, labels are not rounded or normalized, making the dataset applicable across heterogeneous Agile settings.

A small sample of the dataset is provided in data/sample.csv
, with a description in data/README.md
.

4. Project Archetype

This project follows a decision-support machine learning system archetype.
The model predicts story point estimates from user story text to support Agile sprint planning decisions.

Conceptually, the project aligns with optimization-oriented ML systems, where natural-language input is transformed into intermediate representations that can support structured reasoning and future constraint-based extensions.

5. Literature Review
ID (Authors, Year)	Technique	Dataset	Results
Choetkiertikul et al., 2016	LSTM + Recurrent Highway Network (LD-RNN)	23,313 issues from 16 Agile projects	Outperformed traditional baselines in MAE and standardized accuracy
Pavlič & Saklamaeva, 2024	Large Language Models	Agile estimation datasets	Demonstrated competitive performance compared to human estimation
Yalçıner et al., 2024	SBERT + Gradient Boosted Trees	Industrial Agile datasets	Improved estimation accuracy over classical ML models

These studies demonstrate that machine learning models consistently outperform traditional expert-based approaches for Agile effort estimation, validating the feasibility of this project.

A complete list of references is available in references/references.md
.

6. Baseline Model

To validate feasibility, a simple and reproducible baseline model is adopted.

Baseline Specification

Text representation: TF-IDF (title + description)

Prediction model: Linear Regression

Justification

Widely used in text-based regression tasks

Easy to interpret and reproduce

Serves as a clear reference point for more advanced models

The baseline model can be retrained using the notebook
notebooks/baseline_retrain.ipynb
.

The baseline model is documented in
models/baseline/model_card.md
.

7. Evaluation Metrics

The following metrics are used to evaluate performance with respect to business objectives:

Mean Absolute Error (MAE)
Measures average deviation between predicted and actual story points.

Root Mean Squared Error (RMSE)
Penalizes larger estimation errors.

Accuracy@±1
Measures the proportion of predictions within ±1 story point, reflecting acceptable tolerance in Agile planning.

8. Repository Structure and Assets

Relevant assets supporting this milestone include:

Dataset sample: data/sample.csv

Dataset description: data/README.md

Baseline retraining notebook: notebooks/baseline_retrain.ipynb

Baseline model card: models/baseline/model_card.md

References: references/references.md

Presentation materials: presentation/

9. Recorded Presentation

A short recorded presentation summarizing the business case, dataset, baseline model, and evaluation metrics is provided in the presentation/ folder.
