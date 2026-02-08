# Milestone 1 – Project Inception
## AI-Based Story Point Estimation for Optimization-Driven Agile Planning

## Business Case Description
Accurate effort estimation is central to Agile development because it directly affects sprint planning, resource allocation, and delivery predictability. In practice, teams estimate effort using story points (e.g., planning poker). While effective, this process is often subjective, time-consuming, and inconsistent across teams and sprints.

This project aims to automatically estimate story points from user story text (title + description) using machine learning. The system is designed as a decision-support tool that assists Agile teams during backlog refinement and sprint planning by providing consistent, data-driven estimates.

## Business Value of Using ML
Story point estimation depends on variable natural-language descriptions and historical estimation behavior. Rule-based approaches cannot robustly capture this semantic variability.

Machine learning enables:
- Consistency: reduces estimation variability across time and team members
- Efficiency: speeds up grooming and planning by reducing manual estimation effort
- Scalability: supports estimation across multiple projects/teams
- Decision support: helps less-experienced teams calibrate estimates using historical data

##  Dataset Overview

### Source and Origin
This project uses the **Agile User Story Point Estimation** dataset, a publicly available benchmark dataset originally introduced by **Choetkiertikul et al.** in *A Deep Learning Model for Estimating Story Points*. The dataset was collected from **JIRA issue trackers** and has since become a **standard reference benchmark** for research on automated story point estimation.

The same dataset (or directly derived variants) is reused in multiple subsequent studies, including deep learning–based approaches, GPT-based models (e.g., GPT2SP), and more recent LLM-based estimators (e.g., Llama3SP), which highlights its relevance and robustness.

---

### Size and Coverage
- **Total instances:** 23,313 user stories / issues  
- **Projects:** 16 large open-source Agile projects  
- **Repositories:** Multiple well-known OSS ecosystems (e.g., Apache, Moodle, Spring, Atlassian)  
- **Data source:** Real-world Agile development artifacts extracted from JIRA

This scale and diversity make the dataset representative of realistic Agile planning scenarios rather than synthetic or toy examples.

---

### Core Fields (Conceptual View)
At a conceptual level, the dataset contains:
- **Title:** short textual summary of the user story  
- **Description (or user story):** detailed natural-language description of the requirement  
- **Story points:** numeric effort estimate assigned by Agile teams  

In this milestone, story points are modeled as a **continuous numeric target**, following prior work.

---

### Preprocessing and Representations
Different repositories derived from the original dataset provide **alternative but semantically equivalent representations** to facilitate modeling. In particular:
- Textual fields may be merged into a single column (e.g., `concat = title + description`)
- The target label may appear under different names (e.g., `point` instead of `story_points`)

These variations are **purely preprocessing choices** and do not alter the underlying semantics of the data. In this project, the model operates on the **combined textual content** of the title and description, consistent with prior studies.

---

### Why This Dataset Is Powerful
This dataset is particularly well suited for this project because:
- It captures **real-world Agile estimation behavior** rather than synthetic labels  
- It contains **rich natural-language descriptions**, enabling NLP-based modeling  
- It has been **validated and reused across multiple independent studies**, ensuring comparability  
- It supports both **classical ML baselines** and **modern LLM-based approaches**

As a result, the dataset provides a strong foundation for evaluating story point estimation models and for integrating learned estimates into downstream optimization workflows in later milestones.

---

### Repository Assets
- **Sample file:** `data/sample.csv` (for quick inspection and reproducibility)  
- **Full dataset:** referenced via the public source link above


## Project Archetype
This project is framed as a decision-support ML system embedded within an optimization-driven Agile planning workflow.

Predictive component (ML)
- Given a user story’s title + description, the model predicts its story points.

Optimization component (business decision)
Sprint planning can be formulated as a constrained optimization problem (e.g., knapsack-style backlog selection):
- Decision variables: select which stories to include in the next sprint
- Constraints: capacity/velocity, dependencies, deadlines, WIP limits, priorities, team availability
- Objective: maximize delivered value (or minimize risk/penalties) under constraints

A simple form:
- choose stories x_i ∈ {0, 1}
- capacity constraint Σ x_i · sp_i ≤ capacity
- maximize value Σ x_i · value_i

Here, estimated story points are inputs to the optimization, not the final output.

Why code generation matters
Optimization constraints typically do not live inside the story-point dataset. They come from planning context (capacity notes, dependencies, team constraints, sprint goals, etc.). In later milestones, an LLM will be used to:
1) extract decision variables / constraints / objective from natural-language planning context, and
2) generate solver-ready symbolic code (e.g., OR-Tools / Pyomo / MiniZinc / CP-SAT model).

Milestone 1 scope: validate feasibility using a reproducible story point estimation baseline only.
Later milestones: integrate solver-based optimization and LLM-generated symbolic formulations.

## Feasibility Analysis and Related Work
Prior research shows that predicting story points from Agile text is feasible on public benchmarks and that more advanced architectures (deep learning and LLM-based approaches) can improve accuracy. The works below are selected to (1) establish the benchmark dataset and task feasibility, (2) show a deep-learning refinement on the same benchmark family, (3) provide a Transformer-based baseline with public artifacts, and (4) represent a recent resource-efficient LLM approach with released code/models.

Table — Representative studies on automated story point estimation

| ID | Reference | Model / Technique | Dataset | Evaluation Metrics | Key Contribution |
|---|---|---|---|---|---|
| 1 | Choetkiertikul et al. | LD-RNN (LSTM + Recurrent Highway Network) | 23,313 stories, 16 OSS projects (JIRA) | MAE, SA | Introduced the benchmark dataset and showed deep learning outperforms common baselines |
| 2 | Mittal, Arsalan, Garg | Novel deep learning model | 23,313 issues, 16 OSS projects (JIRA) | MAE, SA (reported) | Confirms feasibility and reports improved performance comparisons vs earlier baselines |
| 3 | Fu, Tantithamthavorn | GPT2SP (Transformer/GPT-2 based) | 23,313 issues, 16 OSS projects | MAE (reported in paper) | Provides a Transformer-based approach and an openly available replication package |
| 4 | Sepúlveda Montoya, Ríos Gómez, Jaramillo Villegas | Llama3SP (fine-tuned LLaMA 3.2 with QLoRA) | 23,313 issues, 16 OSS projects | MAE, RMSE, tolerance-based metrics (reported) | Resource-efficient LLM estimator; code + pretrained models released |

## Baseline Specification
Baseline choice (what & why)
To satisfy the milestone’s reproducibility requirement (trained model binary + retraining notebook available), we select GPT2SP as the baseline.

GPT2SP is an established Transformer-based story point estimator built on GPT-2. It is accompanied by a public replication package that provides:
- trained model binaries (hosted on Hugging Face Model Hub; also mirrored via Google Drive), and
- training notebooks documenting the full retraining process.

This baseline provides a reproducible reference point for future milestones, where we will integrate estimates into optimization (sprint backlog selection) and compare against more recent LLM estimators such as Llama3SP.

Reproducibility pointers (baseline artifacts)
- Replication package (code + training notebooks): https://github.com/awsm-research/gpt2sp
- Example trained model on Hugging Face: https://huggingface.co/MickyMike/GPT2SP


## Metrics for Business Goal Evaluation

The business goal of this project is to enable reliable downstream optimization by providing **accurate story point estimates** from user story text. In Milestone 1, the labeled signal is the story point value assigned by Agile teams; therefore, evaluation focuses on **estimation accuracy**, which is a necessary prerequisite for meaningful optimization in later stages.

The following metrics are used:

- **Mean Absolute Error (MAE):** measures the average absolute difference between predicted and actual story points, providing an intuitive and directly interpretable measure of estimation quality in the same units used by Agile teams.

- **Root Mean Squared Error (RMSE):** penalizes larger estimation errors more strongly, highlighting cases where inaccurate estimates could significantly distort constraint-based optimization formulations (e.g., capacity limits).

- **Accuracy@±1 story point:** measures the proportion of predictions that fall within a tolerance of ±1 story point, reflecting common Agile practice where small estimation deviations are acceptable and unlikely to affect decision-making materially.

These metrics are consistently used in prior work on automated story point estimation, including Choetkiertikul et al., GPT2SP, and Llama3SP, enabling direct comparability with published results while ensuring that the learned estimation component is sufficiently reliable to support later symbolic optimization and code generation.


## Repository Structure / Assets
- data/
  - sample.csv
- models/
  - model_card.md (baseline model card)
- references/
  - references.md
- presentation/
  - recorded presentation

## References
See: references/references.md
