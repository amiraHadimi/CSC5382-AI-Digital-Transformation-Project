# Baseline Model Card

## Model Name
TF-IDF + Linear Regression for Story Point Estimation

## Intended Use
This model is intended as a feasibility baseline for predicting Agile story points from user story text.

## Training Data
Agile user story dataset from 16 open-source projects (23,313 issues).

## Input / Output
- Input: concatenated title and description
- Output: continuous story point value

## Evaluation Metrics
MAE, RMSE, Accuracy@±1

## Limitations
This model does not capture deep semantic relationships and is intended only as a baseline.

