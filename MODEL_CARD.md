# Model Card: Heart Disease Classification

## Model

Logistic Regression with a fitted scikit-learn preprocessing pipeline.

## Intended use

Educational demonstration of tabular binary classification,
preprocessing, cross-validation, evaluation and model packaging.

## Out-of-scope use

Clinical diagnosis, triage, treatment planning, population screening,
or any patient-care decision.

## Data

Source: UCI Heart Disease dataset (ID 45), commonly used Cleveland setup.
Original target values 1-4 were grouped into class 1; value 0 became class 0.

## Evaluation setup

- Training split: 242 records
- Held-out test split: 61 records
- Stratified split, random state 42
- Five-fold stratified cross-validation used during model selection
- Default classification threshold: 0.5

## Held-out results

| Metric               |   Test score |
|:---------------------|-------------:|
| Accuracy             |     0.885246 |
| Precision            |     0.83871  |
| Recall / Sensitivity |     0.928571 |
| Specificity          |     0.848485 |
| F1-score             |     0.881356 |
| Balanced Accuracy    |     0.888528 |
| ROC-AUC              |     0.965368 |
| Average Precision    |     0.962246 |

Confusion matrix: TN=28, FP=5,
FN=2, TP=26.

## Key limitations

The dataset is small and historical. The evaluation uses one test split,
without independent external validation. The test results should not be
interpreted as reliable estimates of performance in clinical practice.
Model probabilities have not been established as calibrated clinical risks.

Permutation feature importance is exploratory and does not demonstrate
causality or clinical importance.
