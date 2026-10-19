# CodeAlpha — Disease Prediction

An educational machine-learning project for binary classification
using the UCI Heart Disease dataset.

> **Important:** This model is for educational use only. It is not a
> medical diagnostic tool. Its reported performance comes from one
> 61-record held-out split and has not been independently validated
> on an external clinical dataset.

## Objective

Predict the dataset-defined presence or absence of heart disease from
13 structured input features.

**Dataset:** [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

The original target column `num` contains values 0 through 4. This
project maps value 0 to class 0 and values 1 through 4 to class 1.

This binary mapping is a common benchmark convention; it is not a
general clinical diagnosis or an estimate of future disease risk.

## Workflow

1. Inspect the raw dataset, feature types, missing values and target.
2. Prepare the binary target.
3. Split the data using a stratified 80/20 train/test split
   (`random_state=42`).
4. Impute and standardize numerical features.
5. Impute and one-hot encode categorical features.
6. Compare Logistic Regression, SVM, Random Forest and XGBoost.
7. Tune candidate models using five-fold cross-validation on the
   training split.
8. Evaluate the selected Logistic Regression model on the held-out test set.
9. Examine exploratory permutation feature importance.
10. Save, reload and smoke-test the prediction pipeline.

Preprocessing is kept inside the scikit-learn pipeline to avoid
fitting learned preprocessing on held-out evaluation data.

## Cross-validation comparison

The table shows mean scores for the selected configuration of each
model, measured across five folds of the training data.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.830 | 0.823 | 0.810 | 0.813 | 0.903 |
| SVM (RBF) | 0.814 | 0.830 | 0.765 | 0.789 | 0.888 |
| Random Forest | 0.814 | 0.824 | 0.764 | 0.788 | 0.900 |
| XGBoost | 0.834 | 0.855 | 0.783 | 0.811 | 0.892 |

Logistic Regression was selected using mean cross-validation ROC-AUC.
These cross-validation metrics were used during model selection and
should not be confused with the held-out test results below.

## Held-out test results

| Metric | Test score |
|---|---:|
| Accuracy | 0.8852 |
| Precision | 0.8387 |
| Recall / Sensitivity | 0.9286 |
| Specificity | 0.8485 |
| F1-score | 0.8814 |
| Balanced Accuracy | 0.8885 |
| ROC-AUC | 0.9654 |
| Average Precision | 0.9622 |

### Confusion matrix

| Actual class | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Disease absent (0) | 28 | 5 |
| Disease present (1) | 2 | 26 |

The test split contains 61 records. These results
describe one split of a small historical dataset; individual
observations can materially affect the resulting metrics.

## Project structure

```text
CodeAlpha_Disease_Prediction/
├── models/
│   └── heart_disease_logistic_pipeline.joblib
├── examples/
│   ├── sample_input.csv
│   └── sample_predictions.csv
├── outputs/
│   └── .gitkeep
├── predict.py
├── metadata.json
├── requirements.txt
├── README.md
├── MODEL_CARD.md
├── REFERENCES.md
├── CITATION.cff
└── .gitignore
```

The Colab notebook documenting the full development workflow should
also be exported and added to the repository root.

## Installation

Use a compatible Python environment. The dependency versions used
during model development are listed in `requirements.txt`.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate `.venv` before running the following command.

## Run a prediction

```bash
python predict.py --input examples/sample_input.csv
```

By default, predictions are saved to `outputs/predictions.csv`.

To choose a different destination:

```bash
python predict.py --input examples/sample_input.csv --output outputs/custom_predictions.csv
```

The input CSV must contain all 13 original feature columns described
in `metadata.json`. Column order does not matter; the script aligns
the data to the recorded feature schema.

The prediction output retains the input columns and adds:

- `predicted_label`: class 0 or class 1 at the model's default threshold.
- `predicted_class`: human-readable description of the dataset target mapping.
- `class_1_model_score`: the model's estimated probability for class 1.

The score has not been established as a calibrated clinical risk.
Predictions must not be used for diagnosis or treatment decisions.

## Model interpretation

`permutation_importance.csv` records exploratory permutation
importance on the held-out test data. Importance indicates how
predictive performance changes when a feature is shuffled; it does
not demonstrate causation or clinical importance.

The test split was also used for this exploratory analysis. Its
importance rankings were not used to select features, tune parameters
or change the decision threshold.

## Reproducibility

- Dataset source and target mapping: `metadata.json`
- Feature schema and preprocessing groups: `metadata.json`
- Selected parameters and cross-validation score: `metadata.json`
- Dependency versions: `requirements.txt`
- Evaluation reports: CSV files in the repository root

Scikit-learn's `joblib` persistence format requires a compatible
environment. Only load model files from trusted sources.

## Limitations

- The dataset is small and historical.
- Only one held-out split was used for final evaluation.
- No independent external validation or prospective study has been performed.
- Performance may differ in other populations and settings.
- Permutation importance is exploratory and not causal.
- Neither accuracy nor class-1 scores establish clinical reliability.

## References

1. UCI Machine Learning Repository — Heart Disease:
   https://archive.ics.uci.edu/dataset/45/heart+disease
2. Scikit-learn — Pipelines and composite estimators:
   https://scikit-learn.org/stable/modules/compose.html
3. Scikit-learn — Cross-validation:
   https://scikit-learn.org/stable/modules/cross_validation.html
4. Scikit-learn — Permutation feature importance:
   https://scikit-learn.org/stable/modules/permutation_importance.html
5. Scikit-learn — Model persistence:
   https://scikit-learn.org/stable/model_persistence.html
6. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement:
   updated guidance for reporting clinical prediction models using
   regression or machine learning methods. BMJ. 2024;385:e078378.
   https://doi.org/10.1136/bmj-2023-078378
