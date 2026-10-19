"""Run the packaged CodeAlpha heart-disease classification pipeline.

Usage from the repository root:
    python predict.py --input examples/sample_input.csv
    python predict.py --input your_data.csv --output outputs/custom_predictions.csv

This is an educational machine-learning demonstration, not a medical diagnostic tool.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Resolve packaged files relative to this script, not the caller's working directory.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "heart_disease_logistic_pipeline.joblib"
METADATA_PATH = PROJECT_DIR / "metadata.json"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "outputs" / "predictions.csv"


def resolve_input_path(value: str) -> Path:
    """Resolve an input path from the current directory or project directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    current_directory_path = path.resolve()
    if current_directory_path.is_file():
        return current_directory_path

    return (PROJECT_DIR / path).resolve()


def resolve_output_path(value: str | None) -> Path:
    """Resolve output paths relative to the project directory by default."""
    if not value:
        return DEFAULT_OUTPUT_PATH

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    return (PROJECT_DIR / path).resolve()


def fail(parser: argparse.ArgumentParser, message: str) -> None:
    """Print a command-line usage error and exit cleanly."""
    parser.error(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict the dataset-defined binary outcome for rows in a CSV. "
            "This project is educational and is not a medical diagnostic tool."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV file containing the 13 original model input features.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Destination CSV. Defaults to outputs/predictions.csv "
            "inside the project directory."
        ),
    )
    args = parser.parse_args()

    # Check that all packaged resources exist before attempting inference.
    if not MODEL_PATH.is_file():
        fail(
            parser,
            "Trained model not found. Expected it at: "
            f"{MODEL_PATH}\nMake sure the repository includes the models/ folder "
            "and heart_disease_logistic_pipeline.joblib.",
        )

    if not METADATA_PATH.is_file():
        fail(
            parser,
            f"Project metadata not found: {METADATA_PATH}. "
            "Restore metadata.json from the repository.",
        )

    input_path = resolve_input_path(args.input)
    if not input_path.is_file():
        fail(parser, f"Input CSV not found: {input_path}")

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        fail(parser, f"Could not read metadata.json: {error}")

    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        fail(parser, "metadata.json does not contain a valid feature_columns list.")

    try:
        input_data = pd.read_csv(input_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        fail(parser, f"Could not read input CSV: {error}")
    except pd.errors.EmptyDataError:
        fail(parser, "The input CSV is empty or has no readable header.")

    if input_data.empty:
        fail(parser, "The input CSV contains no data rows.")

    missing_columns = [
        column for column in feature_columns if column not in input_data.columns
    ]
    if missing_columns:
        fail(
            parser,
            "Input CSV is missing required feature columns: "
            f"{missing_columns}\nSee examples/sample_input.csv for the required schema.",
        )

    extra_columns = [
        column for column in input_data.columns if column not in feature_columns
    ]
    if extra_columns:
        print(
            "Note: extra input columns will be preserved in the output but "
            f"ignored by the model: {extra_columns}"
        )

    # Preserve the exact training-time feature names and ordering.
    X = input_data.loc[:, feature_columns].copy()

    # The training pipeline expects numeric encodings for all 13 source features.
    # Blank cells are permitted and handled by the fitted imputers. Non-numeric
    # text is rejected so it is not silently mistaken for a missing value.
    for column in feature_columns:
        original = X[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_values = original.notna() & converted.isna()

        if invalid_values.any():
            examples = original.loc[invalid_values].astype(str).unique()[:5].tolist()
            fail(
                parser,
                f"Feature '{column}' contains non-numeric values, for example "
                f"{examples}. Supply the dataset's numeric feature codes.",
            )

        X[column] = converted

    # Completely empty rows cannot meaningfully describe an input observation.
    entirely_missing_rows = X.isna().all(axis=1)
    if entirely_missing_rows.any():
        row_numbers = (np.flatnonzero(entirely_missing_rows.to_numpy()) + 2).tolist()
        fail(
            parser,
            "Some CSV rows have all 13 feature values missing. "
            f"CSV line numbers (including the header): {row_numbers[:10]}",
        )

    missing_value_count = int(X.isna().sum().sum())
    if missing_value_count:
        print(
            f"Note: {missing_value_count} missing feature value(s) detected. "
            "The fitted pipeline will impute them using its training-time settings."
        )

    # Only load serialized model artifacts from a trusted repository.
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as error:
        fail(
            parser,
            f"Could not load the trained model: {error}. Check the installed "
            "package versions in requirements.txt.",
        )

    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        fail(parser, "The saved artifact is not a probability-capable classifier pipeline.")

    try:
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        classes = list(model.named_steps["classifier"].classes_)
        positive_index = classes.index(1)
    except (ValueError, KeyError, AttributeError) as error:
        fail(parser, f"Model prediction failed: {error}")

    class_1_scores = probabilities[:, positive_index]

    # The score is the model's estimate for dataset class 1, not a clinical risk.
    output = input_data.copy()
    output["predicted_label"] = predictions.astype(int)
    output["predicted_class"] = np.where(
        predictions.astype(int) == 1,
        "Disease present according to dataset target",
        "Disease absent according to dataset target",
    )
    output["class_1_model_score"] = class_1_scores

    output_path = resolve_output_path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
    except OSError as error:
        fail(parser, f"Could not write predictions to {output_path}: {error}")

    print("\nPrediction completed successfully.")
    print(f"Input file: {input_path}")
    print(f"Rows processed: {len(output)}")
    print(f"Predictions saved to: {output_path}")
    print("Labels represent the dataset's binary target mapping.")
    print("Class-1 scores are model outputs, not calibrated clinical risks or diagnoses.")


if __name__ == "__main__":
    main()
