# ============================================================
# MACHINE LEARNING ASSIGNMENT 2
# BEED - Bangalore EEG Epilepsy Dataset
# Train and evaluate five classification models
# ============================================================

import json
from pathlib import Path

import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42
TARGET_COLUMN = "y"

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}


def load_dataset():
    """Download the BEED dataset from the UCI repository."""
    print("Downloading BEED dataset from UCI...")
    dataset = fetch_ucirepo(id=1134)

    features = dataset.data.features.copy()
    target = dataset.data.targets.copy()

    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError("Expected exactly one target column in the UCI dataset.")
        target = target.iloc[:, 0]

    data = features.copy()
    data[TARGET_COLUMN] = target.values
    data = data.dropna().reset_index(drop=True)

    X = data.drop(columns=[TARGET_COLUMN]).apply(pd.to_numeric, errors="raise")
    y = data[TARGET_COLUMN]

    print("\nDataset shape:", data.shape)
    print("Feature count:", X.shape[1])
    print("Number of rows:", len(data))
    print("Target classes:", sorted(pd.Series(y).unique().tolist()))
    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    if X.shape[1] < 12:
        raise ValueError("Dataset does not meet the minimum 12-feature requirement.")
    if len(data) < 500:
        raise ValueError("Dataset does not meet the minimum 500-instance requirement.")

    return X, y


def create_models():
    """Create the five classifiers required for the assignment."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
        ])
    }


def calculate_metrics(model, X_test, y_test):
    """Calculate all six evaluation metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = model.classes_

    if len(classes) == 2:
        auc = roc_auc_score(y_test, probabilities[:, 1])
    else:
        auc = roc_auc_score(
            y_test, probabilities, multi_class="ovr",
            average="weighted", labels=classes
        )

    return {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "AUC": float(auc),
        "Precision": float(precision_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "Recall": float(recall_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "F1 Score": float(f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "MCC": float(matthews_corrcoef(y_test, predictions))
    }


def main():
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Save a labelled hold-out set for testing in the Streamlit application.
    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test.values
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    results = {}
    for model_name, model in create_models().items():
        print("\nTraining:", model_name)
        model.fit(X_train, y_train)

        model_filename = MODEL_FILENAMES[model_name]
        joblib.dump(model, MODEL_DIR / model_filename)
        results[model_name] = calculate_metrics(model, X_test, y_test)

        print("Saved:", model_filename)
        print("Metrics:", results[model_name])

    with open(ROOT / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    with open(ROOT / "feature_columns.json", "w", encoding="utf-8") as file:
        json.dump(list(X.columns), file, indent=4)

    print("\nAll five models trained and saved successfully.")
    print("Test data saved as test_data.csv")
    print("Metrics saved as metrics.json")


if __name__ == "__main__":
    main()
