# ============================================================
# STREAMLIT APPLICATION
# Compare five trained ML models on uploaded BEED test data
# ============================================================

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET_COLUMN = "y"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

st.set_page_config(page_title="BEED ML Model Comparison", page_icon="🧠", layout="wide")
st.title("🧠 BEED – EEG Classification Model Comparison")
st.write(
    "Upload a labelled test CSV to evaluate the five saved classification models. "
    "The application does not train or modify models using uploaded test data."
)
st.info(
    "The CSV must contain all input feature columns and the target column 'y'. "
    "Use the supplied test_data.csv as an example."
)

feature_file = ROOT / "feature_columns.json"
if not feature_file.exists():
    st.error("Preprocessing information is missing. Run train_models.py first.")
    st.stop()

with open(feature_file, "r", encoding="utf-8") as file:
    feature_columns = json.load(file)

uploaded_file = st.file_uploader("Upload labelled test data (CSV)", type=["csv"])

if uploaded_file is None:
    default_test = ROOT / "test_data.csv"
    if default_test.exists():
        st.caption("No file uploaded. The included test_data.csv will be used.")
        test_data = pd.read_csv(default_test)
    else:
        st.warning("Upload a test CSV or run train_models.py to create test_data.csv.")
        st.stop()
else:
    try:
        test_data = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read the CSV file: {error}")
        st.stop()

required_columns = feature_columns + [TARGET_COLUMN]
missing_columns = [column for column in required_columns if column not in test_data.columns]
if missing_columns:
    st.error("Missing required columns: " + ", ".join(missing_columns))
    st.stop()

if test_data.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

X_test = test_data[feature_columns].apply(pd.to_numeric, errors="coerce")
y_test = test_data[TARGET_COLUMN]
if X_test.isnull().any().any() or y_test.isnull().any():
    st.error("The test CSV contains missing or non-numeric values in required columns.")
    st.stop()

st.subheader("Test dataset preview")
st.write(f"Rows: **{len(test_data)}** | Input features: **{len(feature_columns)}**")
st.dataframe(test_data.head(10), use_container_width=True)

selected_model = st.selectbox("Choose a classification model", list(MODEL_FILES.keys()))
model_path = MODEL_DIR / MODEL_FILES[selected_model]
if not model_path.exists():
    st.error(f"Saved model not found: {model_path.name}. Run train_models.py first.")
    st.stop()

model = joblib.load(model_path)

try:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    if len(model.classes_) == 2:
        auc = roc_auc_score(y_test, probabilities[:, 1])
    else:
        auc = roc_auc_score(
            y_test, probabilities, multi_class="ovr",
            average="weighted", labels=model.classes_
        )

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": auc,
        "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions)
    }

    st.subheader(f"Evaluation results – {selected_model}")
    metric_columns = st.columns(3)
    for index, (metric_name, metric_value) in enumerate(metrics.items()):
        metric_columns[index % 3].metric(metric_name, f"{metric_value:.4f}")

    st.subheader("Confusion matrix")
    matrix = confusion_matrix(y_test, predictions, labels=model.classes_)
    figure, axis = plt.subplots(figsize=(7, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title(selected_model)
    st.pyplot(figure)
    plt.close(figure)

    st.subheader("Classification report")
    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

except Exception as error:
    st.error(
        "Evaluation failed. Check that uploaded target labels match the classes "
        f"used during training. Details: {error}"
    )
    st.stop()

metrics_file = ROOT / "metrics.json"
if metrics_file.exists():
    with open(metrics_file, "r", encoding="utf-8") as file:
        saved_metrics = json.load(file)
    st.subheader("Original hold-out test comparison")
    st.caption("Metrics calculated by train_models.py on its original test split.")
    st.dataframe(pd.DataFrame(saved_metrics).transpose(), use_container_width=True)
