# BEED EEG Classification – Machine Learning Assignment 2

## Problem statement

Build and compare five supervised machine-learning classification models using the
BEED (Bangalore EEG Epilepsy Dataset). A Streamlit application allows a user to upload
labelled test data and evaluate a saved model.

This is an academic machine-learning demonstration, not a medical diagnostic tool.

## Dataset

- **Dataset:** BEED – Bangalore EEG Epilepsy Dataset
- **Source:** https://archive.ics.uci.edu/dataset/1134/beed:+bangalore+eeg+epilepsy+dataset
- **Target column:** y
- **Expected input features:** 16 (the training script reports the actual feature count)
- **Task:** Classification
- **Citation:** Najmusseher & Nizar Banu, P.K. (2024). BEED: Bangalore EEG Epilepsy Dataset.
  UCI Machine Learning Repository. https://doi.org/10.24432/C5K33B.

## Models

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All models use the same stratified 80/20 train/test split.

## Evaluation metrics

Accuracy, AUC, weighted Precision, weighted Recall, weighted F1 Score and
Matthews Correlation Coefficient (MCC) are calculated for every model.

Run the training script to generate the real values in `metrics.json`. No
performance values or winning model are invented in advance.

## Project structure

```text
ML-ASSGNMENT-2/
|-- app.py
|-- model.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv             # generated after training
|-- metrics.json              # generated after training
|-- feature_columns.json      # generated after training
|-- model/                    # saved model files generated after training
```

## Installation and execution

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

The training script downloads the dataset from UCI and generates five saved models,
a labelled hold-out `test_data.csv`, `metrics.json`, and `feature_columns.json`.

The Streamlit app evaluates uploaded labelled CSV data using the saved models. It
does not retrain models on uploaded test data. The CSV must contain the same feature
columns used during training and the target column `y`.

## Streamlit application

- CSV upload for labelled test data
- Dropdown to select one of the five saved classifiers
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix and classification report
- Original hold-out comparison table

## Deployment

1. Push the project to this GitHub repository.
2. Open https://streamlit.io/cloud and sign in with GitHub.
3. Create a new app and select this repository, branch `main`, and entry file `app.py`.
4. Deploy after the saved model files and generated data files are present in the repository.

## Submission checklist

- [ ] Train all five models on the same dataset
- [ ] Calculate all six evaluation metrics
- [ ] Write observations based on actual metric values
- [ ] Include test_data.csv, metrics.json, feature_columns.json and saved models
- [ ] Deploy and test the Streamlit app
- [ ] Add the live app link and required screenshots to the final submission
