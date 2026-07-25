# 🛡️ AI Fraud Detection Engine

A machine learning system that scores credit card transactions for fraud risk in real time — trained on a 10,000-transaction dataset with a realistic ~1.5% fraud rate, benchmarked across three classifiers, and deployed as a full-featured Streamlit app with single-transaction scoring, batch CSV scanning, and live model-performance dashboards.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Explainability (SHAP)](#explainability-shap)
- [App Features](#app-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [Model File Handling](#model-file-handling)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Overview

Credit card fraud detection is a textbook case of an **extreme class-imbalance problem** — genuine transactions vastly outnumber fraudulent ones, so a naive model can look "accurate" while catching almost no real fraud. This project is built specifically around that challenge:

1. **Data exploration** — inspected transaction volume, fraud rate, and feature correlations.
2. **Class imbalance handling** — applied **SMOTE** (Synthetic Minority Over-sampling) to the training set only, keeping the test set untouched so evaluation reflects real-world conditions.
3. **Model benchmarking** — compared Logistic Regression, Random Forest, and XGBoost on an identical stratified split.
4. **Explainability** — used **SHAP** to understand which features actually drive the model's fraud predictions.
5. **Deployment** — packaged the final XGBoost model into a three-tab Streamlit app: single-transaction check, batch CSV scoring, and a live model-insights dashboard (confusion matrix, ROC curve, feature importance).

---

## Dataset

- **Source:** credit card fraud detection dataset (Kaggle), 10,000 transactions
- **Fraud rate:** ~1.51% (151 fraud cases out of 10,000) — a realistic, highly imbalanced distribution
- **No missing values or duplicates**

| Feature | Description |
|---|---|
| `amount` | Transaction amount ($) |
| `transaction_hour` | Hour of day (0–23) the transaction occurred |
| `merchant_category` | Clothing / Electronics / Food / Grocery / Travel |
| `foreign_transaction` | Whether the transaction was made abroad |
| `location_mismatch` | Whether billing and transaction location disagree |
| `device_trust_score` | Trust score of the device used (0–100) |
| `velocity_last_24h` | Number of transactions by the same card in the last 24h |
| `cardholder_age` | Age of the cardholder |
| `is_fraud` | Target label |

`merchant_category` was one-hot encoded (drop-first) before modeling, and `transaction_id` was dropped as a non-predictive identifier.

---

## Handling Class Imbalance

With only ~1.5% of transactions labeled fraud, a model that predicts "genuine" for everything would already score ~98.5% accuracy without learning anything useful. To address this properly:

- The data was split **80/20 with stratification** to preserve the fraud ratio in both sets.
- **SMOTE** was applied *only to the training set* (X_train, y_train), synthetically balancing the classes for training — the test set was left in its original, real-world imbalanced state.
- This means the reported test metrics reflect genuine held-out performance on realistic data, not performance inflated by oversampling the evaluation set.

---

## Modeling Approach

Three classifiers were trained on the SMOTE-balanced training data and evaluated on the untouched, imbalanced test set:

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000`, `class_weight='balanced'` |
| Random Forest | `n_estimators=100`, `class_weight='balanced'`, `random_state=42` |
| **XGBoost (final model)** | `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric='logloss'` |

---

## Results

| Model | Test Accuracy |
|---|---|
| Logistic Regression | 92.50% |
| Random Forest | 98.25% |
| **XGBoost** ✅ | **98.80%** |

**XGBoost was selected as the final model.** Because of the severe class imbalance, the deployed app's **Model Insights** tab goes beyond raw accuracy and also reports **precision, recall, F1, and ROC-AUC** on the same held-out split, plus a confusion matrix and ROC curve — since for fraud detection, catching actual fraud cases (recall) and minimizing false alarms (precision) matter far more than accuracy alone.

---

## Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was used on a 1,000-transaction sample from the test set to interpret the trained XGBoost model:

- **Beeswarm plot** — shows how each feature pushes individual predictions toward "fraud" or "genuine," and by how much.
- **Bar plot** — ranks features by their overall (mean absolute) impact on the model's output.

This gives a transparent view into *why* the model flags a transaction, rather than treating it as a black box.

---

## App Features

The Streamlit app (`app.py`) is organized into three tabs:

### 🔍 Check a Transaction
Enter a single transaction's details (amount, hour, merchant category, device trust score, velocity, foreign/mismatch flags, cardholder age) and get:
- An instant **High / Low Fraud Risk** verdict with a recommendation
- A fraud probability score and progress bar
- Plain-language **contributing risk factors** (e.g. "Foreign transaction with location mismatch," "Low device trust score," "Large transaction amount")

### 📁 Batch Scan
- Upload a CSV of transactions and score them all at once
- Downloadable **sample template** matching the expected column format
- Summary metrics (transactions scanned, flagged count, flag rate) plus a sortable results table
- Download the fully scored CSV

### 📊 Model Insights
- Live-computed **Accuracy, Precision, Recall, F1, and ROC-AUC** on the held-out test split
- **Confusion matrix** and **ROC curve** visualizations
- **Feature importance** chart
- Dataset snapshot (total transactions, fraud cases, fraud rate)

---

## Tech Stack

- **Language:** Python 3
- **Modeling:** scikit-learn (Logistic Regression, Random Forest), XGBoost
- **Imbalance handling:** imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit
- **Development environment:** Google Colab (training) → local deployment

---

## Project Structure

```
ai-fraud-detection-engine/
├── AI_Fraud_detection_engine.ipynb   # Full pipeline: EDA → SMOTE → 3-model benchmark → SHAP → export
├── app.py                             # Streamlit app (single check / batch scan / insights dashboard)
├── xgb_model.json                     # Trained XGBoost model — native format (preferred, load-order priority)
├── xgb_model.pkl                      # Trained XGBoost model — pickle fallback
├── credit_card_fraud_10k.csv          # Source dataset (also powers the Insights tab + batch template)
└── requirements.txt                    # Python dependencies
```

> All three data/model files (`xgb_model.json`, `xgb_model.pkl`, `credit_card_fraud_10k.csv`) must sit in the **same folder** as `app.py` — paths are resolved automatically relative to the script, so no manual path editing is needed as long as the folder structure above is kept intact.

---

## Setup & Installation

### 1. Prerequisites
- Python 3.9–3.11

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

No API keys or external services are required — everything runs locally, entirely offline.

---

## Running the App

```bash
streamlit run app.py
```

Then open your browser to:
```
http://localhost:8501
```

---

## Model File Handling

The app is built to **prefer the native XGBoost format over pickle**, and falls back gracefully:

1. If `xgb_model.json` exists, it's loaded via `XGBClassifier().load_model()` — this format is self-contained and immune to the "input stream corrupted" errors that pickle files can throw across different XGBoost/Python versions.
2. If only `xgb_model.pkl` is present, it falls back to a standard `pickle.load()`.
3. If neither file loads successfully, the app shows a friendly on-screen error (with an expandable technical traceback) instead of crashing silently — and tells you exactly which files need to be in the app's folder.

---

## Limitations

- Trained on a **synthetic** 10K-row dataset — real-world fraud patterns (stolen-card testing, card-not-present schemes, merchant collusion, etc.) are more complex and varied.
- SMOTE generates synthetic minority-class samples for training; while it helps the model learn fraud patterns, it doesn't replace having more genuine, diverse fraud examples.
- The 0.5 probability threshold for flagging fraud is a simple default — in production, this threshold would typically be tuned against the cost of false positives (blocking genuine customers) vs. false negatives (missing fraud).
- No hyperparameter tuning (e.g. grid/random search) was performed — the three models were compared using hand-picked configurations.

## Future Work

- Hyperparameter tuning for the XGBoost model (e.g. `GridSearchCV`, `Optuna`)
- Precision-recall curve and threshold-tuning tool built into the app itself
- Per-transaction SHAP explanations surfaced directly in the "Check a Transaction" tab (not just aggregate insights)
- Testing against a larger, more diverse real-world transaction dataset
- Adding authentication/access control before any real deployment handling sensitive financial data

---

*Part of an end-to-end machine learning portfolio — trained and benchmarked in Google Colab, deployed as an interactive Streamlit application.*
