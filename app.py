"""
AI Fraud Detection Engine
--------------------------------
A Streamlit app for real-time and batch credit-card fraud scoring,
built on top of a trained XGBoost classifier.

To run locally:
    streamlit run app.py
"""

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI Fraud Detection Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Paths — resolved automatically relative to this file.
# As long as xgb_model.pkl and credit_card_fraud_10k.csv sit in the
# SAME folder as app.py, nothing here needs to be edited by hand.
# ------------------------------------------------------------------
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "xgb_model.json"   # native XGBoost format — avoids pickle/version corruption issues
MODEL_PATH_PKL = APP_DIR / "xgb_model.pkl"  # fallback if only the pickle is present
DATA_PATH = APP_DIR / "credit_card_fraud_10k.csv"

MERCHANT_CATEGORIES = ["Clothing", "Electronics", "Food", "Grocery", "Travel"]
# Exact column order the model was trained on — must match training notebook.
FEATURE_ORDER = [
    "amount", "transaction_hour", "foreign_transaction", "location_mismatch",
    "device_trust_score", "velocity_last_24h", "cardholder_age",
    "merchant_category_Electronics", "merchant_category_Food",
    "merchant_category_Grocery", "merchant_category_Travel",
]

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a1f36;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f8f9fb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 14px 16px;
    }
    .risk-card {
        padding: 22px 26px;
        border-radius: 14px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .risk-high {
        background-color: #fef2f2;
        border: 1.5px solid #fca5a5;
        color: #991b1b;
    }
    .risk-low {
        background-color: #f0fdf4;
        border: 1.5px solid #86efac;
        color: #166534;
    }
    .risk-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .risk-sub {
        font-size: 0.95rem;
        opacity: 0.85;
    }
    .factor-chip {
        display: inline-block;
        padding: 4px 12px;
        margin: 4px 6px 4px 0;
        border-radius: 999px;
        font-size: 0.85rem;
        background-color: #eef2ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    # Preferred: native XGBoost format (xgb_model.json). This format is
    # self-contained and not sensitive to pickle/library-version mismatches,
    # which is what causes "input stream corrupted" errors with .pkl files.
    if MODEL_PATH.exists():
        clf = xgb.XGBClassifier()
        clf.load_model(str(MODEL_PATH))
        return clf
    # Fallback: legacy pickle file, if that's all that's present.
    with open(MODEL_PATH_PKL, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def get_test_split(_df):
    """Reproduces the exact train/test split used during training
    (test_size=0.2, random_state=42, stratified) so the metrics
    shown reflect genuine held-out performance."""
    df = _df.copy()
    df = pd.get_dummies(df, columns=["merchant_category"], drop_first=True)
    X = df.drop(["is_fraud", "transaction_id"], axis=1)
    y = df["is_fraud"]
    for col in FEATURE_ORDER:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_ORDER]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def build_feature_row(amount, hour, foreign, mismatch, trust, velocity, age, category):
    row = {
        "amount": amount,
        "transaction_hour": hour,
        "foreign_transaction": int(foreign),
        "location_mismatch": int(mismatch),
        "device_trust_score": trust,
        "velocity_last_24h": velocity,
        "cardholder_age": age,
        "merchant_category_Electronics": 1 if category == "Electronics" else 0,
        "merchant_category_Food": 1 if category == "Food" else 0,
        "merchant_category_Grocery": 1 if category == "Grocery" else 0,
        "merchant_category_Travel": 1 if category == "Travel" else 0,
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


def risk_factors(amount, hour, foreign, mismatch, trust, velocity, age):
    flags = []
    if foreign and mismatch:
        flags.append("Foreign transaction with location mismatch")
    elif foreign:
        flags.append("Foreign transaction")
    elif mismatch:
        flags.append("Location mismatch")
    if trust < 40:
        flags.append("Low device trust score")
    if velocity >= 5:
        flags.append("High transaction velocity (24h)")
    if hour <= 5:
        flags.append("Late-night transaction")
    if amount >= 800:
        flags.append("Large transaction amount")
    return flags


# ------------------------------------------------------------------
# Load model + data (with a friendly error screen if it fails)
# ------------------------------------------------------------------
try:
    model = load_model()
    df_raw = load_data()
    load_error = None
except Exception:
    model, df_raw = None, None
    load_error = traceback.format_exc()

if load_error:
    st.error("The app couldn't load the model or dataset.")
    with st.expander("Show technical details"):
        st.code(load_error)
    st.info(
        f"Make sure **xgb_model.pkl** and **credit_card_fraud_10k.csv** are "
        f"in the same folder as app.py:\n\n`{APP_DIR}`"
    )
    st.stop()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown('<p class="main-header">🛡️ AI Fraud Detection Engine</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">XGBoost-powered credit card fraud scoring — '
    'single transaction checks, batch scans, and model insights.</p>',
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_insights = st.tabs(
    ["🔍 Check a Transaction", "📁 Batch Scan", "📊 Model Insights"]
)

# ------------------------------------------------------------------
# TAB 1 — Single transaction check
# ------------------------------------------------------------------
with tab_single:
    st.write("Enter the transaction details and the model will estimate its fraud risk.")

    left, right = st.columns(2)
    with left:
        amount = st.number_input("Transaction amount ($)", min_value=0.0, value=150.0, step=10.0)
        hour = st.slider("Transaction hour (24h)", 0, 23, 14)
        category = st.selectbox("Merchant category", MERCHANT_CATEGORIES)
        age = st.slider("Cardholder age", 18, 90, 40)
    with right:
        trust = st.slider("Device trust score (0-100)", 0, 100, 70)
        velocity = st.number_input("Transactions in last 24h", min_value=0, value=1, step=1)
        foreign = st.checkbox("Foreign transaction")
        mismatch = st.checkbox("Billing/location mismatch")

    st.write("")
    check = st.button("🔎 Analyze Transaction", type="primary", use_container_width=True)

    if check:
        X_row = build_feature_row(amount, hour, foreign, mismatch, trust, velocity, age, category)
        prob = float(model.predict_proba(X_row)[0, 1])
        is_fraud = prob >= 0.5

        st.markdown("---")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            css_class = "risk-high" if is_fraud else "risk-low"
            label = "⚠️ High Fraud Risk" if is_fraud else "✅ Low Fraud Risk"
            recommendation = (
                "Recommend manual review or step-up verification before approval."
                if is_fraud else
                "No red flags detected — transaction looks routine."
            )
            st.markdown(
                f"""<div class="risk-card {css_class}">
                        <div class="risk-title">{label}</div>
                        <div class="risk-sub">{recommendation}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

            flags = risk_factors(amount, hour, foreign, mismatch, trust, velocity, age)
            if flags:
                st.write("**Contributing factors:**")
                st.markdown(
                    "".join(f'<span class="factor-chip">{f}</span>' for f in flags),
                    unsafe_allow_html=True,
                )
            else:
                st.write("No specific risk factors stood out for this transaction.")

        with c2:
            st.metric("Fraud probability", f"{prob * 100:.1f}%")
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption("Model flags a transaction as fraud when probability ≥ 50%.")

# ------------------------------------------------------------------
# TAB 2 — Batch scan
# ------------------------------------------------------------------
with tab_batch:
    st.write(
        "Upload a CSV of transactions to score them all at once. "
        "Expected columns: `amount, transaction_hour, merchant_category, "
        "foreign_transaction, location_mismatch, device_trust_score, "
        "velocity_last_24h, cardholder_age`."
    )

    template = df_raw.drop(columns=["transaction_id", "is_fraud"]).head(3)
    st.download_button(
        "⬇️ Download a sample template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="transaction_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            work_df = batch_df.copy()
            for cat in MERCHANT_CATEGORIES:
                col = f"merchant_category_{cat}"
                work_df[col] = (work_df["merchant_category"] == cat).astype(int)
            for col in FEATURE_ORDER:
                if col not in work_df.columns:
                    work_df[col] = 0
            X_batch = work_df[FEATURE_ORDER]

            probs = model.predict_proba(X_batch)[:, 1]
            results = batch_df.copy()
            results["fraud_probability"] = (probs * 100).round(2)
            results["flagged"] = np.where(probs >= 0.5, "Yes", "No")

            n_flagged = int((results["flagged"] == "Yes").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Transactions scanned", len(results))
            m2.metric("Flagged as fraud", n_flagged)
            m3.metric("Flag rate", f"{n_flagged / len(results) * 100:.1f}%")

            st.dataframe(
                results.sort_values("fraud_probability", ascending=False),
                use_container_width=True,
                height=400,
            )
            st.download_button(
                "⬇️ Download scored results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="scored_transactions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception:
            st.error("Couldn't process that file — check that its columns match the template.")
            with st.expander("Show technical details"):
                st.code(traceback.format_exc())

# ------------------------------------------------------------------
# TAB 3 — Model insights
# ------------------------------------------------------------------
with tab_insights:
    X_test, y_test = get_test_split(df_raw)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    st.subheader("Held-out test performance")
    st.caption("Computed on the same 20% stratified test split used during training.")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc*100:.1f}%")
    m2.metric("Precision", f"{prec*100:.1f}%")
    m3.metric("Recall", f"{rec*100:.1f}%")
    m4.metric("F1 score", f"{f1*100:.1f}%")
    m5.metric("ROC-AUC", f"{auc:.3f}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Genuine", "Fraud"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Genuine", "Fraud"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig2, ax2 = plt.subplots(figsize=(4, 3.5))
        ax2.plot(fpr, tpr, color="#4f46e5", linewidth=2, label=f"AUC = {auc:.3f}")
        ax2.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend(loc="lower right")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("What drives the model's decisions")
    importances = pd.Series(model.feature_importances_, index=FEATURE_ORDER)
    importances = importances.sort_values(ascending=True)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.barh(importances.index, importances.values, color="#4f46e5")
    ax3.set_xlabel("Feature importance")
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("Dataset snapshot")
    d1, d2, d3 = st.columns(3)
    d1.metric("Total transactions", f"{len(df_raw):,}")
    d2.metric("Fraud cases", int(df_raw["is_fraud"].sum()))
    d3.metric("Fraud rate", f"{df_raw['is_fraud'].mean()*100:.2f}%")
    st.caption(
        "The dataset is highly imbalanced (~1.5% fraud), so the model was trained "
        "on a SMOTE-balanced training set while evaluation above uses the original, "
        "untouched test distribution."
    )
