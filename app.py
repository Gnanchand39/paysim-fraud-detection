"""Streamlit demo app for the PaySim fraud-detection portfolio project.

Thin by design: this file handles page layout, widgets, and wiring only.
Feature construction lives in src/app_features.py, model loading/prediction
in src/prediction.py, and threshold classification in src/risk.py.
"""

import matplotlib.pyplot as plt
import streamlit as st

from src.app_features import (
    ADVANCED_HISTORICAL_FIELDS,
    AdvancedDemoInput,
    InputValidationError,
    QuickDemoInput,
    build_feature_vector,
)
from src.features import TRANSACTION_TYPES
from src.prediction import (
    ModelArtifactError,
    PredictionError,
    load_feature_importance,
    load_metrics,
    load_model,
    predict_fraud_probability,
)
from src.risk import DEFAULT_THRESHOLD, classify

st.set_page_config(page_title="Mobile-Money Fraud Detection", page_icon="🔍", layout="wide")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Mobile-Money Fraud Detection")
st.caption("AI-powered transaction risk analysis using synthetic PaySim data")
st.info(
    "**Portfolio demonstration using synthetic PaySim data** — not connected to NPCI, UPI, "
    "banks, or real customer accounts. No real financial decisions should be made from this tool."
)

# ---------------------------------------------------------------------------
# Sidebar: mode + inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Analysis Mode")
mode = st.sidebar.radio("Choose a mode", ["Quick Demo", "Advanced Demo"], label_visibility="collapsed")

st.sidebar.header("Transaction Inputs")
transaction_type = st.sidebar.selectbox("Transaction Type", TRANSACTION_TYPES, index=TRANSACTION_TYPES.index("TRANSFER"))
amount = st.sidebar.number_input("Amount", min_value=0.0, value=10000.0, step=100.0)
sender_balance = st.sidebar.number_input("Sender Previous Balance", min_value=0.0, value=10000.0, step=100.0)
receiver_balance = st.sidebar.number_input("Receiver Previous Balance", min_value=0.0, value=0.0, step=100.0)
simulated_step = st.sidebar.number_input(
    "Simulated Step (1-743)", min_value=1, max_value=743, value=200, step=1,
    help="PaySim step represents simulated time; it is not a real-world timestamp.",
)
st.sidebar.caption("PaySim step represents simulated time (1 step = 1 simulated hour); it is not a real-world timestamp.")

advanced_input = None
if mode == "Advanced Demo":
    st.sidebar.header("Historical Context (Demo)")
    st.sidebar.caption(
        "Hypothetical demonstration values, not observed customer history. "
        "Each field below is pre-filled with its no-history default (0, or -1 for "
        "\"steps since last transaction\") — edit a value to explore a hypothetical scenario."
    )
    advanced_values = {}
    field_labels = {
        "prior_sender_transaction_count": "Sender: prior transaction count",
        "prior_sender_total_amount": "Sender: prior total amount",
        "prior_sender_mean_amount": "Sender: prior mean amount",
        "prior_sender_max_amount": "Sender: prior max amount",
        "prior_sender_time_since_last_transaction": "Sender: steps since last transaction (-1 = no history)",
        "prior_sender_transfer_count": "Sender: prior TRANSFER count",
        "prior_sender_cashout_count": "Sender: prior CASH_OUT count",
        "prior_sender_distinct_destinations": "Sender: distinct prior destinations",
        "prior_receiver_transaction_count": "Receiver: prior transaction count",
        "prior_receiver_total_amount": "Receiver: prior total amount received",
        "prior_receiver_mean_amount": "Receiver: prior mean amount received",
        "prior_receiver_max_amount": "Receiver: prior max amount received",
        "prior_receiver_time_since_last_transaction": "Receiver: steps since last transaction (-1 = no history)",
        "prior_receiver_distinct_senders": "Receiver: distinct prior senders",
        "sender_transactions_previous_1_step": "Sender: transactions in previous 1 step",
        "sender_transactions_previous_6_steps": "Sender: transactions in previous 6 steps",
        "sender_transactions_previous_24_steps": "Sender: transactions in previous 24 steps",
        "sender_amount_previous_24_steps": "Sender: amount in previous 24 steps",
        "receiver_transactions_previous_24_steps": "Receiver: transactions in previous 24 steps",
    }
    with st.sidebar.expander("Historical Context (Demo)", expanded=False):
        for field_name in ADVANCED_HISTORICAL_FIELDS:
            default_val = -1.0 if "time_since_last" in field_name else 0.0
            advanced_values[field_name] = st.number_input(
                field_labels[field_name], value=default_val, step=1.0, key=f"adv_{field_name}",
            )
    advanced_input = AdvancedDemoInput(**advanced_values)

analyze_clicked = st.sidebar.button("Analyze Transaction", type="primary")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.subheader("Transaction Summary")
summary_cols = st.columns(5)
summary_cols[0].metric("Type", transaction_type)
summary_cols[1].metric("Amount", f"{amount:,.2f}")
summary_cols[2].metric("Sender Balance", f"{sender_balance:,.2f}")
summary_cols[3].metric("Receiver Balance", f"{receiver_balance:,.2f}")
summary_cols[4].metric("Simulated Step", str(simulated_step))

if mode == "Quick Demo":
    st.caption(
        "**Demo assumption:** sender and receiver have no prior transaction history. "
        "Historical behavioral features are therefore initialized to their validated "
        "no-history defaults. This is a demonstration assumption, not observed customer history."
    )
else:
    st.caption(
        "**Advanced Demo:** historical context values are hypothetical demonstration inputs "
        "and are not observed customer history. Fields not shown above remain at their "
        "validated no-history defaults."
    )

if analyze_clicked:
    st.divider()
    st.subheader("Model Prediction")
    try:
        quick = QuickDemoInput(
            transaction_type=transaction_type, amount=amount,
            sender_balance=sender_balance, receiver_balance=receiver_balance,
            simulated_step=simulated_step,
        )
        feature_row = build_feature_vector(quick, advanced_input)
        model = load_model()
        probability = predict_fraud_probability(feature_row, model=model)
        # Threshold is sourced from the model-card artifact (single source of truth),
        # falling back to the code default only if the artifact is unavailable.
        threshold = load_metrics().get("threshold", DEFAULT_THRESHOLD)
        label = classify(probability, threshold=threshold)

        col_a, col_b = st.columns(2)
        col_a.metric("Fraud Probability", f"{probability * 100:.2f}%")
        col_b.metric("Prediction", label)

        st.subheader("Risk Visualization")
        fig, ax = plt.subplots(figsize=(8, 1.2))
        bar_color = "#C44E52" if label == "FRAUD" else "#4C72B0"
        ax.barh([0], [probability], color=bar_color, height=0.5)
        ax.barh([0], [1], color="none", edgecolor="black", height=0.5, linewidth=0.8)
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Predicted fraud probability")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"{threshold} is the model's classification threshold. This bar is a probability "
            f"visualization only — not a validated business risk policy."
        )

    except InputValidationError as e:
        st.error(f"Input validation error: {e}")
    except ModelArtifactError as e:
        st.error(f"Model artifact not found. Please verify the project structure. ({e})")
    except PredictionError as e:
        st.error(f"Model error: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Model-level feature importance
# ---------------------------------------------------------------------------
st.subheader("Model-Level Feature Importance")
try:
    importance_df = load_feature_importance()
    top10 = importance_df.sort_values("importance", ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top10["feature"], top10["importance"], color="#4C72B0")
    ax.set_xlabel("Gain-based importance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Feature importance reflects model-level predictive usefulness. It is not a causal "
        "explanation and does not explain an individual prediction."
    )
except ModelArtifactError as e:
    st.warning(f"Feature importance unavailable: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
st.subheader("Model Performance")
st.caption("Final held-out test results on synthetic PaySim data")
try:
    metrics = load_metrics()
    perf_cols = st.columns(5)
    perf_cols[0].metric("Precision", f"{metrics['precision']:.4f}")
    perf_cols[1].metric("Recall", f"{metrics['recall']:.4f}")
    perf_cols[2].metric("F1", f"{metrics['f1']:.4f}")
    perf_cols[3].metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    perf_cols[4].metric("PR-AUC", f"{metrics['pr_auc']:.4f}")

    st.divider()
    st.subheader("About the Model")
    info_cols = st.columns(2)
    with info_cols[0]:
        st.markdown(
            f"""
- **Model:** {metrics['model_type']}
- **Features:** {metrics['feature_count']} causal features
- **Threshold:** {metrics['threshold']}
- **Training data:** {metrics['training_data']}
"""
        )
    with info_cols[1]:
        st.markdown(
            f"""
- **Dataset:** Synthetic PaySim
- **Training rows:** {metrics['training_rows']:,} (fraud: {metrics['training_fraud_count']:,})
- **Test rows:** {metrics['test_rows']:,} (fraud: {metrics['test_fraud_count']:,})
- **Excluded features:** {", ".join(metrics['excluded_features'])}
"""
        )
except ModelArtifactError as e:
    st.warning(f"Model metrics unavailable: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------
st.subheader("Limitations")
st.markdown(
    """
- PaySim is a **synthetic** dataset.
- This application is a **portfolio demonstration**.
- It is **not connected** to NPCI, UPI, banks, or real financial systems.
- Historical context in demo modes may be **assumed or hypothetical**, not observed customer history.
- Model performance on PaySim **does not establish real-world fraud-detection performance**.
- Global feature importance is **not causal** and does not explain an individual prediction.
"""
)
