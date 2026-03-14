"""
=============================================================
 Network Intrusion Detection System — Streamlit Demo
 Run: streamlit run app.py  (after running train.py first)
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🔒",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# Load saved model artifacts
# @st.cache_resource = load once, stay in memory (faster app)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("model/rf_model.pkl")
    scaler        = joblib.load("model/scaler.pkl")
    encoders      = joblib.load("model/label_encoders.pkl")
    feature_names = joblib.load("model/feature_names.pkl")
    return model, scaler, encoders, feature_names

model, scaler, encoders, feature_names = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔒 NIDS")
st.sidebar.markdown("**Network Intrusion Detection System**")
st.sidebar.markdown("Model: Random Forest | Dataset: NSL-KDD")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Live Predictor", "Model Performance", "About"])

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — LIVE PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
if page == "Live Predictor":
    st.title("🔍 Live Traffic Classification")
    st.markdown("Adjust network traffic parameters and click **Predict**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Connection Info**")
        protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
        service       = st.selectbox("Service", ["http", "ftp", "smtp", "ssh",
                                                  "dns", "private", "other"])
        flag          = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTO", "RSTR", "SH", "OTH"])
        duration      = st.number_input("Duration (sec)", 0, 60000, 0)
        src_bytes     = st.number_input("Source Bytes", 0, 1000000, 200)
        dst_bytes     = st.number_input("Dest Bytes", 0, 1000000, 0)
        logged_in     = st.selectbox("Logged In", [0, 1])

    with col2:
        st.markdown("**Login & Access**")
        num_failed_logins = st.number_input("Failed Logins", 0, 10, 0)
        root_shell        = st.selectbox("Root Shell", [0, 1])
        su_attempted      = st.selectbox("SU Attempted", [0, 1])
        num_root          = st.number_input("Num Root Accesses", 0, 100, 0)
        num_compromised   = st.number_input("Compromised Conditions", 0, 100, 0)
        num_file_creations = st.number_input("File Creations", 0, 100, 0)
        num_shells        = st.number_input("Shell Prompts", 0, 10, 0)

    with col3:
        st.markdown("**Traffic Pattern**")
        count             = st.number_input("Count (same host/2s)", 0, 512, 1)
        srv_count         = st.number_input("Srv Count (same svc/2s)", 0, 512, 1)
        dst_host_count    = st.number_input("Dst Host Count", 0, 255, 1)
        dst_host_srv_count = st.number_input("Dst Host Srv Count", 0, 255, 1)
        serror_rate       = st.slider("SYN Error Rate", 0.0, 1.0, 0.0, 0.01)
        rerror_rate       = st.slider("REJ Error Rate", 0.0, 1.0, 0.0, 0.01)
        same_srv_rate     = st.slider("Same Service Rate", 0.0, 1.0, 1.0, 0.01)
        diff_srv_rate     = st.slider("Diff Service Rate", 0.0, 1.0, 0.0, 0.01)

    if st.button("🔍 Predict", type="primary"):

        # Build full 41-feature dict — start with all zeros
        input_dict = {f: 0.0 for f in feature_names}

        # Update with user inputs
        input_dict["duration"]           = float(duration)
        input_dict["src_bytes"]          = float(src_bytes)
        input_dict["dst_bytes"]          = float(dst_bytes)
        input_dict["logged_in"]          = float(logged_in)
        input_dict["num_failed_logins"]  = float(num_failed_logins)
        input_dict["root_shell"]         = float(root_shell)
        input_dict["su_attempted"]       = float(su_attempted)
        input_dict["num_root"]           = float(num_root)
        input_dict["num_compromised"]    = float(num_compromised)
        input_dict["num_file_creations"] = float(num_file_creations)
        input_dict["num_shells"]         = float(num_shells)
        input_dict["count"]              = float(count)
        input_dict["srv_count"]          = float(srv_count)

        # Mirror traffic pattern features across related dst_host_* columns
        # These are the most important features — the model needs them populated
        input_dict["serror_rate"]                = serror_rate
        input_dict["srv_serror_rate"]            = serror_rate
        input_dict["dst_host_serror_rate"]       = serror_rate
        input_dict["dst_host_srv_serror_rate"]   = serror_rate
        input_dict["rerror_rate"]                = rerror_rate
        input_dict["srv_rerror_rate"]            = rerror_rate
        input_dict["dst_host_rerror_rate"]       = rerror_rate
        input_dict["dst_host_srv_rerror_rate"]   = rerror_rate
        input_dict["same_srv_rate"]              = same_srv_rate
        input_dict["dst_host_same_srv_rate"]     = same_srv_rate
        input_dict["diff_srv_rate"]              = diff_srv_rate
        input_dict["dst_host_diff_srv_rate"]     = diff_srv_rate
        input_dict["dst_host_count"]             = float(dst_host_count)
        input_dict["dst_host_srv_count"]         = float(dst_host_srv_count)
        input_dict["dst_host_same_src_port_rate"] = same_srv_rate

        # Encode categorical features using saved encoders
        for col, val in [("protocol_type", protocol_type),
                         ("service", service), ("flag", flag)]:
            try:
                input_dict[col] = float(encoders[col].transform([val])[0])
            except ValueError:
                input_dict[col] = 0.0

        # Build DataFrame in exact same column order as training
        X_input = pd.DataFrame([input_dict], columns=feature_names)
        X_scaled = scaler.transform(X_input)

        prediction    = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        confidence    = max(probabilities) * 100

        color_map = {
            "Normal": "🟢", "DoS": "🔴", "Probe": "🟡",
            "R2L": "🟠", "U2R": "🔴", "Other": "⚪"
        }
        icon = color_map.get(prediction, "⚪")

        st.markdown("---")
        st.subheader("Prediction Result")
        st.markdown(f"### {icon}  **{prediction}**")
        st.markdown(f"**Confidence:** {confidence:.1f}%")

        # Probability bar chart for all classes
        classes = model.classes_
        prob_df = pd.DataFrame({"Class": classes, "Probability": probabilities})
        prob_df = prob_df.sort_values("Probability", ascending=True)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(prob_df["Class"], prob_df["Probability"],
                color=["#d62728" if c == prediction else "#aec7e8"
                       for c in prob_df["Class"]])
        ax.set_xlabel("Probability")
        ax.set_title("Class Probabilities")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Show what features were passed — helpful for debugging
        with st.expander("🔎 See feature values passed to model"):
            st.dataframe(pd.DataFrame([input_dict]).T.rename(columns={0: "Value"}))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.markdown("Generated after running `train.py`")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        try:
            img = mpimg.imread("outputs/confusion_matrix.png")
            st.image(img)
        except FileNotFoundError:
            st.info("Run train.py first to generate this chart.")

    with col2:
        st.subheader("Top 20 Feature Importances")
        try:
            img = mpimg.imread("outputs/feature_importance.png")
            st.image(img)
        except FileNotFoundError:
            st.info("Run train.py first to generate this chart.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### Network Intrusion Detection System

    This system classifies network traffic into 5 categories using a
    **Random Forest** classifier trained on the **NSL-KDD** benchmark dataset.

    | Category | Description |
    |----------|-------------|
    | **Normal** | Legitimate network traffic |
    | **DoS** | Denial of Service — resource exhaustion attacks |
    | **Probe** | Network scanning / reconnaissance |
    | **R2L** | Remote to Local — unauthorized remote access |
    | **U2R** | User to Root — privilege escalation |

    **Tech Stack:** Python · Scikit-learn · Random Forest · Pandas · Streamlit

    **Dataset:** NSL-KDD — 125,973 training samples, 41 features per connection.

    **Key design decisions:**
    - StandardScaler applied to normalize feature ranges
    - class_weight=balanced to handle rare attack classes (U2R, R2L)
    - fit only on train data to prevent data leakage
    - Feature names saved to ensure correct column order at inference
    """)