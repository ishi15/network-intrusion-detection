# Network Intrusion Detection System

Multi-class classifier detecting DoS, Probe, R2L, U2R attacks on the NSL-KDD dataset.

## 🧠 How it works
- Feature engineering + StandardScaler across 41 network traffic features
- Random Forest with class-weight balancing for imbalanced attack classes
- Interactive Streamlit dashboard for real-time classification

## 🛠️ Tech Stack
Python · Scikit-learn · Pandas · Streamlit

## 🚀 Run locally
pip install -r requirements.txt
streamlit run app.py
