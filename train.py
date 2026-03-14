"""
=============================================================
 Network Intrusion Detection System
 Dataset : NSL-KDD
 Model   : Random Forest Classifier
 Author  : Ishita Singhal
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DEFINE COLUMN NAMES
# ─────────────────────────────────────────────────────────────────────────────
# NSL-KDD has no header row, so we manually define all 43 column names.
# 41 features + 1 label (attack type) + 1 difficulty score (we drop this).

COLUMNS = [
    "duration",           # length of connection in seconds
    "protocol_type",      # tcp, udp, icmp
    "service",            # network service e.g. http, ftp, smtp
    "flag",               # connection status e.g. SF=normal, S0=no reply
    "src_bytes",          # bytes sent from source to destination
    "dst_bytes",          # bytes sent from destination to source
    "land",               # 1 if src and dst are same host/port
    "wrong_fragment",     # number of wrong fragments
    "urgent",             # number of urgent packets
    "hot",                # number of hot indicators (privileged ops attempted)
    "num_failed_logins",  # number of failed login attempts
    "logged_in",          # 1 if successfully logged in
    "num_compromised",    # number of compromised conditions
    "root_shell",         # 1 if root shell is obtained
    "su_attempted",       # 1 if su root command attempted
    "num_root",           # number of root accesses
    "num_file_creations", # number of file creation operations
    "num_shells",         # number of shell prompts
    "num_access_files",   # number of operations on access control files
    "num_outbound_cmds",  # number of outbound commands in ftp session
    "is_host_login",      # 1 if login belongs to host list
    "is_guest_login",     # 1 if login is guest
    "count",              # connections to same host in past 2 seconds
    "srv_count",          # connections to same service in past 2 seconds
    "serror_rate",        # % connections with SYN errors
    "srv_serror_rate",    # % connections to same service with SYN errors
    "rerror_rate",        # % connections with REJ errors
    "srv_rerror_rate",    # % connections to same service with REJ errors
    "same_srv_rate",      # % connections to same service
    "diff_srv_rate",      # % connections to different services
    "srv_diff_host_rate", # % connections to different hosts
    "dst_host_count",     # count of connections to same destination host
    "dst_host_srv_count", # count of connections to same dst host+service
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",       # attack type — this is what we predict
    "difficulty"   # difficulty score — not a real feature, drop it
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MAP GRANULAR ATTACK LABELS → 5 CATEGORIES
# ─────────────────────────────────────────────────────────────────────────────
# The dataset has 39 specific attack types (e.g. "neptune", "ipsweep").
# We group them into 5 broader categories so the model learns general patterns,
# not just memorising specific attack names it may never see in production.

ATTACK_MAP = {
    "normal": "Normal",
    # DoS — overwhelm system resources so it crashes or stops responding
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    # Probe — scanning the network to find vulnerabilities before attacking
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L — attacker sends packets from remote machine to gain local access
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L", "sendmail": "R2L",
    "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L",
    # U2R — attacker escalates from normal user to root/admin privileges
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "httptunnel": "U2R", "ps": "U2R",
    "sqlattack": "U2R", "xterm": "U2R",
}

os.makedirs("data",    exist_ok=True)
os.makedirs("model",   exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("[1/4] Loading dataset...")

    # header=None tells pandas there's no header row in the file
    # names=COLUMNS assigns our column names manually
    train = pd.read_csv("data/KDDTrain+.txt", header=None, names=COLUMNS)
    test  = pd.read_csv("data/KDDTest+.txt",  header=None, names=COLUMNS)

    # difficulty column is a meta-score added by NSL-KDD creators
    # it tells how hard a record is to classify — not a real network feature
    train.drop("difficulty", axis=1, inplace=True)
    test.drop("difficulty",  axis=1, inplace=True)

    print(f"    Train: {train.shape[0]:,} rows | Test: {test.shape[0]:,} rows")
    print(f"    Raw attack types found: {train['label'].nunique()}")
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(train, test):
    print("[2/4] Preprocessing...")

    # 4a. Map 39 specific attacks → 5 broad categories
    # fillna("Other") handles any unknown labels in the test set
    train["label"] = train["label"].map(ATTACK_MAP).fillna("Other")
    test["label"]  = test["label"].map(ATTACK_MAP).fillna("Other")
    print(f"    Label distribution (train):\n{train['label'].value_counts().to_string()}\n")

    # 4b. Encode categorical text columns → integers
    # ML models only understand numbers, not text like "tcp" or "http"
    # LabelEncoder converts: ["icmp","tcp","udp"] → [0, 1, 2]
    # We fit on train+test COMBINED so we don't get errors on unseen values
    cat_cols = ["protocol_type", "service", "flag"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]))  # learn all possible values
        train[col] = le.transform(train[col])
        test[col]  = le.transform(test[col])
        encoders[col] = le  # save for use in app.py later

    # 4c. Separate features (X) from labels (y)
    X_train = train.drop("label", axis=1)
    y_train = train["label"]
    X_test  = test.drop("label", axis=1)
    y_test  = test["label"]

    # 4d. Scale with StandardScaler
    # Problem: "src_bytes" can be 0–1,000,000 but "logged_in" is just 0 or 1
    # Without scaling, large-value features dominate the model unfairly
    # StandardScaler transforms each feature to: mean=0, standard deviation=1
    #
    # CRITICAL RULE: fit_transform on TRAIN only, then transform TEST
    # If you fit on test data too → data leakage (model sees test info early)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

    # Save scaler and encoders — app.py needs these to preprocess new inputs
    joblib.dump(scaler,   "model/scaler.pkl")
    joblib.dump(encoders, "model/label_encoders.pkl")

    print("    Preprocessing done.")
    return X_train_scaled, X_test_scaled, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_train, y_train):
    print("[3/4] Training Random Forest...")

    # WHY RANDOM FOREST?
    # A single decision tree overfits — it memorises training data.
    # Random Forest builds many trees (n_estimators), each on a random
    # subset of data and features. Final answer = majority vote across all trees.
    # Result: robust, generalises well, handles mixed feature types well.

    clf = RandomForestClassifier(
        n_estimators=150,        # build 150 trees — more trees = more stable
        max_depth=25,            # each tree can go 25 levels deep max
                                 # without this limit trees overfit badly
        min_samples_split=5,     # a node needs at least 5 samples to split
                                 # prevents the tree fitting single data points
        class_weight="balanced", # U2R and R2L attacks are rare in the dataset
                                 # balanced = auto-increase weight of rare classes
                                 # so the model doesn't just ignore them
        random_state=42,         # fix the random seed for reproducibility
        n_jobs=-1                # use all available CPU cores — faster training
    )

    clf.fit(X_train, y_train)
    joblib.dump(clf, "model/rf_model.pkl")
    print("    Model saved → model/rf_model.pkl")
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(clf, X_test, y_test):
    print("[4/4] Evaluating...")

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    # weighted F1 accounts for class imbalance — better metric than accuracy alone
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n    Accuracy : {acc*100:.2f}%")
    print(f"    F1 Score : {f1:.4f}")
    print("\n    Per-class breakdown:")
    print(classification_report(y_test, y_pred))

    # ── Plot 1: Confusion Matrix ──────────────────────────────────────────
    # Rows = actual labels, Columns = predicted labels
    # Diagonal cells = correct predictions (want these HIGH)
    # Off-diagonal cells = mistakes (want these LOW)
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix — Network Intrusion Detection", fontsize=13)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()
    print("    Saved → outputs/confusion_matrix.png")

    # ── Plot 2: Feature Importance ────────────────────────────────────────
    # Shows which of the 41 features were most useful to the model.
    # High importance = that feature contributed most to correct splits.
    # Useful for interviews: "src_bytes and dst_bytes are top features
    # because attack traffic has unusual byte transfer patterns."
    feat_imp = pd.Series(clf.feature_importances_, index=X_test.columns)
    top20 = feat_imp.nlargest(20)

    plt.figure(figsize=(8, 6))
    top20.sort_values().plot(kind="barh", color="steelblue", edgecolor="white")
    plt.title("Top 20 Feature Importances", fontsize=13)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("    Saved → outputs/feature_importance.png")

    return acc, f1


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — runs all steps in order
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_df, test_df           = load_data()
    X_tr, X_te, y_tr, y_te     = preprocess(train_df, test_df)
    model                       = train_model(X_tr, y_tr)
    acc, f1                     = evaluate(model, X_te, y_te)

    print(f"\n{'='*50}")
    print(f"  Final Accuracy : {acc*100:.2f}%")
    print(f"  Final F1 Score : {f1:.4f}")
    print(f"{'='*50}")
    print("\nNext step → run: streamlit run app.py")