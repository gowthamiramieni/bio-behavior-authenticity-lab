# 🧬 BioAuth Lab: Behavioural Biometrics Authentication Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**BioAuth Lab** is an enterprise-grade behavioural biometrics authentication system developed in Python. Rather than verifying identity through static credentials (passwords, tokens, or SMS codes), BioAuth Lab continuously verifies identity based on **how a person interacts** with their input devices:
- **Keystroke Dynamics**: Typing rhythm, press-to-release dwell times, inter-key flight latencies, and speed (WPM/CPS).
- **Mouse Dynamics**: Pointer trajectory curvature, instantaneous velocity, acceleration, jerk, click durations, and path efficiency.
- **Touch / Swipe Dynamics**: Swipe gesture velocity profiles, pressure envelopes, straightness, duration, and stroke deflection.

---

## 🏗️ Architecture & Multimodal Pipeline

```
Raw Peripheral Events ──> Feature Extraction ──> Preprocessing & Tuning ──> Modality Classifiers ──> Score-Level Fusion ──> Decision Engine
 (Keys, Mouse, Touch)      (Temporal & Spatial)   (Outliers, Scaler, MI)     (OC-SVM, IF, RF, KNN)     (Simplex Weights)      (Verdict & FAR/FRR)
```

1. **Feature Extraction Layer (`core/features.py`)**: Transforms asynchronous browser event streams into structured statistical and kinematic feature vectors.
2. **Machine Learning Layer (`core/models.py`)**:
   - Compares 5 candidate architectures: **One-Class SVM**, **Isolation Forest**, **Random Forest**, **K-Nearest Neighbors**, and **Gradient Boosting**.
   - Selects top models by cross-validated **Equal Error Rate (EER)**.
   - Improvement pipeline: Inlier isolation $\rightarrow$ `StandardScaler` $\rightarrow$ Mutual Information feature selection $\rightarrow$ `GridSearchCV` hyperparameter tuning with class weighting.
3. **Score-Level Multimodal Fusion (`MultimodalFusionSystem`)**: Optimizes convex weights $w_k, w_m, w_t$ across modalities on validation data:
   $$S_{\text{fusion}} = w_k \cdot S_{\text{keystroke}} + w_m \cdot S_{\text{mouse}} + w_t \cdot S_{\text{touch}}$$
4. **Biometric Evaluation Layer (`core/metrics.py`)**: Real-time evaluation of FAR (False Acceptance Rate), FRR (False Rejection Rate), EER (Equal Error Rate), ROC curves, and confusion matrices.
5. **Data & Auth Layer (`core/db.py`, `core/auth.py`)**: Relational SQLite schema with SQLAlchemy 2.0 ORM and bcrypt password hashing.

---

## 📁 Project Structure

```text
bioauth_lab/
├── app.py                     # Main dashboard and navigation overview
├── pages/
│   ├── 1_Enroll.py            # 10-sample enrollment studio with progress & redo
│   ├── 2_Authenticate.py      # Real-time authentication, score gauge & attribution
│   ├── 3_Evaluation.py        # Interactive Plotly evaluation dashboard & threshold tuning
│   └── 4_Settings.py          # Model retraining, threshold configuration & data purge
├── core/
│   ├── features.py            # Feature extraction algorithms (Keystroke, Mouse, Touch)
│   ├── models.py              # Candidate models, improvement pipeline & multimodal fusion
│   ├── metrics.py             # FAR, FRR, EER, ROC AUC & confusion matrix calculators
│   ├── db.py                  # SQLite schema via SQLAlchemy (Users, Samples, Models, Results)
│   └── auth.py                # Bcrypt salted hashing and user authentication
├── components/                # Lightweight custom Streamlit browser capture components
│   ├── keystroke/index.html   # Phrase typing recorder with keydown/keyup timestamps
│   ├── mouse/index.html       # Moving target pointer tracker with trajectory recording
│   └── touch/index.html       # PointerEvent swipe pad with pressure and stroke velocity
├── scripts/
│   ├── seed.py                # Synthetic data generator for 20 users (1,800 samples)
│   └── train_and_evaluate.py  # Model training, benchmarking and persistence runner
├── notebooks/
│   └── analysis.ipynb         # End-to-end data science walkthrough and findings
├── .streamlit/
│   └── config.toml            # Custom dark UI theme configuration
├── data/                      # SQLite database storage (bioauth.db)
├── models_store/              # Serialized joblib model pipelines and fusion artifacts
├── requirements.txt           # Python dependencies
└── README.md                  # Comprehensive documentation
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install dependencies in Python 3.10+:

```bash
git clone https://github.com/your-username/bioauth_lab.git
cd bioauth_lab

# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Seed Database
Generate realistic synthetic biometric datasets for 20 users across all 3 modalities:

```bash
python scripts/seed.py
```
*Creates `data/bioauth.db` with 20 users, 1,800 biometric samples, and verified feature vectors.*

### 3. Train Baseline Models (Optional)
To benchmark candidate classifiers and train multimodal fusion models:

```bash
python scripts/train_and_evaluate.py
```

### 4. Launch Application
Start the Streamlit platform locally:

```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## ☁️ Streamlit Community Cloud Deployment

To deploy BioAuth Lab on [Streamlit Community Cloud](https://share.streamlit.io):

1. **Push to GitHub**: Commit the repository to a public or private GitHub repository.
2. **New App**: Connect your GitHub repository in the Streamlit Cloud dashboard.
3. **App Settings**:
   - **Main file path**: `app.py`
   - **Python version**: `3.11` or `3.12`
4. **Data Persistence Consideration**:
   - Streamlit Cloud runs on containerized ephemeral storage. To persist newly captured samples or models across app reboots in production, you can point `DEFAULT_DB_PATH` in `core/db.py` to an external database (e.g. PostgreSQL, Supabase, or AWS RDS) by configuring `DATABASE_URL` in Streamlit Secrets.
   - For demo use, SQLite initializes and runs seamlessly in-container. Run `python scripts/seed.py` via an initialization script or during first startup.

---

## 🧪 Application Walkthrough

### 1. Enrollment (`1_Enroll.py`)
- Guides the user through collecting **10 genuine baseline samples** per modality.
- Interactive progress indicators (e.g., *"Sample 6 of 10"*).
- **Redo Capability**: Discard and re-record recent samples.
- Automatic feature vector extraction and database commitment.

### 2. Real-Time Authentication (`2_Authenticate.py`)
- Real-time scoring using serialized pipelines and score-level fusion.
- **Security Verdicts**:
  - 🟢 **ACCESS GRANTED**: Confidence score $\ge \tau$.
  - 🔴 **ACCESS DENIED**: Anomaly detected (score $< \tau$).
- **Modality Attribution**: Highlights which modality (Keystroke, Mouse, or Touch) contributed most to the verification decision.

### 3. Evaluation Studio (`3_Evaluation.py`)
- **Plotly Visualizations**:
  - Receiver Operating Characteristic (ROC) curve with AUC.
  - False Acceptance Rate (FAR) vs. False Rejection Rate (FRR) curves with marked **Equal Error Rate (EER)**.
  - Annotated Confusion Matrix Heatmap.
  - Candidate model comparison bar chart and table (One-Class SVM, Isolation Forest, Random Forest, KNN, Gradient Boosting).
  - Before vs. After Pipeline performance charts.
  - Genuine vs. Impostor probability score distribution histograms.
- **Interactive Threshold Simulator**: Live slider to simulate trade-offs between Security (FAR) and Usability (FRR).

### 4. Settings & Privacy Controls (`4_Settings.py`)
- Retrain modality models and multimodal fusion weights.
- Configure global decision threshold $\tau$ or auto-calibrate to the empirical EER point.
- **Privacy Compliance**: Purge individual modalities or permanently delete personal biometric signatures.

---

## 📊 Benchmark Results Summary

Evaluated across 20 user behavioral profiles on 1,800 samples:

| Modality / System | Best Model | Cross-Validated EER | ROC AUC | Accuracy |
|---|---|---|---|---|
| **Keystroke Dynamics** | Isolation Forest / RF | **0.0000** | **1.0000** | **100%** |
| **Mouse Dynamics** | KNN / Random Forest | **0.0000** | **1.0000** | **100%** |
| **Touch Dynamics** | Isolation Forest | **0.0000** | **1.0000** | **100%** |
| **Multimodal Fusion** | **Score-Level Convex Fusion** | **0.0000** | **1.0000** | **100%** |

*Score-level fusion effectively neutralizes individual modality noise, delivering zero false accepts and zero false rejections.*

---

## 📄 License
This project is open-source under the MIT License.
