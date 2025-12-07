# Explainable AI for Robot Telemetry Analysis 🤖🔍

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6%2B-green)](https://xgboost.readthedocs.io/)

This repository contains the code, models, and analysis for a Machine Learning project focused on detecting **Cyber-Physical Attacks (DoS)** and **System Malfunctions** in robotic drones using telemetry data. 

A core focus of this project is **Explainable AI (XAI)**—using SHAP and LIME to interpret *why* the models predict a specific failure mode, providing transparency for safety-critical autonomous systems.

---

## 📂 Project Structure

```text
├── saved_models/                         # Trained models (.json, .keras) and scalers (.pkl)
├── Datasets/                             # .csv files used in assignment
├── Robot_Telemetry_XAI_Assignment.ipynb  # Main Jupyter Notebook with all training & analysis code
├── AAI project report.pdf                # Comprehensive 10-page technical report
└── requirements.txt                      # List of Python dependencies
```
---

## 🚀 Key Features

### 1. Robust Data Preprocessing
- **Asynchronous Data Handling:** Implemented Forward Fill (`ffill`) logic to align sensor streams (GPS, IMU, Battery) that publish at different frequencies.
- **Noise Reduction:** Applied statistical cleaning and Standardization (`StandardScaler`) to normalize high-variance sensor data.

### 2. Multi-Model Architecture
We implemented and compared three distinct architectures:
- **Model 2.2: 1D-CNN (Convolutional Neural Network):** Captures temporal patterns and local dependencies in sensor sequences.
- **Model 2.4: XGBoost (Gradient Boosting):** High-performance decision tree ensemble for structured telemetry data.
- **Model 2.6: FNN (Feedforward Neural Network):** Deep learning baseline with Batch Normalization and Dropout.

### 3. Explainable AI (XAI)
- **SHAP (SHapley Additive exPlanations):** Global interpretation of feature impact (e.g., "How does low battery voltage influence the DoS prediction?").
- **LIME (Local Interpretable Model-agnostic Explanations):** Local interpretation of specific crash instances (e.g., "Why was this specific flight segment flagged as a Malfunction?").

---

## 📊 Performance & Results

| Model | Accuracy | Strengths |
| :--- | :--- | :--- |
| **XGBoost** | **99.2%** | Best overall accuracy; highly efficient training; excellent handling of sensor thresholds. |
| **1D-CNN** | 96.5% | Good at detecting "jittery" malfunctions; requires more data for temporal learning. |
| **FNN** | 97.8% | Strong baseline; captures complex non-linear interactions. |

> **Key Insight:** The XAI analysis revealed that **Battery Voltage** dynamics are the primary indicator for DoS attacks (due to telemetry freezing), while **Angular Velocity (Gyroscope)** variance is the fingerprint for physical malfunctions.

---

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Robot-Telemetry-XAI.git
cd Robot-Telemetry-XAI
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Code
You can open the Robot_Telemetry_XAI_Assignment.ipynb in Jupyter Lab, VS Code, or Google Colab.
To load the pre-trained models for inference:

```Python
import joblib
import xgboost as xgb
import tensorflow as tf

# Load Scaler & Encoder
scaler = joblib.load('saved_models/scaler.pkl')
le = joblib.load('saved_models/label_encoder.pkl')

# Load XGBoost
model_xgb = xgb.XGBClassifier()
model_xgb.load_model('saved_models/xgboost_model.json')

# Load FNN
model_fnn = tf.keras.models.load_model('saved_models/fnn_model.keras')

print("Models loaded successfully!")

```

### 📈 Visualizations
- **SHAP Summary Plot:** Shows the global impact of top features like Battery Voltage and IMU Yaw Rate.
- **LIME Explanation:** Explains a single "Malfunction" prediction based on high angular velocity.

***(See the AAI project report.pdf and Robot_Telemetry_XAI_Assignment.ipynb for full high-resolution plots)***
