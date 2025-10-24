<p align="center">
  <img src="https://raw.githubusercontent.com/KaanSulkalar/NetworkAnomalyDetection/main/assets/header_banner.png" alt="AI-Driven Network Traffic Anomaly Detection" width="100%">
</p>

# AI-Driven Network Traffic Anomaly Detection System

## Project Overview
This project aims to detect anomalies in network traffic using Machine Learning techniques.

**Goal:** Identify malicious or abnormal network behavior from traffic datasets such as DDoS, Port Scans, or Brute-force attacks.  
**Tech Stack:** Python, scikit-learn, pandas, matplotlib, plotly, Streamlit  
**Datasets:** UNSW-NB15 (recommended), CICIDS2017 (optional)  
**Platform:** macOS / Linux / Windows

---

## Run Instructions

### 1. Environment Setup
```bash
git clone https://github.com/KaanSulkalar/NetworkAnomalyDetection.git
cd NetworkAnomalyDetection
pip install -r requirements.txt
2. Run Jupyter Notebooks
cd notebooks
jupyter notebook
3. Launch Streamlit Dashboard
cd dashboard
streamlit run app.py
The app will open in your browser at:
http://localhost:8502
Project Pipeline
1. Data Preprocessing
Load and clean the UNSW-NB15 dataset
Encode categorical features
Normalize numeric features
Split data into training and testing sets
2. Model Training
Train supervised and unsupervised models
Random Forest for classification
Isolation Forest for anomaly detection
3. Evaluation Metrics
Accuracy, Precision, Recall, and F1-Score
ROC-AUC and PR Curves
Confusion Matrix and Feature Importance
4. Dashboard Visualization
Upload CSV data and perform predictions
Interactive performance charts and anomaly distribution
Display model results and downloadable outputs
Directory Structure
NetworkAnomalyDetection/
│
├── data/
│   └── UNSW_NB15.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_dashboard_demo.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── realtime_capture.py
│
├── dashboard/
│   └── app.py
│
├── reports/
│   ├── Project_Report.pdf
│   └── metrics.xlsx
│
└── requirements.txt
Deployed Version
Live app (Streamlit Community Cloud):
https://kaansulkalar-networkanomalydetection.streamlit.app
Future Enhancements
Deep learning (Autoencoder) anomaly detection module
Real-time PCAP traffic capture and scoring
Dockerized cloud deployment
Integration with Azure Monitor or ELK Stack for SIEM analysis
Dashboard Preview
<p align="center"> <img src="https://raw.githubusercontent.com/KaanSulkalar/NetworkAnomalyDetection/main/assets/dashboard_preview.png" alt="Dashboard Preview" width="80%"> </p> ```