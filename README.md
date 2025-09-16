# Predictive Analytics in Healthcare
📌 Overview

This project develops a predictive analytics system for healthcare that forecasts:

1. Patient readmissions
2. Disease outbreaks
3. Medication adherence

The goal is to support proactive decision-making by healthcare providers, improving patient outcomes and optimising resource allocation.
Built using Python (Flask backend), React (frontend), and MySQL (database), the system integrates advanced machine learning models into a user-friendly web application.

🎯 Key Features

1. Patient Readmission Prediction – XGBoost & Gradient Boosting Classifier
2. Disease Outbreak Forecasting – SARIMA & LSTM models
3. Medication Adherence Prediction – Logistic Regression & SVM
4. Synthetic Healthcare Data – Generated with Synthea for privacy compliance
5. Web Application – Interactive React dashboard with Flask backend APIs
6. Data Pipeline – Cleaning, preprocessing, feature engineering, and bias mitigation
7. Evaluation – Metrics include accuracy, precision, recall, F1-score, ROC-AUC, RMSE

🏗️ System Architecture

1. Frontend: React (user input forms, visualisations, dashboards)
2. Backend: Flask (API endpoints, model integration, authentication)
3. Database: MySQL (synthetic EHR-like data – patients, encounters, conditions, medications, immunisations)
4. Models: Trained and validated in Jupyter Notebook, then deployed via Flask APIs
5. Deployment: Local testing environment, modular design for future scalability

📊 Dataset

1. Source: Synthea synthetic healthcare data
2. Tables: Patients, Encounters, Conditions, Medications, Immunisations
3. Preprocessing: Missing value handling, outlier treatment, normalisation, log transformation, feature engineering (age bins, interaction terms, immunisation counts)

🚀 Installation & Setup

Prerequisites:
1. Python 3.9+
2. Node.js & npm
3. MySQL

Steps:
1. Clone the repo
git clone git@github.com:tanvi5-hub/Predictive-Analytics-in-Healthcare.git
cd Predictive-Analytics-in-Healthcare

2. Backend setup (Flask + Models)
cd backend
pip install -r requirements.txt
flask run

3. Frontend setup (React)
cd frontend
npm install
npm start

4. Database setup (MySQL)
Import synthetic dataset (SQL dump in /data/ if provided).
Update database connection in config.py.

📈 Results

1. Patient Readmission Prediction: Accuracy ~85% (XGBoost performed best)
2. Disease Outbreak Prediction: LSTM superior for long-term forecasting, SARIMA for seasonal trends
3. Medication Adherence Prediction: SVM slightly outperformed Logistic Regression (higher recall & AUC)
4. Models demonstrated robust performance across multiple validation techniques

🔒 Ethical Considerations

1. Uses synthetic data to ensure privacy (HIPAA/GDPR compliant)
2. Bias mitigation strategies applied (balanced datasets, fairness checks)
3. Designed for potential real-world extension with strict ethical governance

📌 Future Work

1. Add Explainable AI for model transparency
2. Integrate real-time EHR data via secure APIs
3. Enhance scalability with cloud deployment (Docker/Kubernetes)
4. Extend predictive models to cover more healthcare outcomes

🧑‍💻 Author

Tanvi Patil
MSc Computer Science, University of Southampton (2024)
