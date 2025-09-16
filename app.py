import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Static list of users
users = {
    'admin': {'password': 'adminpass', 'role': 'Admin'},
    'doctor': {'password': 'doctorpass', 'role': 'Doctor'},
    'nurse': {'password': 'nursepass', 'role': 'Nurse'}
}

# Load models and scaler
readmission_model = joblib.load('readmission_model.pkl')
disease_outbreak_model, scaler, feature_names = joblib.load('disease_outbreak_model.pkl')
medication_adherence_model = joblib.load('medication_adherence_model.pkl')

def safe_date_parse(date_str):
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except Exception as e:
        return pd.NaT

def calculate_age(born):
    today = datetime.now()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

patients = pd.read_csv('SQL_CSV/patients_202408091143.csv')
encounters = pd.read_csv('SQL_CSV/encounters_202408091142.csv')
medications = pd.read_csv('SQL_CSV/medications_202408091143.csv')
conditions = pd.read_csv('SQL_CSV/conditions_202408091142.csv')
immunizations = pd.read_csv('SQL_CSV/immunizations_202408091143.csv')
claims = pd.read_csv('SQL_CSV/claims_202408091142.csv')
claims_transactions = pd.read_csv('SQL_CSV/claims_transactions_202408091142.csv')
payer_transitions = pd.read_csv('SQL_CSV/payer_transitions_202408091144.csv')

patients['BIRTHDATE'] = patients['BIRTHDATE'].apply(safe_date_parse)
encounters['START'] = pd.to_datetime(encounters['START'], errors='coerce')
encounters['STOP'] = pd.to_datetime(encounters['STOP'], errors='coerce')
medications['START'] = pd.to_datetime(medications['START'], errors='coerce')
medications['STOP'] = pd.to_datetime(medications['STOP'], errors='coerce')
conditions['START'] = pd.to_datetime(conditions['START'], errors='coerce')
immunizations['DATE'] = pd.to_datetime(immunizations['DATE'], errors='coerce')
claims['CURRENTILLNESSDATE'] = pd.to_datetime(claims['CURRENTILLNESSDATE'], errors='coerce')
claims['SERVICEDATE'] = pd.to_datetime(claims['SERVICEDATE'], errors='coerce')

# Utility function to create plot URLs
def create_plot():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = users.get(username)
        if user and user['password'] == password:
            session['username'] = username
            session['role'] = user['role']
            return redirect(url_for('index'))
        else:
            return "Invalid credentials", 401
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', role=session['role'])

@app.route('/univariate_analysis')
def univariate_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    plt.figure(figsize=(20, 20))
    
    patients['AGE'] = patients['BIRTHDATE'].apply(lambda x: calculate_age(x) if pd.notnull(x) else None)
    
    plt.subplot(3, 2, 1)
    sns.histplot(patients['AGE'].dropna(), kde=True)
    plt.title('Age Distribution')
    
    plt.subplot(3, 2, 2)
    patients['GENDER'].value_counts().plot(kind='bar')
    plt.title('Gender Distribution')
    
    plt.subplot(3, 2, 3)
    encounters['ENCOUNTERCLASS'].value_counts().plot(kind='bar')
    plt.title('Encounter Class Distribution')
    
    plt.subplot(3, 2, 4)
    medications['DESCRIPTION'].value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Medications')
    
    plt.subplot(3, 2, 5)
    conditions['DESCRIPTION'].value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Conditions')
    
    plt.subplot(3, 2, 6)
    immunizations['DESCRIPTION'].value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Immunizations')
    
    plt.tight_layout()
    plot_url = create_plot()
    plt.close()
    
    return render_template('univariate_analysis.html', plot_url=plot_url)

@app.route('/bivariate_analysis')
def bivariate_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    plt.figure(figsize=(20, 20))
    
    merged_data = encounters.merge(patients, left_on='PATIENT', right_on='Id')
    merged_data['AGE'] = merged_data['BIRTHDATE'].apply(lambda x: calculate_age(x) if pd.notnull(x) else None)
    
    plt.subplot(3, 2, 1)
    sns.scatterplot(data=merged_data, x='AGE', y='TOTAL_CLAIM_COST')
    plt.title('Age vs Total Claim Cost')
    
    plt.subplot(3, 2, 2)
    sns.boxplot(data=merged_data, x='GENDER', y='TOTAL_CLAIM_COST')
    plt.title('Gender vs Total Claim Cost')
    
    plt.subplot(3, 2, 3)
    sns.boxplot(data=merged_data, x='ENCOUNTERCLASS', y='TOTAL_CLAIM_COST')
    plt.title('Encounter Class vs Total Claim Cost')
    
    plt.subplot(3, 2, 4)
    medication_counts = medications.groupby('PATIENT').size().reset_index(name='MED_COUNT')
    merged_med_data = merged_data.merge(medication_counts, left_on='PATIENT', right_on='PATIENT')
    sns.scatterplot(data=merged_med_data, x='AGE', y='MED_COUNT')
    plt.title('Age vs Number of Medications')
    
    plt.subplot(3, 2, 5)
    condition_counts = conditions.groupby('PATIENT').size().reset_index(name='COND_COUNT')
    merged_cond_data = merged_data.merge(condition_counts, left_on='PATIENT', right_on='PATIENT')
    sns.scatterplot(data=merged_cond_data, x='AGE', y='COND_COUNT')
    plt.title('Age vs Number of Conditions')
    
    plt.subplot(3, 2, 6)
    sns.scatterplot(data=merged_cond_data, x='COND_COUNT', y='TOTAL_CLAIM_COST')
    plt.title('Number of Conditions vs Total Claim Cost')
    
    plt.tight_layout()
    
    plot_url = create_plot()
    plt.close()

    return render_template('bivariate_analysis.html', plot_url=plot_url)

@app.route('/temporal_analysis')
def temporal_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    plt.figure(figsize=(20, 20))
    
    plt.subplot(3, 2, 1)
    encounters_ts = encounters['START'].dt.to_period('M').value_counts().sort_index()
    encounters_ts.index = encounters_ts.index.to_timestamp()
    encounters_ts = encounters_ts.resample('M').asfreq().fillna(0)
    encounters_ts.plot()
    plt.title('Encounters Over Time')
    
    plt.subplot(3, 2, 2)
    conditions_ts = conditions['START'].dt.to_period('M').value_counts().sort_index()
    conditions_ts.index = conditions_ts.index.to_timestamp()
    conditions_ts = conditions_ts.resample('M').asfreq().fillna(0)
    conditions_ts.plot()
    plt.title('Conditions Over Time')
    
    plt.subplot(3, 2, 3)
    medications_ts = medications['START'].dt.to_period('M').value_counts().sort_index()
    medications_ts.index = medications_ts.index.to_timestamp()
    medications_ts = medications_ts.resample('M').asfreq().fillna(0)
    medications_ts.plot()
    plt.title('Medications Over Time')
    
    plt.subplot(3, 2, 4)
    immunizations_ts = immunizations['DATE'].dt.to_period('M').value_counts().sort_index()
    immunizations_ts.index = immunizations_ts.index.to_timestamp()
    immunizations_ts = immunizations_ts.resample('M').asfreq().fillna(0)
    immunizations_ts.plot()
    plt.title('Immunizations Over Time')
    
    plt.subplot(3, 2, 5)
    if len(encounters_ts) >= 24:
        result = seasonal_decompose(encounters_ts, model='additive', period=12)
        result.plot()
        plt.title('Seasonal Decomposition of Encounters')
    else:
        plt.text(0.5, 0.5, "Insufficient data for seasonal decomposition", 
                 ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.subplot(3, 2, 6)
    claim_cost_ts = encounters.set_index('START')['TOTAL_CLAIM_COST'].resample('M').mean()
    claim_cost_ts.plot()
    plt.title('Average Claim Cost Over Time')
    
    plt.tight_layout()
    
    plot_url = create_plot()
    plt.close()

    return render_template('temporal_analysis.html', plot_url=plot_url)

@app.route('/cohort_analysis')
def cohort_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    plt.figure(figsize=(20, 20))
    
    plt.subplot(2, 2, 1)
    patients['AGE'] = patients['BIRTHDATE'].apply(lambda x: calculate_age(x) if pd.notnull(x) else None)
    age_cohort = pd.cut(patients['AGE'], bins=[0, 18, 30, 50, 65, 100])
    cohort_data = encounters.merge(patients, left_on='PATIENT', right_on='Id')
    cohort_data.groupby(age_cohort)['TOTAL_CLAIM_COST'].mean().plot(kind='bar')
    plt.title('Age Cohort vs Average Claim Cost')
    
    plt.subplot(2, 2, 2)
    cohort_data.groupby('GENDER')['TOTAL_CLAIM_COST'].mean().plot(kind='bar')
    plt.title('Gender Cohort vs Average Claim Cost')
    
    plt.subplot(2, 2, 3)
    encounters.groupby('ENCOUNTERCLASS')['TOTAL_CLAIM_COST'].mean().plot(kind='bar')
    plt.title('Encounter Class Cohort vs Average Claim Cost')
    
    plt.subplot(2, 2, 4)
    condition_cohort = conditions.groupby('PATIENT').size().reset_index(name='CONDITION_COUNT')
    condition_cohort['CONDITION_GROUP'] = pd.cut(condition_cohort['CONDITION_COUNT'], bins=[0, 1, 3, 5, 10, 100])
    cohort_data = encounters.merge(condition_cohort, left_on='PATIENT', right_on='PATIENT')
    cohort_data.groupby('CONDITION_GROUP')['TOTAL_CLAIM_COST'].mean().plot(kind='bar')
    plt.title('Condition Count Cohort vs Average Claim Cost')
    
    plt.tight_layout()
    plot_url = create_plot()
    plt.close()
    
    return render_template('cohort_analysis.html', plot_url=plot_url)

@app.route('/data_quality')
def data_quality():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    plt.figure(figsize=(20, 20))
    
    datasets = {
        'Patients': patients,
        'Encounters': encounters,
        'Medications': medications,
        'Conditions': conditions,
        'Immunizations': immunizations,
        'Claims': claims
    }
    
    for i, (name, df) in enumerate(datasets.items(), 1):
        plt.subplot(3, 2, i)
        missing_data = df.isnull().sum() / len(df) * 100
        missing_data.plot(kind='bar')
        plt.title(f'Missing Data in {name} Dataset')
        plt.ylabel('% of Missing Values')
    
    plt.tight_layout()
    plot_url = create_plot()
    plt.close()
    
    return render_template('data_quality.html', plot_url=plot_url)

@app.route('/readmission', methods=['GET', 'POST'])
def readmission():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        patient_id = request.form['patient_id']
        
        # Fetch patient data from the dataset
        patient_data = patients[patients['Id'] == patient_id].iloc[0]
        encounter_data = encounters[encounters['PATIENT'] == patient_id].sort_values(by='START', ascending=False).iloc[0]

        age = (pd.to_datetime(encounter_data['START']).year - pd.to_datetime(patient_data['BIRTHDATE']).year)
        gender = 1 if patient_data['GENDER'] == 'M' else 0
        length_of_stay = (pd.to_datetime(encounter_data['STOP']) - pd.to_datetime(encounter_data['START'])).days
        total_cost = encounter_data['TOTAL_CLAIM_COST']

        input_data = np.array([[age, gender, length_of_stay, total_cost]])
        prediction = readmission_model.predict(input_data)[0]

        # Create a prediction message based on the model output
        if prediction == 1:
            prediction_message = "The patient is likely to be readmitted within 30 days."
            explanation = "Factors such as age, gender, length of stay, and total cost have contributed to this prediction."
        else:
            prediction_message = "The patient is not likely to be readmitted within 30 days."
            explanation = "Factors such as age, gender, length of stay, and total cost have contributed to this prediction."

        return render_template('readmission_result.html', 
                               patient_id=patient_id,
                               age=age, 
                               gender='Male' if gender == 1 else 'Female',
                               length_of_stay=length_of_stay, 
                               total_cost=total_cost, 
                               prediction_message=prediction_message,
                               explanation=explanation)
    
    return render_template('readmission_form.html')

@app.route('/disease_outbreak', methods=['GET', 'POST'])
def disease_outbreak():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        patient_id = request.form['patient_id']
        month = int(request.form['month'])
        year = int(request.form['year'])
        
        # Fetch the patient's data from the dataset
        patient_data = patients[patients['Id'] == patient_id]
        
        if patient_data.empty:
            return "Patient ID not found", 404
        
        # Extract features from patient data
        age = (datetime(year, month, 1) - pd.to_datetime(patient_data['BIRTHDATE'].values[0])).days // 365
        
        # Get the past conditions
        past_conditions_raw = conditions[conditions['PATIENT'] == patient_id]['DESCRIPTION'].tolist()
        
        # Clean the conditions list: remove duplicates and focus on key conditions
        past_conditions_set = set(past_conditions_raw)  # Remove duplicates by converting to set
        ignore_terms = ['Full-time employment (finding)', 'Part-time employment (finding)', 'Social isolation (finding)']
        significant_conditions = [condition for condition in past_conditions_set if condition not in ignore_terms]

        # Vaccination status
        vaccination_status = immunizations[immunizations['PATIENT'] == patient_id].shape[0]
        
        # Geographical region
        region = patient_data['STATE'].values[0]
        
        # Input data preparation
        input_data = np.array([[
            month,
            year,
            datetime(year, month, 1).timetuple().tm_yday,
            age,
            len(significant_conditions),  # Number of significant past conditions
            vaccination_status  # Number of vaccines received
        ]])
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Make the prediction
        prediction = disease_outbreak_model.predict(input_data_scaled)[0]
        
        # Convert prediction to a binary outcome
        prediction_binary = 1 if prediction > 0.5 else 0  # Assuming a threshold of 0.5 for high/low risk
        
        return render_template('disease_outbreak_result.html', 
                               patient_id=patient_id, 
                               month=month, 
                               year=year, 
                               prediction=prediction_binary,
                               age=age,
                               past_conditions=significant_conditions,
                               vaccination_status=vaccination_status,
                               region=region)
    
    return render_template('disease_outbreak_form.html')

@app.route('/medication_adherence', methods=['GET', 'POST'])
def medication_adherence():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        patient_id = request.form['patient_id']
        
        # Fetch the patient's data from the dataset
        patient_data = patients[patients['Id'] == patient_id]
        medication_data = medications[medications['PATIENT'] == patient_id]
        encounter_data = encounters[encounters['PATIENT'] == patient_id]
        
        if patient_data.empty or medication_data.empty or encounter_data.empty:
            return "Patient ID not found or insufficient data", 404
        
        # Extract necessary details
        age = (pd.to_datetime('today') - pd.to_datetime(patient_data['BIRTHDATE'].values[0])).days // 365
        gender = 1 if patient_data['GENDER'].values[0] == 'M' else 0
        total_cost = medication_data['TOTALCOST'].sum()
        
        # Prepare the input data with only the features the model expects
        input_data = np.array([[age, gender, total_cost]])
        prediction_prob = medication_adherence_model.predict_proba(input_data)[0][1]
        
        # Convert prediction to a binary outcome
        prediction = 1 if prediction_prob > 0.5 else 0
        
        return render_template('medication_adherence_result.html', 
                               prediction=prediction, 
                               age=age, 
                               gender='Male' if gender == 1 else 'Female', 
                               total_cost=total_cost,
                               num_medications=None,  # Not used
                               num_encounters=None,  # Not used
                               chronic_conditions=None)  # Not used)
    
    return render_template('medication_adherence_form.html')

if __name__ == '__main__':
    app.run(debug=True)
