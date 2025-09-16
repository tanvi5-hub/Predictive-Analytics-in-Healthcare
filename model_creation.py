

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

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


patients['BIRTHDATE'] = patients['BIRTHDATE'].apply(safe_date_parse)
encounters['START'] = pd.to_datetime(encounters['START'], errors='coerce')
encounters['STOP'] = pd.to_datetime(encounters['STOP'], errors='coerce')
medications['START'] = pd.to_datetime(medications['START'], errors='coerce')
conditions['START'] = pd.to_datetime(conditions['START'], errors='coerce')


def create_readmission_model():
    print("Creating readmission model...")
    encounters['LENGTH_OF_STAY'] = (encounters['STOP'] - encounters['START']).dt.days
    encounters['READMISSION'] = encounters.groupby('PATIENT')['START'].diff().dt.days.shift(-1) <= 30
    
    data = encounters.merge(patients[['Id', 'BIRTHDATE', 'GENDER']], left_on='PATIENT', right_on='Id')
    data['AGE'] = data['BIRTHDATE'].apply(lambda x: calculate_age(x) if pd.notnull(x) else None)
    
    X = data[['AGE', 'GENDER', 'LENGTH_OF_STAY', 'TOTAL_CLAIM_COST']].dropna()
    y = data['READMISSION'].dropna()
    
    X = pd.get_dummies(X, columns=['GENDER'], drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'readmission_model.pkl')
    print("Readmission model saved as 'readmission_model.pkl'")



def create_disease_outbreak_model():
    print("Creating disease outbreak model...")
    conditions = pd.read_csv('SQL_CSV/conditions_202408091142.csv')
    conditions['START'] = pd.to_datetime(conditions['START'], errors='coerce')
    
    
    monthly_counts = conditions.groupby(conditions['START'].dt.to_period('M')).size().reset_index(name='COUNT')
    monthly_counts['START'] = monthly_counts['START'].dt.to_timestamp()
    
    
    monthly_counts['MONTH'] = monthly_counts['START'].dt.month
    monthly_counts['YEAR'] = monthly_counts['START'].dt.year
    monthly_counts['DAY_OF_YEAR'] = monthly_counts['START'].dt.dayofyear
    
    
    for i in range(1, 4):  
        monthly_counts[f'LAG_{i}'] = monthly_counts['COUNT'].shift(i)
    
    
    monthly_counts = monthly_counts.dropna()
    
    
    X = monthly_counts[['MONTH', 'YEAR', 'DAY_OF_YEAR', 'LAG_1', 'LAG_2', 'LAG_3']]
    y = monthly_counts['COUNT']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    
    joblib.dump((model, scaler, X.columns.tolist()), 'disease_outbreak_model.pkl')
    print("Disease outbreak model saved as 'disease_outbreak_model.pkl'")
    print(f"Number of features: {X.shape[1]}")



def safe_date_parse(date_str):
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except Exception as e:
        print(f"Error parsing date: {date_str}. Error: {e}")
        return pd.NaT

def calculate_age(born):
    today = datetime.now()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def create_medication_adherence_model():
    print("Creating medication adherence model...")
    
    
    medications = pd.read_csv('SQL_CSV/medications_202408091143.csv')
    patients = pd.read_csv('SQL_CSV/patients_202408091143.csv')
    
    
    print("Medications DataFrame Info:")
    print(medications.info())
    print("\nPatients DataFrame Info:")
    print(patients.info())
    
    
    medications['START'] = medications['START'].apply(safe_date_parse)
    medications['STOP'] = medications['STOP'].apply(safe_date_parse)
    patients['BIRTHDATE'] = patients['BIRTHDATE'].apply(safe_date_parse)
    
    
    med_data = medications.merge(patients[['Id', 'BIRTHDATE', 'GENDER']], left_on='PATIENT', right_on='Id')
    
    
    med_data['AGE'] = med_data['BIRTHDATE'].apply(lambda x: calculate_age(x) if pd.notnull(x) else None)
    
    
    med_data['DURATION'] = (med_data['STOP'] - med_data['START']).dt.total_seconds() / (24 * 60 * 60)
    med_data['ADHERENCE'] = med_data['DURATION'].ge(30).astype(float)
    
    
    print("\nMerged Data Info:")
    print(med_data.info())
    print("\nSample of merged data:")
    print(med_data.head())
    
    
    valid_data = med_data.dropna(subset=['AGE', 'GENDER', 'TOTALCOST', 'ADHERENCE'])
    
    
    print("\nValid Data Info:")
    print(valid_data.info())
    
    
    X = valid_data[['AGE', 'GENDER', 'TOTALCOST']]
    y = valid_data['ADHERENCE']
    
    X = pd.get_dummies(X, columns=['GENDER'], drop_first=True)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    
    joblib.dump(model, 'medication_adherence_model.pkl')
    print("Medication adherence model saved as 'medication_adherence_model.pkl'")



if __name__ == "__main__":
    
    create_disease_outbreak_model()
    
    print("All models created successfully!")