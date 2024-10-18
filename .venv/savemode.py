import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import pickle

# Load and preprocess the data
def load_and_preprocess_data():
    # Load the dataset from CSV file
    df = pd.read_csv('soil.csv')
    
    # Split data into features and target
    X = df[['ph_cacl2', 'ph_h20', 'ec', 'oc', 'caco3', 'p', 'n', 'k']]
    y = df['Rel_soil_fertility_category']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train the models
def train_models(X_train, y_train):
    dt_model = DecisionTreeClassifier(random_state=42)
    svm_model = SVC(probability=True, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(random_state=42)
    
    dt_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    
    ensemble_model = VotingClassifier(
        estimators=[
            ('dt', dt_model),
            ('svm', svm_model),
            ('rf', rf_model),
            ('xgb', xgb_model)
        ],
        voting='soft'
    )
    ensemble_model.fit(X_train, y_train)
    
    return dt_model, svm_model, rf_model, xgb_model, ensemble_model

# Save models
def save_models(dt_model, svm_model, rf_model, xgb_model, ensemble_model, scaler):
    with open('dt_model.pkl', 'wb') as f:
        pickle.dump(dt_model, f)
    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    with open('ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    dt_model, svm_model, rf_model, xgb_model, ensemble_model = train_models(X_train, y_train)
    save_models(dt_model, svm_model, rf_model, xgb_model, ensemble_model, scaler)
    print("Models trained and saved successfully.")