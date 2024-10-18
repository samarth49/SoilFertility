from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained models and scaler
with open('dt_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('ensemble_model.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Home route to display the HTML form
@app.route('/')
def home():
    return render_template('prediction.html')

# Route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input_data = {
        'ph_cacl2': float(request.form['ph_cacl2']),
        'ph_h20': float(request.form['ph_h20']),
        'ec': float(request.form['ec']),
        'oc': float(request.form['oc']),
        'caco3': float(request.form['caco3']),
        'p': float(request.form['p']),
        'n': float(request.form['n']),
        'k': float(request.form['k'])
    }

    # Prepare the input data
    input_df = pd.DataFrame([input_data])

    # Normalize the input
    input_data_scaled = scaler.transform(input_df)

    # Make predictions using individual models and ensemble
    dt_pred = dt_model.predict(input_data_scaled)
    svm_pred = svm_model.predict(input_data_scaled)
    rf_pred = rf_model.predict(input_data_scaled)
    xgb_pred = xgb_model.predict(input_data_scaled)
    ensemble_pred = ensemble_model.predict(input_data_scaled)

    # Mapping prediction to fertility category
    fertility_map = {0: 'Low', 1: 'Medium', 2: 'High'}

    # Prepare results
    results = {
        'Decision Tree': fertility_map[dt_pred[0]],
        'SVM': fertility_map[svm_pred[0]],
        'Random Forest': fertility_map[rf_pred[0]],
        'XGBoost': fertility_map[xgb_pred[0]],
        'Ensemble': fertility_map[ensemble_pred[0]]
    }

    # Return the results back to the user
    return render_template('prediction.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)