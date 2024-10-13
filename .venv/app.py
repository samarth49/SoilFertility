from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the pre-trained model and scaler (assuming these are pre-trained and saved)
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Home route to display the HTML form
@app.route('/')
def home():
    return render_template('prediction.html')

# Route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    ph_cacl2 = float(request.form['ph_cacl2'])
    ph_h20 = float(request.form['ph_h20'])
    ec = float(request.form['ec'])
    oc = float(request.form['oc'])
    caco3 = float(request.form['caco3'])
    p = float(request.form['p'])
    n = float(request.form['n'])
    k = float(request.form['k'])

    # Prepare the input data
    input_data = pd.DataFrame([[ph_cacl2, ph_h20, ec, oc, caco3, p, n, k]], 
                              columns=['ph_cacl2', 'ph_h20', 'ec', 'oc', 'caco3', 'p', 'n', 'k'])
    
    # Normalize the input
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(input_data_scaled)

    # Mapping prediction to fertility category
    fertility_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    result = fertility_map[prediction[0]]

    # Return the result back to the user
    return render_template('prediction.html', prediction_text=f"Predicted Soil Fertility Level: {result}")

if __name__ == '__main__':
    app.run(debug=True)
