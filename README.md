# Fertility Prediction System

The Fertility Prediction System is a Flask-based web application designed to predict soil fertility categories (Low, Medium, High) based on various soil properties. The application uses machine learning models to provide individual and ensemble predictions for better accuracy and reliability.

## Features

- **Machine Learning Models**: Includes Decision Tree, Support Vector Machine (SVM), Random Forest, XGBoost, and an Ensemble model.
- **Data Normalization**: Utilizes a pre-trained scaler for accurate predictions.
- **User-Friendly Interface**: Web-based form to input soil parameters and view results.
- **Comprehensive Results**: Displays predictions from all individual models and the ensemble model.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or above
- Flask
- Pandas
- Pickle (for loading pre-trained models)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FertilityPrediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd FertilityPrediction
   ```

3. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Open your web browser and go to `http://127.0.0.1:5000/`.

## Usage

1. Fill in the form with the following soil parameters:
   - **pH (CaCl2)**
   - **pH (H2O)**
   - **EC (Electrical Conductivity)**
   - **OC (Organic Carbon)**
   - **CaCO3**
   - **P (Phosphorus)**
   - **N (Nitrogen)**
   - **K (Potassium)**

2. Submit the form to get predictions for:
   - Decision Tree
   - SVM
   - Random Forest
   - XGBoost
   - Ensemble Model

3. View the predicted fertility category (Low, Medium, or High) for each model.

## Models and Methodology

- **Decision Tree**: A simple and interpretable model for classification.
- **SVM**: Robust in handling complex data boundaries.
- **Random Forest**: An ensemble method for improved accuracy.
- **XGBoost**: High-performance gradient boosting algorithm.
- **Ensemble Model**: Combines predictions from all models for a more reliable output.

## File Structure

- **app.py**: Main Flask application file.
- **templates/prediction.html**: HTML file for the web interface.
- **dt_model.pkl**: Pre-trained Decision Tree model.
- **svm_model.pkl**: Pre-trained SVM model.
- **rf_model.pkl**: Pre-trained Random Forest model.
- **xgb_model.pkl**: Pre-trained XGBoost model.
- **ensemble_model.pkl**: Pre-trained Ensemble model.
- **scaler.pkl**: Pre-trained scaler for data normalization.


## Contact

- **Name**: Samarth Otari
- **GitHub**: https://github.com/samarth49
- **Email**: otarisamarth49@gmail.com

