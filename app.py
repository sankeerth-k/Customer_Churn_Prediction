from flask import Flask, request, render_template, send_file
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the saved model
with open('logistic_regression_model.joblib', 'rb') as f:
    model = joblib.load(f)

# Updated selected features for the model after preprocessing
selected_features = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check','StreamingBoth', 'AvgMonthlyCharges'
]

# Route for the homepage with a form
@app.route('/')
def home():
    return render_template('index.html')  # Ensure the template exists

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting form data and converting to a pandas DataFrame
        data = request.form

        # Prepare input data based on the features provided in the form
        input_data = {
            'gender': [data.get('gender')],
            'SeniorCitizen': [int(data.get('SeniorCitizen'))],
            'Partner': [data.get('Partner')],
            'Dependents': [data.get('Dependents')],
            'tenure': [float(data.get('tenure'))],
            'PhoneService': [data.get('PhoneService')],
            'MultipleLines': [data.get('MultipleLines')],
            'InternetService': [data.get('InternetService')],
            'OnlineSecurity': [data.get('OnlineSecurity')],
            'OnlineBackup': [data.get('OnlineBackup')],
            'DeviceProtection': [data.get('DeviceProtection')],
            'TechSupport': [data.get('TechSupport')],
            'StreamingTV': [data.get('StreamingTV')],
            'StreamingMovies': [data.get('StreamingMovies')],
            'Contract': [data.get('Contract')],
            'PaperlessBilling': [data.get('PaperlessBilling')],
            'PaymentMethod': [data.get('PaymentMethod')],
            'MonthlyCharges': [float(data.get('MonthlyCharges'))],
            'TotalCharges': [float(data.get('TotalCharges'))]
        }

        # Creating a DataFrame from the input data
        df = pd.DataFrame(input_data)

        # Preprocessing: One-Hot Encoding for categorical features
        df_encoded = pd.get_dummies(df, drop_first=True)

        # Scaling numerical features
        scaler = MinMaxScaler()
        numerical_features = ['TotalCharges']
        df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

        # Feature Engineering
        # New feature: Does the customer subscribe to both StreamingTV and StreamingMovies?
        df_encoded['StreamingBoth'] = np.where(
            (df_encoded.get('StreamingTV_Yes', 0) == 1) & (df_encoded.get('StreamingMovies_Yes', 0) == 1), 1, 0)

        # New feature: Average Monthly Charges over tenure
        df_encoded['AvgMonthlyCharges'] = df_encoded['TotalCharges'] / df_encoded['tenure']
        df_encoded['AvgMonthlyCharges'].fillna(0, inplace=True)  # Handling division by zero

        # Align the DataFrame with the selected features (to prevent mismatch)
        df_encoded = df_encoded.reindex(columns=selected_features, fill_value=0)

        # Model prediction
        prediction = model.predict(df_encoded)

        # Sending prediction result to the front-end
        result = 'Churn' if prediction[0] == 1 else 'No Churn'
        return render_template('index.html', prediction=result)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if no port is provided
    app.run(host='0.0.0.0', port=port)
