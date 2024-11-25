from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Load the trained model and columns
model = joblib.load('stroke_prediction_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Stroke Prediction API! Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON data
    data = request.json
    
    # Convert to DataFrame
    user_data = pd.DataFrame([data])
    
    # Reorder columns to match the model's training data
    for col in model_columns:
        if col not in user_data.columns:
            user_data[col] = 0  # Add missing columns
    user_data = user_data[model_columns]
    
    # Handle NaN values (e.g., from 'children' or 'Unknown')
    user_data.fillna(0, inplace=True)
    
    # Make prediction
    probability = model.predict_proba(user_data)[:, 1][0]  # Probability for stroke
    return jsonify({'stroke_likelihood': f"{probability * 100:.2f}%"})

if __name__ == '__main__':
    app.run(debug=True)
