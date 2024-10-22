from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# File to store user input data
DATA_FILE = 'user_data.csv'

# Initialize the data file if not present
if not os.path.exists(DATA_FILE):
    df = pd.DataFrame(columns=['result1', 'result2', 'result3', 'result4', 'result5', 'result6', 'predicted_result'])
    df.to_csv(DATA_FILE, index=False)

# Function to train a model from the stored data
def train_model():
    # Read the logged data
    df = pd.read_csv(DATA_FILE)
    if len(df) < 5:  # Ensure there is enough data to train
        return None, None
    
    # Use the first 6 columns (inputs) and last column (output) to train the model
    X = df[['result1', 'result2', 'result3', 'result4', 'result5', 'result6']].values
    y = df['predicted_result'].values

    # Train the model
    model = LinearRegression()
    model.fit(X, y)
    return model, X

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    results = data.get('results', [])

    # Make sure there are at least 3 valid numbers
    if len(results) < 3:
        return jsonify({"error": "Please enter at least 3 valid results."}), 400

    # Load the model
    model, X = train_model()

    if model is None:
        return jsonify({"error": "Not enough data to make predictions."}), 400

    # Format the input for prediction (if less than 6 results, pad with 0s)
    padded_results = results + [0] * (6 - len(results))

    # Make prediction
    predicted_result = model.predict([padded_results])[0]

    # Calculate confidence (based on variance)
    mean = np.mean(X, axis=0)
    variance = np.var(X - mean, axis=0).sum()
    confidence = max(0, 100 - variance * 10)

    # Log the new data (for further model training)
    df = pd.read_csv(DATA_FILE)
    new_data = pd.DataFrame([padded_results + [predicted_result]], 
                            columns=['result1', 'result2', 'result3', 'result4', 'result5', 'result6', 'predicted_result'])
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

    return jsonify({
        'predicted_result': predicted_result,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
