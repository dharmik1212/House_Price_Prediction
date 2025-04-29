import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

# Load saved model, scalers, and encoders
with open('optimized_xgb_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('minmax_X_scaler.pkl', 'rb') as f:
    loaded_X_scaler = pickle.load(f)

with open('minmax_y_scaler.pkl', 'rb') as f:
    loaded_y_scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    loaded_encoders = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Preprocessing function
def preprocess_input(user_input):
    input_df = pd.DataFrame([user_input])

    categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in categorical_cols:
        input_df[col] = loaded_encoders[col].transform(input_df[col])

    numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    input_df[numerical_cols] = loaded_X_scaler.transform(input_df[numerical_cols])

    return input_df

# API route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.get_json()

        # Preprocess the input
        processed_input = preprocess_input(user_input)

        # Predict the price (scaled)
        predicted_price_scaled = loaded_model.predict(processed_input)

        # Reverse scale the prediction
        predicted_price = loaded_y_scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0][0]

        # Convert numpy.float32 to native Python float before jsonify
        return jsonify({'predicted_price': float(predicted_price)})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app on localhost
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
