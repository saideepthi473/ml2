
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

# Load the model and scaler
model = tf.keras.models.load_model('pcos_model.h5')
scaler = joblib.load('scaler.save')

# Load columns information
with open('columns.json') as f:
    columns = json.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Prepare the data (ensure the input matches the columns in 'columns.json')
    input_data = np.array([data[col] for col in columns]).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    
    # Return prediction result (1 = PCOS, 0 = No PCOS)
    return jsonify({'prediction': int(prediction[0][0] > 0.5)})

if __name__ == '__main__':
    app.run(debug=True)
