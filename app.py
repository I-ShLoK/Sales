from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        month = int(request.form['month'])
        hour = int(request.form['hour'])
        city = int(request.form['city'])
        quantity_ordered = int(request.form['quantity_ordered'])
        price_each = float(request.form['price_each'])
        
        # Prepare input
        features = np.array([[month, hour, city, quantity_ordered, price_each]])
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        sales = round(float(prediction), 2)
        
        return render_template('result.html', prediction=sales)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
