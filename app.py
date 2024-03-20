from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])

def predict():
    data = request.get_json(force=True)
    prediction = model.predict(data['text'])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
