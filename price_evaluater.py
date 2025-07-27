from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import joblib

app = Flask(__name__)

MODEL_PATH = 'price_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
CSV_PATH = 'cars_with_desc.csv'

def train_and_save_model():
    df = pd.read_csv(CSV_PATH)
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['desc'])
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    train_and_save_model()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.route('/predict', methods=['POST'])
def predict_price():
    desc = request.form.get('desc') or (request.json and request.json.get('desc'))
    if not desc:
        return jsonify({'error': 'desc field is required'}), 400
    try:
        desc_vec = vectorizer.transform([desc])
        predicted_price = model.predict(desc_vec)[0]
        return jsonify({'predicted_price': round(predicted_price, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='192.168.43.167', port=5001, debug=True) 