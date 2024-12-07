from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)
CORS(app)

# Load data and train the model
train_data = pd.read_csv('training.csv')
test_data = pd.read_csv('testing.csv')
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', [])
    manual_disease = data.get('manual_disease', None)

    # Convert symptoms into feature vector
    symptom_vector = [0] * len(X_train.columns)
    for symptom in symptoms:
        if symptom in X_train.columns:
            symptom_vector[X_train.columns.get_loc(symptom)] = 1

    # Predict disease
    disease_probs = model.predict_proba([symptom_vector])[0]
    disease_classes = model.classes_
    top_index = np.argmax(disease_probs)
    top_disease = disease_classes[top_index]
    top_probability = disease_probs[top_index]

    # Calculate manual accuracy if a manual disease is provided
    manual_accuracy = None
    if manual_disease and manual_disease in disease_classes:
        manual_accuracy = disease_probs[disease_classes.tolist().index(manual_disease)]

    return jsonify({
        'predicted_disease': top_disease,
        'predicted_probability': top_probability,
        'manual_accuracy': manual_accuracy
    })

if __name__ == '__main__':
    app.run(debug=True)
