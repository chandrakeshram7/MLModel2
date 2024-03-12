from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Read the dataset
data = pd.read_csv("Crop_recommendation.csv")

# Crop mapping dictionary
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7,
    'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
    'pomegranate': 14, 'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}
data['crop_num']=data['label'].map(crop_dict)
# Feature and target variables
X = data.drop(['crop_num', 'label'], axis=1)
y = data['crop_num']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
ms = MinMaxScaler()
sc = StandardScaler()

X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Train RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Convert and save the model to ONNX format
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(rfc, initial_types=initial_type)
onnx_model_path = "model.onnx"
onnx.save_model(onnx_model, onnx_model_path)

# Flask setup
app = Flask(__name__)
CORS(app, methods=['GET', 'POST', 'OPTIONS'])

# Load the ONNX model for inference
sess = ort.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Flask routes
@app.route('/')
def start():
    return "Server chalu jhala"

@app.route('/recommend', methods=['POST'])
def recommendation():
    try:
        # Get input parameters from JSON request
        data = request.get_json()
        N, P, k, temperature, humidity, ph, rainfall = (
            data['N'], data['P'], data['K'], data['temperature'],
            data['humidity'], data['ph'], data['rainfall']
        )

        # Make the recommendation
        features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
        transformed_features = ms.transform(features)
        transformed_features = sc.transform(transformed_features)
        prediction = sess.run([output_name], {input_name: transformed_features})[0].reshape(1, -1)

        # Get the recommended crop
        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            return jsonify({"crop": crop})
        else:
            return jsonify({"error": "Unable to recommend a proper crop for this environment"})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
