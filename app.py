from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load minimal scikit-learn components
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load only the necessary models
from sklearn.ensemble import RandomForestClassifier

# Load only the necessary scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = pd.read_csv("Crop_recommendation.csv")
print(data.head())

crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7,
    'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
    'pomegranate': 14, 'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}
data['crop_num'] = data['label'].map(crop_dict)
X = data.drop(['crop_num', 'label'], axis=1)
y = data['crop_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

app = Flask(__name__)
CORS(app, methods=['GET', 'POST', 'OPTIONS'])

# Use only the necessary scaler
ms = MinMaxScaler()
sc = StandardScaler()

X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Use only the necessary model
rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)
ypred = rfc.predict(X_test)
print(f"Random Forest with accuracy: {accuracy_score(y_test, ypred)}")

@app.route('/')
def start():
    return "Server is running."

@app.route('/recommend')
def recommendation():
    try:
        # Get input parameters from JSON request
        N = 122
        P = 42
        k = 43
        temperature = 30.8
        humidity = 82.443
        ph = 6.565
        rainfall = 101

        # Make the recommendation
        features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
        transformed_features = ms.transform(features)
        transformed_features = sc.transform(transformed_features)

        # Retrain RandomForestClassifier with the updated features
        rfc.fit(X_train, y_train)

        # Make predictions with the retrained model
        prediction = rfc.predict(transformed_features).reshape(1, -1)

        # Get the recommended crop
        crop = prediction[0].tolist()
        return jsonify({"crop": crop})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
