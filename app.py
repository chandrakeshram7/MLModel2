from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from fuzzywuzzy import fuzz
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Crop_recommendation.csv")
print(data.head())



crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
             8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
             14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
             19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
data['crop_num']=data['label'].map(crop_dict)
X = data.drop(['crop_num','label'],axis=1)
y = data['crop_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



app = Flask(__name__)
CORS(app, methods=['GET', 'POST', 'OPTIONS'])


rfc = RandomForestClassifier()
ms = MinMaxScaler()
sc = StandardScaler()

X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

@app.route('/')
def start():
    return "Server chalu jhala"
@app.route('/recommend', methods=['POST'])
def recommendation():
    try:
        # Get input parameters from JSON request
        data = request.get_json()
        print("Received farmer profile:", data)

        N = data['N']
        P = data['P']
        k = data['k']
        temperature = data['temperature']
        humidity = data['humidity']
        ph = data['ph']
        rainfall = data['rainfall']

        # Make the recommendation
        features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
        transformed_features = ms.transform(features)
        transformed_features = sc.transform(transformed_features)
        prediction = rfc.predict(transformed_features).reshape(1, -1)

        # Get the recommended crop
        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            return jsonify({"crop": crop})
        else:
            return jsonify({"error": "Unable to recommend a proper crop for this environment"})

    except Exception as e:
        return jsonify({"error": str(e)})

    




from app import app
if __name__ == '__main__':
    app.run(debug=True, port=5000)

