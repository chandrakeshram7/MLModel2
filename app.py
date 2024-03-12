from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from fuzzywuzzy import fuzz
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = pd.read_csv("Crop_recommendation.csv")
print(data.head())



crop_dict = {
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
}
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

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# create instances of all models
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreeClassifier(),
}


for name, md in models.items():
    md.fit(X_train,y_train)
    ypred = md.predict(X_test)
    
    print(f"{name}  with accuracy : {accuracy_score(y_test,ypred)}")

rfc.fit(X_train,y_train)
ypred = rfc.predict(X_test)
accuracy_score(y_test,ypred)
@app.route('/')
def start():
    return "Server chalu jhala"
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





from app import app
if __name__ == '__main__':
    app.run(debug=True, port=5000)

