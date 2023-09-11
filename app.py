from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Age = float(request.form['Age'])
        Fare = float(request.form['Fare'])
        Pclass_2 = float(request.form['Pclass_2'])
        Pclass_3 = float(request.form['Pclass_3'])
        Sex_male = float(request.form['Sex_male'])
        Embarked_Q = float(request.form['Embarked_Q'])
        Embarked_S = float(request.form['Embarked_S'])
        family_size_Large = float(request.form['family_size_Large'])
        family_size_Medium = float(request.form['family_size_Medium'])

        prediction = model.predict([[Age, Fare, Pclass_2, Pclass_3, Sex_male, Embarked_Q, Embarked_S, family_size_Large, family_size_Medium]])
        
        if prediction == 1:
            return render_template('index.html',prediction_text="Survived".format(prediction))
        elif prediction == 0:
            return render_template('index.html',prediction_text="Not Survived {}".format(prediction))
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
