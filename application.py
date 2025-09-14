from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


application = Flask(__name__)
app = application

ridge = pickle.load(open('Models/ridge.pkl','rb'))
scaler = pickle.load(open('Models/scaler.pkl','rb'))

@app.route("/")
def home():
    return render_template('index.html')
@app.route('/prediction',methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = int(request.form.get('Classes'))
        Region = int(request.form.get('Region'))

        new_data = scaler.transform([[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge.predict(new_data)

        return render_template('home.html',results = result[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
