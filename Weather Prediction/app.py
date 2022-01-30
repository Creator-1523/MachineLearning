from typing import final
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
# headings=("Summary","windspeed")


# df=pd.read_csv("weatherHistory.csv")

@app.route('///')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features= np.array(int_features)
    # final_features = df[df.Temperature==int_features[0]]
    prediction = model.predict([final_features])
    
    # data=((int_features[0],))
    
    #output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=' Temperature (C) {}'.format(prediction))


def data():
     return render_template('index.html', data=' Temperature (C) {}'.format(100))
 
 
# @app.route("/sub",methods=['POST'])
# def submit():
#     # html to .py
#     if request.method == "POST":
#         name=request.form["humidity"]
#     # py to html
#     return render_template("table.html",n=name)

#@app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)