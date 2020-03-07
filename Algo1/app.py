import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from sklearn.externals import joblib
import flask
import pickle

app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')

with open('model.pickle', 'rb') as handle:
    model = pickle.load(handle)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    news=request.get_data(as_text=True)[5:]

    pred=model.predict([news])
    return render_template('index.html', prediction_text='The News should be {}'.format(pred[0]))

if __name__ == "__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)