import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    s=[request.form['News']]
    #s = list(request.form.get('type', None))
    # print(s)
    m=tfidf_vectorizer.transform(s)
    prediction=model.predict(m)


    return render_template('index.html', prediction_text='News should be $ {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)