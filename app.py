import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('LinearModel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features =[]
    int_features.append(float(request.form['T']))
    int_features.append(float(request.form['TM']))
    int_features.append(float(request.form['Tm']))
    int_features.append(float(request.form['SLP']))
    int_features.append(float(request.form['H']))
    int_features.append(float(request.form['VV']))
    int_features.append(float(request.form['V']))
    int_features.append(float(request.form['VM']))
    # int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 4)

    return render_template('index.html', prediction_text='PM2.5 = {}'.format(output))

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