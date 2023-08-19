# for flask application

import pickle
import numpy as np
from flask import Flask, request, render_template

model = None
app = Flask(__name__, template_folder='templates', static_folder='static')
weights = None


def load_model():
    global model
    with open('pharm_model.pkl', 'rb') as f:
        model = pickle.load(f)


def load_weights():
    global weights
    with open('norm_weights.pkl', 'rb') as f:
        weights = pickle.load(f)


def inverse_norm(result):
    '''with open('norm_weights.pkl', 'rb') as f:
        data = pickle.load(f)
        ceil = data[0]
        floor = data[1]
        return result*(ceil-floor) + floor'''
    ceil = weights[0]
    floor = weights[1]
    return result*(ceil-floor) + floor

# normalize input feature values using pickled weights (min and max values of dataset)


def normalize(data):
    for index in range(len(data)):
        ceil = weights[index+2]
        floor = weights[index+3]
        data[index] = (data[index] - floor)/(ceil - floor)
    return data


@app.route('/')
def home_endpoint():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        features = []
        for i in range(10):
            features.append(float(request.form[f"feat_{i+1}"]))
        data = np.array(features)
        data = normalize(data)
        prediction = model.predict(data.reshape(1, -1))
        prediction = inverse_norm(prediction)
        return render_template('predict.html', pred=round(prediction[0], 3))


if __name__ == '__main__':
    load_model()
    load_weights()
    app.run(host='0.0.0.0', port=80)
