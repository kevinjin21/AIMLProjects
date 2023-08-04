# for flask application

import pickle
import numpy as np
from flask import Flask, request

model = None
app = Flask(__name__)


def load_model():
    global model
    with open('pharm_model.pkl', 'rb') as f:
        model = pickle.load(f)


@app.route('/')
def home_endpoint():
    return 'Welcome to Pharmaceutical Prediction App'


@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        data = request.get_json()
        data = np.array(data['data'])[np.newaxis, :]
        #test_array = np.array(data['data'])
        prediction = model.predict(data)
    return str(prediction[0])


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=80)
