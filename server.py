from flask import Flask, request, jsonify
from utils import load_ensemble
from analytics import ensebmle_predict
import json

app = Flask(__name__)

MODEL_DIR = "Saved Models/"

#Load Ensemble
enseble = load_ensemble(MODEL_DIR)

@app.route('/', methods = ['POST'])
def predict():
    #Load Dataa
    data = json.loads(request.data)

    #Collect features from data
    features = data["features"]
    prediction = ensebmle_predict(enseble, features)

    return jsonify({"prediction": str(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0')