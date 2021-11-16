import flask
from flask import Flask, render_template, request, Markup
from flask import jsonify
import numpy as np
from keras.models import load_model
import pickle

classifier = load_model('Trained_model.h5')
classifier._make_predict_function()

crop_recommendation_model_path = 'Crop_Recommendation.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

app = Flask(__name__)


@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")


@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")

@app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        json_data = flask.request.json
        # N = int(request.form['nitrogen'])
        # P = int(request.form['phosphorous'])
        # K = int(request.form['potassium'])
        # ph = float(request.form['ph'])
        # rainfall = float(request.form['rainfall'])
        # temperature = float(request.form['temperature'])
        # humidity = float(request.form['humidity'])
        N = json_data["nitrogen"]
        P = json_data["phosphorous"]
        K = json_data["potassium"]
        ph = json_data["ph"]
        rainfall = json_data["rainfall"]
        temperature = json_data["temperature"]
        humidity = json_data["humidity"]
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        response = {
            "predict": [
                {
                    "crop": final_prediction,
                    "photo": 'img/crop/'+final_prediction+'.jpg'
                }
            ]
        }

        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
