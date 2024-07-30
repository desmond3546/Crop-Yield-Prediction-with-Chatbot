from flask import Flask,request, render_template
import numpy as np
import pickle
import sklearn
print(sklearn.__version__)
#loading models
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocesser.pkl','rb'))

#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        solar_radiation = request.form['solar_radiation']
        soil_organic_matter = request.form['soil_organic_matter']
        soil_nitrogen = request.form['soil_nitrogen']
        soil_phosphorus = request.form['soil_phosphorus']
        soil_potassium = request.form['soil_potassium']
        Area = request.form['Area']
        Item  = request.form['Item']


        features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,solar_radiation,soil_organic_matter,soil_nitrogen,soil_phosphorus,soil_potassium,Area,Item]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1,-1)

        return render_template('index.html',prediction = prediction)

if __name__=="__main__":
    app.run(debug=True)