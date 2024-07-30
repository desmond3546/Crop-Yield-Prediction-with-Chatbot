# Import necessary libraries
from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn

# Print the version of sklearn
print(sklearn.__version__)

# Load the pre-trained models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocesser.pkl', 'rb'))

# Initialize the Flask application
app = Flask(__name__)


# Define the route for the home page
@app.route('/')
def index():
    # Render the home page template
    return render_template('index.html')


# Define the route for making predictions
@app.route("/predict", methods=['POST'])
def predict():
    # Check if the request method is POST
    if request.method == 'POST':
        # Get the input data from the form
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
        Item = request.form['Item']

        # Create a numpy array of the input features
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, solar_radiation,
                              soil_organic_matter, soil_nitrogen, soil_phosphorus, soil_potassium, Area, Item]],
                            dtype=object)

        # Transform the input features using the preprocessor
        transformed_features = preprocessor.transform(features)

        # Make a prediction using the Decision Tree Regressor model
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        # Render the home page template with the prediction result
        return render_template('index.html', prediction=prediction)


# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)

