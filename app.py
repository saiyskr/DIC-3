# Importing the necessary libraries
from flask import Flask, request, render_template
from flask_cors import CORS
import pickle

# Creating a Flask application
app = Flask(__name__)
CORS(app)

# Loading the ML model, minimum and maximum values needed for normalization using the pickle file
with open('models\wine_quality_model.pkl', 'rb') as file:
    model, min_values, max_values = pickle.load(file)
    min_values = dict(min_values)
    max_values = dict(max_values)

# Route for the home page
@app.route('/')
def home():
    """
    Render the home page with form inputs for wine quality prediction.

    Returns:
    - HTML template rendering the home page.
    """
    return render_template('navbars.html', min_values=min_values, max_values=max_values)

# Route for predicting wine quality
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle the prediction request based on user input.

    Returns:
    - HTML template rendering the prediction result.
    """
    # Extracting features from the form data
    wine_type = request.form['wine_type']
    wine_type_encoded = 0 if wine_type.lower() == 'red' else 1

    features = [int(wine_type_encoded),
                float(request.form['fixed_acidity']),
                float(request.form['volatile_acidity']),
                float(request.form['citric_acid']),
                float(request.form['residual_sugar']),
                float(request.form['chlorides']),
                float(request.form['free_sulfur_dioxide']),
                float(request.form['total_sulfur_dioxide']),
                float(request.form['density']),
                float(request.form['pH']),
                float(request.form['sulphates']),
                float(request.form['alcohol'])]

    # Normalize the features using min-max normalization
    normalized_features = []
    normalized_features.append(wine_type_encoded)
    for key, x in zip(min_values.keys(), features[1:]):
        key = key.replace("_", " ")
        if key == 'pH' or key == 'total sulfur dioxide':
            pass
        else:
            denominator = max_values[key] - min_values[key]
            normalized_features.append((x - min_values[key]) / denominator)

    # Make the prediction using the loaded machine learning model
    prediction = model.predict([normalized_features])[0]

    # Process the prediction result for rendering
    prediction_percent = int(prediction * 100)
    prediction_category = "Good" if prediction == 1 else "Bad"

    # Render the result template with the prediction information
    return render_template('result.html', prediction=prediction, prediction_percent=prediction_percent,
                           prediction_category=prediction_category)

# Main function to run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
