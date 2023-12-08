from flask import Flask, request
from flask_cors import CORS
from flask import Flask, render_template
import pickle

app = Flask(__name__, static_folder="/static")
CORS(app)

with open('models\wine_quality_model.pkl', 'rb') as file:
    model, min_values, max_values = pickle.load(file)
    min_values = dict(min_values)
    max_values = dict(max_values)

@app.route('/')
def home():
    return render_template('navbars.html',min_values = min_values,max_values= max_values)

@app.route('/predict', methods=['POST'])
def predict():
    features = []
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
    
    normalized_features =[]
    normalized_features.append(wine_type_encoded)
    for key, x in zip(min_values.keys(), features[1:]):
        key = key.replace("_"," ")
        if(key == 'pH' or key == 'total sulfur dioxide'):
            pass
        else:
            denominator = max_values[key] - min_values[key]
            normalized_features.append((x - min_values[key]) / denominator)

    prediction = model.predict([normalized_features])[0]
    
    prediction_percent = int(prediction * 100)
    if prediction == 1:
        prediction_category = "Good"
    else:
        prediction_category = "Bad"
    
    return render_template('result.html', prediction=prediction, prediction_percent=prediction_percent,prediction_category=prediction_category)

if __name__ == '__main__':
    app.static_folder = 'static'
    app.run(debug=True)
