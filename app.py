from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from flask import Flask, render_template, send_file
import matplotlib.pyplot as plt
from io import BytesIO
import pickle

app = Flask(__name__, static_folder="/static")
CORS(app)

with open('models\wine_quality_model.pkl', 'rb') as file:
    model, min_values, max_values = pickle.load(file)
    # list_representation = min_values.tolist()
    min_values = dict(min_values.round(1))
    max_values = dict(max_values.round(1))
    print(min_values)
    print(max_values)

@app.route('/')
def home():
    return render_template('index.html',min_values = min_values,max_values= max_values)

@app.route('/predict', methods=['POST'])
def predict():
    features = []
    wine_type = request.form['wine_type']
    wine_type_encoded = 0 if wine_type.lower() == 'red' else 1
    # features.append(wine_type_encoded)
    
    features = [int(wine_type_encoded),
                float(request.form['fixed_acidity']),
                float(request.form['volatile_acidity']),
                float(request.form['citric_acid']),
                float(request.form['residual_sugar']),
                float(request.form['chlorides']),
                float(request.form['free_sulfur_dioxide']),
                # float(request.form['total_sulfur_dioxide']),
                float(request.form['density']),
                # float(request.form['pH']),
                float(request.form['sulphates']),
                float(request.form['alcohol'])]
    
    # print(features)
    normalized_features =[]
    normalized_features.append(wine_type_encoded)
    
    for key, x in zip(min_values.keys(), features[1:]):
        key = key.replace("_"," ")
        denominator = max_values[key] - min_values[key]
        print(key)
        if(denominator!=0):
            normalized_features.append((x - min_values[key]) / denominator)
        else:
            normalized_features.append(0)
            

    print("Original Features:")
    print(features)

    print("\nNormalized Features:")
    print(normalized_features)
    
    # model:any
    prediction = model.predict([normalized_features])[0]
    
    prediction_percent = int(prediction * 100)
    if prediction == 1:
        prediction_category = "Good"
    else:
        prediction_category = "Bad"
    
    return render_template('result.html', prediction=prediction, prediction_percent=prediction_percent,prediction_category=prediction_category)
# @app.route('/')
# def hello():
#     return 'Hello, Flask is running!'

@app.route('/getpic', methods=['GET'])
def getpic():
    # Generate a simple plot
    plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Simple Plot')

    # Save the plot to a BytesIO object
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close()

    # Send the plot to the frontend
    return send_file(img_buf, mimetype='image/png')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Assuming the file is a CSV file, read it into a Pandas DataFrame
        # df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        # accuracy = MLModelling(file)
        # Perform any necessary operations on the DataFrame
        # For example, you could run your machine learning model on this data

        # Return a response, or send the processed data back to the frontend
        return jsonify({'success': True, "message": "File uploaded successfully"})

    except Exception as e:
        return jsonify({'error': str(e)})
    


if __name__ == '__main__':
    app.static_folder = 'static'
    app.run(debug=True)
