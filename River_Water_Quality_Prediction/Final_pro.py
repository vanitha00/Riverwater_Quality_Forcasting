from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model_filename = 'riverwater1_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the scaler
scaler_filename = 'scaler1.pkl'
scaler = joblib.load(scaler_filename)

@app.route('/')
def index():
    return render_template('Index_water.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the HTML form
        features = [float(request.form[name]) for name in ['ph', 'hardness', 'solids', 'chloramines', 'sulfate', 'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity']]

        # Transform the features using the scaler
        scaled_features = scaler.transform([features])

        # Make predictions using the loaded model
        prediction = model.predict(scaled_features)

        # Determine the output based on the prediction
        if prediction[0] == 1:
            output = "Water is Good you can Drink"
        elif prediction[0] == 0:
            output = "Don't Drink The Water"
        else:
            output = "Not sure"

        # Pass the prediction result to the HTML template
        return render_template('results_water.html', prediction_text=output)

    except Exception as e:
        # Handle errors gracefully and provide an appropriate message
        return render_template('results_water.html', prediction_text='An error occurred. Please check your input.')

if __name__ == '__main__':
    app.run(debug=True,port=5007)
