import joblib
from flask import Flask, render_template, request
import pandas as pd
import csv

app = Flask(__name__)

# Load the model once when the application starts
model = joblib.load('crop_app.pkl')

@app.route('/')
def home():
    return render_template('Home_1.html')

@app.route('/Predict')
def prediction():
    return render_template('Index.html')

@app.route('/form', methods=["POST"])
def brain():
    try:
        # Get the values from the form
        Nitrogen = float(request.form['Nitrogen'])
        Phosphorus = float(request.form['Phosphorus'])
        Potassium = float(request.form['Potassium'])
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        Ph = float(request.form['ph'])
        Rainfall = float(request.form['Rainfall'])
        
        # Create a DataFrame for the input values
        data = {
            'N': [Nitrogen],
            'P': [Phosphorus],
            'K': [Potassium],
            'temperature': [Temperature],
            'humidity': [Humidity],
            'ph': [Ph],
            'rainfall': [Rainfall]
        }
        input_df = pd.DataFrame(data)
        csv_file = r'C:\Users\gawan\Desktop\Cropv2\Crop_recommendation_system\crop_app\input\input_data.csv'
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([Nitrogen, Phosphorus, Potassium,Temperature,Humidity,Ph,Rainfall])

        # Validate input values
        if 0 < Ph <= 14 and Temperature < 100 and Humidity > 0:
            acc = model.predict(input_df)
            print(f"Predicted Class: {acc}, Input: {input_df.values[0]}")  # Print the predicted class and input for debugging
            
            return render_template('prediction.html', prediction=str(acc[0]))
        else:
            return "Sorry... Error in entered values in the form. Please check the values and fill it again."
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)