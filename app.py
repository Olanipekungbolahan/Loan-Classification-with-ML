# app.py (Flask)

from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the pre-trained model and scaler
rf_model = pickle.load(open('rf_model.pkl','rb'))  # Load your trained model
scaler = pickle.load(open('scaler.pkl','rb')) # Load your scaler

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive the input data from the request
    input_data = request.json
    
    # Create a DataFrame from the input data
    input_df = pd.DataFrame(input_data)
    
    # Perform One Hot Encoding for categorical feature 'purpose'
    input_df_encoded = pd.get_dummies(input_df, columns=['purpose'])
    
    # Normalize the data using the loaded scaler
    input_normalized = scaler.transform(input_df_encoded)
    
    # Make predictions using the pre-trained model
    predictions = rf_model.predict(input_normalized)
    
    # Prepare the response
    response = {'predictions': predictions.tolist()}
    
    return jsonify(response)
    

if __name__ == '__main__':
    app.run(debug=True)
