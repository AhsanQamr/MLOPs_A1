from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved Logistic Regression model
lr = joblib.load('logistic_regression_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Make predictions using the loaded model
        scaled_data = data['data']  # Assuming 'data' is the key for the input data
        prediction = lr.predict(scaled_data)

        # Prepare a response
        response = {'prediction': prediction.tolist()}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run the server on port 5000