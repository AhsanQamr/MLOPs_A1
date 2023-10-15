import unittest
from app import app  # Import your Flask app
import json

class FlaskAppTest(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_endpoint(self):
        input_data = {
            "data": [[1.0, 2.0, 3.0, 4.0, 5.0]]  # Modify this input data to match your expected input format
        }
        expected_response = {"prediction": [1]}  # Modify this to match your expected model output

        response = self.app.post('/predict', data=json.dumps(input_data), content_type='application/json')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data), expected_response)

    def test_invalid_input(self):
        input_data = {
            "invalid_key": [[1.0, 2.0, 3.0, 4.0, 5.0]]  # Sending invalid input without 'data' key
        }

        response = self.app.post('/predict', data=json.dumps(input_data), content_type='application/json')

        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue('error' in response_data)

if __name__ == '__main':
    unittest.main()