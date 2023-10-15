import unittest
import numpy as np
import pandas as pd
from LogisticRegression import clean_data, scaled_X_train, scaled_X_test, lr

class TestLogisticRegression(unittest.TestCase):

    def test_load_and_clean_data(data_path):
        data = pd.read_csv(data_path)
        clean_data = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

        for col in clean_data:
            data[col] = data[col].replace(0, np.NaN)
            mean = int(data[col].mean(skipna=True))
            data[col] = data[col].replace(np.NaN, mean)

        x = data.drop(labels="Outcome", axis=1)
        y = data["Outcome"]

        return x, y

    def test_scaled_X_train_and_test(self):
        # Test the scaled_X_train and scaled_X_test DataFrames
        self.assertIsInstance(scaled_X_train, pd.DataFrame)
        self.assertIsInstance(scaled_X_test, pd.DataFrame)
        # Add more specific tests for your DataFrames

    def test_logistic_regression(self):
        # Test the logistic regression model
        self.assertTrue(lr is not None)
        # Add more tests for the model's attributes and functionality


if __name__ == '__main__':
    unittest.main()