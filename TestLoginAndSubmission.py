import unittest
from app import app, db
from flask import json

class TestLoginAndSubmission(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.user_data = {
            "username": "testuser",
            "password": "testpassword"
        }
        self.prediction_data = {
            "age": 65,
            "condition_count": 3,
            "immunizations": 5
        }

    def test_login(self):
        # Test login endpoint
        response = self.app.post('/login', data=json.dumps(self.user_data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn("token", json.loads(response.data))

    def test_data_submission(self):
        # Assuming login is successful and we get a token
        login_response = self.app.post('/login', data=json.dumps(self.user_data), content_type='application/json')
        token = json.loads(login_response.data).get("token")

        # Test data submission for prediction
        headers = {'Authorization': f'Bearer {token}'}
        response = self.app.post('/submit-data', data=json.dumps(self.prediction_data), content_type='application/json', headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction_result", json.loads(response.data))

if __name__ == '__main__':
    unittest.main()
