"""Small client script for testing the churn prediction API."""

import requests


payload = {
    "tenure": 12,
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0,
    "Contract": 0,
    "PaymentMethod": 2,
    "InternetService": 1,
}


response = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
print(response.json())
