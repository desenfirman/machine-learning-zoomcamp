import requests
import json

url = "http://localhost:8000/batch-predict"

with open('./data/sample_for_predict_test.json', 'r') as fp:
    payload = json.load(fp)
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, json=payload)

    print(response.text)