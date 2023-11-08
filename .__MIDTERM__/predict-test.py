import requests
import json

url = "https://predict-player-trait-xv6leznboq-et.a.run.app/batch-predict/"

with open('./data/sample_for_predict_test.json', 'r') as fp:
    payload = json.load(fp)
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, json=payload)

    print(response.text)