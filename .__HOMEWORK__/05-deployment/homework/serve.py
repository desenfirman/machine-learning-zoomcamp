import pickle

from flask import Flask
from flask import request
from flask import jsonify


with open('./model2.bin', 'rb') as f:
    clf = pickle.load(f)
with open('./dv.bin', 'rb') as f:
    dv = pickle.load(f)

app = Flask('credit-prediction')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = clf.predict_proba(X)[0, 1]
    decision = y_pred >= 0.5

    result = {
        'probability': round(y_pred, 3),
        'decision': bool(decision)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)