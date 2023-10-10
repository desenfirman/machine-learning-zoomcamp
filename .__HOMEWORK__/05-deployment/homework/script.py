import pickle

with open('./model1.bin', 'rb') as f:
    clf = pickle.load(f)
with open('./dv.bin', 'rb') as f:
    dv = pickle.load(f)

X = {"job": "retired", "duration": 445, "poutcome": "success"}
X = dv.transform([X])

print("The probability of the client is", round(clf.predict_proba(X)[0,1], 3))