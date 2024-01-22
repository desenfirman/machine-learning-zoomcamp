from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from typing import List
import pickle

app = FastAPI()
MODEL_PATH = './model/final_model.bin'
DV_PATH = './model/final_dv.bin'

def get_model_and_components():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(DV_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

@app.on_event("startup")
async def startup_event():
    # This code will run when the application starts
    print("Loading model and components...")
    model, vectorizer = get_model_and_components()
    app.state.model = model
    app.state.vectorizer = vectorizer

@app.post("/batch-predict/")
async def batch_predict(data: List[dict], model_and_vectorizer = Depends(get_model_and_components)):
    model, vectorizer = model_and_vectorizer

    X = vectorizer.transform(data)
    y_pred = model.predict_proba(X)[:, 1]
    decision = y_pred >= 0.5

    predictions = [
        {   
            'song': f"{data[i]['track_name']} -{data[i]['artists']}", 
            'decision': bool(decision[i]),
            'probability': float(round(y_pred[i], 3)),
            'description': (
                f"This song match with my taste! Recommended to be saved in my favourite song list"
                if decision[i]
                else f"Well, good song. I'll hear it later"
            )
        }
        for i in range(len(data))
    ]
    return JSONResponse(content={"predictions": predictions})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
