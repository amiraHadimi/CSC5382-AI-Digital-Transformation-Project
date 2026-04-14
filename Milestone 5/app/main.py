from fastapi import FastAPI, HTTPException
from app.schemas import PredictionRequest, PredictionResponse
from app.model import predict_story_points

app = FastAPI(title="LLAMA3SP Story Point Estimation API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        prediction, model_used = predict_story_points(
            request.title,
            request.description
        )
        return PredictionResponse(
            story_points=prediction,
            model_used=model_used
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))