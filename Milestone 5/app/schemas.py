from pydantic import BaseModel


class PredictionRequest(BaseModel):
    title: str
    description: str


class PredictionResponse(BaseModel):
    story_points: float
    model_used: str