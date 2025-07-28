from pydantic import BaseModel

class HeadlineRequest(BaseModel):
    headline: str

class PredictionResponse(BaseModel):
    headline: str
    predicted_category: str
    confidence: float
    all_probabilities: dict