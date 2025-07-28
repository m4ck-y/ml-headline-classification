from pydantic import BaseModel
from typing import Dict

class HeadlineRequest(BaseModel):
    headline: str

class PredictionResponse(BaseModel):
    headline: str
    predicted_category: str
    confidence: float
    all_probabilities: Dict[str, float]