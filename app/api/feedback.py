from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.scoring_service import ScoringService
from pydantic import BaseModel

router = APIRouter()

class FeedbackRequest(BaseModel):
    conversation_text: str
    duration: int

class FeedbackResponse(BaseModel):
    overall_score: float
    fluency_score: float
    vocabulary_score: float
    suggestions: list

@router.post("/analyze", response_model=FeedbackResponse)
async def analyze_performance(request: FeedbackRequest, db: Session = Depends(get_db)):
    scoring_service = ScoringService()
    scores = scoring_service.calculate_scores(request.conversation_text, request.duration)
    return FeedbackResponse(
        overall_score=scores["overall"],
        fluency_score=scores["fluency"],
        vocabulary_score=scores["vocabulary"],
        suggestions=scores["suggestions"]
    )