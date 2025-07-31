from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.vocab_analyzer import VocabAnalyzer
from pydantic import BaseModel
from typing import List

router = APIRouter()

class VocabularyAnalysis(BaseModel):
    text: str

class VocabularyResponse(BaseModel):
    complexity_score: float
    suggestions: List[str]
    new_words: List[str]

@router.post("/analyze", response_model=VocabularyResponse)
async def analyze_vocabulary(analysis: VocabularyAnalysis, db: Session = Depends(get_db)):
    analyzer = VocabAnalyzer()
    result = analyzer.analyze_text(analysis.text)
    return VocabularyResponse(
        complexity_score=result["complexity_score"],
        suggestions=result["suggestions"],
        new_words=result["new_words"]
    )

@router.get("/weekly")
async def get_weekly_vocabulary(db: Session = Depends(get_db)):
    """Get this week's vocabulary words"""
    # Mock data for now
    sample_words = [
        {
            "id": "1",
            "word": "sophisticated",
            "definition": "complex or refined in design or quality",  
            "difficulty": "advanced",
            "category": "descriptive",
            "examples": ["The restaurant has a sophisticated atmosphere"],
            "pronunciation": "sə-ˈfis-tə-ˌkā-təd",
            "usageCount": 0,
            "masteryLevel": 0
        },
        {
            "id": "2",
            "word": "fascinating", 
            "definition": "extremely interesting",
            "difficulty": "intermediate",
            "category": "descriptive",
            "examples": ["That documentary was absolutely fascinating"],
            "pronunciation": "ˈfa-sə-ˌnā-tiŋ",
            "usageCount": 1,
            "masteryLevel": 30
        }
    ]
    
    return {
        "words": sample_words,
        "stats": {
            "totalWords": 150,
            "masteredWords": 45,
            "practiceStreak": 7,
            "weeklyProgress": [5, 8, 6, 10, 7, 9, 8]
        }
    }

@router.post("/{word_id}/usage")
async def update_word_usage(word_id: str, db: Session = Depends(get_db)):
    """Mark a word as used"""
    return {
        "success": True,
        "message": f"Word {word_id} usage updated"
    }