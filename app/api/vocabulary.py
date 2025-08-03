from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.vocab_analyzer import VocabAnalyzer
from app.services.vocabulary_tier_service import VocabularyTierService
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter()

class VocabularyAnalysis(BaseModel):
    text: str

class VocabularyResponse(BaseModel):
    complexity_score: float
    suggestions: List[str]
    new_words: List[str]
    tier: str = "mid"
    score: int = 50

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
async def get_weekly_vocabulary(tier: str = "mid", db: Session = Depends(get_db)):
    """Get this week's vocabulary words based on user's tier"""
    try:
        vocabulary_service = VocabularyTierService()
        recommendations = vocabulary_service.get_vocabulary_recommendations(tier, [])
        
        # Convert to expected format with additional metadata
        weekly_words = []
        for i, rec in enumerate(recommendations):
            weekly_words.append({
                "id": str(i + 1),
                "word": rec["word"],
                "definition": rec["definition"],
                "difficulty": rec["difficulty"],
                "category": "general",  # Could be enhanced based on context
                "examples": [f"Example sentence using '{rec['word']}' in context."],
                "pronunciation": "",  # Could be added later
                "usageCount": 0,
                "masteryLevel": 0
            })
        
        return {
            "words": weekly_words,
            "tier": tier,
            "stats": {
                "totalWords": 150,
                "masteredWords": 45,
                "practiceStreak": 7,
                "weeklyProgress": [5, 8, 6, 10, 7, 9, 8]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get vocabulary: {str(e)}")

# Get tier-specific recommendations
@router.get("/recommendations/{tier}")
async def get_tier_recommendations(tier: str, db: Session = Depends(get_db)):
    """Get vocabulary recommendations for specific tier"""
    if tier not in ["basic", "mid", "top"]:
        raise HTTPException(status_code=400, detail="Invalid tier. Must be 'basic', 'mid', or 'top'")
    
    try:
        vocabulary_service = VocabularyTierService()
        recommendations = vocabulary_service.get_vocabulary_recommendations(tier, [])
        
        return {
            "tier": tier,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

# Enhanced tier analysis endpoint
@router.post("/tier-analysis")
async def analyze_tier(request: Dict, db: Session = Depends(get_db)):
    """Detailed vocabulary tier analysis"""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        vocabulary_service = VocabularyTierService()
        analysis = vocabulary_service.analyze_vocabulary_tier(text)
        
        return {
            "tier": analysis.tier,
            "score": analysis.score,
            "wordCount": analysis.word_count,
            "complexWords": analysis.complex_word_count,
            "averageWordLength": analysis.average_word_length,
            "analysis": analysis.analysis_details,
            "recommendations": vocabulary_service.get_vocabulary_recommendations(analysis.tier, [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/{word_id}/usage")
async def update_word_usage(word_id: str, db: Session = Depends(get_db)):
    """Mark a word as used"""
    return {
        "success": True,
        "message": f"Word {word_id} usage updated"
    }