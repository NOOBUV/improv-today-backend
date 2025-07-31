from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.simple_openai import SimpleOpenAIService
from pydantic import BaseModel
from typing import List, Dict, Optional

router = APIRouter()

class ConversationRequest(BaseModel):
    message: str
    target_vocabulary: Optional[List[Dict]] = []
    session_type: str = "daily"
    topic: Optional[str] = ""

class ConversationResponse(BaseModel):
    response: str
    feedback: Dict
    usage_analysis: Optional[Dict] = None
    success: bool = True

# Main endpoint that frontend expects: POST /api/conversation
@router.post("", response_model=ConversationResponse)
async def handle_conversation(request: ConversationRequest, db: Session = Depends(get_db)):
    """
    Main conversation endpoint - receives transcript and returns AI response with feedback
    """
    try:
        openai_service = SimpleOpenAIService()
        
        # Generate AI response
        ai_response = await openai_service.generate_vocabulary_focused_response(
            request.message, 
            request.target_vocabulary,
            request.topic
        )
        
        # Create basic feedback (can be enhanced later)
        feedback = {
            "clarity": 85,
            "fluency": _estimate_fluency(request.message),
            "vocabularyUsage": [],
            "suggestions": ["Great conversation!", "Keep practicing!"],
            "overallRating": 4
        }
        
        print(f"🎯 Conversation Request: {request.message}")
        print(f"🤖 AI Response: {ai_response}")
        
        return ConversationResponse(
            response=ai_response,
            feedback=feedback,
            usage_analysis=None
        )
        
    except Exception as e:
        print(f"❌ Conversation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversation failed: {str(e)}")

# Legacy endpoint for backward compatibility
@router.post("/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest, db: Session = Depends(get_db)):
    return await handle_conversation(request, db)

@router.get("/history")
async def get_conversation_history(db: Session = Depends(get_db)):
    return {"conversations": []}

def _estimate_fluency(transcript: str) -> int:
    """Estimate fluency based on transcript characteristics"""
    words = transcript.split()
    word_count = len(words)
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Simple heuristic scoring
    base_score = 70
    if word_count > 5: base_score += 10
    if word_count > 10: base_score += 5
    if avg_word_length > 4: base_score += 10
    
    return min(100, base_score)