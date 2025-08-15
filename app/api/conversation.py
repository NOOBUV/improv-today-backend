from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.services.simple_openai import SimpleOpenAIService
from app.services.vocabulary_tier_service import VocabularyTierService
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

router = APIRouter()

class ConversationRequest(BaseModel):
    message: str
    session_id: Optional[int] = None
    personality: Optional[str] = None  # will default from session if not provided
    target_vocabulary: Optional[List[Dict]] = []
    session_type: str = "daily"
    topic: Optional[str] = ""
    last_ai_reply: Optional[str] = None

class ConversationResponse(BaseModel):
    response: str
    feedback: Dict
    vocabulary_tier: Optional[Dict] = None
    usage_analysis: Optional[Dict] = None
    success: bool = True

# Helper function to get or create user from Auth0 token
def get_or_create_user(db: Session, auth0_user: Dict) -> int:
    """Get or create user from Auth0 claims and return user_id"""
    from app.models.user import User
    
    auth0_sub = auth0_user.get("sub")
    email = auth0_user.get("email")
    
    if not auth0_sub:
        raise HTTPException(status_code=400, detail="Auth0 subject not found in token")
    
    # Look for existing user by auth0_sub
    user = db.query(User).filter(User.auth0_sub == auth0_sub).first()
    
    if not user:
        # Create new authenticated user
        user = User(
            auth0_sub=auth0_sub,
            email=email,
            is_anonymous=False,
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"✅ Created new user for Auth0 sub: {auth0_sub}")
    
    return user.id

# Main endpoint that frontend expects: POST /api/conversation
@router.post("", response_model=ConversationResponse)
async def handle_conversation(
    request: ConversationRequest, 
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Main conversation endpoint - receives transcript and returns AI response with feedback
    """
    try:
        # Get or create user from Auth0 token
        user_id = get_or_create_user(db, current_user)
        
        openai_service = SimpleOpenAIService()
        vocabulary_service = VocabularyTierService()
        
        # Load personality and topic from session if session_id provided
        session_personality = request.personality
        if request.session_id:
            from app.models.session import Session as SessionModel
            session_row = db.query(SessionModel).filter(SessionModel.id == request.session_id).first()
            if not session_row:
                raise HTTPException(status_code=404, detail="Session not found")
            session_personality = session_personality or (session_row.personality or "friendly_neutral")
            # update last_message_at
            session_row.last_message_at = datetime.utcnow()
            db.commit()

        # Analyze vocabulary tier
        tier_analysis = vocabulary_service.analyze_vocabulary_tier(request.message)
        
        # Generate AI response with personality
        # Map minimalist personalities to existing prompt keys
        personality_map = {
            "friendly": "friendly_neutral",
            "sassy": "sassy_english",
            "blunt": "blunt_american",
        }
        effective_personality = personality_map.get(session_personality or "friendly", session_personality or "friendly_neutral")

        ai_response = await openai_service.generate_personality_response(
            request.message,
            effective_personality,
            request.target_vocabulary,
            request.topic,
            previous_ai_reply=request.last_ai_reply,
        )
        
        # Create enhanced feedback with vocabulary tier
        feedback = {
            "clarity": 85,
            "fluency": _estimate_fluency(request.message),
            "vocabularyTier": tier_analysis.tier,
            "vocabularyScore": tier_analysis.score,
            "vocabularyUsage": [],
            "suggestions": _generate_tier_suggestions(tier_analysis),
            "overallRating": min(5, max(1, int((tier_analysis.score + _estimate_fluency(request.message)) / 25)))
        }
        
        # Vocabulary tier details for frontend
        vocabulary_tier_data = {
            "tier": tier_analysis.tier,
            "score": tier_analysis.score,
            "wordCount": tier_analysis.word_count,
            "complexWords": tier_analysis.complex_word_count,
            "averageWordLength": tier_analysis.average_word_length,
            "analysis": tier_analysis.analysis_details,
            "recommendations": vocabulary_service.get_vocabulary_recommendations(tier_analysis.tier, [])
        }
        
        print(f"🎯 Conversation Request: {request.message}")
        print(f"📊 Vocabulary Tier: {tier_analysis.tier} (Score: {tier_analysis.score})")
        print(f"🤖 AI Response: {ai_response}")
        
        return ConversationResponse(
            response=ai_response,
            feedback=feedback,
            vocabulary_tier=vocabulary_tier_data,
            usage_analysis=None
        )
        
    except Exception as e:
        print(f"❌ Conversation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversation failed: {str(e)}")

# Add vocabulary tier analysis endpoint
@router.post("/analyze-tier")
async def analyze_vocabulary_tier(
    request: Dict, 
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Analyze vocabulary tier of provided text"""
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

# Legacy endpoint for backward compatibility
@router.post("/chat", response_model=ConversationResponse)
async def chat(
    request: ConversationRequest, 
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    return await handle_conversation(request, db, current_user)

# Get personality options
@router.get("/personalities")
async def get_personality_options():
    """Get available personality options for conversation"""
    return {
        "personalities": [
            {
                "id": "sassy_english",
                "name": "Sassy English",
                "description": "Witty and playful with a charming British flair",
                "example_response": "Oh brilliant! That's quite fascinating, isn't it?"
            },
            {
                "id": "blunt_american",
                "name": "Blunt American", 
                "description": "Direct and no-nonsense, but supportive",
                "example_response": "Alright, let's get straight to the point here."
            },
            {
                "id": "friendly_neutral",
                "name": "Friendly Neutral",
                "description": "Warm and encouraging conversation partner",
                "example_response": "That's wonderful! I'd love to hear more about that."
            }
        ]
    }

# Get welcome message for first-time users
@router.get("/welcome")
async def get_welcome_message(personality: str = "friendly_neutral", db: Session = Depends(get_db)):
    """Get personalized welcome message for first-time users"""
    try:
        openai_service = SimpleOpenAIService()
        welcome_message = await openai_service.generate_welcome_message(personality)
        
        return {
            "message": welcome_message,
            "personality": personality,
            "isWelcome": True
        }
    except Exception as e:
        print(f"❌ Welcome Message Error: {str(e)}")
        # Fallback welcome messages
        fallback_messages = {
            "sassy_english": "Well hello there! Welcome to ImprovToday, darling! What name shall I call you, and do tell me - how's your day been?",
            "blunt_american": "Hey there, welcome to ImprovToday! What should I call you, and how was your day?",
            "friendly_neutral": "Welcome to ImprovToday! I'm so glad you're here. What name should I address you with? How has your day been?"
        }
        
        return {
            "message": fallback_messages.get(personality, fallback_messages["friendly_neutral"]),
            "personality": personality,
            "isWelcome": True
        }

@router.get("/history")
async def get_conversation_history(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    # Get user and return their conversation history
    user_id = get_or_create_user(db, current_user)
    return {"conversations": [], "user_id": user_id}

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

def _generate_tier_suggestions(tier_analysis) -> List[str]:
    """Generate suggestions based on vocabulary tier analysis"""
    suggestions = []
    
    if tier_analysis.tier == "basic":
        suggestions.extend([
            "Try using more descriptive words instead of 'good', 'bad', or 'nice'",
            "Consider expanding your sentences with more details",
            "Practice using words with more than 4-5 letters"
        ])
    elif tier_analysis.tier == "mid":
        suggestions.extend([
            "Great vocabulary variety! Try incorporating more advanced words",
            "Your word choice shows good progression",
            "Consider using more sophisticated synonyms"
        ])
    else:  # top
        suggestions.extend([
            "Excellent vocabulary sophistication!",
            "Your word choice demonstrates advanced language skills",
            "Keep up the great use of complex vocabulary"
        ])
    
    # Add word count specific suggestions
    if tier_analysis.word_count < 10:
        suggestions.append("Try expressing your thoughts in more detail")
    elif tier_analysis.word_count > 50:
        suggestions.append("Great detailed response! Practice being concise too")
    
    return suggestions[:3]  # Return top 3 suggestions