from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.auth.dependencies import verify_protected_token
from app.services.simple_openai import SimpleOpenAIService, OpenAICoachingResponse, WordUsageStatus
from app.services.enhanced_conversation_service import EnhancedConversationService
from app.services.redis_service import RedisService
from app.services.vocabulary_tier_service import VocabularyTierService
from app.services.suggestion_service import SuggestionService
from app.models.vocabulary import VocabularySuggestion
from app.models.conversation_v2 import Conversation
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timezone
import uuid
import re

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
    suggestion: Optional[Dict] = None
    used_suggestion_id: Optional[str] = None
    remediation_feedback: Optional[str] = None  # AC: 4 - Include remediation feedback in API response
    simulation_context: Optional[Dict] = None  # New field for simulation integration
    selected_backstory_types: Optional[List[str]] = None  # What backstory content was used
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



def get_most_recent_active_suggestion(db: Session, user_id: str) -> Optional[VocabularySuggestion]:
    """Get the most recent active suggestion for the user (shown or used_incorrectly)"""
    return db.query(VocabularySuggestion).filter(
        VocabularySuggestion.user_id == user_id,
        VocabularySuggestion.status.in_(["shown", "used_incorrectly"])  # AC: 5 - Include incorrectly used suggestions
    ).order_by(VocabularySuggestion.created_at.desc()).first()



def detect_word_usage(corrected_transcript: str, suggested_word: str) -> bool:
    """
    Detect if the suggested word is used in the corrected transcript.
    
    Implements case-insensitive matching with word boundaries and plural handling.
    Uses regex to avoid partial matches (e.g., "run" in "running").
    
    Args:
        corrected_transcript: The corrected transcript text to search
        suggested_word: The word to look for
        
    Returns:
        bool: True if word is found with proper word boundaries
        
    Examples:
        >>> detect_word_usage("I want to elaborate on this", "elaborate")
        True
        >>> detect_word_usage("I want to elaborate on this", "labor")  # partial match
        False
        >>> detect_word_usage("Multiple books on shelves", "book")  # plural
        True
    """
    if not corrected_transcript or not suggested_word:
        return False
    
    # Normalize both text and word for comparison
    normalized_transcript = corrected_transcript.lower().strip()
    normalized_word = suggested_word.lower().strip()
    
    # Create word boundary pattern to avoid partial matches
    # This ensures "run" doesn't match in "running" but does match in "I run daily"
    word_pattern = r'\b' + re.escape(normalized_word) + r'\b'
    
    # Also check for common plural forms (word + s)
    # This handles most English plurals: book->books, cat->cats
    plural_pattern = r'\b' + re.escape(normalized_word) + r's\b'
    
    return bool(re.search(word_pattern, normalized_transcript) or
                re.search(plural_pattern, normalized_transcript))

# Main endpoint that frontend expects: POST /api/conversation
@router.post("", response_model=ConversationResponse)

async def handle_conversation(
    request: ConversationRequest,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(verify_protected_token)
):
    """
    Main conversation endpoint - receives transcript and returns AI response with feedback
    """
    try:
        # Get or create user from Auth0 token
        user_id = get_or_create_user(db, current_user)

        openai_service = SimpleOpenAIService()
        enhanced_conversation_service = EnhancedConversationService()
        vocabulary_service = VocabularyTierService()
        suggestion_service = SuggestionService()
        redis_service = RedisService()
        
        # Load personality and topic from session if session_id provided
        session_personality = request.personality
        if request.session_id:
            from app.models.session import Session as SessionModel
            session_row = db.query(SessionModel).filter(SessionModel.id == request.session_id).first()
            if not session_row:
                raise HTTPException(status_code=404, detail="Session not found")
            session_personality = session_personality or (session_row.personality or "friendly_neutral")
            # update last_message_at
            session_row.last_message_at = datetime.now(timezone.utc)
            db.commit()

        # Check for existing active suggestion for usage detection (AC: 5)
        recent_suggestion = get_most_recent_active_suggestion(db, str(user_id))
        used_suggestion_id = None
        
        # Map minimalist personalities to existing prompt keys
        personality_map = {
            "friendly": "friendly_neutral",
            "sassy": "sassy_english",
            "blunt": "blunt_american",
        }
        effective_personality = personality_map.get(session_personality or "friendly", session_personality or "friendly_neutral")

        # Find existing conversation for this session or create new one
        conversation = None

        if request.session_id:
            # Look for existing active conversation for this session
            try:
                conversation = db.query(Conversation).filter(
                    Conversation.session_id == request.session_id,
                    Conversation.user_id == str(user_id),
                    Conversation.status == 'active'
                ).first()
            except Exception as e:
                conversation = None

        if not conversation:
            # Create new conversation if none exists
            conversation_id = uuid.uuid4()
            conversation = Conversation(
                id=str(conversation_id),
                user_id=str(user_id),
                session_id=request.session_id,
                status='active',
                personality=effective_personality
            )
            db.add(conversation)
            db.flush()  # Get the ID without committing
            # Note: Conversation will be committed later with messages
        else:
            conversation_id = conversation.id
        
        # Get conversation history from Redis with database fallback (AC: 1, IV1)
        conversation_history_data = redis_service.get_conversation_history(str(conversation_id), db)
        print(f"📚 Retrieved conversation history: {len(conversation_history_data)} messages for conversation {conversation_id}")
        conversation_context = redis_service.build_conversation_context(conversation_history_data)

        # Note: Enhanced conversation service will use its own SessionStateService for conversation history
        # The old conversation_context is kept for fallback and suggestion turn counting compatibility
        
        # Get suggested word for usage evaluation if available
        suggested_word = recent_suggestion.suggested_word if recent_suggestion else None
        
        # Check for graceful suggestion replacement (IV2)
        should_replace_suggestion = False
        if recent_suggestion and suggested_word:
            # Count conversation turns since suggestion was created
            turns_since_suggestion = len([msg for msg in conversation_history_data 
                                        if msg.get('timestamp') and 
                                        msg.get('timestamp') > recent_suggestion.created_at.isoformat()])
            
            # Replace suggestion after 3-4 turns without any usage attempt
            if turns_since_suggestion >= 4:  # 4 turns = 2 user messages + 2 AI responses
                should_replace_suggestion = True
                print(f"🔄 Graceful replacement: {turns_since_suggestion} turns since suggestion created")
        
        # Generate enhanced coaching response with simulation context integration (Story 2.6)
        try:
            # Retrieve user preferences from session for personalization
            from app.services.session_state_service import SessionStateService
            session_service = SessionStateService()
            try:
                session_state = await session_service.get_session_state(str(user_id), str(conversation_id))
                user_preferences = session_state.get("personalization", {}) if session_state else {}
            except Exception as e:
                print(f"⚠️ Could not retrieve session personalization, using defaults: {str(e)}")
                user_preferences = {}

            enhanced_response = await enhanced_conversation_service.generate_enhanced_response(
                user_message=request.message,
                user_id=str(user_id),
                conversation_id=str(conversation_id),
                conversation_history=None,  # Let enhanced service use its own SessionStateService
                personality=effective_personality,
                target_vocabulary=request.target_vocabulary,
                suggested_word=suggested_word,
                user_preferences=user_preferences
            )

            ai_response = enhanced_response["ai_response"]
            corrected_transcript = enhanced_response["corrected_transcript"]
            word_usage_status = enhanced_response["word_usage_status"]
            usage_feedback = enhanced_response["usage_correctness_feedback"]
            simulation_context = enhanced_response.get("simulation_context", {})
            selected_backstory_types = enhanced_response.get("selected_backstory_types", [])

        except Exception as e:
            print(f"⚠️ Enhanced conversation service failed, falling back to simple service: {str(e)}")
            # Fallback to original service
            coaching_response = await openai_service.generate_coaching_response(
                request.message,
                conversation_context,
                effective_personality,
                request.target_vocabulary,
                suggested_word
            )

            ai_response = coaching_response.ai_response
            corrected_transcript = coaching_response.corrected_transcript
            word_usage_status = coaching_response.word_usage_status
            usage_feedback = coaching_response.usage_correctness_feedback
            simulation_context = None
            selected_backstory_types = None
        
        # Update suggestion status based on coaching response analysis (AC: 3, 4)
        if recent_suggestion:
            try:
                if word_usage_status == WordUsageStatus.USED_CORRECTLY:
                    recent_suggestion.status = "used"
                    used_suggestion_id = str(recent_suggestion.id)
                    print(f"✅ Word usage correct: '{recent_suggestion.suggested_word}' marked as used (ID: {used_suggestion_id})")
                elif word_usage_status == WordUsageStatus.USED_INCORRECTLY:
                    # Update status to track incorrect usage (AC: 4)
                    recent_suggestion.status = "used_incorrectly"
                    print(f"⚠️ Word used incorrectly: '{recent_suggestion.suggested_word}' - {usage_feedback}")
                elif should_replace_suggestion:
                    # Graceful replacement without negative feedback (IV2)
                    recent_suggestion.status = "ignored"
                    print(f"🔄 Gracefully replacing suggestion '{recent_suggestion.suggested_word}' after timeout")
                    # Save the status change to database first
                    db.add(recent_suggestion)
                    db.flush()  # Ensure the update is included in transaction
                    # Clear the recent_suggestion since it's now ignored and should not block new suggestions
                    recent_suggestion = None
                else:
                    print(f"ℹ️ Word not used: '{recent_suggestion.suggested_word}'")
                    db.add(recent_suggestion)
                    db.flush()  # Ensure the update is included in transaction
            except Exception as e:
                print(f"⚠️ Failed to update suggestion status: {str(e)}")
                # Continue without blocking the conversation
        
        # Analyze vocabulary tier on corrected transcript for better accuracy
        tier_analysis = vocabulary_service.analyze_vocabulary_tier(corrected_transcript)
        
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
        
        # Save conversation messages to database and Redis cache
        try:
            from app.models.conversation_v2 import ConversationMessage
            
            # Save user message to database (AC: 5 - use corrected_transcript)
            user_message = ConversationMessage(
                conversation_id=conversation_id,
                role="user",
                content=corrected_transcript,  # Use corrected transcript as per AC: 5
                timestamp=datetime.now(timezone.utc)
            )
            db.add(user_message)
            db.flush()
            
            # Save AI response to database
            ai_message = ConversationMessage(
                conversation_id=conversation_id,
                role="assistant",
                content=ai_response,
                timestamp=datetime.now(timezone.utc)
            )
            db.add(ai_message)
            db.flush()
            
            # Cache messages in Redis for future conversation history (AC: 2)
            redis_service.cache_message(str(conversation_id), "user", corrected_transcript, user_message.timestamp)
            redis_service.cache_message(str(conversation_id), "assistant", ai_response, ai_message.timestamp)
            
            # Commit conversation and messages to ensure persistence
            db.commit()
            print(f"✅ Conversation and messages committed to database")
            
        except Exception as e:
            print(f"⚠️ Message saving failed: {str(e)}")
            # Continue without blocking conversation
        
        # Generate vocabulary suggestion and save to DB
        suggestion_data = None
        
        # Generate new suggestion if no active suggestion or if graceful replacement needed
        if not recent_suggestion or should_replace_suggestion or word_usage_status == WordUsageStatus.USED_CORRECTLY:
            print(f"🔍 Suggestion generation condition met: recent_suggestion={bool(recent_suggestion)}, should_replace={should_replace_suggestion}, word_usage={word_usage_status}")
            try:
                suggestion = suggestion_service.generate_suggestion(request.message)
                print(f"🔍 Suggestion service returned: {suggestion}")
                if suggestion:
                    # Save suggestion to database
                    db_suggestion = VocabularySuggestion(
                        conversation_id=conversation_id,
                        user_id=str(user_id),
                        suggested_word=suggestion["word"],
                        status="shown"
                    )
                    db.add(db_suggestion)
                    db.commit()
                    db.refresh(db_suggestion)
                    
                    suggestion_data = {
                        "id": str(db_suggestion.id),
                        "word": suggestion["word"],
                        "definition": suggestion["definition"],
                        "exampleSentence": suggestion["exampleSentence"]
                    }
                    
                    if should_replace_suggestion:
                        print(f"🔄 Generated replacement suggestion: {suggestion['word']}")
                        
            except Exception as e:
                print(f"⚠️ Suggestion generation failed: {str(e)}")
                # Continue without suggestion as per IV1 requirement
                # Note: Don't rollback here as conversation and messages are already committed
        
        # Create enhanced usage analysis with word usage feedback
        usage_analysis = None
        if word_usage_status != WordUsageStatus.NOT_USED:
            usage_analysis = {
                "word_usage_status": word_usage_status.value,
                "suggested_word": suggested_word,
                "usage_feedback": usage_feedback,
                "conversation_context_used": bool(conversation_context)
            }
        
        print(f"🎯 Original Message: {request.message}")
        print(f"✅ Corrected Transcript: {corrected_transcript}")
        print(f"📊 Vocabulary Tier: {tier_analysis.tier} (Score: {tier_analysis.score})")
        print(f"🤖 AI Response: {ai_response}")
        print(f"📚 Word Usage Status: {word_usage_status.value}")
        if usage_feedback:
            print(f"💬 Usage Feedback: {usage_feedback}")
        if used_suggestion_id:
            print(f"🎉 Used Suggestion ID: {used_suggestion_id}")
        if suggestion_data:
            print(f"💡 New Suggestion: {suggestion_data['word']} - {suggestion_data['definition']}")
        print(f"🗨️ Conversation History Length: {len(conversation_history_data)} messages")
        
        return ConversationResponse(
            response=ai_response,
            feedback=feedback,
            vocabulary_tier=vocabulary_tier_data,
            usage_analysis=usage_analysis,
            suggestion=suggestion_data,
            used_suggestion_id=used_suggestion_id,
            remediation_feedback=usage_feedback,  # AC: 4 - Include remediation feedback in response
            simulation_context=simulation_context,  # Story 2.6 - Simulation context integration
            selected_backstory_types=selected_backstory_types  # Story 2.6 - Backstory content used
        )
        
    except Exception as e:
        print(f"❌ Conversation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversation failed: {str(e)}")

# Add vocabulary tier analysis endpoint
@router.post("/analyze-tier")
async def analyze_vocabulary_tier(
    request: Dict, 
    db: Session = Depends(get_db),
    current_user: Dict = Depends(verify_protected_token)
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
    current_user: Dict = Depends(verify_protected_token)
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
    current_user: Dict = Depends(verify_protected_token)
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
    if word_count > 5:
        base_score += 10
    if word_count > 10:
        base_score += 5
    if avg_word_length > 4:
        base_score += 10
    
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