from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List
from datetime import datetime
import uuid
import logging

from app.core.database import get_db
from app.auth.subscription_guard import require_active_subscription
from app.models.user import User
from app.models.clara_state import ClaraState
from app.schemas.clara import (
    ConversationRequest,
    ConversationResponse,
    EmotionalState,
    EmotionType,
    ClaraStateRead,
    ClaraStateCreate,
    ClaraStateUpdate
)
from app.services.character_content_service import CharacterContentService
from app.services.conversation_prompt_service import ConversationPromptService, EmotionType as ServiceEmotionType
from app.services.clara_llm_service import ClaraLLMService
from app.services.enhanced_conversation_service import EnhancedConversationService
from app.services.event_selection_service import EventSelectionService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/conversation", response_model=ConversationResponse)
async def conversation(
    request: ConversationRequest, 
    current_user: User = Depends(require_active_subscription),
    db: Session = Depends(get_db)
):
    """
    Main conversation endpoint for interacting with Clara.

    Implements Story 1.4: V0 Conversational Logic
    - Loads Clara's backstory and guiding principles
    - Constructs LLM prompt with foundational context
    - Calls premium LLM for authentic responses
    - Returns structured response with emotion tagging
    """
    try:
        logger.info(f"Processing conversation request for user {current_user.id}: {request.message[:100]}...")

        # Initialize services
        enhanced_service = EnhancedConversationService()
        event_service = EventSelectionService()

        # Handle conversation tracking with session_id
        if request.session_id:
            # Use session_id as conversation_id for consistency
            conversation_id = f"session_{request.session_id}_user_{current_user.id}"
            logger.info(f"🔄 Using session-based conversation ID: {conversation_id}")
        else:
            # Generate new conversation ID if no session_id provided
            conversation_id = request.conversation_id or str(uuid.uuid4())
            logger.info(f"🆕 Generated new conversation ID: {conversation_id}")

        # Get fresh events to avoid repetition before generating response
        logger.info("Getting fresh events for conversation context...")
        fresh_events = await event_service.get_contextual_events(
            user_id=str(current_user.id),
            conversation_id=conversation_id,
            user_message=request.message,
            max_events=2  # Limit to 2 events to avoid overwhelming Clara
        )
        logger.info(f"Retrieved {len(fresh_events)} fresh events for conversation context")

        # Use enhanced service to generate response with simulation context and fresh events
        logger.info("Generating enhanced response with simulation context...")
        enhanced_response = await enhanced_service.generate_enhanced_response(
            user_message=request.message,
            user_id=str(current_user.id),
            conversation_id=conversation_id,
            conversation_history=None,  # Enhanced service handles its own conversation history via SessionStateService
            personality=request.personality or "friendly_neutral",
            fresh_events=fresh_events  # Pass fresh events to avoid repetition
        )

        # Extract response data from enhanced service
        ai_message = enhanced_response.get("ai_response", "I'm sorry, I'm having trouble responding right now.")
        fallback_mode = enhanced_response.get("fallback_mode", False)
        enhanced_mode = enhanced_response.get("enhanced_mode", False)
        events_mentioned = enhanced_response.get("events_mentioned", [])

        if fallback_mode:
            logger.warning("Enhanced conversation service returned fallback response")

        # Track events mentioned in Clara's response to prevent future repetition
        if events_mentioned:
            logger.info(f"Tracking {len(events_mentioned)} events mentioned in response...")
            await event_service.track_events_mentioned_in_response(
                user_id=str(current_user.id),
                conversation_id=conversation_id,
                events_mentioned=events_mentioned
            )
            logger.info("Events tracked successfully")
        else:
            logger.info("No events mentioned in response - no tracking needed")

        # Default emotion type (enhanced service doesn't return structured emotion yet)
        emotion_type = EmotionType.CALM  # TODO: Extract emotion from enhanced service

        # Construct emotional state for response
        emotional_state = EmotionalState(
            emotion=emotion_type,
            mood=emotion_type.value,
            energy=7,  # Default values - could be enhanced with state management
            stress=6   # Normalized stress level (1-10 scale) based on story requirements
        )

        # Build final response (AC: 5)
        response = ConversationResponse(
            message=ai_message,
            emotion=emotion_type,
            emotional_state=emotional_state,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            context_used=enhanced_mode  # True if using enhanced context
        )

        logger.info(f"Generated enhanced response: {ai_message[:100]}... (enhanced_mode: {enhanced_mode})")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing conversation: {str(e)}"
        )


@router.get("/state", response_model=List[ClaraStateRead])
async def get_clara_states(db: Session = Depends(get_db)):
    """Get all Clara's current state traits."""
    states = db.query(ClaraState).all()
    return states


@router.get("/state/{trait_name}", response_model=ClaraStateRead)
async def get_clara_state(trait_name: str, db: Session = Depends(get_db)):
    """Get a specific state trait by name."""
    state = db.query(ClaraState).filter(ClaraState.trait_name == trait_name).first()
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"State trait '{trait_name}' not found"
        )
    return state


@router.post("/state", response_model=ClaraStateRead)
async def create_clara_state(
    state_data: ClaraStateCreate,
    db: Session = Depends(get_db)
):
    """Create a new state trait for Clara."""
    try:
        new_state = ClaraState(**state_data.model_dump())
        db.add(new_state)
        db.commit()
        db.refresh(new_state)
        return new_state
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"State trait '{state_data.trait_name}' already exists"
        )


@router.put("/state/{trait_name}", response_model=ClaraStateRead)
async def update_clara_state(
    trait_name: str,
    state_data: ClaraStateUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing state trait."""
    state = db.query(ClaraState).filter(ClaraState.trait_name == trait_name).first()
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"State trait '{trait_name}' not found"
        )
    
    state.value = state_data.value
    db.commit()
    db.refresh(state)
    return state


@router.post("/conversation/stream")
async def stream_conversation(
    request: ConversationRequest,
    current_user: User = Depends(require_active_subscription),
    db: Session = Depends(get_db)
):
    """
    Stream conversation response using Server-Sent Events (SSE) for optimized performance.

    Provides progressive response delivery to reduce perceived latency from ~6.4s to <500ms
    for first meaningful content. Uses enhanced service with stream=True for feature parity.
    """
    try:
        from fastapi.responses import StreamingResponse
        from app.services.event_selection_service import EventSelectionService

        logger.info(f"Processing streaming conversation request for user {current_user.id}: {request.message[:100]}...")

        # Initialize services (same as normal /conversation endpoint)
        enhanced_service = EnhancedConversationService()
        event_service = EventSelectionService()

        # Handle conversation tracking with session_id (same as normal endpoint)
        if request.session_id:
            conversation_id = f"session_{request.session_id}_user_{current_user.id}"
            logger.info(f"🔄 Using session-based conversation ID: {conversation_id}")
        else:
            conversation_id = request.conversation_id or str(uuid.uuid4())
            logger.info(f"🆕 Generated new conversation ID: {conversation_id}")

        # Get fresh events BEFORE streaming (same as normal endpoint)
        logger.info("Getting fresh events for conversation context...")
        fresh_events = await event_service.get_contextual_events(
            user_id=str(current_user.id),
            conversation_id=conversation_id,
            user_message=request.message,
            max_events=2  # Limit to 2 events to avoid overwhelming Clara
        )
        logger.info(f"Retrieved {len(fresh_events)} fresh events for conversation context")

        # Use enhanced service with stream=True for SSE streaming
        logger.info("Generating streaming enhanced response with simulation context...")
        response_stream = await enhanced_service.generate_enhanced_response(
            user_message=request.message,
            user_id=str(current_user.id),
            conversation_id=conversation_id,
            conversation_history=None,  # Enhanced service handles via SessionStateService
            personality=request.personality or "friendly_neutral",
            fresh_events=fresh_events,
            stream=True  # Enable SSE streaming
        )

        # Return SSE streaming response
        return StreamingResponse(
            response_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Critical: Disable nginx/proxy buffering
                "X-Content-Type-Options": "nosniff",  # Prevent browser MIME sniffing buffering
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )

    except Exception as e:
        logger.error(f"❌ Streaming Conversation Error: {str(e)}")
        # Rollback database transaction to prevent aborted transaction state
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error(f"Database rollback failed: {str(rollback_error)}")
        raise HTTPException(status_code=500, detail=f"Streaming conversation failed: {str(e)}")