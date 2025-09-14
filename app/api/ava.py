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
from app.models.ava_state import AvaState
from app.schemas.ava import (
    ConversationRequest, 
    ConversationResponse, 
    EmotionalState,
    EmotionType,
    AvaStateRead,
    AvaStateCreate,
    AvaStateUpdate
)
from app.services.character_content_service import CharacterContentService
from app.services.conversation_prompt_service import ConversationPromptService, EmotionType as ServiceEmotionType
from app.services.ava_llm_service import AvaLLMService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/conversation", response_model=ConversationResponse)
async def conversation(
    request: ConversationRequest, 
    current_user: User = Depends(require_active_subscription),
    db: Session = Depends(get_db)
):
    """
    Main conversation endpoint for interacting with Ava.
    
    Implements Story 1.4: V0 Conversational Logic
    - Loads Ava's backstory and guiding principles
    - Constructs LLM prompt with foundational context
    - Calls premium LLM for authentic responses
    - Returns structured response with emotion tagging
    """
    try:
        logger.info(f"Processing conversation request for user {current_user.id}: {request.message[:100]}...")
        
        # Initialize services
        content_service = CharacterContentService()
        prompt_service = ConversationPromptService()
        llm_service = AvaLLMService()
        
        # Generate conversation ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Load Ava's character content (AC: 1)
        logger.info("Loading Ava's character content...")
        character_backstory = content_service.get_consolidated_backstory()
        
        if not character_backstory:
            logger.warning("No character content loaded, using minimal backstory")
            character_backstory = "You are Ava, a bright, dry-witted 22-year-old creative strategist."
        
        # Determine conversation emotion based on user message (AC: 4)
        conversation_emotion, emotion_reasoning = prompt_service.determine_emotion_from_context(
            request.message,
            conversation_history=None,  # TODO: Load from conversation history in future
            global_mood="stressed"
        )
        
        logger.info(f"Selected emotion: {conversation_emotion} - {emotion_reasoning}")
        
        # Construct LLM prompt following Pattern B architecture (AC: 2)
        conversation_prompt = prompt_service.construct_conversation_prompt(
            character_backstory=character_backstory,
            user_message=request.message,
            conversation_emotion=conversation_emotion,
            global_mood="stressed",
            stress_level=65,
            conversation_history=None  # TODO: Add conversation history support
        )
        
        logger.info(f"Constructed conversation prompt: {len(conversation_prompt)} characters")
        
        # Generate response using OpenAI API (AC: 3)
        ava_response = await llm_service.generate_ava_response(
            prompt=conversation_prompt,
            max_tokens=200,
            temperature=0.8,
            timeout=30
        )
        
        if not ava_response.success:
            logger.warning("LLM service returned fallback response")
        
        # Construct emotional state for response
        emotional_state = EmotionalState(
            emotion=EmotionType(ava_response.emotion.value),
            mood=conversation_emotion.value,
            energy=7,  # Default values - could be enhanced with state management
            stress=6   # Normalized stress level (1-10 scale) based on story requirements
        )
        
        # Build final response (AC: 5)
        response = ConversationResponse(
            message=ava_response.message,
            emotion=EmotionType(ava_response.emotion.value),
            emotional_state=emotional_state,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            context_used=False  # No conversation history used in V0
        )
        
        logger.info(f"Generated response: {ava_response.message[:100]}... (emotion: {ava_response.emotion})")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing conversation: {str(e)}"
        )


@router.get("/state", response_model=List[AvaStateRead])
async def get_ava_states(db: Session = Depends(get_db)):
    """Get all Ava's current state traits."""
    states = db.query(AvaState).all()
    return states


@router.get("/state/{trait_name}", response_model=AvaStateRead)
async def get_ava_state(trait_name: str, db: Session = Depends(get_db)):
    """Get a specific state trait by name."""
    state = db.query(AvaState).filter(AvaState.trait_name == trait_name).first()
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"State trait '{trait_name}' not found"
        )
    return state


@router.post("/state", response_model=AvaStateRead)
async def create_ava_state(
    state_data: AvaStateCreate, 
    db: Session = Depends(get_db)
):
    """Create a new state trait for Ava."""
    try:
        new_state = AvaState(**state_data.model_dump())
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


@router.put("/state/{trait_name}", response_model=AvaStateRead)
async def update_ava_state(
    trait_name: str,
    state_data: AvaStateUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing state trait."""
    state = db.query(AvaState).filter(AvaState.trait_name == trait_name).first()
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"State trait '{trait_name}' not found"
        )
    
    state.value = state_data.value
    db.commit()
    db.refresh(state)
    return state