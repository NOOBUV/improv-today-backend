from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List
from datetime import datetime
import uuid

from app.core.database import get_db
from app.models.ava_state import AvaState
from app.schemas.ava import (
    ConversationRequest, 
    ConversationResponse, 
    EmotionalState,
    AvaStateRead,
    AvaStateCreate,
    AvaStateUpdate
)

router = APIRouter()


@router.post("/conversation", response_model=ConversationResponse)
async def conversation(
    request: ConversationRequest, 
    db: Session = Depends(get_db)
):
    """
    Main conversation endpoint for interacting with Ava.
    
    Receives a user message and returns Ava's response with emotional state.
    Currently returns a hardcoded response as specified in the requirements.
    """
    try:
        # For now, return a hardcoded sample response with varying emotional states
        # This will be replaced with actual consciousness generation logic
        
        # Sample emotional state data
        sample_emotional_state = EmotionalState(
            mood="happy",
            energy=7,
            stress=3
        )
        
        # Generate a simple conversation ID for this session
        conversation_id = str(uuid.uuid4())
        
        # Sample hardcoded responses based on message content
        sample_responses = [
            "That's really interesting! I've been thinking about similar things lately.",
            "I appreciate you sharing that with me. It reminds me of something from my own experiences.",
            "You know, that brings up some fascinating questions. I'd love to explore this more with you.",
            "I can really relate to that feeling. Sometimes I find myself pondering the same kinds of things."
        ]
        
        # Simple logic to vary the response
        response_index = len(request.message) % len(sample_responses)
        sample_message = sample_responses[response_index]
        
        response = ConversationResponse(
            message=sample_message,
            emotional_state=sample_emotional_state,
            timestamp=datetime.now(),
            conversation_id=conversation_id
        )
        
        return response
        
    except Exception as e:
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