from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class ConversationRequest(BaseModel):
    """Request schema for conversation endpoint."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's message to Ava")
    user_id: Optional[str] = Field(None, description="Optional user identifier")


class EmotionalState(BaseModel):
    """Schema for representing Ava's emotional state."""
    mood: str = Field(..., description="Current mood (e.g., 'cheerful', 'thoughtful', 'excited')")
    energy: int = Field(..., ge=1, le=10, description="Energy level from 1-10")
    stress: int = Field(..., ge=1, le=10, description="Stress level from 1-10")


class ConversationResponse(BaseModel):
    """Response schema for conversation endpoint."""
    message: str = Field(..., description="Ava's response message")
    emotional_state: EmotionalState = Field(..., description="Ava's current emotional state")
    timestamp: datetime = Field(..., description="Response timestamp")
    conversation_id: Optional[str] = Field(None, description="Conversation session identifier")


class AvaStateRead(BaseModel):
    """Schema for reading AvaState data."""
    state_id: int
    trait_name: str
    value: str
    last_updated: datetime

    class Config:
        from_attributes = True
        # Note: This will be updated to ConfigDict in future Pydantic 3.0 migration


class AvaStateCreate(BaseModel):
    """Schema for creating AvaState data."""
    trait_name: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., min_length=1, max_length=500)


class AvaStateUpdate(BaseModel):
    """Schema for updating AvaState data."""
    value: str = Field(..., min_length=1, max_length=500)