from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class EmotionType(str, Enum):
    """Standardized emotion types for Clara"""
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    STRESSED = "stressed"
    SASSY = "sassy"


class ConversationRequest(BaseModel):
    """Request schema for conversation endpoint."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's message to Clara")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")
    session_id: Optional[int] = Field(None, description="Session ID for conversation tracking")
    personality: Optional[str] = Field(None, description="Conversation personality style")


class EmotionalState(BaseModel):
    """Schema for representing Clara's emotional state with standardized emotions."""
    emotion: EmotionType = Field(..., description="Current primary emotion")
    mood: str = Field(..., description="Current mood description")
    energy: int = Field(..., ge=1, le=10, description="Energy level from 1-10")
    stress: int = Field(..., ge=1, le=10, description="Stress level from 1-10")


class ConversationResponse(BaseModel):
    """Enhanced response schema for conversation endpoint with emotion tagging."""
    message: str = Field(..., description="Clara's response message")
    emotion: EmotionType = Field(..., description="Primary emotion tag for this response")
    emotional_state: EmotionalState = Field(..., description="Clara's detailed emotional state")
    timestamp: datetime = Field(..., description="Response timestamp")
    conversation_id: Optional[str] = Field(None, description="Conversation session identifier")
    context_used: bool = Field(default=False, description="Whether conversation history was used")


class ClaraStateRead(BaseModel):
    """Schema for reading ClaraState data."""
    state_id: int
    trait_name: str
    value: str
    last_updated: datetime

    model_config = ConfigDict(from_attributes=True)


class ClaraStateCreate(BaseModel):
    """Schema for creating ClaraState data."""
    trait_name: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., min_length=1, max_length=500)


class ClaraStateUpdate(BaseModel):
    """Schema for updating ClaraState data."""
    value: str = Field(..., min_length=1, max_length=500)