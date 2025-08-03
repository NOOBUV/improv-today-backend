"""
WebSocket message schemas for real-time conversation synchronization.
These schemas define the message types and structure for WebSocket communication
as outlined in the redesign plan.
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal
from datetime import datetime


class WebSocketMessage(BaseModel):
    """Base WebSocket message structure"""
    type: str
    conversation_id: str
    timestamp: datetime
    payload: Dict[str, Any]


class StateUpdateMessage(BaseModel):
    """State update message for conversation state changes"""
    type: Literal["state_update"] = "state_update"
    conversation_id: str
    timestamp: datetime
    payload: Dict[str, Any]  # Contains: current_state, speech_recognition_active, speech_synthesis_active


class TranscriptUpdateMessage(BaseModel):
    """Transcript synchronization message"""
    type: Literal["transcript_update"] = "transcript_update" 
    conversation_id: str
    timestamp: datetime
    payload: Dict[str, Any]  # Contains: interim_transcript, accumulated_transcript, confidence


class AIResponseMessage(BaseModel):
    """AI response delivery message"""
    type: Literal["ai_response"] = "ai_response"
    conversation_id: str
    timestamp: datetime
    payload: Dict[str, Any]  # Contains: message content, audio_url, feedback


class SpeechEventMessage(BaseModel):
    """Speech coordination events"""
    type: Literal["speech_event"] = "speech_event"
    conversation_id: str
    timestamp: datetime
    payload: Dict[str, Any]  # Contains: event_type, data


class ErrorMessage(BaseModel):
    """Error message for WebSocket communication"""
    type: Literal["error"] = "error"
    conversation_id: str
    timestamp: datetime
    payload: Dict[str, Any]  # Contains: error_type, message, retry_after


class ConnectionMessage(BaseModel):
    """Connection status messages"""
    type: Literal["connection_status"] = "connection_status"
    conversation_id: str
    timestamp: datetime
    payload: Dict[str, Any]  # Contains: status (connected/disconnected), client_count


# Union type for all possible WebSocket messages
WebSocketMessageType = (
    StateUpdateMessage |
    TranscriptUpdateMessage |
    AIResponseMessage |
    SpeechEventMessage |
    ErrorMessage |
    ConnectionMessage
)