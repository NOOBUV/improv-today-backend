"""
ConversationStateManager Service

This service manages the real-time state of conversations, providing thread-safe
state transitions, transcript synchronization, and WebSocket coordination as
outlined in the redesign plan.

Key responsibilities:
- Thread-safe conversation state management
- Real-time transcript synchronization
- Speech event coordination
- WebSocket message broadcasting
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import update
from app.core.database import SessionLocal
from app.models.conversation_v2 import Conversation, SessionState, ConversationMessage
from app.schemas.websocket import (
    StateUpdateMessage, 
    TranscriptUpdateMessage, 
    AIResponseMessage, 
    SpeechEventMessage,
    ErrorMessage
)

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Conversation state machine states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    WAITING_FOR_USER = "waiting_for_user"
    ERROR = "error"
    ENDED = "ended"


@dataclass
class ActiveConnection:
    """Represents an active WebSocket connection"""
    websocket: object  # WebSocket instance
    conversation_id: str
    connected_at: datetime
    last_heartbeat: datetime


class ConversationStateManager:
    """
    Thread-safe conversation state manager for real-time synchronization.
    Manages conversation states, transcript updates, and WebSocket connections.
    """
    
    def __init__(self):
        self._connections: Dict[str, List[ActiveConnection]] = {}  # conversation_id -> connections
        self._conversation_states: Dict[str, ConversationState] = {}
        self._transcript_cache: Dict[str, str] = {}  # conversation_id -> current transcript
        self._interim_transcript_cache: Dict[str, str] = {}  # conversation_id -> interim transcript
        self._lock = asyncio.Lock()
        
    async def add_connection(self, conversation_id: str, websocket) -> None:
        """Add a new WebSocket connection for a conversation"""
        async with self._lock:
            if conversation_id not in self._connections:
                self._connections[conversation_id] = []
            
            connection = ActiveConnection(
                websocket=websocket,
                conversation_id=conversation_id,
                connected_at=datetime.now(),
                last_heartbeat=datetime.now()
            )
            
            self._connections[conversation_id].append(connection)
            logger.info(f"WebSocket connection added for conversation {conversation_id}")
            
            # Send current state to new connection
            await self._send_current_state(conversation_id, websocket)
    
    async def remove_connection(self, conversation_id: str, websocket) -> None:
        """Remove a WebSocket connection"""
        async with self._lock:
            if conversation_id in self._connections:
                self._connections[conversation_id] = [
                    conn for conn in self._connections[conversation_id]
                    if conn.websocket != websocket
                ]
                
                # Clean up empty connection lists
                if not self._connections[conversation_id]:
                    del self._connections[conversation_id]
                    
                logger.info(f"WebSocket connection removed for conversation {conversation_id}")
    
    async def update_conversation_state(
        self, 
        conversation_id: str, 
        new_state: ConversationState,
        metadata: Optional[Dict] = None
    ) -> None:
        """Update conversation state and broadcast to all connected clients"""
        async with self._lock:
            old_state = self._conversation_states.get(conversation_id, ConversationState.IDLE)
            
            # Validate state transition
            if not self._is_valid_transition(old_state, new_state):
                logger.warning(f"Invalid state transition from {old_state} to {new_state} for conversation {conversation_id}")
                return
            
            # Update in-memory state
            self._conversation_states[conversation_id] = new_state
            
            # Update database
            await self._update_db_state(conversation_id, new_state)
            
            # Broadcast state update to all connections
            await self._broadcast_state_update(conversation_id, new_state, metadata or {})
            
            logger.info(f"Conversation {conversation_id} state updated: {old_state} -> {new_state}")
    
    async def update_transcript(
        self, 
        conversation_id: str, 
        interim_transcript: Optional[str] = None,
        final_transcript: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> None:
        """Update transcript and broadcast to connected clients"""
        async with self._lock:
            # Update interim transcript
            if interim_transcript is not None:
                self._interim_transcript_cache[conversation_id] = interim_transcript
            
            # Update final transcript
            if final_transcript is not None:
                current_transcript = self._transcript_cache.get(conversation_id, "")
                if current_transcript:
                    self._transcript_cache[conversation_id] = f"{current_transcript} {final_transcript}".strip()
                else:
                    self._transcript_cache[conversation_id] = final_transcript
                
                # Clear interim transcript when we have final text
                self._interim_transcript_cache[conversation_id] = ""
                
                # Update database
                await self._update_db_transcript(conversation_id, self._transcript_cache[conversation_id])
            
            # Broadcast transcript update
            await self._broadcast_transcript_update(
                conversation_id, 
                interim_transcript or self._interim_transcript_cache.get(conversation_id, ""),
                self._transcript_cache.get(conversation_id, ""),
                confidence
            )
    
    async def send_ai_response(
        self, 
        conversation_id: str, 
        message_content: str,
        audio_url: Optional[str] = None,
        feedback: Optional[Dict] = None
    ) -> None:
        """Send AI response to all connected clients"""
        # Store message in database
        message_id = await self._store_ai_message(conversation_id, message_content, audio_url, feedback)
        
        # Broadcast AI response
        await self._broadcast_ai_response(conversation_id, message_content, audio_url, feedback, message_id)
    
    async def send_speech_event(
        self, 
        conversation_id: str, 
        event_type: str,
        event_data: Dict
    ) -> None:
        """Send speech coordination events"""
        await self._broadcast_speech_event(conversation_id, event_type, event_data)
    
    async def get_conversation_state(self, conversation_id: str) -> ConversationState:
        """Get current conversation state"""
        return self._conversation_states.get(conversation_id, ConversationState.IDLE)
    
    async def get_current_transcript(self, conversation_id: str) -> Dict[str, str]:
        """Get current transcript state"""
        return {
            "final_transcript": self._transcript_cache.get(conversation_id, ""),
            "interim_transcript": self._interim_transcript_cache.get(conversation_id, "")
        }
    
    def get_active_connections_count(self, conversation_id: str) -> int:
        """Get number of active connections for a conversation"""
        return len(self._connections.get(conversation_id, []))
    
    async def cleanup_conversation(self, conversation_id: str) -> None:
        """Clean up conversation data when conversation ends"""
        async with self._lock:
            # Remove from caches
            self._conversation_states.pop(conversation_id, None)
            self._transcript_cache.pop(conversation_id, None)
            self._interim_transcript_cache.pop(conversation_id, None)
            
            # Close all connections
            if conversation_id in self._connections:
                for connection in self._connections[conversation_id]:
                    try:
                        await connection.websocket.close()
                    except Exception as e:
                        logger.error(f"Error closing WebSocket: {e}")
                
                del self._connections[conversation_id]
            
            logger.info(f"Conversation {conversation_id} cleaned up")
    
    # Private methods
    
    def _is_valid_transition(self, from_state: ConversationState, to_state: ConversationState) -> bool:
        """Validate state transitions based on conversation state machine"""
        valid_transitions = {
            ConversationState.IDLE: {ConversationState.LISTENING, ConversationState.ENDED},
            ConversationState.LISTENING: {ConversationState.PROCESSING, ConversationState.IDLE, ConversationState.ERROR},
            ConversationState.PROCESSING: {ConversationState.SPEAKING, ConversationState.ERROR, ConversationState.WAITING_FOR_USER},
            ConversationState.SPEAKING: {ConversationState.WAITING_FOR_USER, ConversationState.IDLE, ConversationState.ERROR},
            ConversationState.WAITING_FOR_USER: {ConversationState.LISTENING, ConversationState.IDLE, ConversationState.ERROR},
            ConversationState.ERROR: {ConversationState.IDLE, ConversationState.ENDED},
            ConversationState.ENDED: set()  # No transitions from ended state
        }
        
        return to_state in valid_transitions.get(from_state, set())
    
    async def _update_db_state(self, conversation_id: str, state: ConversationState) -> None:
        """Update conversation state in database"""
        try:
            db = SessionLocal()
            try:
                # Update or create session state
                result = db.execute(
                    update(SessionState)
                    .where(SessionState.conversation_id == conversation_id)
                    .values(state=state.value, updated_at=datetime.now())
                )
                
                if result.rowcount == 0:
                    # Create new session state if none exists
                    from app.models.conversation_v2 import SessionState as SessionStateModel
                    session_state = SessionStateModel(
                        conversation_id=conversation_id,
                        state=state.value
                    )
                    db.add(session_state)
                
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error updating database state: {e}")
    
    async def _update_db_transcript(self, conversation_id: str, transcript: str) -> None:
        """Update transcript in database"""
        try:
            db = SessionLocal()
            try:
                db.execute(
                    update(SessionState)
                    .where(SessionState.conversation_id == conversation_id)
                    .values(transcript=transcript, updated_at=datetime.now())
                )
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error updating database transcript: {e}")
    
    async def _store_ai_message(
        self, 
        conversation_id: str, 
        content: str, 
        audio_url: Optional[str],
        feedback: Optional[Dict]
    ) -> str:
        """Store AI message in database and return message ID"""
        try:
            db = SessionLocal()
            try:
                message = ConversationMessage(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=content,
                    audio_url=audio_url,
                    feedback=feedback
                )
                db.add(message)
                db.commit()
                db.refresh(message)
                return str(message.id)
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error storing AI message: {e}")
            return ""
    
    async def _send_current_state(self, conversation_id: str, websocket) -> None:
        """Send current conversation state to a specific connection"""
        try:
            current_state = self._conversation_states.get(conversation_id, ConversationState.IDLE)
            transcript_data = await self.get_current_transcript(conversation_id)
            
            state_message = StateUpdateMessage(
                conversation_id=conversation_id,
                timestamp=datetime.now(),
                payload={
                    "current_state": current_state.value,
                    "speech_recognition_active": current_state == ConversationState.LISTENING,
                    "speech_synthesis_active": current_state == ConversationState.SPEAKING,
                    "transcript": transcript_data
                }
            )
            
            await websocket.send_text(state_message.model_dump_json())
        except Exception as e:
            logger.error(f"Error sending current state: {e}")
    
    async def _broadcast_state_update(
        self, 
        conversation_id: str, 
        state: ConversationState, 
        metadata: Dict
    ) -> None:
        """Broadcast state update to all connections"""
        if conversation_id not in self._connections:
            return
        
        message = StateUpdateMessage(
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            payload={
                "current_state": state.value,
                "speech_recognition_active": state == ConversationState.LISTENING,
                "speech_synthesis_active": state == ConversationState.SPEAKING,
                **metadata
            }
        )
        
        await self._broadcast_message(conversation_id, message.model_dump_json())
    
    async def _broadcast_transcript_update(
        self, 
        conversation_id: str, 
        interim_transcript: str,
        accumulated_transcript: str, 
        confidence: Optional[float]
    ) -> None:
        """Broadcast transcript update to all connections"""
        if conversation_id not in self._connections:
            return
        
        message = TranscriptUpdateMessage(
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            payload={
                "interim_transcript": interim_transcript,
                "accumulated_transcript": accumulated_transcript,
                "confidence": confidence
            }
        )
        
        await self._broadcast_message(conversation_id, message.model_dump_json())
    
    async def _broadcast_ai_response(
        self, 
        conversation_id: str, 
        content: str, 
        audio_url: Optional[str],
        feedback: Optional[Dict],
        message_id: str
    ) -> None:
        """Broadcast AI response to all connections"""
        if conversation_id not in self._connections:
            return
        
        message = AIResponseMessage(
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            payload={
                "message_id": message_id,
                "content": content,
                "audio_url": audio_url,
                "feedback": feedback
            }
        )
        
        await self._broadcast_message(conversation_id, message.model_dump_json())
    
    async def _broadcast_speech_event(
        self, 
        conversation_id: str, 
        event_type: str, 
        event_data: Dict
    ) -> None:
        """Broadcast speech event to all connections"""
        if conversation_id not in self._connections:
            return
        
        message = SpeechEventMessage(
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            payload={
                "event_type": event_type,
                "data": event_data
            }
        )
        
        await self._broadcast_message(conversation_id, message.model_dump_json())
    
    async def _broadcast_message(self, conversation_id: str, message: str) -> None:
        """Broadcast message to all connections for a conversation"""
        if conversation_id not in self._connections:
            return
        
        # Create a copy of connections to avoid modification during iteration
        connections = self._connections[conversation_id].copy()
        disconnected_connections = []
        
        for connection in connections:
            try:
                await connection.websocket.send_text(message)
                connection.last_heartbeat = datetime.now()
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                disconnected_connections.append(connection)
        
        # Remove disconnected connections
        if disconnected_connections:
            async with self._lock:
                for conn in disconnected_connections:
                    try:
                        self._connections[conversation_id].remove(conn)
                    except ValueError:
                        pass  # Connection already removed


# Global singleton instance
conversation_state_manager = ConversationStateManager()