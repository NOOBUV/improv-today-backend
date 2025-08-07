"""
WebSocket API for real-time conversation synchronization.

This module implements the WebSocket endpoints for real-time communication
between frontend and backend as outlined in the redesign plan.

Key features:
- WebSocket endpoint `/ws/conversations/{conversation_id}`
- Connection management with auto-reconnection support
- Message handling for state updates, transcript sync, and AI responses
- Error handling and connection recovery
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.conversation_v2 import Conversation, SessionState
from app.services.conversation_state_manager import (
    conversation_state_manager, 
    ConversationState
)
from app.schemas.websocket import ErrorMessage, ConnectionMessage

logger = logging.getLogger(__name__)

router = APIRouter()


class WebSocketConnectionManager:
    """
    Manages WebSocket connections and handles message routing.
    Provides connection lifecycle management and error recovery.
    """
    
    def __init__(self):
        self.active_connections: Dict[str, list] = {}
    
    async def connect(self, websocket: WebSocket, conversation_id: str):
        """Accept WebSocket connection and initialize conversation state"""
        await websocket.accept()
        
        # Add connection to state manager
        await conversation_state_manager.add_connection(conversation_id, websocket)
        
        # Send connection confirmation
        connection_message = ConnectionMessage(
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            payload={
                "status": "connected",
                "client_count": conversation_state_manager.get_active_connections_count(conversation_id)
            }
        )
        
        await websocket.send_text(connection_message.model_dump_json())
        logger.info(f"WebSocket connected for conversation: {conversation_id}")
    
    async def disconnect(self, websocket: WebSocket, conversation_id: str):
        """Handle WebSocket disconnection"""
        await conversation_state_manager.remove_connection(conversation_id, websocket)
        logger.info(f"WebSocket disconnected for conversation: {conversation_id}")
    
    async def handle_message(self, websocket: WebSocket, conversation_id: str, message: str):
        """Handle incoming WebSocket messages from clients"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            payload = data.get("payload", {})
            
            logger.info(f"Received WebSocket message: {message_type} for conversation {conversation_id}")
            
            # Route message based on type
            if message_type == "transcript_update":
                await self._handle_transcript_update(conversation_id, payload)
            elif message_type == "state_change_request":
                await self._handle_state_change_request(conversation_id, payload)
            elif message_type == "speech_event":
                await self._handle_speech_event(conversation_id, payload)
            elif message_type == "heartbeat":
                await self._handle_heartbeat(websocket, conversation_id)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                await self._send_error(websocket, conversation_id, "unknown_message_type", f"Unknown message type: {message_type}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in WebSocket message: {e}")
            await self._send_error(websocket, conversation_id, "invalid_json", "Invalid JSON format")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self._send_error(websocket, conversation_id, "message_handling_error", str(e))
    
    async def _handle_transcript_update(self, conversation_id: str, payload: Dict[str, Any]):
        """Handle transcript update from frontend"""
        interim_transcript = payload.get("interim_transcript")
        final_transcript = payload.get("final_transcript")
        confidence = payload.get("confidence")
        
        await conversation_state_manager.update_transcript(
            conversation_id=conversation_id,
            interim_transcript=interim_transcript,
            final_transcript=final_transcript,
            confidence=confidence
        )
    
    async def _handle_state_change_request(self, conversation_id: str, payload: Dict[str, Any]):
        """Handle state change requests from frontend"""
        requested_state = payload.get("state")
        metadata = payload.get("metadata", {})
        
        if requested_state:
            try:
                new_state = ConversationState(requested_state)
                await conversation_state_manager.update_conversation_state(
                    conversation_id=conversation_id,
                    new_state=new_state,
                    metadata=metadata
                )
            except ValueError:
                logger.error(f"Invalid state requested: {requested_state}")
    
    async def _handle_speech_event(self, conversation_id: str, payload: Dict[str, Any]):
        """Handle speech coordination events from frontend"""
        event_type = payload.get("event_type")
        event_data = payload.get("data", {})
        
        if event_type:
            await conversation_state_manager.send_speech_event(
                conversation_id=conversation_id,
                event_type=event_type,
                event_data=event_data
            )
    
    async def _handle_heartbeat(self, websocket: WebSocket, conversation_id: str):
        """Handle heartbeat messages to keep connection alive"""
        heartbeat_response = {
            "type": "heartbeat_response",
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "payload": {
                "status": "alive",
                "server_time": datetime.now().isoformat()
            }
        }
        await websocket.send_text(json.dumps(heartbeat_response, default=str))
    
    async def _send_error(self, websocket: WebSocket, conversation_id: str, error_type: str, message: str):
        """Send error message to WebSocket client"""
        error_message = ErrorMessage(
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            payload={
                "error_type": error_type,
                "message": message,
                "retry_after": None
            }
        )
        
        try:
            await websocket.send_text(error_message.model_dump_json())
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")


# Global connection manager
connection_manager = WebSocketConnectionManager()


@router.websocket("/ws/conversations/{conversation_id}")
async def websocket_conversation_endpoint(
    websocket: WebSocket,
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time conversation synchronization.
    
    Endpoint: /ws/conversations/{conversation_id}
    
    Handles:
    - Real-time state updates
    - Transcript synchronization  
    - AI response delivery
    - Speech event coordination
    - Connection management
    """
    
    # Validate conversation exists
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        await websocket.close(code=1008, reason="Conversation not found")
        return
    
    # Initialize conversation state if needed
    current_state = await conversation_state_manager.get_conversation_state(conversation_id)
    if current_state == ConversationState.IDLE:
        # Load state from database if available
        session_state = db.query(SessionState).filter(
            SessionState.conversation_id == conversation_id
        ).first()
        
        if session_state and session_state.state:
            try:
                db_state = ConversationState(session_state.state)
                await conversation_state_manager.update_conversation_state(
                    conversation_id, db_state
                )
            except ValueError:
                logger.warning(f"Invalid state in database: {session_state.state}")
    
    # Connect WebSocket
    await connection_manager.connect(websocket, conversation_id)
    
    try:
        while True:
            # Wait for messages from client
            message = await websocket.receive_text()
            await connection_manager.handle_message(websocket, conversation_id, message)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for conversation: {conversation_id}")
    except Exception as e:
        logger.error(f"WebSocket error for conversation {conversation_id}: {e}")
    finally:
        # Clean up connection
        await connection_manager.disconnect(websocket, conversation_id)


@router.get("/conversations/{conversation_id}/state")
async def get_conversation_state(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """
    Get current conversation state via REST API.
    Useful for initial state loading and debugging.
    """
    # Validate conversation exists
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Get current state from manager
    current_state = await conversation_state_manager.get_conversation_state(conversation_id)
    transcript_data = await conversation_state_manager.get_current_transcript(conversation_id)
    
    return {
        "conversation_id": conversation_id,
        "current_state": current_state.value,
        "speech_recognition_active": current_state == ConversationState.LISTENING,
        "speech_synthesis_active": current_state == ConversationState.SPEAKING,
        "transcript": transcript_data,
        "active_connections": conversation_state_manager.get_active_connections_count(conversation_id),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/conversations/{conversation_id}/state")
async def update_conversation_state(
    conversation_id: str,
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Update conversation state via REST API.
    Useful for external integrations and testing.
    """
    # Validate conversation exists
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    new_state_value = request.get("state")
    metadata = request.get("metadata", {})
    
    if not new_state_value:
        raise HTTPException(status_code=400, detail="State is required")
    
    try:
        new_state = ConversationState(new_state_value)
        await conversation_state_manager.update_conversation_state(
            conversation_id=conversation_id,
            new_state=new_state,
            metadata=metadata
        )
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "new_state": new_state.value,
            "timestamp": datetime.now().isoformat()
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid state: {new_state_value}")
    except Exception as e:
        logger.error(f"Error updating conversation state: {e}")
        raise HTTPException(status_code=500, detail="Failed to update conversation state")


@router.post("/conversations/{conversation_id}/transcript")
async def update_transcript(
    conversation_id: str,
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Update transcript via REST API.
    Useful for external integrations and testing.
    """
    # Validate conversation exists
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    interim_transcript = request.get("interim_transcript")
    final_transcript = request.get("final_transcript")
    confidence = request.get("confidence")
    
    await conversation_state_manager.update_transcript(
        conversation_id=conversation_id,
        interim_transcript=interim_transcript,
        final_transcript=final_transcript,
        confidence=confidence
    )
    
    return {
        "success": True,
        "conversation_id": conversation_id,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/conversations/{conversation_id}/ai-response")
async def send_ai_response(
    conversation_id: str,
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Send AI response via REST API.
    Useful for external AI services and testing.
    """
    # Validate conversation exists
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    message_content = request.get("content")
    audio_url = request.get("audio_url")
    feedback = request.get("feedback")
    
    if not message_content:
        raise HTTPException(status_code=400, detail="Message content is required")
    
    await conversation_state_manager.send_ai_response(
        conversation_id=conversation_id,
        message_content=message_content,
        audio_url=audio_url,
        feedback=feedback
    )
    
    return {
        "success": True,
        "conversation_id": conversation_id,
        "timestamp": datetime.now().isoformat()
    }


@router.websocket("/ws")
async def simple_websocket_endpoint(websocket: WebSocket):
    """
    Simple WebSocket endpoint for frontend integration.
    
    This endpoint handles the frontend's expected `/ws` connection
    and manages conversation sessions dynamically through messages.
    """
    try:
        await websocket.accept()
        logger.info("Simple WebSocket connected from frontend")
        
        # Get database session manually
        from app.core.database import SessionLocal
        db = SessionLocal()
        
        # Initialize session state
        current_conversation_id = None
        
        while True:
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                message_type = data.get("type")
                message_data = data.get("data", {})
                
                logger.info(f"Received message type: {message_type}")
                
                if message_type == "ping":
                    # Handle heartbeat
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                
                elif message_type == "start_conversation":
                    # Create a new conversation and start session
                    personality = message_data.get("personality", "friendly")
                    skip_welcome = message_data.get("skipWelcome", False)
                    user_name = message_data.get("userName", "User")
                    
                    # Create new conversation in database
                    conversation_metadata = {
                        "user_name": user_name,
                        "start_time": datetime.now().isoformat(),
                        "skip_welcome": skip_welcome
                    }
                    
                    conversation = Conversation(
                        user_id=f"anonymous_{datetime.now().timestamp()}",  # Anonymous user
                        personality=personality,
                        conversation_metadata=conversation_metadata,
                        status="active"
                    )
                    db.add(conversation)
                    db.commit()
                    db.refresh(conversation)
                    
                    current_conversation_id = str(conversation.id)  # Convert UUID to string
                    logger.info(f"Created new conversation: {current_conversation_id}")
                    
                    # Send conversation started message
                    welcome_message = None if skip_welcome else f"Hello {user_name}! I'm ready to help you practice English. What would you like to talk about?"
                    
                    await websocket.send_text(json.dumps({
                        "type": "conversation_started",
                        "data": {
                            "conversationId": current_conversation_id,
                            "welcomeMessage": welcome_message
                        },
                        "timestamp": datetime.now().isoformat()
                    }))
                
                elif message_type == "conversation_message":
                    # Handle conversation message
                    if not current_conversation_id:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "data": {
                                "message": "No active conversation. Please start a conversation first.",
                                "code": "no_active_conversation"
                            },
                            "timestamp": datetime.now().isoformat()
                        }))
                        continue
                    
                    # Extract message details
                    user_message = message_data.get("message", "")
                    personality = message_data.get("personality", "friendly")
                    
                    if not user_message:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "data": {
                                "message": "Message content is required",
                                "code": "missing_message"
                            },
                            "timestamp": datetime.now().isoformat()
                        }))
                        continue
                    
                    # For now, send a simple response based on personality
                    # In a full implementation, this would call the AI service
                    responses = {
                        "friendly": f"That's really interesting about '{user_message}'! I'd love to hear more about your experience with that.",
                        "sassy": f"Oh, '{user_message}'? How utterly fascinating, darling! Do tell me more - I'm positively riveted!",
                        "blunt": f"'{user_message}' - okay, got it. What's your point exactly?"
                    }
                    
                    ai_response = responses.get(personality, responses["friendly"])
                    
                    # Send AI response
                    await websocket.send_text(json.dumps({
                        "type": "conversation_response",
                        "data": {
                            "response": ai_response,
                            "conversationId": current_conversation_id,
                            "feedback": {
                                "clarity": 85,
                                "fluency": 80,
                                "vocabularyUsage": [],
                                "suggestions": ["Great conversation!"],
                                "overallRating": 4
                            }
                        },
                        "timestamp": datetime.now().isoformat()
                    }))
                
                else:
                    logger.warning(f"Unknown message type: {message_type}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {
                            "message": f"Unknown message type: {message_type}",
                            "code": "unknown_message_type"
                        },
                        "timestamp": datetime.now().isoformat()
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": {
                        "message": "Invalid JSON format",
                        "code": "invalid_json"
                    },
                    "timestamp": datetime.now().isoformat()
                }))
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": {
                        "message": "Internal server error",
                        "code": "internal_error"
                    },
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect as e:
        logger.info(f"Simple WebSocket disconnected: code={e.code}, reason={getattr(e, 'reason', 'No reason')}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        try:
            db.close()
        except:
            pass
        logger.info("Simple WebSocket connection closed")


@router.get("/conversations/{conversation_id}/connections")
async def get_active_connections(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """
    Get information about active WebSocket connections.
    Useful for monitoring and debugging.
    """
    # Validate conversation exists
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    connection_count = conversation_state_manager.get_active_connections_count(conversation_id)
    
    return {
        "conversation_id": conversation_id,
        "active_connections": connection_count,
        "timestamp": datetime.now().isoformat()
    }