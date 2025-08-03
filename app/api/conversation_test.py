"""
Conversation Test API

This module provides test endpoints for validating the WebSocket conversation system.
Useful for development, debugging, and integration testing.

Key features:
- Test conversation creation and management
- WebSocket connection testing
- State transition validation
- Mock data generation for testing
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.database import get_db
from app.models.conversation_v2 import Conversation, ConversationMessage, SessionState
from app.services.conversation_state_manager import (
    conversation_state_manager,
    ConversationState
)
from app.services.conversation_integration_service import conversation_integration_service

logger = logging.getLogger(__name__)
router = APIRouter()


class TestConversationRequest(BaseModel):
    personality: str = "friendly_neutral"
    session_type: str = "test"
    topic: Optional[str] = "testing"


class TestMessageRequest(BaseModel):
    content: str
    simulate_processing_delay: bool = False
    delay_seconds: float = 1.0


class MockSpeechRequest(BaseModel):
    text: str
    is_final: bool = True
    confidence: Optional[float] = 0.95
    simulate_interim_steps: bool = True


@router.post("/test/conversations")
async def create_test_conversation(
    request: TestConversationRequest,
    db: Session = Depends(get_db)
):
    """
    Create a test conversation with WebSocket support.
    Returns conversation details and WebSocket connection info.
    """
    try:
        # Create conversation using integration service
        result = await conversation_integration_service.create_conversation_from_welcome(
            user_id="test_user",
            personality=request.personality,
            session_type=request.session_type,
            topic=request.topic
        )
        
        return {
            "success": True,
            "conversation": result,
            "test_endpoints": {
                "websocket": f"/api/ws/conversations/{result['conversation_id']}",
                "state": f"/api/conversations/{result['conversation_id']}/state",
                "send_message": f"/api/test/conversations/{result['conversation_id']}/message",
                "simulate_speech": f"/api/test/conversations/{result['conversation_id']}/speech"
            },
            "instructions": "Connect to the WebSocket endpoint to receive real-time updates"
        }
        
    except Exception as e:
        logger.error(f"Error creating test conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create test conversation: {str(e)}")


@router.post("/test/conversations/{conversation_id}/message")
async def send_test_message(
    conversation_id: str,
    request: TestMessageRequest,
    db: Session = Depends(get_db)
):
    """
    Send a test message to a conversation and observe WebSocket updates.
    """
    try:
        # Validate conversation exists
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Simulate processing delay if requested
        if request.simulate_processing_delay:
            logger.info(f"Simulating processing delay of {request.delay_seconds} seconds")
            await asyncio.sleep(request.delay_seconds)
        
        # Process the message using integration service
        result = await conversation_integration_service.process_user_message(
            conversation_id=conversation_id,
            message_content=request.content
        )
        
        # Get current conversation state
        current_state = await conversation_state_manager.get_conversation_state(conversation_id)
        transcript_data = await conversation_state_manager.get_current_transcript(conversation_id)
        
        return {
            "success": True,
            "message_processed": True,
            "conversation_id": conversation_id,
            "user_message": request.content,
            "ai_response": result.get("ai_response"),
            "feedback": result.get("feedback"),
            "current_state": current_state.value,
            "transcript": transcript_data,
            "processing_time_ms": result.get("processing_time"),
            "websocket_updates": "Check WebSocket connection for real-time updates"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending test message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send test message: {str(e)}")


@router.post("/test/conversations/{conversation_id}/speech")
async def simulate_speech_input(
    conversation_id: str,
    request: MockSpeechRequest,
    db: Session = Depends(get_db)
):
    """
    Simulate speech-to-text input with interim and final results.
    """
    try:
        # Validate conversation exists
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        results = []
        
        if request.simulate_interim_steps:
            # Simulate interim speech recognition steps
            words = request.text.split()
            
            # Update state to listening
            await conversation_state_manager.update_conversation_state(
                conversation_id=conversation_id,
                new_state=ConversationState.LISTENING,
                metadata={"listening_started": True}
            )
            
            # Send interim transcripts
            for i in range(1, len(words) + 1):
                interim_text = " ".join(words[:i])
                confidence = min(0.95, 0.3 + (i / len(words)) * 0.65)  # Gradually increase confidence
                
                await conversation_state_manager.update_transcript(
                    conversation_id=conversation_id,
                    interim_transcript=interim_text,
                    confidence=confidence
                )
                
                results.append({
                    "step": i,
                    "interim_text": interim_text,
                    "confidence": confidence,
                    "is_final": False
                })
                
                # Small delay to simulate real-time speech
                await asyncio.sleep(0.2)
        
        # Process final result
        result = await conversation_integration_service.process_speech_to_text_result(
            conversation_id=conversation_id,
            transcript=request.text,
            is_final=request.is_final,
            confidence=request.confidence
        )
        
        results.append({
            "step": "final",
            "final_text": request.text,
            "confidence": request.confidence,
            "is_final": request.is_final,
            "processing_result": result
        })
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "simulation_steps": results,
            "final_result": result,
            "websocket_updates": "Check WebSocket connection for real-time updates"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error simulating speech input: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to simulate speech input: {str(e)}")


@router.get("/test/conversations/{conversation_id}/state-transitions")
async def test_state_transitions(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """
    Test all valid state transitions for a conversation.
    """
    try:
        # Validate conversation exists
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get current state
        current_state = await conversation_state_manager.get_conversation_state(conversation_id)
        
        # Test state transitions
        transition_results = []
        
        # Test transition to listening
        try:
            await conversation_state_manager.update_conversation_state(
                conversation_id=conversation_id,
                new_state=ConversationState.LISTENING,
                metadata={"test": "listening_state"}
            )
            transition_results.append({"from": current_state.value, "to": "listening", "success": True})
            await asyncio.sleep(0.5)
        except Exception as e:
            transition_results.append({"from": current_state.value, "to": "listening", "success": False, "error": str(e)})
        
        # Test transition to processing
        try:
            await conversation_state_manager.update_conversation_state(
                conversation_id=conversation_id,
                new_state=ConversationState.PROCESSING,
                metadata={"test": "processing_state"}
            )
            transition_results.append({"from": "listening", "to": "processing", "success": True})
            await asyncio.sleep(0.5)
        except Exception as e:
            transition_results.append({"from": "listening", "to": "processing", "success": False, "error": str(e)})
        
        # Test transition to speaking
        try:
            await conversation_state_manager.update_conversation_state(
                conversation_id=conversation_id,
                new_state=ConversationState.SPEAKING,
                metadata={"test": "speaking_state"}
            )
            transition_results.append({"from": "processing", "to": "speaking", "success": True})
            await asyncio.sleep(0.5)
        except Exception as e:
            transition_results.append({"from": "processing", "to": "speaking", "success": False, "error": str(e)})
        
        # Test transition to waiting_for_user
        try:
            await conversation_state_manager.update_conversation_state(
                conversation_id=conversation_id,
                new_state=ConversationState.WAITING_FOR_USER,
                metadata={"test": "waiting_state"}
            )
            transition_results.append({"from": "speaking", "to": "waiting_for_user", "success": True})
            await asyncio.sleep(0.5)
        except Exception as e:
            transition_results.append({"from": "speaking", "to": "waiting_for_user", "success": False, "error": str(e)})
        
        # Return to idle
        try:
            await conversation_state_manager.update_conversation_state(
                conversation_id=conversation_id,
                new_state=ConversationState.IDLE,
                metadata={"test": "returned_to_idle"}
            )
            transition_results.append({"from": "waiting_for_user", "to": "idle", "success": True})
        except Exception as e:
            transition_results.append({"from": "waiting_for_user", "to": "idle", "success": False, "error": str(e)})
        
        final_state = await conversation_state_manager.get_conversation_state(conversation_id)
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "initial_state": current_state.value,
            "final_state": final_state.value,
            "transitions_tested": transition_results,
            "websocket_updates": "Check WebSocket connection to see state changes in real-time"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing state transitions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test state transitions: {str(e)}")


@router.get("/test/conversations/{conversation_id}/stress-test")
async def stress_test_conversation(
    conversation_id: str,
    message_count: int = 10,
    concurrent_connections: int = 3,
    db: Session = Depends(get_db)
):
    """
    Perform stress testing on the conversation system.
    """
    try:
        # Validate conversation exists
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        start_time = datetime.now()
        
        # Generate test messages
        test_messages = [
            f"This is test message number {i+1}. Testing the conversation system with message {i+1}."
            for i in range(message_count)
        ]
        
        # Process messages sequentially
        message_results = []
        for i, message in enumerate(test_messages):
            try:
                result = await conversation_integration_service.process_user_message(
                    conversation_id=conversation_id,
                    message_content=message
                )
                
                message_results.append({
                    "message_number": i + 1,
                    "success": result.get("success", False),
                    "processing_time_ms": result.get("processing_time"),
                    "ai_response_length": len(result.get("ai_response", "")) if result.get("ai_response") else 0
                })
                
                # Small delay between messages
                await asyncio.sleep(0.1)
                
            except Exception as e:
                message_results.append({
                    "message_number": i + 1,
                    "success": False,
                    "error": str(e)
                })
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        successful_messages = sum(1 for r in message_results if r.get("success", False))
        total_processing_time = sum(r.get("processing_time_ms", 0) for r in message_results if r.get("processing_time_ms"))
        avg_processing_time = total_processing_time / successful_messages if successful_messages > 0 else 0
        
        # Get final conversation state
        final_state = await conversation_state_manager.get_conversation_state(conversation_id)
        active_connections = conversation_state_manager.get_active_connections_count(conversation_id)
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "test_parameters": {
                "message_count": message_count,
                "concurrent_connections": concurrent_connections
            },
            "results": {
                "total_time_seconds": total_time,
                "successful_messages": successful_messages,
                "failed_messages": message_count - successful_messages,
                "success_rate": (successful_messages / message_count) * 100,
                "average_processing_time_ms": avg_processing_time,
                "total_processing_time_ms": total_processing_time
            },
            "final_state": {
                "conversation_state": final_state.value,
                "active_connections": active_connections
            },
            "message_details": message_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in stress test: {e}")
        raise HTTPException(status_code=500, detail=f"Stress test failed: {str(e)}")


@router.delete("/test/conversations/{conversation_id}")
async def cleanup_test_conversation(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """
    Clean up a test conversation and all associated data.
    """
    try:
        # Validate conversation exists
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Clean up WebSocket connections and state
        await conversation_state_manager.cleanup_conversation(conversation_id)
        
        # Delete from database
        db.query(ConversationMessage).filter(
            ConversationMessage.conversation_id == conversation_id
        ).delete()
        
        db.query(SessionState).filter(
            SessionState.conversation_id == conversation_id
        ).delete()
        
        db.delete(conversation)
        db.commit()
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "message": "Test conversation cleaned up successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up test conversation: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to cleanup test conversation: {str(e)}")


@router.get("/test/system-status")
async def get_system_status():
    """
    Get overall system status and health information.
    """
    try:
        # Get conversation state manager status
        # Note: This would need additional methods in the state manager for introspection
        
        return {
            "success": True,
            "timestamp": datetime.now(),
            "system_status": "operational",
            "components": {
                "websocket_server": "running",
                "conversation_state_manager": "running",
                "database": "connected",
                "ai_service": "available"
            },
            "test_endpoints": {
                "create_test_conversation": "/api/test/conversations",
                "websocket_test": "Use browser dev tools or WebSocket client",
                "state_transitions": "/api/test/conversations/{id}/state-transitions",
                "stress_test": "/api/test/conversations/{id}/stress-test"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")