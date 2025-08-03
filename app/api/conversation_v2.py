"""
Enhanced Conversation API (v2) with WebSocket Integration

This module provides the enhanced conversation API that integrates with the
WebSocket system for real-time synchronization as outlined in the redesign plan.

Key features:
- Integration with ConversationStateManager
- Real-time conversation flow management
- WebSocket-aware AI response handling
- State-synchronized conversation processing
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
import uuid

from app.core.database import get_db
from app.models.conversation_v2 import Conversation, ConversationMessage, SessionState
from app.services.conversation_state_manager import (
    conversation_state_manager, 
    ConversationState
)
from app.services.simple_openai import SimpleOpenAIService
from app.services.vocabulary_tier_service import VocabularyTierService

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models for API requests/responses
class CreateConversationRequest(BaseModel):
    user_id: Optional[str] = None
    personality: str = "friendly_neutral"
    session_type: str = "daily"
    topic: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    user_id: Optional[str]
    status: str
    personality: str
    created_at: datetime
    conversation_metadata: Optional[Dict] = None


class ProcessMessageRequest(BaseModel):
    content: str
    role: str = "user"  # 'user' or 'assistant'
    audio_url: Optional[str] = None


class ProcessMessageResponse(BaseModel):
    success: bool
    message_id: str
    ai_response: Optional[str] = None
    feedback: Optional[Dict] = None
    vocabulary_analysis: Optional[Dict] = None
    conversation_state: str


class ConversationStatusResponse(BaseModel):
    conversation_id: str
    status: str
    current_state: str
    message_count: int
    last_activity: datetime
    active_connections: int


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: CreateConversationRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new conversation with WebSocket support.
    Initializes conversation state for real-time synchronization.
    """
    try:
        # Create conversation in database
        conversation = Conversation(
            user_id=request.user_id,
            personality=request.personality,
            status="active",
            conversation_metadata={
                "session_type": request.session_type,
                "topic": request.topic,
                "created_via": "api_v2"
            }
        )
        
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        
        # Initialize session state
        session_state = SessionState(
            conversation_id=conversation.id,
            state=ConversationState.IDLE.value
        )
        
        db.add(session_state)
        db.commit()
        
        # Initialize conversation state in manager
        await conversation_state_manager.update_conversation_state(
            conversation_id=str(conversation.id),
            new_state=ConversationState.IDLE
        )
        
        logger.info(f"Created conversation {conversation.id} with WebSocket support")
        
        return ConversationResponse(
            id=str(conversation.id),
            user_id=conversation.user_id,
            status=conversation.status,
            personality=conversation.personality,
            created_at=conversation.created_at,
            conversation_metadata=conversation.conversation_metadata
        )
        
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")


@router.post("/conversations/{conversation_id}/messages", response_model=ProcessMessageResponse)
async def process_message(
    conversation_id: str,
    request: ProcessMessageRequest,
    db: Session = Depends(get_db)
):
    """
    Process a message in the conversation with real-time WebSocket updates.
    Handles both user messages and generates AI responses with state management.
    """
    try:
        # Validate conversation exists
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Store user message
        user_message = ConversationMessage(
            conversation_id=conversation_id,
            role=request.role,
            content=request.content,
            audio_url=request.audio_url
        )
        
        db.add(user_message)
        db.commit()
        db.refresh(user_message)
        
        # Update conversation state to processing
        await conversation_state_manager.update_conversation_state(
            conversation_id=conversation_id,
            new_state=ConversationState.PROCESSING,
            metadata={"processing_message": str(user_message.id)}
        )
        
        ai_response = None
        feedback = None
        vocabulary_analysis = None
        
        # Generate AI response if this is a user message
        if request.role == "user":
            try:
                # Initialize services
                openai_service = SimpleOpenAIService()
                vocabulary_service = VocabularyTierService()
                
                # Analyze vocabulary
                tier_analysis = vocabulary_service.analyze_vocabulary_tier(request.content)
                vocabulary_analysis = {
                    "tier": tier_analysis.tier,
                    "score": tier_analysis.score,
                    "word_count": tier_analysis.word_count,
                    "complex_words": tier_analysis.complex_word_count,
                    "average_word_length": tier_analysis.average_word_length,
                    "analysis": tier_analysis.analysis_details
                }
                
                # Generate AI response
                ai_response = await openai_service.generate_personality_response(
                    request.content,
                    conversation.personality,
                    [],  # target_vocabulary - could be enhanced later
                    conversation.conversation_metadata.get("topic") if conversation.conversation_metadata else None
                )
                
                # Create feedback
                feedback = {
                    "clarity": 85,
                    "fluency": _estimate_fluency(request.content),
                    "vocabulary_tier": tier_analysis.tier,
                    "vocabulary_score": tier_analysis.score,
                    "suggestions": _generate_tier_suggestions(tier_analysis),
                    "overall_rating": min(5, max(1, int((tier_analysis.score + _estimate_fluency(request.content)) / 25)))
                }
                
                # Send AI response through WebSocket
                await conversation_state_manager.send_ai_response(
                    conversation_id=conversation_id,
                    message_content=ai_response,
                    audio_url=None,  # TTS could be added here
                    feedback=feedback
                )
                
                # Update state to waiting for user
                await conversation_state_manager.update_conversation_state(
                    conversation_id=conversation_id,
                    new_state=ConversationState.WAITING_FOR_USER,
                    metadata={"last_ai_response": ai_response[:100] + "..."}
                )
                
            except Exception as ai_error:
                logger.error(f"Error generating AI response: {ai_error}")
                
                # Update state to error
                await conversation_state_manager.update_conversation_state(
                    conversation_id=conversation_id,
                    new_state=ConversationState.ERROR,
                    metadata={"error": str(ai_error)}
                )
                
                # Still return the user message as successfully processed
                ai_response = "I apologize, but I'm having trouble generating a response right now. Please try again."
        
        # Get current conversation state
        current_state = await conversation_state_manager.get_conversation_state(conversation_id)
        
        logger.info(f"Processed message in conversation {conversation_id}: {request.role} message")
        
        return ProcessMessageResponse(
            success=True,
            message_id=str(user_message.id),
            ai_response=ai_response,
            feedback=feedback,
            vocabulary_analysis=vocabulary_analysis,
            conversation_state=current_state.value
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


@router.get("/conversations/{conversation_id}", response_model=ConversationStatusResponse)
async def get_conversation_status(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive conversation status including WebSocket connection info.
    """
    try:
        # Validate conversation exists
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get message count
        message_count = db.query(ConversationMessage).filter(
            ConversationMessage.conversation_id == conversation_id
        ).count()
        
        # Get current state from manager
        current_state = await conversation_state_manager.get_conversation_state(conversation_id)
        
        # Get active connections
        active_connections = conversation_state_manager.get_active_connections_count(conversation_id)
        
        return ConversationStatusResponse(
            conversation_id=conversation_id,
            status=conversation.status,
            current_state=current_state.value,
            message_count=message_count,
            last_activity=conversation.updated_at,
            active_connections=active_connections
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation status: {str(e)}")


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get conversation messages with pagination.
    """
    try:
        # Validate conversation exists
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get messages with pagination
        messages = db.query(ConversationMessage).filter(
            ConversationMessage.conversation_id == conversation_id
        ).order_by(ConversationMessage.timestamp.desc()).offset(offset).limit(limit).all()
        
        # Convert to response format
        message_list = []
        for msg in messages:
            message_list.append({
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "audio_url": msg.audio_url,
                "feedback": msg.feedback,
                "processing_time": msg.processing_time
            })
        
        return {
            "conversation_id": conversation_id,
            "messages": message_list,
            "total_messages": db.query(ConversationMessage).filter(
                ConversationMessage.conversation_id == conversation_id
            ).count(),
            "limit": limit,
            "offset": offset
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation messages: {str(e)}")


@router.put("/conversations/{conversation_id}/end")
async def end_conversation(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """
    End a conversation and clean up WebSocket connections.
    """
    try:
        # Validate conversation exists
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Update conversation status
        conversation.status = "ended"
        conversation.updated_at = datetime.now()
        db.commit()
        
        # Update conversation state to ended
        await conversation_state_manager.update_conversation_state(
            conversation_id=conversation_id,
            new_state=ConversationState.ENDED
        )
        
        # Clean up conversation resources
        await conversation_state_manager.cleanup_conversation(conversation_id)
        
        logger.info(f"Ended conversation {conversation_id}")
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "status": "ended",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending conversation: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to end conversation: {str(e)}")


@router.get("/conversations")
async def list_conversations(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    List conversations with optional filtering.
    """
    try:
        query = db.query(Conversation)
        
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        
        if status:
            query = query.filter(Conversation.status == status)
        
        conversations = query.order_by(
            Conversation.updated_at.desc()
        ).offset(offset).limit(limit).all()
        
        # Convert to response format
        conversation_list = []
        for conv in conversations:
            # Get current state if conversation is active
            current_state = "unknown"
            active_connections = 0
            
            if conv.status == "active":
                try:
                    state = await conversation_state_manager.get_conversation_state(str(conv.id))
                    current_state = state.value
                    active_connections = conversation_state_manager.get_active_connections_count(str(conv.id))
                except Exception:
                    pass  # Use defaults if state manager fails
            
            conversation_list.append({
                "id": str(conv.id),
                "user_id": conv.user_id,
                "status": conv.status,
                "personality": conv.personality,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "current_state": current_state,
                "active_connections": active_connections,
                "conversation_metadata": conv.conversation_metadata
            })
        
        total_count = db.query(Conversation).count()
        if user_id:
            total_count = db.query(Conversation).filter(Conversation.user_id == user_id).count()
        if status:
            total_count = db.query(Conversation).filter(Conversation.status == status).count()
        
        return {
            "conversations": conversation_list,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")


# Helper functions (similar to existing conversation.py)
def _estimate_fluency(transcript: str) -> int:
    """Estimate fluency based on transcript characteristics"""
    words = transcript.split()
    word_count = len(words)
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Simple heuristic scoring
    base_score = 70
    if word_count > 5: base_score += 10
    if word_count > 10: base_score += 5
    if avg_word_length > 4: base_score += 10
    
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