"""
Conversation Integration Service

This service provides utilities for integrating the new WebSocket-based conversation
system with existing services and external APIs. It handles the coordination between
different components and provides migration utilities.

Key responsibilities:
- Coordinating between old and new conversation systems
- Integration with external AI and TTS services
- Migration utilities for existing data
- Service orchestration for complex conversation flows
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.conversation_v2 import Conversation, ConversationMessage, SessionState
from app.models.session import Session as LegacySession  # Legacy session model
from app.services.conversation_state_manager import (
    conversation_state_manager,
    ConversationState
)
from app.services.simple_openai import SimpleOpenAIService
from app.services.vocabulary_tier_service import VocabularyTierService

logger = logging.getLogger(__name__)


class ConversationIntegrationService:
    """
    Service for integrating conversation components and managing transitions
    between different conversation states and external services.
    """
    
    def __init__(self):
        try:
            self.openai_service = SimpleOpenAIService()
        except Exception as e:
            logger.warning(f"OpenAI service initialization failed: {e}")
            self.openai_service = None
        
        try:
            self.vocabulary_service = VocabularyTierService()
        except Exception as e:
            logger.warning(f"Vocabulary service initialization failed: {e}")
            self.vocabulary_service = None
    
    async def migrate_legacy_session_to_conversation(
        self, 
        legacy_session_id: int,
        db: Optional[Session] = None
    ) -> Optional[str]:
        """
        Migrate a legacy session to the new conversation system.
        Returns the new conversation ID if successful.
        """
        if db is None:
            db = SessionLocal()
            should_close = True
        else:
            should_close = False
        
        try:
            # Get legacy session
            legacy_session = db.query(LegacySession).filter(
                LegacySession.id == legacy_session_id
            ).first()
            
            if not legacy_session:
                logger.warning(f"Legacy session {legacy_session_id} not found")
                return None
            
            # Create new conversation
            conversation = Conversation(
                user_id=getattr(legacy_session, 'user_id', None),
                session_id=legacy_session_id,  # Link to legacy session
                status='active',
                personality=getattr(legacy_session, 'personality', 'friendly_neutral'),
                conversation_metadata={
                    'migrated_from_session': legacy_session_id,
                    'migration_date': datetime.now().isoformat(),
                    'legacy_data': {
                        'created_at': str(legacy_session.created_at) if hasattr(legacy_session, 'created_at') else None,
                        'session_type': getattr(legacy_session, 'session_type', 'unknown')
                    }
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
            
            # Initialize in state manager
            await conversation_state_manager.update_conversation_state(
                conversation_id=str(conversation.id),
                new_state=ConversationState.IDLE
            )
            
            logger.info(f"Migrated legacy session {legacy_session_id} to conversation {conversation.id}")
            return str(conversation.id)
            
        except Exception as e:
            logger.error(f"Error migrating legacy session {legacy_session_id}: {e}")
            db.rollback()
            return None
        finally:
            if should_close:
                db.close()
    
    async def create_conversation_from_welcome(
        self,
        user_id: Optional[str] = None,
        personality: str = "friendly_neutral",
        session_type: str = "daily",
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new conversation and generate a welcome message.
        Returns conversation details and welcome message.
        """
        db = SessionLocal()
        try:
            # Create conversation
            conversation = Conversation(
                user_id=user_id,
                personality=personality,
                status="active",
                conversation_metadata={
                    "session_type": session_type,
                    "topic": topic,
                    "created_via": "welcome_flow"
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
            
            # Initialize in state manager
            await conversation_state_manager.update_conversation_state(
                conversation_id=str(conversation.id),
                new_state=ConversationState.IDLE
            )
            
            # Generate welcome message
            try:
                if self.openai_service:
                    welcome_message = await self.openai_service.generate_welcome_message(personality)
                else:
                    raise Exception("OpenAI service not available")
            except Exception as e:
                logger.error(f"Error generating welcome message: {e}")
                # Fallback welcome messages
                fallback_messages = {
                    "sassy_english": "Well hello there! Welcome to ImprovToday, darling! What name shall I call you, and do tell me - how's your day been?",
                    "blunt_american": "Hey there, welcome to ImprovToday! What should I call you, and how was your day?",
                    "friendly_neutral": "Welcome to ImprovToday! I'm so glad you're here. What name should I address you with? How has your day been?"
                }
                welcome_message = fallback_messages.get(personality, fallback_messages["friendly_neutral"])
            
            # Store welcome message as first AI message
            welcome_msg = ConversationMessage(
                conversation_id=conversation.id,
                role="assistant",
                content=welcome_message
            )
            
            db.add(welcome_msg)
            db.commit()
            db.refresh(welcome_msg)
            
            # Send welcome message through WebSocket
            await conversation_state_manager.send_ai_response(
                conversation_id=str(conversation.id),
                message_content=welcome_message,
                audio_url=None,
                feedback=None
            )
            
            # Update state to waiting for user
            await conversation_state_manager.update_conversation_state(
                conversation_id=str(conversation.id),
                new_state=ConversationState.WAITING_FOR_USER
            )
            
            return {
                "conversation_id": str(conversation.id),
                "welcome_message": welcome_message,
                "personality": personality,
                "status": "ready",
                "websocket_url": f"/ws/conversations/{conversation.id}"
            }
            
        except Exception as e:
            logger.error(f"Error creating conversation from welcome: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    async def process_speech_to_text_result(
        self,
        conversation_id: str,
        transcript: str,
        is_final: bool = True,
        confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process speech-to-text results and update conversation state.
        Handles both interim and final transcripts.
        """
        try:
            if is_final:
                # Update conversation state to processing
                await conversation_state_manager.update_conversation_state(
                    conversation_id=conversation_id,
                    new_state=ConversationState.PROCESSING
                )
                
                # Update final transcript
                await conversation_state_manager.update_transcript(
                    conversation_id=conversation_id,
                    final_transcript=transcript,
                    confidence=confidence
                )
                
                # Process the message and generate AI response
                result = await self.process_user_message(conversation_id, transcript)
                
                return {
                    "success": True,
                    "conversation_id": conversation_id,
                    "transcript": transcript,
                    "is_final": is_final,
                    "confidence": confidence,
                    "ai_response": result.get("ai_response"),
                    "processing_time": result.get("processing_time")
                }
            else:
                # Update interim transcript
                await conversation_state_manager.update_transcript(
                    conversation_id=conversation_id,
                    interim_transcript=transcript,
                    confidence=confidence
                )
                
                return {
                    "success": True,
                    "conversation_id": conversation_id,
                    "interim_transcript": transcript,
                    "is_final": is_final,
                    "confidence": confidence
                }
                
        except Exception as e:
            logger.error(f"Error processing speech-to-text result: {e}")
            # Update state to error
            await conversation_state_manager.update_conversation_state(
                conversation_id=conversation_id,
                new_state=ConversationState.ERROR,
                metadata={"error": str(e)}
            )
            
            return {
                "success": False,
                "error": str(e),
                "conversation_id": conversation_id
            }
    
    async def process_user_message(
        self,
        conversation_id: str,
        message_content: str
    ) -> Dict[str, Any]:
        """
        Process a user message and generate AI response with feedback.
        """
        start_time = datetime.now()
        
        try:
            db = SessionLocal()
            try:
                # Validate conversation exists
                conversation = db.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()
                
                if not conversation:
                    raise ValueError(f"Conversation {conversation_id} not found")
                
                # Store user message
                user_message = ConversationMessage(
                    conversation_id=conversation_id,
                    role="user",
                    content=message_content
                )
                
                db.add(user_message)
                db.commit()
                db.refresh(user_message)
                
                # Analyze vocabulary
                if self.vocabulary_service:
                    tier_analysis = self.vocabulary_service.analyze_vocabulary_tier(message_content)
                else:
                    # Mock tier analysis
                    from types import SimpleNamespace
                    tier_analysis = SimpleNamespace(
                        tier="intermediate",
                        score=75,
                        word_count=len(message_content.split()),
                        complex_word_count=2,
                        average_word_length=5.0,
                        analysis_details="Mock analysis - vocabulary service unavailable"
                    )
                
                # Generate AI response
                if self.openai_service:
                    ai_response = await self.openai_service.generate_personality_response(
                        message_content,
                        conversation.personality,
                        [],  # target_vocabulary
                        conversation.conversation_metadata.get("topic") if conversation.conversation_metadata else None
                    )
                else:
                    # Fallback AI response
                    ai_response = f"That's interesting! Can you tell me more about that? I'm still learning to give better responses."
                
                # Create feedback
                feedback = {
                    "clarity": 85,
                    "fluency": self._estimate_fluency(message_content),
                    "vocabulary_tier": tier_analysis.tier,
                    "vocabulary_score": tier_analysis.score,
                    "suggestions": self._generate_tier_suggestions(tier_analysis),
                    "overall_rating": min(5, max(1, int((tier_analysis.score + self._estimate_fluency(message_content)) / 25)))
                }
                
                # Calculate processing time
                processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
                
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
                    new_state=ConversationState.WAITING_FOR_USER
                )
                
                return {
                    "success": True,
                    "ai_response": ai_response,
                    "feedback": feedback,
                    "vocabulary_analysis": {
                        "tier": tier_analysis.tier,
                        "score": tier_analysis.score,
                        "word_count": tier_analysis.word_count
                    },
                    "processing_time": processing_time
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            
            # Update state to error
            await conversation_state_manager.update_conversation_state(
                conversation_id=conversation_id,
                new_state=ConversationState.ERROR,
                metadata={"error": str(e)}
            )
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": int((datetime.now() - start_time).total_seconds() * 1000)
            }
    
    async def handle_speech_synthesis_request(
        self,
        conversation_id: str,
        text: str,
        voice: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle text-to-speech synthesis requests.
        Updates conversation state during synthesis.
        """
        try:
            # Update state to speaking
            await conversation_state_manager.update_conversation_state(
                conversation_id=conversation_id,
                new_state=ConversationState.SPEAKING,
                metadata={"synthesizing_text": text[:100] + "..."}
            )
            
            # Send speech event
            await conversation_state_manager.send_speech_event(
                conversation_id=conversation_id,
                event_type="synthesis_started",
                event_data={"text": text, "voice": voice}
            )
            
            # TODO: Integrate with actual TTS service
            # For now, simulate TTS processing
            audio_url = None  # Would be returned by TTS service
            
            # Send speech event for completion
            await conversation_state_manager.send_speech_event(
                conversation_id=conversation_id,
                event_type="synthesis_completed",
                event_data={"audio_url": audio_url}
            )
            
            # Update state back to waiting for user
            await conversation_state_manager.update_conversation_state(
                conversation_id=conversation_id,
                new_state=ConversationState.WAITING_FOR_USER
            )
            
            return {
                "success": True,
                "audio_url": audio_url,
                "text": text,
                "voice": voice
            }
            
        except Exception as e:
            logger.error(f"Error handling speech synthesis: {e}")
            
            # Update state to error
            await conversation_state_manager.update_conversation_state(
                conversation_id=conversation_id,
                new_state=ConversationState.ERROR,
                metadata={"error": str(e)}
            )
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def _estimate_fluency(self, transcript: str) -> int:
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
    
    def _generate_tier_suggestions(self, tier_analysis) -> List[str]:
        """Generate suggestions based on vocabulary tier analysis"""
        suggestions = []
        
        tier = getattr(tier_analysis, 'tier', 'intermediate')
        word_count = getattr(tier_analysis, 'word_count', 0)
        
        if tier == "basic":
            suggestions.extend([
                "Try using more descriptive words instead of 'good', 'bad', or 'nice'",
                "Consider expanding your sentences with more details",
                "Practice using words with more than 4-5 letters"
            ])
        elif tier in ["mid", "intermediate"]:
            suggestions.extend([
                "Great vocabulary variety! Try incorporating more advanced words",
                "Your word choice shows good progression",
                "Consider using more sophisticated synonyms"
            ])
        else:  # top/advanced
            suggestions.extend([
                "Excellent vocabulary sophistication!",
                "Your word choice demonstrates advanced language skills",
                "Keep up the great use of complex vocabulary"
            ])
        
        # Add word count specific suggestions
        if word_count < 10:
            suggestions.append("Try expressing your thoughts in more detail")
        elif word_count > 50:
            suggestions.append("Great detailed response! Practice being concise too")
        
        return suggestions[:3]  # Return top 3 suggestions


# Global singleton instance
conversation_integration_service = ConversationIntegrationService()