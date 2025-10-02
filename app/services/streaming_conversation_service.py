"""
Streaming Conversation Service for Story 3.3: Speech Optimization & Clara's Response Performance.

Implements Server-Sent Events (SSE) streaming for progressive response delivery
to reduce perceived latency and improve user experience.
"""
import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, Any, Optional, List
from datetime import datetime, timezone

from fastapi import Depends
from sqlalchemy.orm import Session

from app.services.enhanced_conversation_service import EnhancedConversationService, ConversationPerformanceMonitor
from app.services.simple_openai import SimpleOpenAIService, WordUsageStatus
from app.services.vocabulary_tier_service import VocabularyTierService
from app.services.suggestion_service import SuggestionService
from app.services.redis_service import RedisService
from app.models.vocabulary import VocabularySuggestion
from app.models.conversation_v2 import Conversation, ConversationMessage
from app.core.database import get_db

logger = logging.getLogger(__name__)


class StreamingConversationService:
    """
    Implements progressive response streaming for faster perceived performance.

    Features:
    - Server-Sent Events (SSE) streaming
    - Progressive response chunking with intelligent boundaries
    - Real-time performance monitoring
    - Compatible with existing conversation API contract
    """

    def __init__(self):
        self.enhanced_conversation_service = EnhancedConversationService()
        self.openai_service = SimpleOpenAIService()
        self.vocabulary_service = VocabularyTierService()
        self.suggestion_service = SuggestionService()
        self.redis_service = RedisService()
        self.performance_monitor = ConversationPerformanceMonitor()

    async def stream_conversation_response(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        session_id: Optional[int] = None,
        personality: Optional[str] = None,
        target_vocabulary: Optional[List[Dict]] = None,
        user_preferences: Optional[Dict] = None,
        db: Session = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream conversation response progressively using Server-Sent Events.

        Yields SSE-formatted events:
        - 'processing_start': Initial acknowledgment with correlation ID
        - 'context_ready': Context gathering completed
        - 'consciousness_chunk': Progressive AI response chunks
        - 'analysis_ready': Vocabulary analysis completed
        - 'suggestion_ready': New vocabulary suggestion available
        - 'processing_complete': Final response with all metadata

        Args:
            user_message: User's input message
            user_id: User identifier
            conversation_id: Conversation session ID
            session_id: Optional session ID for personality/topic context
            personality: Conversation personality
            target_vocabulary: Target vocabulary list
            user_preferences: User personalization preferences
            db: Database session

        Yields:
            str: SSE-formatted event data
        """
        correlation_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Start performance monitoring
            timing_context = self.performance_monitor.start_timing_context(correlation_id, "streaming_conversation")

            # Yield initial processing acknowledgment
            yield self._format_sse_event("processing_start", {
                "correlation_id": correlation_id,
                "status": "starting",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Phase 1: Context gathering and preparation (stream immediately when ready)
            context_start = time.time()

            # Note: Conversation record creation is handled by the enhanced service
            # We don't create conversation records here - that's done in the enhanced service

            # Get conversation history and context
            conversation_history_data = self.redis_service.get_conversation_history(conversation_id, db)

            # Check for active suggestions
            recent_suggestion = self._get_most_recent_active_suggestion(db, user_id)
            suggested_word = recent_suggestion.suggested_word if recent_suggestion else None

            context_duration = (time.time() - context_start) * 1000

            # Yield context ready event
            yield self._format_sse_event("context_ready", {
                "correlation_id": correlation_id,
                "context_items": len(conversation_history_data),
                "suggested_word": suggested_word,
                "processing_time_ms": round(context_duration, 2),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Phase 2: Generate AI response with progressive streaming
            consciousness_start = time.time()

            # Resolve personality (enhanced service will handle conversation creation)
            effective_personality = self._resolve_personality(personality, session_id, db)

            # Stream consciousness generation with chunking
            ai_response_chunks = []
            word_usage_status = WordUsageStatus.NOT_USED
            usage_feedback = None
            corrected_transcript = user_message
            simulation_context = {}
            selected_backstory_types = []

            # REAL STREAMING: Direct OpenAI stream with context
            ai_response_chunks = []
            word_usage_status = WordUsageStatus.NOT_USED
            usage_feedback = None
            corrected_transcript = user_message
            simulation_context = {}
            selected_backstory_types = []

            # Get the SAME context as enhanced service
            recent_events, global_state, content_metadata, selected_backstory, conversation_emotion, mood_transition_data = await self._get_enhanced_context(
                user_message, user_id, conversation_id, effective_personality
            )

            # Build conversation history string (same format as enhanced service)
            conversation_history = self.redis_service.build_conversation_context(conversation_history_data)

            # Build IDENTICAL prompt as enhanced service
            from app.services.conversation_prompt_service import ConversationPromptService
            conversation_prompt_service = ConversationPromptService()

            try:
                enhanced_prompt = conversation_prompt_service.construct_conversation_prompt_with_mood(
                    character_backstory=selected_backstory.get("content", ""),
                    user_message=user_message,
                    conversation_emotion=conversation_emotion,
                    mood_transition_data=mood_transition_data,
                    conversation_history=conversation_history
                )
                logger.info(f"✅ Prompt construction successful, length: {len(enhanced_prompt) if enhanced_prompt else 0}")
            except Exception as prompt_error:
                logger.error(f"❌ Prompt construction failed: {str(prompt_error)}")
                # Fallback to simple prompt
                enhanced_prompt = f"You are Clara, a warm conversation partner. Respond to: {user_message}"

            # Add simulation context (same as enhanced service)
            enhanced_prompt += self.enhanced_conversation_service._build_simulation_context_prompt(
                recent_events, global_state, content_metadata
            )

            # Start REAL OpenAI streaming
            if self.enhanced_conversation_service.openai_client:
                stream = self.enhanced_conversation_service.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": enhanced_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=400,
                    temperature=0.7,
                    stream=True  # TRUE STREAMING
                )

                # Stream chunks in REAL-TIME as OpenAI generates them
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        chunk_text = chunk.choices[0].delta.content
                        ai_response_chunks.append(chunk_text)

                        # Send chunk IMMEDIATELY to frontend
                        yield self._format_sse_event("consciousness_chunk", {
                            "correlation_id": correlation_id,
                            "chunk": chunk_text,
                            "total_length": sum(len(c) for c in ai_response_chunks),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })

                # Streaming complete - store conversation messages via SessionStateService
                full_ai_response = "".join(ai_response_chunks)

                # Store conversation messages using the same method as enhanced service
                await self._store_conversation_messages_via_session_service(
                    conversation_id=conversation_id,
                    user_message=user_message,
                    ai_response=full_ai_response,
                    user_id=user_id
                )

            else:
                raise Exception("OpenAI client not available for streaming")

            consciousness_duration = (time.time() - consciousness_start) * 1000
            full_ai_response = "".join(ai_response_chunks)

            total_duration = (time.time() - start_time) * 1000

            # End timing context and log performance metrics
            if timing_context:
                final_metrics = self.performance_monitor.end_timing_context(timing_context)
                self.performance_monitor.log_detailed_timing_breakdown(final_metrics)

            # Yield final completion event
            yield self._format_sse_event("processing_complete", {
                "correlation_id": correlation_id,
                "response": full_ai_response,
                "feedback": {},
                "simulation_context": simulation_context,
                "selected_backstory_types": selected_backstory_types,
                "performance_metrics": {
                    "total_time_ms": round(total_duration, 2),
                    "consciousness_time_ms": round(consciousness_duration, 2),
                    "context_time_ms": round(context_duration, 2),
                    "chunks_delivered": len(ai_response_chunks)
                },
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            logger.error(f"Streaming conversation failed: {str(e)}")
            # Ensure correlation_id is available
            try:
                correlation_id_value = correlation_id
            except NameError:
                correlation_id_value = str(uuid.uuid4())

            # Yield error event
            yield self._format_sse_event("error", {
                "correlation_id": correlation_id_value,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })



    def _format_sse_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """
        Format data as Server-Sent Events (SSE) string.

        Args:
            event_type: SSE event type
            data: Event data dictionary

        Returns:
            SSE-formatted string
        """
        json_data = json.dumps(data)
        return f"event: {event_type}\ndata: {json_data}\n\n"

    def _get_most_recent_active_suggestion(self, db: Session, user_id: str) -> Optional[VocabularySuggestion]:
        """Get the most recent active suggestion for the user."""
        return db.query(VocabularySuggestion).filter(
            VocabularySuggestion.user_id == user_id,
            VocabularySuggestion.status.in_(["shown", "used_incorrectly"])
        ).order_by(VocabularySuggestion.created_at.desc()).first()

    def _resolve_personality(self, personality: Optional[str], session_id: Optional[int], db: Session) -> str:
        """Resolve effective personality from request or session."""
        if personality:
            return personality

        if session_id:
            from app.models.session import Session as SessionModel
            session_row = db.query(SessionModel).filter(SessionModel.id == session_id).first()
            if session_row and session_row.personality:
                return session_row.personality

        return "friendly_neutral"

    def _should_generate_suggestion(
        self,
        recent_suggestion: Optional[VocabularySuggestion],
        word_usage_status: WordUsageStatus,
        conversation_history: List[Dict]
    ) -> bool:
        """Determine if a new vocabulary suggestion should be generated."""
        if not recent_suggestion:
            return True

        if word_usage_status == WordUsageStatus.USED_CORRECTLY:
            return True

        # Check for graceful replacement (4+ turns without usage)
        turns_since_suggestion = len([
            msg for msg in conversation_history
            if msg.get('timestamp') and msg.get('timestamp') > recent_suggestion.created_at.isoformat()
        ])

        return turns_since_suggestion >= 4

    async def _analyze_vocabulary_tier(self, text: str):
        """Analyze vocabulary tier of given text."""
        return self.vocabulary_service.analyze_vocabulary_tier(text)

    async def _generate_vocabulary_suggestion(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        db: Session
    ) -> Optional[Dict]:
        """Generate and save new vocabulary suggestion."""
        try:
            suggestion = self.suggestion_service.generate_suggestion(user_message)
            if not suggestion:
                return None

            # Save suggestion to database
            db_suggestion = VocabularySuggestion(
                conversation_id=conversation_id,
                user_id=user_id,
                suggested_word=suggestion["word"],
                status="shown"
            )
            db.add(db_suggestion)
            db.commit()
            db.refresh(db_suggestion)

            return {
                "id": str(db_suggestion.id),
                "word": suggestion["word"],
                "definition": suggestion["definition"],
                "exampleSentence": suggestion["exampleSentence"]
            }
        except Exception as e:
            logger.error(f"Suggestion generation failed: {str(e)}")
            return None

    async def _create_empty_suggestion(self) -> None:
        """Create empty suggestion for consistency in parallel execution."""
        return None

    def _update_suggestion_status(
        self,
        suggestion: VocabularySuggestion,
        word_usage_status: WordUsageStatus,
        conversation_history: List[Dict],
        db: Session
    ) -> Optional[str]:
        """Update suggestion status based on usage analysis."""
        try:
            if word_usage_status == WordUsageStatus.USED_CORRECTLY:
                suggestion.status = "used"
                db.add(suggestion)
                db.commit()
                return str(suggestion.id)
            elif word_usage_status == WordUsageStatus.USED_INCORRECTLY:
                suggestion.status = "used_incorrectly"
                db.add(suggestion)
                db.commit()
            else:
                # Check for graceful replacement
                turns_since_suggestion = len([
                    msg for msg in conversation_history
                    if msg.get('timestamp') and msg.get('timestamp') > suggestion.created_at.isoformat()
                ])

                if turns_since_suggestion >= 4:
                    suggestion.status = "ignored"
                    db.add(suggestion)
                    db.commit()

            return None
        except Exception as e:
            logger.error(f"Suggestion status update failed: {str(e)}")
            return None

    async def _save_conversation_messages(
        self,
        conversation_id: str,
        user_message: str,
        ai_response: str,
        db: Session
    ):
        """Save conversation messages to database and Redis cache."""
        try:
            # Save user message
            user_message_obj = ConversationMessage(
                conversation_id=conversation_id,
                role="user",
                content=user_message,
                timestamp=datetime.now(timezone.utc)
            )
            db.add(user_message_obj)
            db.flush()

            # Save AI response
            ai_message_obj = ConversationMessage(
                conversation_id=conversation_id,
                role="assistant",
                content=ai_response,
                timestamp=datetime.now(timezone.utc)
            )
            db.add(ai_message_obj)
            db.flush()

            # Cache in Redis
            self.redis_service.cache_message(conversation_id, "user", user_message, user_message_obj.timestamp)
            self.redis_service.cache_message(conversation_id, "assistant", ai_response, ai_message_obj.timestamp)

            db.commit()

        except Exception as e:
            logger.error(f"Message saving failed: {str(e)}")
            raise

    def _create_feedback(self, tier_analysis, corrected_transcript: str) -> Dict:
        """Create feedback dictionary from tier analysis."""
        return {
            "clarity": 85,
            "fluency": self._estimate_fluency(corrected_transcript),
            "vocabularyTier": tier_analysis.tier,
            "vocabularyScore": tier_analysis.score,
            "vocabularyUsage": [],
            "suggestions": self._generate_tier_suggestions(tier_analysis),
            "overallRating": min(5, max(1, int((tier_analysis.score + self._estimate_fluency(corrected_transcript)) / 25)))
        }

    def _create_vocabulary_tier_data(self, tier_analysis) -> Dict:
        """Create vocabulary tier data dictionary."""
        return {
            "tier": tier_analysis.tier,
            "score": tier_analysis.score,
            "wordCount": tier_analysis.word_count,
            "complexWords": tier_analysis.complex_word_count,
            "averageWordLength": tier_analysis.average_word_length,
            "analysis": tier_analysis.analysis_details,
            "recommendations": self.vocabulary_service.get_vocabulary_recommendations(tier_analysis.tier, [])
        }

    def _create_usage_analysis(
        self,
        word_usage_status: WordUsageStatus,
        suggested_word: Optional[str],
        usage_feedback: Optional[str]
    ) -> Optional[Dict]:
        """Create usage analysis dictionary."""
        if word_usage_status == WordUsageStatus.NOT_USED:
            return None

        return {
            "word_usage_status": word_usage_status.value,
            "suggested_word": suggested_word,
            "usage_feedback": usage_feedback,
            "conversation_context_used": True
        }

    def _estimate_fluency(self, transcript: str) -> int:
        """Estimate fluency based on transcript characteristics."""
        words = transcript.split()
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        base_score = 70
        if word_count > 5:
            base_score += 10
        if word_count > 10:
            base_score += 5
        if avg_word_length > 4:
            base_score += 10

        return min(100, base_score)

    def _generate_tier_suggestions(self, tier_analysis) -> List[str]:
        """Generate suggestions based on vocabulary tier analysis."""
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

        if tier_analysis.word_count < 10:
            suggestions.append("Try expressing your thoughts in more detail")
        elif tier_analysis.word_count > 50:
            suggestions.append("Great detailed response! Practice being concise too")

        return suggestions[:3]

    async def _get_enhanced_context(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        personality: str
    ):
        """
        Get the EXACT same context as the enhanced conversation service.
        No optimization, no shortcuts - identical logic.
        """
        try:
            # Use enhanced service's exact context gathering logic
            from app.services.event_selection_service import EventSelectionService
            from app.services.contextual_backstory_service import ContextualBackstoryService
            from app.services.simulation.state_manager import StateManagerService

            # Get recent events (same as enhanced service)
            event_service = EventSelectionService()
            recent_events = await event_service.get_contextual_events(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=user_message,
                max_events=5
            )

            # Get global state (same as enhanced service)
            state_manager = StateManagerService()
            global_state = await state_manager.get_current_global_state()

            # Select backstory content using the correct service
            from app.core.conversation_config import conversation_config
            backstory_service = ContextualBackstoryService(conversation_config)
            selected_backstory = await backstory_service.select_relevant_content(
                user_message=user_message,
                max_chars=int(conversation_config.MAX_BACKSTORY_CHARS * 0.6)
            )

            # Get content metadata (same as enhanced service)
            content_metadata = {
                "content_types": selected_backstory.get("content_types", []),
                "content_selection_reasoning": selected_backstory.get("selection_reasoning", "Default selection")
            }

            # Get mood data - emotion will be determined by state influence service
            # which is called later in the prompt construction
            conversation_emotion = None
            mood_transition_data = {}

            return (
                recent_events,
                global_state,
                content_metadata,
                selected_backstory,
                conversation_emotion,
                mood_transition_data
            )

        except Exception as e:
            logger.warning(f"Failed to get enhanced context: {str(e)}")
            # Return fallback values same structure as enhanced service
            return (
                [],  # recent_events
                {},  # global_state
                {},  # content_metadata
                {"content": "You are Clara, a friendly conversational assistant."},  # selected_backstory
                None,  # conversation_emotion
                {}   # mood_transition_data
            )


    async def _store_conversation_messages_via_session_service(
        self,
        conversation_id: str,
        user_message: str,
        ai_response: str,
        user_id: str
    ) -> None:
        """
        Store conversation messages via SessionStateService (same as enhanced service).
        This maintains conversation history in Redis for context gathering.
        """
        try:
            from app.services.session_state_service import SessionStateService

            session_service = SessionStateService()

            # Add user message
            await session_service.add_conversation_message(
                conversation_id=conversation_id,
                message_content=user_message,
                message_type="user_input",
                user_id=user_id
            )

            # Add AI response
            await session_service.add_conversation_message(
                conversation_id=conversation_id,
                message_content=ai_response,
                message_type="ai_response",
                user_id=user_id
            )

            logger.info(f"✅ Stored conversation messages via SessionStateService for conversation {conversation_id}")

        except Exception as e:
            logger.error(f"❌ Failed to store conversation messages via SessionStateService: {str(e)}")
            # Don't raise - this is not critical for streaming functionality

    async def _store_streaming_conversation_to_db(
        self,
        conversation_id: str,
        user_id: str,
        user_message: str,
        ai_response: str,
        effective_personality: str,
        db
    ) -> None:
        """
        Store the complete streaming conversation to database after streaming completes.
        This ensures database consistency while maintaining real-time streaming UX.
        """
        try:
            from app.models.conversation_v2 import Conversation, ConversationMessage
            import uuid

            # 1. Ensure conversation record exists
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.status == 'active'
            ).first()

            if not conversation:
                # Create new conversation
                conversation = Conversation(
                    id=conversation_id,
                    user_id=user_id,
                    session_id=None,  # No session linking for now
                    status='active',
                    personality=effective_personality
                )
                db.add(conversation)
                db.flush()

            # 2. Store user message
            user_message_record = ConversationMessage(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role='user',
                content=user_message,
                timestamp=datetime.now(timezone.utc)
            )
            db.add(user_message_record)

            # 3. Parse and store AI response
            try:
                import json
                ai_response_json = json.loads(ai_response)
                ai_message_content = ai_response_json.get("message", ai_response)
            except (json.JSONDecodeError, ValueError):
                ai_message_content = ai_response

            ai_message_record = ConversationMessage(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role='assistant',
                content=ai_message_content,
                timestamp=datetime.now(timezone.utc)
            )
            db.add(ai_message_record)

            # 4. Commit all changes
            db.commit()
            logger.info(f"✅ Stored streaming conversation to database: {conversation_id}")

        except Exception as e:
            logger.error(f"❌ Failed to store streaming conversation to DB: {str(e)}")
            db.rollback()
            # Don't raise - streaming was successful, DB storage is secondary