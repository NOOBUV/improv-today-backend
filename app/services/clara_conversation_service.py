"""
Clara Conversation Service: the conversation path behind /api/clara/conversation[/stream].
Orchestrates context gathering from simulation, state, and character services.
"""
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
from datetime import datetime, timezone
import json
import time

from app.core.conversation_config import conversation_config
from app.services.character_content_service import CharacterContentService
from app.services.conversation_performance import ConversationPerformanceMonitor
from app.services.conversation_prompt_service import ConversationPromptService
from app.services.state_influence_service import StateInfluenceService, ConversationScenario
from app.services.simulation.state_manager import StateManagerService
from app.services.session_state_service import SessionStateService
from app.services.event_selection_service import EventSelectionService
from app.core.config import settings
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Gemini speaks OpenAI's wire protocol, so the AsyncOpenAI client stays.
# ponytail: flash-lite over flash for the free-tier limits (15 RPM/500 RPD vs 5 RPM/20 RPD).
CLARA_MODEL = "gemini-3.1-flash-lite"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Gemini 3 thinks by default and its reasoning eats max_tokens: at the default effort
# the 400-token budget is gone before the JSON reply starts (finish_reason=length).
# "low" is the floor this model accepts ("none" is rejected) and keeps TTFT short.
CLARA_REASONING_EFFORT = "low"

# Voice options for the fallback path only (the enhanced path builds its own prompt).
FALLBACK_PERSONALITY_PROMPTS = {
    "sassy": "You are a witty, sassy English conversation partner with a charming British accent in your responses. Be playful, slightly cheeky, but encouraging.",
    "blunt": "You are a direct, no-nonsense American conversation partner. Be straightforward, honest, and practical while remaining supportive.",
    "friendly": "You are a warm, encouraging conversation partner. Be supportive, patient, and genuinely interested in the conversation.",
}


class ClaraConversationService:
    """
    Clara's conversation service: integrates simulation context with conversations.

    This service orchestrates:
    - Global state retrieval from StateManagerService
    - Recent simulation events using configurable time windows
    - Intelligent backstory selection via CharacterContentService
    - State influence calculation via StateInfluenceService
    - Enhanced prompt construction via ConversationPromptService
    """

    def __init__(self):
        self.config = conversation_config
        self.character_content_service = CharacterContentService(self.config)
        self.conversation_prompt_service = ConversationPromptService()
        self.state_influence_service = StateInfluenceService()
        self.state_manager_service = StateManagerService()
        self.session_state_service = SessionStateService()
        self.event_selection_service = EventSelectionService()
        self.openai_client = AsyncOpenAI(api_key=settings.gemini_api_key, base_url=GEMINI_BASE_URL) if settings.gemini_api_key else None
        self.performance_monitor = ConversationPerformanceMonitor()

    async def generate_enhanced_response(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        conversation_history: Optional[str] = None,
        personality: str = "friendly",
        user_preferences: Optional[Dict[str, Any]] = None,
        fresh_events: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False
    ):
        """
        Generate enhanced conversation response with comprehensive performance monitoring.
        Includes detailed timing breakdown, correlation IDs, and threshold alerting.

        Args:
            user_message: User's message
            user_id: User identifier
            conversation_id: Conversation session identifier
            conversation_history: Existing conversation context
            personality: AI personality style
            user_preferences: User-specific preferences
            fresh_events: Pre-selected fresh events to avoid repetition
            stream: If True, returns AsyncGenerator for SSE streaming; if False, returns Dict

        Returns:
            If stream=False: Dict with enhanced conversation response and performance metrics
            If stream=True: AsyncGenerator yielding SSE events progressively
        """
        # Create correlation ID for this conversation request
        correlation_id = self.performance_monitor.create_conversation_correlation_id(user_id, conversation_id)

        # Start comprehensive timing context
        timing_context = self.performance_monitor.start_timing_context(
            correlation_id, "enhanced_conversation_response"
        )

        try:
            logger.info(f"[{correlation_id}] Generating enhanced response for user {user_id}, conversation {conversation_id}")

            # Sub-operation 1: Request parsing and session state setup
            with self.performance_monitor.step(timing_context, "request_parsing") as s:
                # Store user message in session state
                await self.session_state_service.add_conversation_message(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    message_type="user",
                    message_content=user_message
                )

                # Get conversation history from session state if not provided
                if not conversation_history:
                    conversation_history = await self.session_state_service.get_conversation_history(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        max_messages=4  # Reduced from 10 to 4 (2 exchanges) for faster TTFT
                    )

                s.update(
                    user_message_length=len(user_message),
                    history_retrieved=bool(conversation_history),
                    fresh_events_provided=len(fresh_events) if fresh_events else 0
                )

            # Sub-operation 2: Context gathering with detailed breakdown
            logger.debug(f"[{correlation_id}] Starting context gathering...")
            simulation_context = {}

            try:
                with self.performance_monitor.step(timing_context, "context_gathering") as s:
                    simulation_context = await self._gather_simulation_context_with_monitoring(
                        user_message, user_id, conversation_id, user_preferences, fresh_events, timing_context
                    )

                    s.update(
                        context_items_gathered=len(simulation_context),
                        recent_events_count=len(simulation_context.get("recent_events", [])),
                        backstory_chars=simulation_context.get("selected_backstory", {}).get("char_count", 0)
                    )

            except Exception as e:
                self.performance_monitor.log_error_with_context(timing_context, e, "context_gathering")
                logger.warning(f"[{correlation_id}] Context gathering failed, using fallback: {str(e)}")

            # Sub-operation 3: Consciousness generation (enhanced context-aware response)
            if simulation_context and self.openai_client:
                try:
                    # Branch: streaming vs non-streaming response
                    if stream:
                        logger.debug(f"[{correlation_id}] Returning streaming generator (will execute when consumed)")
                        # Return async generator for SSE streaming
                        return self._respond_stream(
                            user_message=user_message,
                            user_id=user_id,
                            conversation_id=conversation_id,
                            simulation_context=simulation_context,
                            conversation_history=conversation_history,
                            timing_context=timing_context,
                            correlation_id=correlation_id
                        )

                    with self.performance_monitor.step(timing_context, "consciousness_generation") as s:
                        # Non-streaming: wait for complete response
                        response = await self._respond(
                            user_message=user_message,
                            simulation_context=simulation_context,
                            conversation_history=conversation_history,
                            timing_context=timing_context
                        )

                        s.update(
                            response_length=len(response.get("ai_response", "")),
                            enhanced_mode=True,
                            emotion_selected=response.get("simulation_context", {}).get("conversation_emotion")
                        )

                    # Sub-operation 4: Response formatting and session state update
                    with self.performance_monitor.step(timing_context, "response_formatting") as s:
                        # Add comprehensive performance metrics
                        final_metrics = self.performance_monitor.end_timing_context(timing_context)

                        response["performance_metrics"] = final_metrics
                        response["enhanced_mode"] = True
                        response["correlation_id"] = correlation_id

                        # Store Clara's response in session state + track events mentioned
                        await self._persist_turn(
                            user_id=user_id,
                            conversation_id=conversation_id,
                            ai_response=response.get("ai_response", ""),
                            response_emotion=response.get("simulation_context", {}).get("conversation_emotion"),
                            correlation_id=correlation_id,
                            simulation_context=simulation_context,
                            global_mood=response.get("simulation_context", {}).get("global_mood")
                        )

                        s.update(
                            session_state_updated=True,
                            events_tracked=len(simulation_context.get("content_selection_metadata", {}).get("fresh_events_used", []))
                        )

                    logger.info(f"[{correlation_id}] Enhanced response completed in {final_metrics['total_duration_ms']:.2f}ms")
                    return response

                except Exception as e:
                    # Log error with full context but continue to fallback
                    self.performance_monitor.log_error_with_context(timing_context, e, "consciousness_generation")
                    logger.warning(f"[{correlation_id}] Enhanced response generation failed: {str(e)}")

            # Sub-operation 5: Fallback response generation
            if stream:
                # stream=True must hand back a generator; a dict here would be iterated
                # by StreamingResponse into its key names.
                logger.info(f"[{correlation_id}] Using fallback SSE stream")
                return self._fallback_sse(
                    user_message, conversation_history, personality, correlation_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    simulation_context=simulation_context
                )

            with self.performance_monitor.step(timing_context, "fallback_response") as s:
                logger.info(f"[{correlation_id}] Using fallback response generation")
                fallback_text = await self._fallback_response(
                    user_message, conversation_history, personality
                )

                s.update(
                    response_length=len(fallback_text),
                    fallback_mode=True
                )

            # Sub-operation 6: Fallback response formatting
            with self.performance_monitor.step(timing_context, "response_formatting") as s:
                final_metrics = self.performance_monitor.end_timing_context(timing_context)

                result = {
                    "ai_response": fallback_text,
                    "corrected_transcript": user_message,
                    "simulation_context": simulation_context,
                    "selected_backstory_types": [],
                    "fallback_mode": True,
                    "enhanced_mode": False,
                    "performance_metrics": final_metrics,
                    "correlation_id": correlation_id
                }

                # Store fallback response in session state (no event tracking)
                await self._persist_turn(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    ai_response=fallback_text,
                    response_emotion=None,
                    correlation_id=correlation_id,
                    simulation_context=simulation_context,
                    fallback=True
                )

                s.update(
                    session_state_updated=True,
                    fallback_mode=True
                )

            logger.info(f"[{correlation_id}] Fallback response completed in {final_metrics['total_duration_ms']:.2f}ms")
            return result

        except Exception as e:
            # Log critical error with full performance context
            self.performance_monitor.log_error_with_context(timing_context, e)
            final_metrics = self.performance_monitor.end_timing_context(timing_context)
            logger.error(f"[{correlation_id}] Critical error in conversation generation after {final_metrics['total_duration_ms']:.2f}ms: {str(e)}")
            raise

    async def _gather_simulation_context_with_monitoring(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        user_preferences: Optional[Dict[str, Any]] = None,
        fresh_events: Optional[List[Dict[str, Any]]] = None,
        timing_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Gather simulation context with detailed performance monitoring."""
        context = {}

        try:
            # Sub-sub-operation: Global state retrieval
            with self.performance_monitor.step(timing_context, "global_state_retrieval") as s:
                global_state = await self.state_manager_service.get_current_global_state()
                context["global_state"] = global_state
                s.update(traits_count=len(global_state))

            logger.debug(f"Retrieved global state: {len(global_state)} traits")

            # Sub-sub-operation: Event selection and processing
            with self.performance_monitor.step(timing_context, "event_selection") as s:
                if fresh_events is not None:
                    logger.info(f"Using pre-selected fresh events: {len(fresh_events)} events")
                    fresh_events_data = fresh_events
                else:
                    logger.info("Fetching fresh events from event selection service")
                    fresh_events_data = await self.event_selection_service.get_contextual_events(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        user_message=user_message,
                        max_events=self.config.MAX_EVENTS_COUNT
                    )

                context["recent_events"] = [event.get("original_event", event) for event in fresh_events_data]
                context["content_selection_metadata"] = {
                    "strategy": "fresh_events_rotation",
                    "entities_found": [],
                    "total_analyzed": len(fresh_events_data),
                    "selected_count": len(fresh_events_data),
                    "fresh_events_used": [event.get("id") for event in fresh_events_data]
                }

                s.update(
                    events_selected=len(fresh_events_data),
                    pre_selected=fresh_events is not None
                )

            logger.info(f"Fresh events selection: {len(fresh_events_data)} events selected")

            # Sub-sub-operation: Backstory content selection
            with self.performance_monitor.step(timing_context, "backstory_selection") as s:
                backstory_context_data = await self.character_content_service.select_relevant_content(
                    user_message=user_message,
                    max_chars=int(self.config.MAX_BACKSTORY_CHARS * 0.6)
                )
                context["selected_backstory"] = backstory_context_data
                s.update(
                    chars_selected=backstory_context_data['char_count'],
                    content_types=len(backstory_context_data['content_types'])
                )

            logger.debug(f"Selected backstory: {backstory_context_data['char_count']} chars, types: {backstory_context_data['content_types']}")

            # Sub-sub-operation: Conversation sentiment analysis
            with self.performance_monitor.step(timing_context, "sentiment_analysis") as s:
                conversation_sentiment_score = self._message_sentiment_score(user_message)
                s.update(sentiment_score=conversation_sentiment_score)

            # Sub-sub-operation: State influence calculation
            with self.performance_monitor.step(timing_context, "state_influence_calculation") as s:
                conversation_context = await self.state_influence_service.build_conversation_context(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    scenario=ConversationScenario.CASUAL_CHAT,
                    user_preferences=user_preferences,
                    conversation_sentiment=conversation_sentiment_score,
                    recent_events=context["recent_events"]
                )
                context["conversation_influence"] = conversation_context
                s.update(influence_factors=len(conversation_context))

            logger.debug(f"Built conversation context with {len(conversation_context)} influence factors")

            return context

        except Exception as e:
            logger.error(f"Error gathering simulation context: {str(e)}")
            return {}

    def _prepare_prompt(
        self,
        user_message: str,
        simulation_context: Dict[str, Any],
        conversation_history: Optional[str] = None,
        timing_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Context extraction -> emotion selection -> prompt construction.

        The single copy shared by the streaming and non-streaming paths.
        """
        with self.performance_monitor.step(timing_context, "context_extraction") as s:
            global_state = simulation_context.get("global_state", {})
            recent_events = simulation_context.get("recent_events", [])
            selected_backstory = simulation_context.get("selected_backstory", {})
            conversation_influence = simulation_context.get("conversation_influence", {})
            content_metadata = simulation_context.get("content_selection_metadata", {})

            mood_transition_data = conversation_influence.get("mood_transition", {})
            blended_mood = mood_transition_data.get("blended_mood_score", 60)
            mood_context = mood_transition_data.get("mood_context", {})

            s.update(
                global_state_items=len(global_state),
                recent_events_count=len(recent_events),
                backstory_chars=selected_backstory.get("char_count", 0),
                blended_mood=blended_mood
            )

        logger.debug(f"Using intelligent content selection: {content_metadata.get('strategy', 'unknown')}")

        with self.performance_monitor.step(timing_context, "emotion_selection") as s:
            conversation_emotion, _ = self.conversation_prompt_service.select_conversation_emotion_with_mood(
                user_message=user_message,
                blended_mood_score=blended_mood,
                mood_transition_data=mood_transition_data
            )
            s.update(
                selected_emotion=conversation_emotion.value,
                blended_mood_score=blended_mood
            )

        with self.performance_monitor.step(timing_context, "prompt_construction") as s:
            # One system message end to end: OpenAI caches the longest stable prefix
            system_prompt = self.conversation_prompt_service.construct_conversation_prompt_with_mood(
                character_backstory=selected_backstory.get("content", ""),
                user_message=user_message,
                conversation_emotion=conversation_emotion,
                mood_transition_data=mood_transition_data,
                conversation_history=conversation_history,
                recent_events=recent_events,
                global_state=global_state,
                content_metadata=content_metadata
            )
            s.update(
                prompt_length=len(system_prompt),
                backstory_chars=len(selected_backstory.get("content", "")),
                events_included=len(recent_events)
            )

        return {
            "system_prompt": system_prompt,
            "emotion": conversation_emotion,
            "blended_mood": blended_mood,
            "mood_context": mood_context
        }

    def _parse_llm_response(self, raw: Optional[str], emotion) -> tuple[str, str]:
        """Parse the model's JSON reply. Tolerates raw text and truncated streams."""
        default_emotion = emotion.value if emotion else "calm"

        try:
            parsed = json.loads(raw)
            return parsed.get("message", raw).strip(), parsed.get("emotion", default_emotion)
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to parse JSON response ({e}), using raw content: {str(raw)[:100]}...")

        # Truncated stream: the closing brace never arrived, pull the message out by hand
        if raw and '"message": "' in raw:
            msg_start = raw.find('"message": "') + 12
            msg_end = raw.find('",', msg_start)
            return (raw[msg_start:msg_end] if msg_end != -1 else raw[msg_start:]), default_emotion

        return (raw or ""), default_emotion

    async def _persist_turn(
        self,
        user_id: str,
        conversation_id: str,
        ai_response: str,
        response_emotion: Optional[str],
        correlation_id: str,
        simulation_context: Dict[str, Any],
        global_mood: Any = None,
        fallback: bool = False
    ) -> None:
        """Store Clara's turn in session state and track the events it consumed."""
        if fallback:
            metadata = {
                "fallback_mode": True,
                "enhanced_mode": False,
                "correlation_id": correlation_id
            }
        else:
            metadata = {
                "conversation_emotion": response_emotion,
                "global_mood": global_mood,
                "enhanced_mode": True,
                "correlation_id": correlation_id
            }

        await self.session_state_service.add_conversation_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message_type="assistant",
            message_content=ai_response,
            metadata=metadata
        )

        if not fallback:
            # Track events mentioned to prevent repetition
            await self._track_events_mentioned(simulation_context, user_id, conversation_id)

    async def _respond(
        self,
        user_message: str,
        simulation_context: Dict[str, Any],
        conversation_history: Optional[str] = None,
        timing_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Non-streaming context-aware response with OpenAI API monitoring."""
        prepared = self._prepare_prompt(
            user_message, simulation_context, conversation_history, timing_context
        )
        system_prompt = prepared["system_prompt"]

        with self.performance_monitor.step(timing_context, "openai_api_call") as s:
            response = await self.openai_client.chat.completions.create(
                model=CLARA_MODEL,
                reasoning_effort=CLARA_REASONING_EFFORT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=400,
                temperature=0.7
            )

            ai_response_raw = response.choices[0].message.content

            s.update(
                model=CLARA_MODEL,
                prompt_tokens=len(system_prompt.split()),
                max_tokens=400,
                response_length=len(ai_response_raw) if ai_response_raw else 0
            )

        with self.performance_monitor.step(timing_context, "response_parsing") as s:
            ai_response, response_emotion = self._parse_llm_response(ai_response_raw, prepared["emotion"])
            s.update(
                json_parsed=ai_response != ai_response_raw,
                final_response_length=len(ai_response)
            )

        selected_backstory = simulation_context.get("selected_backstory", {})
        mood_context = prepared["mood_context"]

        return {
            "ai_response": ai_response,
            "corrected_transcript": user_message,
            "simulation_context": {
                "recent_events_count": len(simulation_context.get("recent_events", [])),
                "global_mood": mood_context.get("current_mood", 60),
                "stress_level": mood_context.get("stress_level", 50),
                "selected_content_types": selected_backstory.get("content_types", []),
                "conversation_emotion": response_emotion
            },
            "selected_backstory_types": selected_backstory.get("content_types", []),
            "fallback_mode": False
        }

    async def _respond_stream(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        simulation_context: Dict[str, Any],
        conversation_history: Optional[str] = None,
        timing_context: Optional[Dict[str, Any]] = None,
        correlation_id: str = None
    ) -> AsyncGenerator[str, None]:
        """Same prep/parse/persist as _respond, assembled as Server-Sent Events.

        Yields: processing_start -> context_ready -> consciousness_chunk* -> processing_complete
        (or a single error event if anything above it raises).
        """
        start_time = time.time()
        logger.debug(f"[{correlation_id}] STREAM START - Beginning context-aware response generation")

        try:
            yield self._format_sse_event("processing_start", correlation_id, {
                "status": "starting"
            })

            prepared = self._prepare_prompt(
                user_message, simulation_context, conversation_history, timing_context
            )
            conversation_emotion = prepared["emotion"]
            recent_events = simulation_context.get("recent_events", [])

            yield self._format_sse_event("context_ready", correlation_id, {
                "recent_events_count": len(recent_events),
                "conversation_emotion": conversation_emotion.value if conversation_emotion else None
            })

            # AsyncOpenAI streams without blocking the event loop; chunks go out as they land
            stream = await self.openai_client.chat.completions.create(
                model=CLARA_MODEL,
                reasoning_effort=CLARA_REASONING_EFFORT,
                messages=[
                    {"role": "system", "content": prepared["system_prompt"]},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=400,
                temperature=0.7,
                stream=True
            )

            accumulated_response = ""
            chunk_count = 0
            first_chunk_time = None

            async for chunk in stream:
                if chunk.choices[0].delta.content is None:
                    continue

                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    logger.info(f"[{correlation_id}] FIRST TOKEN RECEIVED: {(first_chunk_time - start_time) * 1000:.0f}ms from stream start")

                chunk_text = chunk.choices[0].delta.content
                accumulated_response += chunk_text
                chunk_count += 1

                # Yield IMMEDIATELY - no buffering, no delays
                yield self._format_sse_event("consciousness_chunk", correlation_id, {
                    "chunk": chunk_text
                })

            logger.info(f"[{correlation_id}] Stream completed: {chunk_count} chunks, {len(accumulated_response)} chars")

            full_ai_response, response_emotion = self._parse_llm_response(
                accumulated_response, conversation_emotion
            )

            await self._persist_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                ai_response=full_ai_response,
                response_emotion=response_emotion,
                correlation_id=correlation_id,
                simulation_context=simulation_context,
                global_mood=prepared["blended_mood"]
            )

            final_metrics = self.performance_monitor.end_timing_context(timing_context)

            yield self._format_sse_event("processing_complete", correlation_id, {
                "response": full_ai_response,  # Already parsed message text only
                "simulation_context": {
                    "conversation_emotion": response_emotion,
                    "global_mood": prepared["blended_mood"],
                    "recent_events_count": len(recent_events)
                },
                "performance_metrics": final_metrics,
                "success": True
            })

        except Exception as e:
            logger.error(f"Streaming response failed: {str(e)}")
            yield self._format_sse_event("error", correlation_id, {
                "error": str(e)
            })

    async def _fallback_response(
        self,
        user_message: str,
        conversation_history: Optional[str],
        personality: str = "friendly"
    ) -> str:
        """Clara's plain reply when the enhanced path is unavailable.

        No simulation context, no JSON envelope - just a line in character.
        """
        if self.openai_client is None:
            return "That's really interesting! Can you tell me more about that?"

        base_prompt = FALLBACK_PERSONALITY_PROMPTS.get(
            personality, FALLBACK_PERSONALITY_PROMPTS["friendly"]
        )
        history_context = f"\n\nRecent conversation:\n{conversation_history}" if conversation_history else ""

        response = await self.openai_client.chat.completions.create(
            model=CLARA_MODEL,
            reasoning_effort=CLARA_REASONING_EFFORT,
            messages=[
                {"role": "system", "content": (
                    f"{base_prompt}\n\n"
                    "You are Clara. Reply in character as plain text (no JSON), "
                    "1-2 conversational sentences, and ask a follow-up question."
                    f"{history_context}"
                )},
                {"role": "user", "content": user_message}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return (response.choices[0].message.content or "").strip()

    async def _fallback_sse(
        self,
        user_message: str,
        conversation_history: Optional[str],
        personality: str,
        correlation_id: str,
        user_id: str,
        conversation_id: str,
        simulation_context: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """The fallback reply dressed as the normal SSE sequence.

        Yields: processing_start -> one consciousness_chunk (whole text) -> processing_complete.
        The client parses the same events either way.
        """
        start_time = time.time()

        try:
            yield self._format_sse_event("processing_start", correlation_id, {
                "status": "starting"
            })

            text = await self._fallback_response(user_message, conversation_history, personality)

            yield self._format_sse_event("consciousness_chunk", correlation_id, {
                "chunk": text
            })

            # Same persist point as _respond_stream: history must carry the reply,
            # not just the user turn that prompted it.
            await self._persist_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                ai_response=text,
                response_emotion=None,
                correlation_id=correlation_id,
                simulation_context=simulation_context,
                fallback=True
            )

            yield self._format_sse_event("processing_complete", correlation_id, {
                "response": text,
                "simulation_context": {
                    "conversation_emotion": None,
                    "global_mood": None,
                    "recent_events_count": 0
                },
                "performance_metrics": {"total_duration_ms": (time.time() - start_time) * 1000},
                "fallback_mode": True,
                "success": True
            })

        except Exception as e:
            logger.error(f"[{correlation_id}] Fallback stream failed: {str(e)}")
            yield self._format_sse_event("error", correlation_id, {
                "error": str(e)
            })

    def _format_sse_event(self, event_type: str, correlation_id: str, data: Dict[str, Any]) -> str:
        """Format data as an SSE string, stamping correlation_id + timestamp."""
        payload = {
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

    async def _track_events_mentioned(
        self,
        simulation_context: Dict[str, Any],
        user_id: str,
        conversation_id: str
    ) -> None:
        """Track events mentioned in response to prevent future repetition."""
        try:
            events_mentioned = simulation_context.get("content_selection_metadata", {}).get("fresh_events_used", [])
            fresh_events_data = []

            for event_id in events_mentioned:
                for event in simulation_context.get("recent_events", []):
                    if event.get("event_id") == event_id:
                        fresh_events_data.append({"id": event_id, "summary": event.get("summary", "")})
                        break

            if fresh_events_data:
                await self.event_selection_service.track_events_mentioned_in_response(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    events_mentioned=fresh_events_data
                )
                logger.debug(f"Tracked {len(fresh_events_data)} events as mentioned")

        except Exception as e:
            logger.warning(f"Failed to track events mentioned: {str(e)}")

    def _message_sentiment_score(self, user_message: str) -> float:
        """Sentiment of the user's message, -1.0 (negative) to 1.0 (positive).

        Feeds StateInfluenceService.build_conversation_context - the only consumer.
        """
        try:
            words = user_message.strip().split()
            message_lower = user_message.lower().strip()

            positive_keywords = [
                "happy", "great", "awesome", "wonderful", "excited", "amazing", "love",
                "fantastic", "brilliant", "perfect", "excellent", "good", "nice", "fun"
            ]

            negative_keywords = [
                "sad", "terrible", "awful", "hate", "angry", "frustrated", "disappointed",
                "upset", "worried", "stressed", "anxious", "bad", "horrible", "difficult"
            ]

            # Special resonance indicators that suggest personal investment
            romantic_indicators = [
                "excited to meet", "special friend", "can't wait to see", "butterflies",
                "nervous about", "thinking about", "looking forward to meeting"
            ]

            safety_urgency = [
                "crashed", "accident", "hurt", "injured", "emergency", "help", "broke",
                "damaged", "hospital", "bleeding", "pain"
            ]

            personal_interests = [
                "coffee", "stressed about work", "deadline", "project", "tired",
                "exhausted", "can't sleep", "overthinking", "anxiety"
            ]

            positive_count = sum(1 for word in words if any(pos in word for pos in positive_keywords))
            negative_count = sum(1 for word in words if any(neg in word for neg in negative_keywords))

            romantic_score = sum(1 for phrase in romantic_indicators if phrase in message_lower)
            safety_score = sum(1 for phrase in safety_urgency if phrase in message_lower) * 3  # High priority
            interest_score = sum(1 for phrase in personal_interests if phrase in message_lower)

            # If safety concerns detected, return high negative sentiment to trigger priority response
            if safety_score > 0:
                logger.info(f"Safety concern detected in message: {safety_score} indicators")
                sentiment_score = -0.8  # Strong negative to trigger urgent response
            # If romantic undertones detected, adjust sentiment to reflect excitement
            elif romantic_score > 0:
                logger.info(f"Romantic subtext detected: {romantic_score} indicators")
                sentiment_score = min(1.0, 0.3 + (romantic_score * 0.2))  # Boost positive sentiment
            # Standard sentiment calculation
            elif positive_count == 0 and negative_count == 0:
                sentiment_score = 0.1 if interest_score > 0 else 0.0
            else:
                # Calculate base sentiment with personal interest boost
                sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
                if interest_score > 0:
                    sentiment_score += 0.1  # Small boost for personal resonance

            sentiment_score = max(-1.0, min(1.0, sentiment_score))

            logger.debug(f"Sentiment: '{user_message[:50]}...' -> {sentiment_score}")
            return sentiment_score

        except Exception as e:
            logger.error(f"Error analyzing message sentiment: {e}")
            return 0.0
