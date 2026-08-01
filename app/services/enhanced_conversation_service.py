"""
Enhanced Conversation Service for Story 2.6: Enhanced Conversational Context Integration.
Orchestrates context gathering from simulation, state, and character services.
"""
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
from datetime import datetime, timezone
import json
import time

from app.core.conversation_config import conversation_config
from app.services.contextual_backstory_service import ContextualBackstoryService
from app.services.conversation_performance import ConversationPerformanceMonitor, maybe_step
from app.services.conversation_prompt_service import ConversationPromptService
from app.services.state_influence_service import StateInfluenceService, ConversationScenario
from app.services.simulation.state_manager import StateManagerService
from app.services.simple_openai import SimpleOpenAIService
from app.services.session_state_service import SessionStateService
from app.services.event_selection_service import EventSelectionService
from app.core.config import settings
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class EnhancedConversationService:
    """
    Enhanced conversation service that integrates simulation context with conversations.

    This service orchestrates:
    - Global state retrieval from StateManagerService
    - Recent simulation events using configurable time windows
    - Intelligent backstory selection via ContextualBackstoryService
    - State influence calculation via StateInfluenceService
    - Enhanced prompt construction via ConversationPromptService
    """

    def __init__(self):
        self.config = conversation_config
        self.contextual_backstory_service = ContextualBackstoryService(self.config)
        self.conversation_prompt_service = ConversationPromptService()
        self.state_influence_service = StateInfluenceService()
        self.state_manager_service = StateManagerService()
        self.simple_openai_service = SimpleOpenAIService()
        self.session_state_service = SessionStateService()
        self.event_selection_service = EventSelectionService()
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.performance_monitor = ConversationPerformanceMonitor()

    async def generate_enhanced_response(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        conversation_history: Optional[str] = None,
        personality: str = "friendly_neutral",
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

                s.meta(
                    user_message_length=len(user_message),
                    history_retrieved=bool(conversation_history),
                    fresh_events_provided=len(fresh_events) if fresh_events else 0
                )

            # Sub-operation 2: Context gathering with detailed breakdown
            context_start = time.time()
            print(f"⏱️  [{correlation_id}] Starting context gathering...", flush=True)
            simulation_context = {}

            try:
                with self.performance_monitor.step(timing_context, "context_gathering") as s:
                    simulation_context = await self._gather_simulation_context_with_monitoring(
                        user_message, user_id, conversation_id, user_preferences, fresh_events, timing_context
                    )

                    context_time = (time.time() - context_start) * 1000
                    print(f"⏱️  [{correlation_id}] Context gathering completed: {context_time:.0f}ms", flush=True)

                    s.meta(
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
                        print(f"⏱️  [{correlation_id}] Returning streaming generator (will execute when consumed)", flush=True)
                        # Return async generator for SSE streaming
                        return self._stream_context_aware_response(
                            user_message=user_message,
                            user_id=user_id,
                            conversation_id=conversation_id,
                            simulation_context=simulation_context,
                            conversation_history=conversation_history,
                            personality=personality,
                            timing_context=timing_context,
                            correlation_id=correlation_id
                        )

                    with self.performance_monitor.step(timing_context, "consciousness_generation") as s:
                        # Non-streaming: wait for complete response
                        response = await self._generate_context_aware_response_with_monitoring(
                            user_message=user_message,
                            simulation_context=simulation_context,
                            conversation_history=conversation_history,
                            personality=personality,
                            timing_context=timing_context
                        )

                        s.meta(
                            response_length=len(response.get("ai_response", "")),
                            enhanced_mode=True,
                            emotion_selected=response.get("simulation_context", {}).get("conversation_emotion")
                        )

                    # Sub-operation 4: Response formatting and session state update
                    with self.performance_monitor.step(timing_context, "response_formatting") as s:
                        # Add comprehensive performance metrics
                        final_metrics = self.performance_monitor.end_timing_context(timing_context)

                        # Log detailed timing breakdown for analysis
                        self.performance_monitor.log_detailed_timing_breakdown(final_metrics)

                        response["performance_metrics"] = final_metrics
                        response["enhanced_mode"] = True
                        response["correlation_id"] = correlation_id

                        # Store Clara's response in session state
                        await self.session_state_service.add_conversation_message(
                            user_id=user_id,
                            conversation_id=conversation_id,
                            message_type="assistant",
                            message_content=response.get("ai_response", ""),
                            metadata={
                                "conversation_emotion": response.get("simulation_context", {}).get("conversation_emotion"),
                                "global_mood": response.get("simulation_context", {}).get("global_mood"),
                                "enhanced_mode": True,
                                "correlation_id": correlation_id
                            }
                        )

                        # Track events mentioned to prevent repetition
                        await self._track_events_mentioned(simulation_context, user_id, conversation_id)

                        s.meta(
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
            with self.performance_monitor.step(timing_context, "fallback_response") as s:
                logger.info(f"[{correlation_id}] Using fallback response generation")
                fallback_response = await self.simple_openai_service.generate_coaching_response(
                    message=user_message,
                    conversation_history=conversation_history or "",
                    personality=personality
                )

                s.meta(
                    response_length=len(fallback_response.ai_response),
                    fallback_mode=True
                )

            # Sub-operation 6: Fallback response formatting
            with self.performance_monitor.step(timing_context, "response_formatting") as s:
                final_metrics = self.performance_monitor.end_timing_context(timing_context)

                # Log detailed timing breakdown for fallback analysis
                self.performance_monitor.log_detailed_timing_breakdown(final_metrics)

                result = {
                    "ai_response": fallback_response.ai_response,
                    "corrected_transcript": fallback_response.corrected_transcript,
                    "simulation_context": simulation_context,
                    "selected_backstory_types": [],
                    "fallback_mode": True,
                    "enhanced_mode": False,
                    "performance_metrics": final_metrics,
                    "correlation_id": correlation_id
                }

                # Store fallback response in session state
                await self.session_state_service.add_conversation_message(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    message_type="assistant",
                    message_content=fallback_response.ai_response,
                    metadata={
                        "fallback_mode": True,
                        "enhanced_mode": False,
                        "correlation_id": correlation_id
                    }
                )

                s.meta(
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
            with maybe_step(self.performance_monitor, timing_context, "global_state_retrieval") as s:
                global_state = await self.state_manager_service.get_current_global_state()
                context["global_state"] = global_state
                s.meta(traits_count=len(global_state))

            logger.debug(f"Retrieved global state: {len(global_state)} traits")

            # Sub-sub-operation: Event selection and processing
            with maybe_step(self.performance_monitor, timing_context, "event_selection") as s:
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

                s.meta(
                    events_selected=len(fresh_events_data),
                    pre_selected=fresh_events is not None
                )

            logger.info(f"Fresh events selection: {len(fresh_events_data)} events selected")

            # Sub-sub-operation: Backstory content selection
            with maybe_step(self.performance_monitor, timing_context, "backstory_selection") as s:
                backstory_context_data = await self.contextual_backstory_service.select_relevant_content(
                    user_message=user_message,
                    max_chars=int(self.config.MAX_BACKSTORY_CHARS * 0.6)
                )
                context["selected_backstory"] = backstory_context_data
                s.meta(
                    chars_selected=backstory_context_data['char_count'],
                    content_types=len(backstory_context_data['content_types'])
                )

            logger.debug(f"Selected backstory: {backstory_context_data['char_count']} chars, types: {backstory_context_data['content_types']}")

            # Sub-sub-operation: Conversation sentiment analysis
            with maybe_step(self.performance_monitor, timing_context, "sentiment_analysis") as s:
                conversation_sentiment_analysis = self._analyze_message_sentiment(user_message)
                conversation_sentiment_score = conversation_sentiment_analysis.get("sentiment_score", 0.0)
                s.meta(
                    sentiment_score=conversation_sentiment_score,
                    complexity=conversation_sentiment_analysis.get("complexity", "unknown")
                )

            # Sub-sub-operation: State influence calculation
            with maybe_step(self.performance_monitor, timing_context, "state_influence_calculation") as s:
                conversation_context = await self.state_influence_service.build_conversation_context(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    scenario=ConversationScenario.CASUAL_CHAT,
                    user_preferences=user_preferences,
                    conversation_sentiment=conversation_sentiment_score,
                    recent_events=context["recent_events"]
                )
                context["conversation_influence"] = conversation_context
                s.meta(influence_factors=len(conversation_context))

            logger.debug(f"Built conversation context with {len(conversation_context)} influence factors")

            return context

        except Exception as e:
            logger.error(f"Error gathering simulation context: {str(e)}")
            return {}

    async def _generate_context_aware_response_with_monitoring(
        self,
        user_message: str,
        simulation_context: Dict[str, Any],
        conversation_history: Optional[str] = None,
        personality: str = "friendly_neutral",
        timing_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate context-aware response with detailed OpenAI API monitoring."""

        try:
            # Sub-sub-operation: Extract context components
            with maybe_step(self.performance_monitor, timing_context, "context_extraction") as s:
                global_state = simulation_context.get("global_state", {})
                recent_events = simulation_context.get("recent_events", [])
                selected_backstory = simulation_context.get("selected_backstory", {})
                conversation_influence = simulation_context.get("conversation_influence", {})
                content_metadata = simulation_context.get("content_selection_metadata", {})

                mood_transition_data = conversation_influence.get("mood_transition", {})
                blended_mood = mood_transition_data.get("blended_mood_score", 60)
                mood_context = mood_transition_data.get("mood_context", {})

                s.meta(
                    global_state_items=len(global_state),
                    recent_events_count=len(recent_events),
                    backstory_chars=selected_backstory.get("char_count", 0),
                    blended_mood=blended_mood
                )

            logger.debug(f"Using intelligent content selection: {content_metadata.get('strategy', 'unknown')}")

            # Sub-sub-operation: Emotion selection with mood awareness
            with maybe_step(self.performance_monitor, timing_context, "emotion_selection") as s:
                conversation_emotion, emotion_reasoning = self.conversation_prompt_service.select_conversation_emotion_with_mood(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    blended_mood_score=blended_mood,
                    mood_transition_data=mood_transition_data
                )
                s.meta(
                    selected_emotion=conversation_emotion.value,
                    blended_mood_score=blended_mood
                )

            # Sub-sub-operation: Prompt construction
            with maybe_step(self.performance_monitor, timing_context, "prompt_construction") as s:
                enhanced_prompt = self.conversation_prompt_service.construct_conversation_prompt_with_mood(
                    character_backstory=selected_backstory.get("content", ""),
                    user_message=user_message,
                    conversation_emotion=conversation_emotion,
                    mood_transition_data=mood_transition_data,
                    conversation_history=conversation_history
                )

                enhanced_prompt += self._build_simulation_context_prompt(
                    recent_events, global_state, content_metadata
                )

                s.meta(
                    prompt_length=len(enhanced_prompt),
                    backstory_chars=len(selected_backstory.get("content", "")),
                    events_included=len(recent_events)
                )

            # Sub-sub-operation: OpenAI API call
            with maybe_step(self.performance_monitor, timing_context, "openai_api_call") as s:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": enhanced_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=400,
                    temperature=0.7
                )

                ai_response_raw = response.choices[0].message.content

                s.meta(
                    model="gpt-4o-mini",
                    prompt_tokens=len(enhanced_prompt.split()),
                    max_tokens=400,
                    response_length=len(ai_response_raw) if ai_response_raw else 0
                )

            # Sub-sub-operation: Response parsing and formatting
            with maybe_step(self.performance_monitor, timing_context, "response_parsing") as s:
                # Parse JSON response from OpenAI
                try:
                    ai_response_json = json.loads(ai_response_raw)
                    ai_response = ai_response_json.get("message", ai_response_raw)
                    response_emotion = ai_response_json.get("emotion", conversation_emotion.value)
                except (json.JSONDecodeError, TypeError):
                    ai_response = ai_response_raw
                    response_emotion = conversation_emotion.value
                    logger.warning(f"Failed to parse JSON response, using raw content: {ai_response_raw[:100]}...")

                s.meta(
                    json_parsed=ai_response != ai_response_raw,
                    final_response_length=len(ai_response)
                )

            return {
                "ai_response": ai_response,
                "corrected_transcript": user_message,
                "simulation_context": {
                    "recent_events_count": len(recent_events),
                    "global_mood": mood_context.get("current_mood", 60),
                    "stress_level": mood_context.get("stress_level", 50),
                    "selected_content_types": selected_backstory.get("content_types", []),
                    "conversation_emotion": response_emotion,
                    "emotion_reasoning": emotion_reasoning
                },
                "selected_backstory_types": selected_backstory.get("content_types", []),
                "fallback_mode": False
            }

        except Exception as e:
            logger.error(f"Error generating context-aware response: {str(e)}")
            raise

    async def _stream_context_aware_response(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        simulation_context: Dict[str, Any],
        conversation_history: Optional[str] = None,
        personality: str = "friendly_neutral",
        timing_context: Optional[Dict[str, Any]] = None,
        correlation_id: str = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream context-aware response progressively using Server-Sent Events.

        Yields SSE-formatted events as OpenAI generates response chunks.
        Handles all the same context integration as non-streaming mode.
        """
        start_time = time.time()
        print(f"⏱️  [{correlation_id}] STREAM START - Beginning context-aware response generation", flush=True)

        try:
            # Yield initial processing acknowledgment
            yield self._format_sse_event("processing_start", {
                "correlation_id": correlation_id,
                "status": "starting",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Extract context components (same as non-streaming)
            context_start = time.time()
            recent_events = simulation_context.get("recent_events", [])
            global_state = simulation_context.get("global_state", {})
            conversation_influence = simulation_context.get("conversation_influence", {})
            selected_backstory = simulation_context.get("selected_backstory", {})
            content_metadata = simulation_context.get("content_selection_metadata", {})
            context_extract_time = (time.time() - context_start) * 1000
            print(f"⏱️  [{correlation_id}] Context extraction: {context_extract_time:.0f}ms", flush=True)

            mood_transition_data = conversation_influence.get("mood_transition", {})
            blended_mood = mood_transition_data.get("blended_mood_score", 60)
            mood_context = mood_transition_data.get("mood_context", {})

            # Determine conversation emotion (same logic as non-streaming)
            emotion_start = time.time()
            conversation_emotion, emotion_reasoning = self.conversation_prompt_service.select_conversation_emotion_with_mood(
                user_message=user_message,
                conversation_history=conversation_history,
                blended_mood_score=blended_mood,
                mood_transition_data=mood_transition_data
            )
            emotion_time = (time.time() - emotion_start) * 1000
            print(f"⏱️  [{correlation_id}] Emotion selection: {emotion_time:.0f}ms", flush=True)

            # Yield context ready event
            yield self._format_sse_event("context_ready", {
                "correlation_id": correlation_id,
                "recent_events_count": len(recent_events),
                "conversation_emotion": conversation_emotion.value if conversation_emotion else None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Construct prompt (same as non-streaming)
            prompt_start = time.time()
            base_prompt = self.conversation_prompt_service.construct_conversation_prompt_with_mood(
                character_backstory=selected_backstory.get("content", ""),
                user_message=user_message,
                conversation_emotion=conversation_emotion,
                mood_transition_data=mood_transition_data,
                conversation_history=conversation_history
            )
            base_prompt_size = len(base_prompt)

            # Add simulation context to prompt
            sim_context = self._build_simulation_context_prompt(
                recent_events, global_state, content_metadata
            )
            sim_context_size = len(sim_context)
            enhanced_prompt = base_prompt + sim_context

            prompt_time = (time.time() - prompt_start) * 1000
            prompt_length = len(enhanced_prompt)
            print(f"⏱️  [{correlation_id}] Prompt construction: {prompt_time:.0f}ms", flush=True)
            print(f"    - Base prompt: {base_prompt_size} chars | Sim context: {sim_context_size} chars | Total: {prompt_length} chars", flush=True)

            # CRITICAL: Using AsyncOpenAI for true async streaming without blocking
            # Chunks will be sent progressively as they arrive from OpenAI

            # Stream OpenAI response with automatic prompt caching (Oct 2024 feature)
            # Caching activates automatically for prompts >1024 tokens (~4000 chars)
            # Best practice: Put ALL content in one system message for max cache hit
            openai_start = time.time()
            print(f"⏱️  [{correlation_id}] Initiating OpenAI stream request (auto-caching enabled)...", flush=True)

            # Combine base_prompt + sim_context in single system message
            # OpenAI caches the longest matching prefix automatically
            # Since base_prompt (11k chars) is stable, it will be cached
            system_prompt = base_prompt + sim_context

            stream = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=400,
                temperature=0.7,
                stream=True  # Enable streaming
            )
            openai_init_time = (time.time() - openai_start) * 1000
            print(f"⏱️  [{correlation_id}] OpenAI stream initialized: {openai_init_time:.0f}ms", flush=True)

            # RAW STREAMING: Send chunks immediately as they arrive, no backend parsing
            # Frontend will handle JSON filtering on-the-fly for maximum real-time performance
            accumulated_response = ""  # Build complete response for final parsing

            # IMMEDIATE STREAMING: Send every token as it arrives from OpenAI
            # This eliminates all buffering to achieve true real-time streaming
            chunk_count = 0
            first_chunk_time = None
            last_chunk_time = None

            print(f"⏱️  [{correlation_id}] Waiting for first chunk from OpenAI...", flush=True)

            async for chunk in stream:
                if chunk.choices[0].delta.content is None:
                    continue

                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    time_to_first_token = (first_chunk_time - start_time) * 1000
                    print(f"⚡ [{correlation_id}] FIRST TOKEN RECEIVED: {time_to_first_token:.0f}ms from stream start", flush=True)

                last_chunk_time = time.time()
                chunk_text = chunk.choices[0].delta.content
                accumulated_response += chunk_text
                chunk_count += 1

                # Yield IMMEDIATELY - no buffering, no delays
                sse_event = self._format_sse_event("consciousness_chunk", {
                    "correlation_id": correlation_id,
                    "chunk": chunk_text,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                if chunk_count == 1:
                    print(f"⚡ [{correlation_id}] YIELDING FIRST CHUNK to frontend", flush=True)

                yield sse_event

                # Log first few chunks and periodically thereafter
                if chunk_count <= 5 or chunk_count % 10 == 0:
                    logger.info(f"[{correlation_id}] Chunk {chunk_count}: {len(chunk_text)} chars - '{chunk_text[:30]}'")

            stream_duration = (last_chunk_time - first_chunk_time) if first_chunk_time and last_chunk_time else 0
            logger.info(f"[{correlation_id}] Stream completed: {chunk_count} chunks, {len(accumulated_response)} chars, {stream_duration:.2f}s duration")

            # Final parsing: Extract complete message and emotion from full JSON
            try:
                ai_response_json = json.loads(accumulated_response)
                full_ai_response = ai_response_json.get("message", "").strip()
                response_emotion = ai_response_json.get("emotion", conversation_emotion.value if conversation_emotion else "calm")
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to parse complete JSON: {e}, raw: {accumulated_response[:200]}")
                # Fallback: manual extraction
                if '"message"' in accumulated_response and '": "' in accumulated_response:
                    try:
                        msg_start = accumulated_response.find('"message": "') + 12
                        msg_end = accumulated_response.find('",', msg_start)
                        full_ai_response = accumulated_response[msg_start:msg_end] if msg_end != -1 else accumulated_response[msg_start:]
                    except:
                        full_ai_response = accumulated_response
                else:
                    full_ai_response = accumulated_response
                response_emotion = conversation_emotion.value if conversation_emotion else "calm"

            # Store messages in session state (same as non-streaming)
            await self.session_state_service.add_conversation_message(
                user_id=user_id,
                conversation_id=conversation_id,
                message_type="assistant",
                message_content=full_ai_response,
                metadata={
                    "conversation_emotion": response_emotion,
                    "global_mood": blended_mood,
                    "enhanced_mode": True,
                    "correlation_id": correlation_id
                }
            )

            # Track events mentioned (same as non-streaming)
            await self._track_events_mentioned(simulation_context, user_id, conversation_id)

            # Calculate performance metrics
            total_duration = (time.time() - start_time) * 1000

            # End timing context
            if timing_context:
                final_metrics = self.performance_monitor.end_timing_context(timing_context)
            else:
                final_metrics = {"total_duration_ms": total_duration}

            # Yield final completion event with parsed message
            yield self._format_sse_event("processing_complete", {
                "correlation_id": correlation_id,
                "response": full_ai_response,  # Already parsed message text only
                "simulation_context": {
                    "conversation_emotion": response_emotion,
                    "global_mood": blended_mood,
                    "recent_events_count": len(recent_events)
                },
                "performance_metrics": final_metrics,
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            logger.error(f"Streaming response failed: {str(e)}")
            yield self._format_sse_event("error", {
                "correlation_id": correlation_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    def _format_sse_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """Format data as Server-Sent Events (SSE) string."""
        json_data = json.dumps(data)
        return f"event: {event_type}\ndata: {json_data}\n\n"

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

    def _build_simulation_context_prompt(
        self,
        recent_events: List[Dict[str, Any]],
        global_state: Dict[str, Any],
        content_metadata: Dict[str, Any] = None
    ) -> str:
        """Build simulation context section with intelligent event prioritization."""

        if not recent_events and not global_state:
            return ""

        context_parts = []

        if recent_events:
            # Use content selection strategy to inform how events are presented
            strategy = content_metadata.get("strategy", "") if content_metadata else ""

            if "current_day" in strategy.lower():
                context_parts.append("\n\nTODAY'S EVENTS (prioritized for current day discussion):")
            elif "specific_person" in strategy.lower():
                context_parts.append("\n\nRELEVANT RECENT INTERACTIONS:")
            elif "recent_life" in strategy.lower():
                context_parts.append("\n\nRECENT LIFE HIGHLIGHTS:")
            else:
                context_parts.append("\n\nRECENT LIFE EVENTS:")

            for event in recent_events:  # Use all intelligently selected events
                hours_ago = event.get("hours_ago", 0)
                if hours_ago < self.config.RECENT_EVENTS_HOURS_BACK:
                    # More detailed time formatting for better conversation context
                    if hours_ago < 1:
                        time_str = "just now"
                    elif hours_ago < 2:
                        time_str = f"{int(hours_ago * 60)} minutes ago"
                    elif hours_ago < 24:
                        time_str = f"{int(hours_ago)} hours ago"
                    elif hours_ago < 48:
                        time_str = "yesterday"
                    else:
                        days_ago = int(hours_ago / 24)
                        time_str = f"{days_ago} days ago"

                    # Include intensity information for better context
                    intensity = event.get("intensity", 0)
                    summary = event.get("summary", '')

                    if intensity >= 7:
                        context_parts.append(f"- {summary} ({time_str}) [significant experience]")
                    else:
                        context_parts.append(f"- {summary} ({time_str})")

        if global_state:
            mood = global_state.get("mood", {}).get("numeric_value", 60)
            stress = global_state.get("stress", {}).get("numeric_value", 50)
            energy = global_state.get("energy", {}).get("numeric_value", 70)

            context_parts.append(f"\n\nCURRENT STATE:")
            context_parts.append(f"- Mood: {mood}/100, Stress: {stress}/100, Energy: {energy}/100")

        # Adaptive instruction based on content selection
        if content_metadata and "specific_person" in content_metadata.get("strategy", "").lower():
            context_parts.append("\n\nFocus on the relevant interactions and experiences with the people mentioned. Share details naturally as they relate to the conversation.")
        elif content_metadata and "current_day" in content_metadata.get("strategy", "").lower():
            context_parts.append("\n\nShare how today has been going, referencing these recent experiences authentically. Don't feel obligated to mention everything - pick what feels natural to share.")
        else:
            context_parts.append("\n\nWeave these recent experiences into the conversation naturally when relevant. Focus on what genuinely connects to what the user is asking about.")

        return "".join(context_parts)

    def _analyze_message_sentiment(self, user_message: str) -> Dict[str, Any]:
        """
        Analyze sentiment and input complexity for mood transition analysis and brevity enforcement.
        Enhanced to identify personally resonant elements and input complexity for selective response.
        Returns dict with sentiment score, complexity assessment, and response guidance.
        """
        try:
            import re

            # Input complexity detection for brevity enforcement
            words = user_message.strip().split()
            word_count = len(words)
            char_count = len(user_message.strip())

            # Determine input complexity level
            if word_count == 1 and char_count < 15:
                complexity = "single_word"
            elif word_count <= 3 or char_count < 30:
                complexity = "fragment"
            elif word_count <= 8 or char_count < 100:
                complexity = "simple"
            elif word_count <= 20 or char_count < 250:
                complexity = "moderate"
            else:
                complexity = "complex"

            message_lower = user_message.lower().strip()

            # Logical patterns for incomplete/ambiguous input detection
            is_incomplete_input = False

            if complexity in ["single_word", "fragment"]:
                # Check if it's a complete thought using logical patterns
                complete_patterns = [
                    # Greetings
                    r'\b(hi|hello|hey|yo|sup)\b',
                    # Yes/No responses
                    r'\b(yes|no|yeah|nah|yep|nope|sure|okay|ok)\b',
                    # Exclamations that are complete
                    r'\b(wow|cool|nice|great|awesome|thanks|bye)\b',
                    # Questions that are complete even if short
                    r'\bwhat\?|why\?|how\?|when\?|where\?|who\?',
                    # Commands that are complete
                    r'\b(stop|wait|go|help|start|continue)\b'
                ]

                is_complete_thought = any(re.search(pattern, message_lower) for pattern in complete_patterns)

                # If it doesn't match complete patterns, it's likely incomplete
                is_incomplete_input = not is_complete_thought

            # Additional check for single words: most nouns without context are incomplete
            if word_count == 1 and not is_incomplete_input:
                single_word = message_lower
                # These are typically incomplete when said alone
                is_incomplete_input = (
                    len(single_word) > 2 and
                    single_word.isalpha() and
                    single_word not in ["yes", "no", "hi", "bye", "ok", "wow", "cool", "nice", "thanks", "help"]
                )

            # Enhanced keyword analysis with personal resonance detection
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

            # Basic sentiment calculation
            positive_count = sum(1 for word in words if any(pos in word for pos in positive_keywords))
            negative_count = sum(1 for word in words if any(neg in word for neg in negative_keywords))

            # Enhanced analysis for resonance detection
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
                total_sentiment_words = positive_count + negative_count
                if total_sentiment_words == 0:
                    sentiment_score = 0.1 if interest_score > 0 else 0.0
                else:
                    # Calculate base sentiment with personal interest boost
                    sentiment_score = (positive_count - negative_count) / total_sentiment_words
                    if interest_score > 0:
                        sentiment_score += 0.1  # Small boost for personal resonance

            # Clamp to valid range
            sentiment_score = max(-1.0, min(1.0, sentiment_score))

            # Generate response guidance based on complexity and content
            response_guidance = {
                "enforce_brevity": complexity in ["single_word", "fragment"],
                "expected_response_words": self._get_expected_response_length(complexity, is_incomplete_input, safety_score > 0),
                "requires_confusion": is_incomplete_input,
                "priority_response": safety_score > 0,
                "romantic_subtext": romantic_score > 0,
                "personal_interest": interest_score > 0
            }

            analysis_result = {
                "sentiment_score": sentiment_score,
                "complexity": complexity,
                "word_count": word_count,
                "char_count": char_count,
                "is_incomplete_input": is_incomplete_input,
                "romantic_score": romantic_score,
                "safety_score": safety_score,
                "interest_score": interest_score,
                "response_guidance": response_guidance
            }

            logger.debug(f"Enhanced sentiment analysis: '{user_message[:50]}...' -> {sentiment_score} complexity:{complexity} incomplete:{is_incomplete_input}")
            return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing message sentiment: {e}")
            return {
                "sentiment_score": 0.0,
                "complexity": "simple",
                "word_count": len(user_message.split()),
                "char_count": len(user_message),
                "is_incomplete_input": False,
                "romantic_score": 0,
                "safety_score": 0,
                "interest_score": 0,
                "response_guidance": {
                    "enforce_brevity": False,
                    "expected_response_words": "8-15",
                    "requires_confusion": False,
                    "priority_response": False,
                    "romantic_subtext": False,
                    "personal_interest": False
                }
            }

    def _get_expected_response_length(self, complexity: str, is_incomplete_input: bool, is_urgent: bool) -> str:
        """Determine expected response length based on input complexity."""
        if is_urgent:
            return "8-15"  # Urgent responses can be longer for clarity

        if complexity == "single_word":
            if is_incomplete_input:
                return "1-5"  # Brief confusion: "Highway?" or "What about it?"
            else:
                return "3-8"  # Simple acknowledgment
        elif complexity == "fragment":
            return "3-10"  # Brief clarifying response
        elif complexity == "simple":
            return "8-15"  # Standard short response
        elif complexity == "moderate":
            return "12-25"  # Can be more detailed
        else:  # complex
            return "15-35"  # Full response allowed
