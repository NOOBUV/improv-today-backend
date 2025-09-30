"""
Enhanced Conversation Service for Story 2.6: Enhanced Conversational Context Integration.
Orchestrates context gathering from simulation, state, and character services.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import json
import time
import uuid
import traceback

from app.core.conversation_config import conversation_config
from app.services.contextual_backstory_service import ContextualBackstoryService
from app.services.conversation_prompt_service import ConversationPromptService, EmotionType
from app.services.state_influence_service import StateInfluenceService, ConversationScenario
from app.services.simulation.state_manager import StateManagerService
from app.services.mood_transition_analyzer import MoodTransitionAnalyzer
from app.services.simple_openai import SimpleOpenAIService, OpenAICoachingResponse, WordUsageStatus
from app.services.dynamic_content_selector import DynamicContentSelector
from app.services.session_state_service import SessionStateService
from app.services.event_selection_service import EventSelectionService
from app.core.config import settings
from openai import OpenAI

logger = logging.getLogger(__name__)


class ConversationPerformanceMonitor:
    """
    Comprehensive performance monitoring for conversation pipeline.
    Tracks timing metrics, generates correlation IDs, and provides detailed logging.
    """

    def __init__(self):
        self.performance_logger = logging.getLogger(f"{__name__}.performance")
        self.alert_thresholds = {
            "total_response_time_ms": 3000,  # <3s total response time requirement
            "consciousness_generation_ms": 2000,  # <2s consciousness generation
            "context_gathering_ms": 1000,  # <1s context processing
            "response_formatting_ms": 500,  # <500ms response formatting

            # Granular conversation response component thresholds
            "context_extraction_ms": 50,  # <50ms to extract simulation context components
            "emotion_selection_ms": 100,  # <100ms for emotion selection with mood awareness
            "prompt_construction_ms": 200,  # <200ms for enhanced prompt building
            "openai_api_call_ms": 1500,  # <1.5s for OpenAI API response
            "response_parsing_ms": 100,  # <100ms for JSON parsing and formatting

            # Context gathering sub-components
            "global_state_retrieval_ms": 100,  # <100ms for database state retrieval
            "event_selection_ms": 200,  # <200ms for event selection service
            "backstory_selection_ms": 150,  # <150ms for backstory content selection
            "sentiment_analysis_ms": 50,  # <50ms for message sentiment analysis
            "state_influence_calculation_ms": 100,  # <100ms for state influence calculations
        }

    def create_conversation_correlation_id(self, user_id: str, conversation_id: str) -> str:
        """Generate unique correlation ID for conversation tracking."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"conv_{user_id[:8]}_{conversation_id[:8]}_{timestamp}_{unique_id}"

    def start_timing_context(self, correlation_id: str, operation: str) -> Dict[str, Any]:
        """Start timing context for a specific operation."""
        start_time = time.time()
        context = {
            "correlation_id": correlation_id,
            "operation": operation,
            "start_time": start_time,
            "start_timestamp": datetime.now(timezone.utc).isoformat(),
            "sub_operations": {}
        }

        self.performance_logger.info(
            f"[{correlation_id}] Starting {operation}",
            extra={
                "correlation_id": correlation_id,
                "operation": operation,
                "event_type": "operation_start",
                "timestamp": context["start_timestamp"]
            }
        )

        return context

    def end_timing_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """End timing context and calculate final metrics."""
        end_time = time.time()
        total_duration_ms = (end_time - context["start_time"]) * 1000

        metrics = {
            "correlation_id": context["correlation_id"],
            "operation": context["operation"],
            "total_duration_ms": round(total_duration_ms, 2),
            "start_timestamp": context["start_timestamp"],
            "end_timestamp": datetime.now(timezone.utc).isoformat(),
            "sub_operations": context["sub_operations"]
        }

        # Check thresholds and alert if exceeded
        self._check_performance_thresholds(metrics)

        self.performance_logger.info(
            f"[{context['correlation_id']}] Completed {context['operation']} in {total_duration_ms:.2f}ms",
            extra={
                **metrics,
                "event_type": "operation_complete"
            }
        )

        return metrics

    def time_sub_operation(self, context: Dict[str, Any], sub_operation: str) -> Dict[str, Any]:
        """Start timing a sub-operation within the main context."""
        start_time = time.time()
        sub_context = {
            "start_time": start_time,
            "start_timestamp": datetime.now(timezone.utc).isoformat()
        }

        self.performance_logger.debug(
            f"[{context['correlation_id']}] Starting sub-operation: {sub_operation}",
            extra={
                "correlation_id": context["correlation_id"],
                "parent_operation": context["operation"],
                "sub_operation": sub_operation,
                "event_type": "sub_operation_start"
            }
        )

        return sub_context

    def end_sub_operation(self, context: Dict[str, Any], sub_operation: str, sub_context: Dict[str, Any], **metadata) -> float:
        """End timing for a sub-operation and record metrics."""
        end_time = time.time()
        duration_ms = (end_time - sub_context["start_time"]) * 1000

        sub_operation_data = {
            "duration_ms": round(duration_ms, 2),
            "start_timestamp": sub_context["start_timestamp"],
            "end_timestamp": datetime.now(timezone.utc).isoformat(),
            **metadata
        }

        context["sub_operations"][sub_operation] = sub_operation_data

        self.performance_logger.debug(
            f"[{context['correlation_id']}] Completed sub-operation {sub_operation} in {duration_ms:.2f}ms",
            extra={
                "correlation_id": context["correlation_id"],
                "parent_operation": context["operation"],
                "sub_operation": sub_operation,
                "duration_ms": duration_ms,
                "event_type": "sub_operation_complete",
                **metadata
            }
        )

        # Check sub-operation thresholds
        threshold_key = f"{sub_operation}_ms"
        if threshold_key in self.alert_thresholds and duration_ms > self.alert_thresholds[threshold_key]:
            self.performance_logger.warning(
                f"[{context['correlation_id']}] {sub_operation} exceeded threshold: {duration_ms:.2f}ms > {self.alert_thresholds[threshold_key]}ms",
                extra={
                    "correlation_id": context["correlation_id"],
                    "sub_operation": sub_operation,
                    "duration_ms": duration_ms,
                    "threshold_ms": self.alert_thresholds[threshold_key],
                    "event_type": "threshold_exceeded"
                }
            )

        return duration_ms

    def _check_performance_thresholds(self, metrics: Dict[str, Any]) -> None:
        """Check if performance metrics exceed defined thresholds."""
        correlation_id = metrics["correlation_id"]
        operation = metrics["operation"]
        total_duration = metrics["total_duration_ms"]

        # Check total response time
        if operation == "enhanced_conversation_response" and total_duration > self.alert_thresholds["total_response_time_ms"]:
            self.performance_logger.warning(
                f"[{correlation_id}] Total conversation response time exceeded threshold: {total_duration:.2f}ms > {self.alert_thresholds['total_response_time_ms']}ms",
                extra={
                    "correlation_id": correlation_id,
                    "total_duration_ms": total_duration,
                    "threshold_ms": self.alert_thresholds["total_response_time_ms"],
                    "event_type": "total_threshold_exceeded",
                    "sub_operations_breakdown": metrics["sub_operations"]
                }
            )

    def log_error_with_context(self, context: Dict[str, Any], error: Exception, operation: str = None) -> None:
        """Log error with full performance context."""
        operation_name = operation or context.get("operation", "unknown")
        correlation_id = context.get("correlation_id", "unknown")

        current_time = time.time()
        duration_so_far = (current_time - context.get("start_time", current_time)) * 1000

        self.performance_logger.error(
            f"[{correlation_id}] Error in {operation_name} after {duration_so_far:.2f}ms: {str(error)}",
            extra={
                "correlation_id": correlation_id,
                "operation": operation_name,
                "error_message": str(error),
                "error_type": type(error).__name__,
                "duration_until_error_ms": duration_so_far,
                "traceback": traceback.format_exc(),
                "event_type": "operation_error",
                "sub_operations_completed": context.get("sub_operations", {})
            }
        )

    def log_detailed_timing_breakdown(self, metrics: Dict[str, Any]) -> None:
        """Log detailed timing breakdown for conversation analysis."""
        correlation_id = metrics["correlation_id"]
        operation = metrics["operation"]
        total_duration = metrics["total_duration_ms"]
        sub_operations = metrics.get("sub_operations", {})

        # Create a detailed breakdown log
        breakdown_lines = [f"[{correlation_id}] DETAILED TIMING BREAKDOWN for {operation} ({total_duration:.2f}ms total):"]

        # Sort sub-operations by duration (slowest first)
        sorted_ops = sorted(sub_operations.items(), key=lambda x: x[1].get("duration_ms", 0), reverse=True)

        for op_name, op_data in sorted_ops:
            duration = op_data.get("duration_ms", 0)
            percentage = (duration / total_duration * 100) if total_duration > 0 else 0

            # Add warning emoji for operations exceeding thresholds
            threshold_key = f"{op_name}_ms"
            warning = ""
            if threshold_key in self.alert_thresholds and duration > self.alert_thresholds[threshold_key]:
                warning = " ⚠️ SLOW"
                exceeded_by = duration - self.alert_thresholds[threshold_key]
                warning += f" (+{exceeded_by:.1f}ms over {self.alert_thresholds[threshold_key]}ms threshold)"

            breakdown_lines.append(f"  • {op_name}: {duration:.2f}ms ({percentage:.1f}%){warning}")

            # Add metadata if available
            metadata = {k: v for k, v in op_data.items() if k not in ['duration_ms', 'start_timestamp', 'end_timestamp']}
            if metadata:
                breakdown_lines.append(f"    └─ {metadata}")

        # Log the complete breakdown
        self.performance_logger.info("\n".join(breakdown_lines), extra={
            "correlation_id": correlation_id,
            "event_type": "detailed_timing_breakdown",
            "total_duration_ms": total_duration,
            "operation": operation,
            "sub_operations_count": len(sub_operations)
        })

    def generate_performance_report(self, correlation_id: str) -> Dict[str, Any]:
        """Generate performance report for analysis (placeholder for future implementation)."""
        return {
            "correlation_id": correlation_id,
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "Performance reporting dashboard to be implemented in future story"
        }


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
        self.mood_transition_analyzer = MoodTransitionAnalyzer()
        self.simple_openai_service = SimpleOpenAIService()
        self.dynamic_content_selector = DynamicContentSelector()
        self.session_state_service = SessionStateService()
        self.event_selection_service = EventSelectionService()
        self.openai_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.performance_monitor = ConversationPerformanceMonitor()

    async def generate_enhanced_response(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        conversation_history: Optional[str] = None,
        personality: str = "friendly_neutral",
        target_vocabulary: Optional[List[Dict]] = None,
        suggested_word: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        fresh_events: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate enhanced conversation response with comprehensive performance monitoring.
        Includes detailed timing breakdown, correlation IDs, and threshold alerting.

        Args:
            user_message: User's message
            user_id: User identifier
            conversation_id: Conversation session identifier
            conversation_history: Existing conversation context
            personality: AI personality style
            target_vocabulary: Target vocabulary words
            suggested_word: Previously suggested word to evaluate
            user_preferences: User-specific preferences
            fresh_events: Pre-selected fresh events to avoid repetition

        Returns:
            Enhanced conversation response with comprehensive performance metrics
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
            parse_context = self.performance_monitor.time_sub_operation(timing_context, "request_parsing")

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
                    max_messages=10
                )

            self.performance_monitor.end_sub_operation(
                timing_context, "request_parsing", parse_context,
                user_message_length=len(user_message),
                history_retrieved=bool(conversation_history),
                fresh_events_provided=len(fresh_events) if fresh_events else 0
            )

            # Sub-operation 2: Context gathering with detailed breakdown
            context_sub_context = self.performance_monitor.time_sub_operation(timing_context, "context_gathering")
            simulation_context = {}

            try:
                simulation_context = await self._gather_simulation_context_with_monitoring(
                    user_message, user_id, conversation_id, user_preferences, fresh_events, timing_context
                )

                self.performance_monitor.end_sub_operation(
                    timing_context, "context_gathering", context_sub_context,
                    context_items_gathered=len(simulation_context),
                    recent_events_count=len(simulation_context.get("recent_events", [])),
                    backstory_chars=simulation_context.get("selected_backstory", {}).get("char_count", 0)
                )

            except Exception as e:
                self.performance_monitor.end_sub_operation(
                    timing_context, "context_gathering", context_sub_context,
                    error=str(e), fallback_triggered=True
                )
                self.performance_monitor.log_error_with_context(timing_context, e, "context_gathering")
                logger.warning(f"[{correlation_id}] Context gathering failed, using fallback: {str(e)}")

            # Sub-operation 3: Consciousness generation (enhanced context-aware response)
            if simulation_context and self.openai_client:
                try:
                    consciousness_context = self.performance_monitor.time_sub_operation(timing_context, "consciousness_generation")

                    response = await self._generate_context_aware_response_with_monitoring(
                        user_message=user_message,
                        simulation_context=simulation_context,
                        conversation_history=conversation_history,
                        personality=personality,
                        suggested_word=suggested_word,
                        timing_context=timing_context
                    )

                    self.performance_monitor.end_sub_operation(
                        timing_context, "consciousness_generation", consciousness_context,
                        response_length=len(response.get("ai_response", "")),
                        enhanced_mode=True,
                        emotion_selected=response.get("simulation_context", {}).get("conversation_emotion")
                    )

                    # Sub-operation 4: Response formatting and session state update
                    formatting_context = self.performance_monitor.time_sub_operation(timing_context, "response_formatting")

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

                    self.performance_monitor.end_sub_operation(
                        timing_context, "response_formatting", formatting_context,
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
            fallback_context = self.performance_monitor.time_sub_operation(timing_context, "fallback_response")

            logger.info(f"[{correlation_id}] Using fallback response generation")
            fallback_response = await self.simple_openai_service.generate_coaching_response(
                message=user_message,
                conversation_history=conversation_history or "",
                personality=personality,
                target_vocabulary=target_vocabulary or [],
                suggested_word=suggested_word
            )

            self.performance_monitor.end_sub_operation(
                timing_context, "fallback_response", fallback_context,
                response_length=len(fallback_response.ai_response),
                fallback_mode=True
            )

            # Sub-operation 6: Fallback response formatting
            formatting_context = self.performance_monitor.time_sub_operation(timing_context, "response_formatting")

            final_metrics = self.performance_monitor.end_timing_context(timing_context)

            # Log detailed timing breakdown for fallback analysis
            self.performance_monitor.log_detailed_timing_breakdown(final_metrics)

            result = {
                "ai_response": fallback_response.ai_response,
                "corrected_transcript": fallback_response.corrected_transcript,
                "word_usage_status": fallback_response.word_usage_status,
                "usage_correctness_feedback": fallback_response.usage_correctness_feedback,
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

            self.performance_monitor.end_sub_operation(
                timing_context, "response_formatting", formatting_context,
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
            if timing_context:
                state_context = self.performance_monitor.time_sub_operation(timing_context, "global_state_retrieval")

            global_state = await self.state_manager_service.get_current_global_state()
            context["global_state"] = global_state

            if timing_context:
                self.performance_monitor.end_sub_operation(
                    timing_context, "global_state_retrieval", state_context,
                    traits_count=len(global_state)
                )

            logger.debug(f"Retrieved global state: {len(global_state)} traits")

            # Sub-sub-operation: Event selection and processing
            if timing_context:
                events_context = self.performance_monitor.time_sub_operation(timing_context, "event_selection")

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

            if timing_context:
                self.performance_monitor.end_sub_operation(
                    timing_context, "event_selection", events_context,
                    events_selected=len(fresh_events_data),
                    pre_selected=fresh_events is not None
                )

            logger.info(f"Fresh events selection: {len(fresh_events_data)} events selected")

            # Sub-sub-operation: Backstory content selection
            if timing_context:
                backstory_context = self.performance_monitor.time_sub_operation(timing_context, "backstory_selection")

            backstory_context_data = await self.contextual_backstory_service.select_relevant_content(
                user_message=user_message,
                max_chars=int(self.config.MAX_BACKSTORY_CHARS * 0.6)
            )
            context["selected_backstory"] = backstory_context_data

            if timing_context:
                self.performance_monitor.end_sub_operation(
                    timing_context, "backstory_selection", backstory_context,
                    chars_selected=backstory_context_data['char_count'],
                    content_types=len(backstory_context_data['content_types'])
                )

            logger.debug(f"Selected backstory: {backstory_context_data['char_count']} chars, types: {backstory_context_data['content_types']}")

            # Sub-sub-operation: Conversation sentiment analysis
            if timing_context:
                sentiment_context = self.performance_monitor.time_sub_operation(timing_context, "sentiment_analysis")

            conversation_sentiment_analysis = self._analyze_message_sentiment(user_message)
            conversation_sentiment_score = conversation_sentiment_analysis.get("sentiment_score", 0.0)

            if timing_context:
                self.performance_monitor.end_sub_operation(
                    timing_context, "sentiment_analysis", sentiment_context,
                    sentiment_score=conversation_sentiment_score,
                    complexity=conversation_sentiment_analysis.get("complexity", "unknown")
                )

            # Sub-sub-operation: State influence calculation
            if timing_context:
                influence_context = self.performance_monitor.time_sub_operation(timing_context, "state_influence_calculation")

            conversation_context = await self.state_influence_service.build_conversation_context(
                user_id=user_id,
                conversation_id=conversation_id,
                scenario=ConversationScenario.CASUAL_CHAT,
                user_preferences=user_preferences,
                conversation_sentiment=conversation_sentiment_score,
                recent_events=context["recent_events"]
            )
            context["conversation_influence"] = conversation_context

            if timing_context:
                self.performance_monitor.end_sub_operation(
                    timing_context, "state_influence_calculation", influence_context,
                    influence_factors=len(conversation_context)
                )

            logger.debug(f"Built conversation context with {len(conversation_context)} influence factors")

            return context

        except Exception as e:
            logger.error(f"Error gathering simulation context: {str(e)}")
            return {}

    async def _gather_simulation_context(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        user_preferences: Optional[Dict[str, Any]] = None,
        fresh_events: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Gather all simulation context for conversation enhancement."""

        context = {}

        try:
            # Get current global state
            global_state = await self.state_manager_service.get_current_global_state()
            context["global_state"] = global_state
            logger.debug(f"Retrieved global state: {len(global_state)} traits")

            # Use provided fresh events or fetch new ones if not provided
            # This prevents Clara from repeating the same events to the same user
            if fresh_events is not None:
                logger.info(f"Using pre-selected fresh events: {len(fresh_events)} events")
                # Use provided fresh events (already filtered for this user)
                fresh_events_data = fresh_events
            else:
                logger.info("Fetching fresh events from event selection service")
                # Fetch fresh, contextually relevant events
                fresh_events_data = await self.event_selection_service.get_contextual_events(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    user_message=user_message,
                    max_events=self.config.MAX_EVENTS_COUNT
                )

            # Convert fresh events back to format expected by rest of system
            context["recent_events"] = [event.get("original_event", event) for event in fresh_events_data]
            context["content_selection_metadata"] = {
                "strategy": "fresh_events_rotation",
                "entities_found": [],
                "total_analyzed": len(fresh_events_data),
                "selected_count": len(fresh_events_data),
                "fresh_events_used": [event.get("id") for event in fresh_events_data]
            }

            logger.info(f"Fresh events selection: {len(fresh_events_data)} events selected")
            logger.debug(f"Selected fresh events: {[event.get('id') for event in fresh_events_data]}")

            # Select relevant backstory content (reduced weight since we now prioritize recent events)
            backstory_context = await self.contextual_backstory_service.select_relevant_content(
                user_message=user_message,
                max_chars=int(self.config.MAX_BACKSTORY_CHARS * 0.6)  # Reduce backstory to make room for events
            )
            context["selected_backstory"] = backstory_context
            logger.debug(f"Selected backstory: {backstory_context['char_count']} chars, types: {backstory_context['content_types']}")

            # Analyze conversation sentiment for mood transitions (enhanced with complexity detection)
            conversation_sentiment_analysis = self._analyze_message_sentiment(user_message)
            conversation_sentiment_score = conversation_sentiment_analysis.get("sentiment_score", 0.0)

            # Build conversation context with state influence including mood transitions
            conversation_context = await self.state_influence_service.build_conversation_context(
                user_id=user_id,
                conversation_id=conversation_id,
                scenario=ConversationScenario.CASUAL_CHAT,
                user_preferences=user_preferences,
                conversation_sentiment=conversation_sentiment_score,
                recent_events=context["recent_events"]
            )
            context["conversation_influence"] = conversation_context
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
        suggested_word: Optional[str] = None,
        timing_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate context-aware response with detailed OpenAI API monitoring."""

        try:
            # Sub-sub-operation: Extract context components
            if timing_context:
                extraction_context = self.performance_monitor.time_sub_operation(timing_context, "context_extraction")

            global_state = simulation_context.get("global_state", {})
            recent_events = simulation_context.get("recent_events", [])
            selected_backstory = simulation_context.get("selected_backstory", {})
            conversation_influence = simulation_context.get("conversation_influence", {})
            content_metadata = simulation_context.get("content_selection_metadata", {})

            mood_transition_data = conversation_influence.get("mood_transition", {})
            blended_mood = mood_transition_data.get("blended_mood_score", 60)
            mood_context = mood_transition_data.get("mood_context", {})

            if timing_context:
                self.performance_monitor.end_sub_operation(
                    timing_context, "context_extraction", extraction_context,
                    global_state_items=len(global_state),
                    recent_events_count=len(recent_events),
                    backstory_chars=selected_backstory.get("char_count", 0),
                    blended_mood=blended_mood
                )

            logger.debug(f"Using intelligent content selection: {content_metadata.get('strategy', 'unknown')}")

            # Sub-sub-operation: Emotion selection with mood awareness
            if timing_context:
                emotion_context = self.performance_monitor.time_sub_operation(timing_context, "emotion_selection")

            conversation_emotion, emotion_reasoning = self.conversation_prompt_service.select_conversation_emotion_with_mood(
                user_message=user_message,
                conversation_history=conversation_history,
                blended_mood_score=blended_mood,
                mood_transition_data=mood_transition_data
            )

            if timing_context:
                self.performance_monitor.end_sub_operation(
                    timing_context, "emotion_selection", emotion_context,
                    selected_emotion=conversation_emotion.value,
                    blended_mood_score=blended_mood
                )

            # Sub-sub-operation: Prompt construction
            if timing_context:
                prompt_context = self.performance_monitor.time_sub_operation(timing_context, "prompt_construction")

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

            if timing_context:
                self.performance_monitor.end_sub_operation(
                    timing_context, "prompt_construction", prompt_context,
                    prompt_length=len(enhanced_prompt),
                    backstory_chars=len(selected_backstory.get("content", "")),
                    events_included=len(recent_events)
                )

            # Sub-sub-operation: OpenAI API call
            if timing_context:
                openai_context = self.performance_monitor.time_sub_operation(timing_context, "openai_api_call")

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": enhanced_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=400,
                temperature=0.7
            )

            ai_response_raw = response.choices[0].message.content

            if timing_context:
                self.performance_monitor.end_sub_operation(
                    timing_context, "openai_api_call", openai_context,
                    model="gpt-4o-mini",
                    prompt_tokens=len(enhanced_prompt.split()),
                    max_tokens=400,
                    response_length=len(ai_response_raw) if ai_response_raw else 0
                )

            # Sub-sub-operation: Response parsing and formatting
            if timing_context:
                parsing_context = self.performance_monitor.time_sub_operation(timing_context, "response_parsing")

            # Parse JSON response from OpenAI
            try:
                ai_response_json = json.loads(ai_response_raw)
                ai_response = ai_response_json.get("message", ai_response_raw)
                response_emotion = ai_response_json.get("emotion", conversation_emotion.value)
            except (json.JSONDecodeError, TypeError):
                ai_response = ai_response_raw
                response_emotion = conversation_emotion.value
                logger.warning(f"Failed to parse JSON response, using raw content: {ai_response_raw[:100]}...")

            # Handle word usage evaluation
            word_usage_status = WordUsageStatus.NOT_USED
            usage_feedback = None

            if suggested_word:
                word_usage_status, usage_feedback = self._evaluate_word_usage(
                    user_message, suggested_word, ai_response
                )

            if timing_context:
                self.performance_monitor.end_sub_operation(
                    timing_context, "response_parsing", parsing_context,
                    json_parsed=ai_response != ai_response_raw,
                    word_evaluation=suggested_word is not None,
                    final_response_length=len(ai_response)
                )

            return {
                "ai_response": ai_response,
                "corrected_transcript": user_message,
                "word_usage_status": word_usage_status,
                "usage_correctness_feedback": usage_feedback,
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

    async def _generate_context_aware_response(
        self,
        user_message: str,
        simulation_context: Dict[str, Any],
        conversation_history: Optional[str] = None,
        personality: str = "friendly_neutral",
        suggested_word: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using enhanced context."""

        try:
            # Extract context components
            global_state = simulation_context.get("global_state", {})
            recent_events = simulation_context.get("recent_events", [])
            selected_backstory = simulation_context.get("selected_backstory", {})
            conversation_influence = simulation_context.get("conversation_influence", {})
            content_metadata = simulation_context.get("content_selection_metadata", {})

            # Extract mood transition data for emotion selection
            mood_transition_data = conversation_influence.get("mood_transition", {})
            blended_mood = mood_transition_data.get("blended_mood_score", 60)
            mood_context = mood_transition_data.get("mood_context", {})

            logger.debug(f"Using intelligent content selection: {content_metadata.get('strategy', 'unknown')}")

            # Determine conversation emotion using mood-aware selection
            conversation_emotion, emotion_reasoning = self.conversation_prompt_service.select_conversation_emotion_with_mood(
                user_message=user_message,
                conversation_history=conversation_history,
                blended_mood_score=blended_mood,
                mood_transition_data=mood_transition_data
            )

            # Construct enhanced prompt using ConversationPromptService with mood transition data
            enhanced_prompt = self.conversation_prompt_service.construct_conversation_prompt_with_mood(
                character_backstory=selected_backstory.get("content", ""),
                user_message=user_message,
                conversation_emotion=conversation_emotion,
                mood_transition_data=mood_transition_data,
                conversation_history=conversation_history
            )

            # Add simulation context to the prompt with intelligent prioritization
            enhanced_prompt += self._build_simulation_context_prompt(
                recent_events, global_state, content_metadata
            )

            # Call OpenAI with enhanced prompt
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": enhanced_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=400,
                temperature=0.7
            )

            ai_response_raw = response.choices[0].message.content

            # Parse JSON response from OpenAI
            try:
                ai_response_json = json.loads(ai_response_raw)
                ai_response = ai_response_json.get("message", ai_response_raw)
                response_emotion = ai_response_json.get("emotion", conversation_emotion.value)
            except (json.JSONDecodeError, TypeError):
                # Fallback if JSON parsing fails
                ai_response = ai_response_raw
                response_emotion = conversation_emotion.value
                logger.warning(f"Failed to parse JSON response, using raw content: {ai_response_raw[:100]}...")

            # Handle word usage evaluation if suggested word provided
            word_usage_status = WordUsageStatus.NOT_USED
            usage_feedback = None

            if suggested_word:
                word_usage_status, usage_feedback = self._evaluate_word_usage(
                    user_message, suggested_word, ai_response
                )

            return {
                "ai_response": ai_response,
                "corrected_transcript": user_message,  # TODO: Add transcript correction
                "word_usage_status": word_usage_status,
                "usage_correctness_feedback": usage_feedback,
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

    def _evaluate_word_usage(
        self,
        user_message: str,
        suggested_word: str,
        ai_response: str
    ) -> tuple[WordUsageStatus, Optional[str]]:
        """Evaluate if the suggested word was used correctly."""

        user_lower = user_message.lower()
        suggested_lower = suggested_word.lower()

        if suggested_lower not in user_lower:
            return WordUsageStatus.NOT_USED, None

        # Simple heuristic for correctness - could be enhanced with NLP
        # For now, assume usage is correct if word appears in context
        return WordUsageStatus.USED_CORRECTLY, None

    async def get_context_summary(
        self,
        user_id: str,
        conversation_id: str
    ) -> Dict[str, Any]:
        """Get summary of available context for debugging/monitoring."""

        try:
            # Get state influence summary
            state_summary = await self.state_influence_service.get_state_influence_summary(
                user_id, conversation_id
            )

            # Get global state
            global_state = await self.state_manager_service.get_current_global_state()

            # Get cache status
            cache_status = self.contextual_backstory_service.get_cache_status()

            return {
                "state_influence": state_summary,
                "global_state_available": len(global_state) > 0,
                "backstory_cache": cache_status,
                "config": {
                    "recent_events_hours": self.config.RECENT_EVENTS_HOURS_BACK,
                    "max_events": self.config.MAX_EVENTS_COUNT,
                    "max_backstory_chars": self.config.MAX_BACKSTORY_CHARS
                }
            }

        except Exception as e:
            logger.error(f"Error getting context summary: {str(e)}")
            return {"error": str(e)}

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

    async def _monitor_mood_analysis_performance(
        self,
        start_time: float,
        mood_analysis_start: float,
        timing_metrics: Dict[str, float]
    ) -> None:
        """Monitor mood analysis performance and log warnings if exceeding thresholds."""
        try:
            mood_analysis_time = (time.time() - mood_analysis_start) * 1000
            timing_metrics["mood_analysis_ms"] = mood_analysis_time

            # Check performance threshold (100ms as specified in Story 2.6 requirements)
            if mood_analysis_time > 100:
                logger.warning(
                    f"Mood analysis exceeded 100ms threshold: {mood_analysis_time:.2f}ms"
                )

            # Add to context gathering time
            timing_metrics["context_gathering_ms"] = timing_metrics.get("context_gathering_ms", 0) + mood_analysis_time

        except Exception as e:
            logger.error(f"Error monitoring mood analysis performance: {e}")


