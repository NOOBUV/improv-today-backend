"""
Enhanced Conversation Service for Story 2.6: Enhanced Conversational Context Integration.
Orchestrates context gathering from simulation, state, and character services.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import json
import time

from app.core.conversation_config import conversation_config
from app.services.contextual_backstory_service import ContextualBackstoryService
from app.services.conversation_prompt_service import ConversationPromptService, EmotionType
from app.services.state_influence_service import StateInfluenceService, ConversationScenario
from app.services.simulation.state_manager import StateManagerService
from app.services.simple_openai import SimpleOpenAIService, OpenAICoachingResponse, WordUsageStatus
from app.core.config import settings
from openai import OpenAI

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
        self.openai_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    async def generate_enhanced_response(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        conversation_history: Optional[str] = None,
        personality: str = "friendly_neutral",
        target_vocabulary: Optional[List[Dict]] = None,
        suggested_word: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate enhanced conversation response with simulation context integration.
        Includes performance monitoring and timing metrics.

        Args:
            user_message: User's message
            user_id: User identifier
            conversation_id: Conversation session identifier
            conversation_history: Existing conversation context
            personality: AI personality style
            target_vocabulary: Target vocabulary words
            suggested_word: Previously suggested word to evaluate
            user_preferences: User-specific preferences

        Returns:
            Enhanced conversation response with simulation context and performance metrics
        """
        start_time = time.time()
        timing_metrics = {}

        try:
            logger.info(f"Generating enhanced response for user {user_id}, conversation {conversation_id}")

            # Fallback handling - if simulation services fail, use original flow
            fallback_response = None
            simulation_context = {}

            try:
                # Step 1: Gather simulation context with timing
                context_start = time.time()
                simulation_context = await self._gather_simulation_context(
                    user_message, user_id, conversation_id, user_preferences
                )
                timing_metrics["context_gathering_ms"] = (time.time() - context_start) * 1000

                # Check if context gathering exceeded threshold
                if timing_metrics["context_gathering_ms"] > self.config.MAX_CONTEXT_PROCESSING_MS:
                    logger.warning(f"Context gathering took {timing_metrics['context_gathering_ms']:.2f}ms (threshold: {self.config.MAX_CONTEXT_PROCESSING_MS}ms)")

            except Exception as e:
                timing_metrics["context_gathering_ms"] = (time.time() - context_start) * 1000
                logger.warning(f"Simulation context gathering failed after {timing_metrics['context_gathering_ms']:.2f}ms, using fallback: {str(e)}")
                # Continue with fallback mode
                pass

            # Step 2: Generate response with enhanced context or fallback
            if simulation_context and self.openai_client:
                try:
                    response_start = time.time()
                    response = await self._generate_context_aware_response(
                        user_message=user_message,
                        simulation_context=simulation_context,
                        conversation_history=conversation_history,
                        personality=personality,
                        suggested_word=suggested_word
                    )
                    timing_metrics["response_generation_ms"] = (time.time() - response_start) * 1000
                    timing_metrics["total_response_time_ms"] = (time.time() - start_time) * 1000

                    # Add performance metrics to response
                    response["performance_metrics"] = timing_metrics
                    response["enhanced_mode"] = True

                    logger.info(f"Enhanced response generated in {timing_metrics['total_response_time_ms']:.2f}ms")
                    return response
                except Exception as e:
                    timing_metrics["response_generation_ms"] = (time.time() - response_start) * 1000
                    logger.warning(f"Context-aware response generation failed after {timing_metrics['response_generation_ms']:.2f}ms: {str(e)}")
                    # Fall through to fallback

            # Step 3: Fallback to original SimpleOpenAIService
            logger.info("Using fallback response generation")
            fallback_start = time.time()
            fallback_response = await self.simple_openai_service.generate_coaching_response(
                message=user_message,
                conversation_history=conversation_history or "",
                personality=personality,
                target_vocabulary=target_vocabulary or [],
                suggested_word=suggested_word
            )
            timing_metrics["fallback_response_ms"] = (time.time() - fallback_start) * 1000
            timing_metrics["total_response_time_ms"] = (time.time() - start_time) * 1000

            result = {
                "ai_response": fallback_response.ai_response,
                "corrected_transcript": fallback_response.corrected_transcript,
                "word_usage_status": fallback_response.word_usage_status,
                "usage_correctness_feedback": fallback_response.usage_correctness_feedback,
                "simulation_context": simulation_context,
                "selected_backstory_types": [],
                "fallback_mode": True,
                "enhanced_mode": False,
                "performance_metrics": timing_metrics
            }

            logger.info(f"Fallback response generated in {timing_metrics['total_response_time_ms']:.2f}ms")
            return result

        except Exception as e:
            timing_metrics["total_response_time_ms"] = (time.time() - start_time) * 1000
            timing_metrics["error"] = str(e)
            logger.error(f"Error in enhanced conversation generation after {timing_metrics['total_response_time_ms']:.2f}ms: {str(e)}")
            raise

    async def _gather_simulation_context(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Gather all simulation context for conversation enhancement."""

        context = {}

        try:
            # Get current global state
            global_state = await self.state_manager_service.get_current_global_state()
            context["global_state"] = global_state
            logger.debug(f"Retrieved global state: {len(global_state)} traits")

            # Get recent simulation events
            recent_events = await self.state_manager_service.get_recent_events(
                hours_back=self.config.RECENT_EVENTS_HOURS_BACK,
                max_count=self.config.MAX_EVENTS_COUNT
            )
            context["recent_events"] = recent_events
            logger.debug(f"Retrieved {len(recent_events)} recent events")

            # Select relevant backstory content
            backstory_context = await self.contextual_backstory_service.select_relevant_content(
                user_message=user_message,
                max_chars=self.config.MAX_BACKSTORY_CHARS
            )
            context["selected_backstory"] = backstory_context
            logger.debug(f"Selected backstory: {backstory_context['char_count']} chars, types: {backstory_context['content_types']}")

            # Build conversation context with state influence
            conversation_context = await self.state_influence_service.build_conversation_context(
                user_id=user_id,
                conversation_id=conversation_id,
                scenario=ConversationScenario.CASUAL_CHAT,
                user_preferences=user_preferences
            )
            context["conversation_influence"] = conversation_context
            logger.debug(f"Built conversation context with {len(conversation_context)} influence factors")

            return context

        except Exception as e:
            logger.error(f"Error gathering simulation context: {str(e)}")
            return {}

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

            # Determine conversation emotion based on state and user message
            current_mood = global_state.get("mood", {}).get("numeric_value", 60)
            stress_level = global_state.get("stress", {}).get("numeric_value", 50)

            conversation_emotion, emotion_reasoning = self.conversation_prompt_service.determine_emotion_from_context(
                user_message=user_message,
                conversation_history=conversation_history,
                global_mood="stressed" if stress_level > 60 else "neutral"
            )

            # Construct enhanced prompt using ConversationPromptService
            enhanced_prompt = self.conversation_prompt_service.construct_conversation_prompt(
                character_backstory=selected_backstory.get("content", ""),
                user_message=user_message,
                conversation_emotion=conversation_emotion,
                global_mood="stressed" if stress_level > 60 else "neutral",
                stress_level=int(stress_level),
                conversation_history=conversation_history
            )

            # Add simulation context to the prompt
            enhanced_prompt += self._build_simulation_context_prompt(recent_events, global_state)

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

            ai_response = response.choices[0].message.content

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
                    "global_mood": current_mood,
                    "stress_level": stress_level,
                    "selected_content_types": selected_backstory.get("content_types", []),
                    "conversation_emotion": conversation_emotion.value,
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
        global_state: Dict[str, Any]
    ) -> str:
        """Build simulation context section for the prompt."""

        if not recent_events and not global_state:
            return ""

        context_parts = []

        if recent_events:
            context_parts.append("\n\nRECENT LIFE EVENTS:")
            for event in recent_events[:3]:  # Limit to most recent 3
                hours_ago = event.get("hours_ago", 0)
                if hours_ago < 24:
                    context_parts.append(f"- {event.get('summary', '')} ({int(hours_ago)} hours ago)")

        if global_state:
            mood = global_state.get("mood", {}).get("numeric_value", 60)
            stress = global_state.get("stress", {}).get("numeric_value", 50)
            energy = global_state.get("energy", {}).get("numeric_value", 70)

            context_parts.append(f"\n\nCURRENT STATE:")
            context_parts.append(f"- Mood: {mood}/100, Stress: {stress}/100, Energy: {energy}/100")

        context_parts.append("\n\nNaturally weave these recent experiences and your current state into the conversation when relevant, but don't force it. Respond authentically to what the user is saying.")

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