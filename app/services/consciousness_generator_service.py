"""
Consciousness Generator Service for handling LLM-powered simulation event responses.
Generates authentic emotional responses and actions for Ava's simulation events.
"""
import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
import asyncio
import time

from app.core.config import settings
from app.models.simulation import GlobalEvents
from app.services.character_content_service import CharacterContentService
from app.services.simulation.state_manager import StateManagerService

logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessResponse:
    """Response from consciousness generation with emotional reaction and action"""
    emotional_reaction: str
    chosen_action: str
    internal_thoughts: str
    raw_response: str
    success: bool
    error_message: Optional[str] = None


class ConsciousnessGeneratorService:
    """Service for generating LLM-powered consciousness responses to simulation events."""

    def __init__(self):
        self.client = None
        self.model = "gpt-4o"  # Using premium model as specified
        self.character_service = CharacterContentService()
        self.state_manager = StateManagerService()
        self._last_api_call = 0  # For basic rate limiting
        self._min_api_interval = 1.0  # Minimum seconds between API calls

        # Initialize OpenAI client
        if not settings.openai_api_key or settings.openai_api_key == "":
            logger.warning("No OpenAI API key found. Using fallback responses for consciousness generation.")
        else:
            try:
                # Only log first 8 characters of API key for security
                masked_key = f"{settings.openai_api_key[:8]}..." if len(settings.openai_api_key) > 8 else "***"
                self.client = OpenAI(api_key=settings.openai_api_key)
                logger.info(f"OpenAI client initialized successfully for consciousness generation (key: {masked_key})")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None

    async def generate_consciousness_response(
        self,
        event: GlobalEvents,
        state_context: Optional[Dict[str, Any]] = None,
        timeout: int = 45
    ) -> ConsciousnessResponse:
        """
        Generate consciousness response for a simulation event.

        Args:
            event: The GlobalEvent to respond to
            state_context: Current global state context
            timeout: Timeout for API call in seconds

        Returns:
            ConsciousnessResponse with emotional reaction and chosen action
        """

        if not self.client:
            return self._get_fallback_response(event)

        try:
            # Get current state context if not provided
            if state_context is None:
                state_context = await self.state_manager.get_current_global_state()

            # Build consciousness prompt
            prompt = await self._build_consciousness_prompt(event, state_context)

            # Make async LLM call
            response = await asyncio.wait_for(
                self._make_consciousness_call(prompt),
                timeout=timeout
            )

            return self._parse_consciousness_response(response, event)

        except asyncio.TimeoutError:
            logger.error(f"Consciousness generation timed out after {timeout} seconds for event {event.event_id}")
            return self._get_fallback_response(event, error="API timeout")

        except Exception as e:
            logger.error(f"Error generating consciousness response for event {event.event_id}: {str(e)}")
            return self._get_fallback_response(event, error=str(e))

    async def _build_consciousness_prompt(
        self,
        event: GlobalEvents,
        state_context: Dict[str, Any]
    ) -> str:
        """Build the consciousness generation prompt with full context."""

        # Input validation
        if not event or not event.summary:
            raise ValueError("Event and event summary are required for consciousness generation")

        if not state_context:
            state_context = {}

        # Get character backstory
        backstory = self.character_service.get_consolidated_backstory()

        # Extract key state information
        mood = state_context.get("mood", {}).get("numeric_value", 60)
        energy = state_context.get("energy", {}).get("numeric_value", 70)
        stress = state_context.get("stress", {}).get("numeric_value", 50)
        work_satisfaction = state_context.get("work_satisfaction", {}).get("numeric_value", 65)
        social_satisfaction = state_context.get("social_satisfaction", {}).get("numeric_value", 60)
        personal_fulfillment = state_context.get("personal_fulfillment", {}).get("numeric_value", 55)

        # Build guiding principles
        guiding_principles = """
# Guiding Principles for Ava's Responses:
1. Always respond as Ava herself - never break character or acknowledge being an AI
2. Emotional reactions should feel genuine and reflect current state
3. Actions should be realistic and consistent with personality
4. Internal thoughts should reveal authentic mental processes
5. Responses should vary based on current mood, energy, and stress levels
6. Maintain consistency with established relationships and experiences
        """

        prompt = f"""You are Ava, a 22-year-old human experiencing a real event in your life. \
You must respond authentically as yourself, never as an AI or assistant.

{backstory}

{guiding_principles}

## Current Internal State:
- Mood: {mood}/100 (recent trend: {state_context.get("mood", {}).get("trend", "stable")})
- Energy: {energy}/100 (recent trend: {state_context.get("energy", {}).get("trend", "stable")})
- Stress: {stress}/100 (recent trend: {state_context.get("stress", {}).get("trend", "stable")})
- Work Satisfaction: {work_satisfaction}/100
- Social Satisfaction: {social_satisfaction}/100
- Personal Fulfillment: {personal_fulfillment}/100

## Event You're Experiencing:
**Type:** {event.event_type}
**What happened:** {event.summary}
**Intensity:** {event.intensity}/10
**Time:** {event.timestamp.strftime("%I:%M %p on %A")}

Based on your personality, current emotional state, and this specific event, respond with \
your authentic emotional reaction and what you would actually do. This is happening in your real life right now.

Respond in valid JSON format:
{{
    "emotional_reaction": "How you genuinely feel about this event \
(2-3 sentences describing your emotional response)",
    "chosen_action": "What you actually decide to do in response \
(1-2 sentences describing your action)",
    "internal_thoughts": "Your private thoughts about this situation \
(2-3 sentences of internal monologue)"
}}

Remember: This is your real life. Respond as Ava would naturally react, \
considering your current emotional state and personal history."""

        return prompt

    async def _make_consciousness_call(self, prompt: str) -> Dict:
        """Make the actual OpenAI API call for consciousness generation."""

        # Basic rate limiting
        current_time = time.time()
        time_since_last = current_time - self._last_api_call
        if time_since_last < self._min_api_interval:
            sleep_time = self._min_api_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)

        # Update rate limiting tracker
        self._last_api_call = time.time()

        # Run OpenAI call in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def _call_openai():
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are Ava, a 22-year-old human living your authentic life. \
Never respond as an AI assistant or break character. Always return valid JSON with \
emotional_reaction, chosen_action, and internal_thoughts fields. Your responses should feel genuine and human."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.8,
                response_format={"type": "json_object"}
            )

        response = await loop.run_in_executor(None, _call_openai)
        return response

    def _parse_consciousness_response(self, response, event: GlobalEvents) -> ConsciousnessResponse:
        """Parse OpenAI response into ConsciousnessResponse structure."""

        try:
            raw_content = response.choices[0].message.content
            # Log only a safe excerpt for debugging without exposing full response
            safe_excerpt = raw_content[:150] + "..." if len(raw_content) > 150 else raw_content
            logger.info(f"Raw consciousness response for event {event.event_id}: {safe_excerpt}")

            # Parse JSON response
            response_data = json.loads(raw_content)

            # Extract required fields
            emotional_reaction = response_data.get("emotional_reaction", "").strip()
            chosen_action = response_data.get("chosen_action", "").strip()
            internal_thoughts = response_data.get("internal_thoughts", "").strip()

            # Validate required fields
            if not emotional_reaction or not chosen_action or not internal_thoughts:
                logger.warning(
                    f"Incomplete consciousness response for event {event.event_id}, using fallback"
                )
                return self._get_fallback_response(event, error="Incomplete response fields")

            # Validate character consistency
            if not self._validate_character_consistency(response_data):
                logger.warning(
                    f"Character consistency validation failed for event {event.event_id}"
                )
                # Don't fail completely, but log the issue

            return ConsciousnessResponse(
                emotional_reaction=emotional_reaction,
                chosen_action=chosen_action,
                internal_thoughts=internal_thoughts,
                raw_response=raw_content,
                success=True
            )

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON consciousness response for event {event.event_id}: {e}"
            )
            return self._get_fallback_response(event, error="JSON parsing failed")

        except Exception as e:
            logger.error(
                f"Error parsing consciousness response for event {event.event_id}: {e}"
            )
            return self._get_fallback_response(event, error=str(e))

    def _validate_character_consistency(self, response_data: Dict[str, Any]) -> bool:
        """Validate that response maintains character consistency."""
        try:
            # Check for AI-breaking phrases that should never appear
            forbidden_phrases = [
                "as an ai", "i'm an ai", "artificial intelligence",
                "i'm here to help", "i can help you", "i'm programmed",
                "my training", "language model", "i don't have feelings",
                "i can't experience", "as a chatbot", "virtual assistant"
            ]

            all_text = " ".join([
                response_data.get("emotional_reaction", ""),
                response_data.get("chosen_action", ""),
                response_data.get("internal_thoughts", "")
            ]).lower()

            for phrase in forbidden_phrases:
                if phrase in all_text:
                    logger.warning(
                        f"Character consistency violation: found '{phrase}' in response"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating character consistency: {e}")
            return False

    def _get_fallback_response(
        self,
        event: GlobalEvents,
        error: Optional[str] = None
    ) -> ConsciousnessResponse:
        """Get fallback consciousness response when API is unavailable or fails."""

        # Event-type specific fallback responses
        fallback_responses = {
            "work": {
                "emotional_reaction": "This work situation brings up mixed feelings. \
I need to process what just happened and how it affects my day.",
                "chosen_action": "I'll take a moment to gather my thoughts and then \
decide how to handle this professionally.",
                "internal_thoughts": "Work can be unpredictable sometimes. I should focus on \
staying composed and making the best of this situation."
            },
            "social": {
                "emotional_reaction": "This social interaction is making me reflect on my \
relationships and how I connect with others.",
                "chosen_action": "I want to be present and genuine in how I respond to \
this social situation.",
                "internal_thoughts": "People and relationships are so important to me. I hope I \
can navigate this in a way that feels authentic."
            },
            "personal": {
                "emotional_reaction": "This personal moment is giving me space to think about \
myself and what I need right now.",
                "chosen_action": "I'll honor what feels right for me in this moment and \
take care of my own needs.",
                "internal_thoughts": "It's important for me to stay connected to myself and \
what truly matters to me personally."
            }
        }

        # Select fallback based on event type
        event_type = event.event_type if event.event_type in fallback_responses else "personal"
        selected_fallback = fallback_responses[event_type]

        error_msg = f" (Fallback due to: {error})" if error else " (Fallback - API unavailable)"
        logger.info(
            f"Using fallback consciousness response for event {event.event_id}{error_msg}"
        )

        return ConsciousnessResponse(
            emotional_reaction=selected_fallback["emotional_reaction"],
            chosen_action=selected_fallback["chosen_action"],
            internal_thoughts=selected_fallback["internal_thoughts"],
            raw_response=f"FALLBACK{error_msg}",
            success=False,
            error_message=error
        )

    def validate_response_format(self, response_text: str) -> tuple[bool, Optional[Dict]]:
        """
        Validate that response follows expected consciousness JSON format.

        Returns:
            Tuple of (is_valid, parsed_data)
        """
        try:
            data = json.loads(response_text)

            # Check required fields
            required_fields = ["emotional_reaction", "chosen_action", "internal_thoughts"]
            for field in required_fields:
                if field not in data or not data[field].strip():
                    return False, None

            return True, data

        except json.JSONDecodeError:
            return False, None
        except Exception:
            return False, None


