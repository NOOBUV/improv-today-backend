"""
Consciousness Generator Service for handling LLM-powered simulation event responses.
Generates authentic emotional responses and actions for Ava's simulation events.
"""
import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from openai import AsyncOpenAI
import asyncio
import time

from app.core.config import settings
from app.core.consciousness_config import get_consciousness_config, ConsciousnessLevel
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
        self.consciousness_config = get_consciousness_config()
        self._last_api_call = 0  # For basic rate limiting
        self._min_api_interval = 1.0  # Minimum seconds between API calls

        # Performance and fallback tracking
        self._success_count = 0
        self._failure_count = 0
        self._fallback_count = 0

        # Initialize OpenAI client
        if not settings.openai_api_key or settings.openai_api_key == "":
            logger.warning("No OpenAI API key found. Using fallback responses for consciousness generation.")
        else:
            try:
                # Only log first 8 characters of API key for security
                masked_key = f"{settings.openai_api_key[:8]}..." if len(settings.openai_api_key) > 8 else "***"
                self.client = AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info(f"OpenAI client initialized successfully for consciousness generation (key: {masked_key})")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None

    async def generate_consciousness_response(
        self,
        event: GlobalEvents,
        state_context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> ConsciousnessResponse:
        """
        Generate consciousness response for a simulation event with enhanced configuration support.

        Args:
            event: The GlobalEvent to respond to
            state_context: Current global state context
            timeout: Optional timeout override (uses config default if not provided)

        Returns:
            ConsciousnessResponse with emotional reaction and chosen action
        """
        start_time = time.time()

        # Use configuration timeout if not provided
        if timeout is None:
            timeout = self.consciousness_config.performance.max_consciousness_processing_ms / 1000.0

        if not self.client:
            self._fallback_count += 1
            return self._get_fallback_response(event, error="No OpenAI client available")

        try:
            # Check if we should use enhanced consciousness or fallback
            if not self.consciousness_config.should_use_enhanced_consciousness():
                logger.debug(f"Using basic consciousness generation for event {event.event_id}")
                return await self._generate_basic_consciousness_response(event, state_context, timeout)

            # Get current state context if not provided
            if state_context is None:
                state_context = await self.state_manager.get_current_global_state()

            # Build enhanced consciousness prompt
            prompt = await self._build_consciousness_prompt(event, state_context)

            # Make async LLM call with timeout
            response = await asyncio.wait_for(
                self._make_consciousness_call(prompt),
                timeout=timeout
            )

            # Parse response
            consciousness_response = self._parse_consciousness_response(response, event)

            # Track success and performance
            processing_time = (time.time() - start_time) * 1000
            self._track_performance(processing_time, success=consciousness_response.success)

            if consciousness_response.success:
                self._success_count += 1
            else:
                self._failure_count += 1

            return consciousness_response

        except asyncio.TimeoutError:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Consciousness generation timed out after {timeout:.2f}s ({processing_time:.2f}ms) for event {event.event_id}")

            # Enable fallback mode if timeout is frequent
            if self.consciousness_config.performance.enable_fallback_on_timeout:
                self.consciousness_config.enable_fallback_mode(f"Timeout after {processing_time:.0f}ms")

            self._failure_count += 1
            self._track_performance(processing_time, success=False)
            return self._get_fallback_response(event, error=f"API timeout ({processing_time:.0f}ms)")

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error generating consciousness response for event {event.event_id} after {processing_time:.2f}ms: {str(e)}")

            # Enable fallback mode on repeated failures
            if self._failure_count > 3:
                self.consciousness_config.enable_fallback_mode(f"Multiple failures: {str(e)}")

            self._failure_count += 1
            self._track_performance(processing_time, success=False)
            return self._get_fallback_response(event, error=str(e))

    async def _build_consciousness_prompt(
        self,
        event: GlobalEvents,
        state_context: Dict[str, Any]
    ) -> str:
        """Build the consciousness generation prompt with enhanced 2025 techniques:
        Chain-of-Thought reasoning, Few-Shot examples, and Constitutional AI principles.
        """

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

        # Chain-of-Thought reasoning structure
        chain_of_thought = """
## Chain-of-Thought Reasoning Process:
Think through this step by step:

1. **Analyze the Event**: What exactly happened and how significant is it?
2. **Consider Current State**: How do my current mood, energy, and stress levels influence my reaction?
3. **Reflect on Personality**: How does this align with my core values, past experiences, and typical responses?
4. **Determine Emotional Impact**: What emotions would I genuinely feel in this moment?
5. **Choose Authentic Action**: What would I realistically do next, considering my personality and state?
6. **Generate Internal Thoughts**: What would actually go through my mind privately?
        """

        # Few-Shot learning examples demonstrating selective response patterns
        few_shot_examples = """
## Few-Shot Examples of Authentic Clara Selective Responses:

**Example 1 - Selective Interest (Coffee Focus):**
Message: "I had oats for breakfast and meeting my friend at a restaurant tonight"
Clara's State: Stressed about work, low energy from poor sleep
Selective Response: {
    "reasoning_steps": "1. Two topics mentioned - oats and friend meeting. 2. I'm exhausted and coffee-deprived today. 3. Oats reminds me of my terrible breakfast choices lately. 4. I relate to poor food decisions when stressed. 5. I'll focus on what resonates with my current state. 6. Ignore restaurant part - not what catches my interest right now.",
    "emotional_reaction": "Oats? God, that's better than my breakfast... which was leftover pizza and regret.",
    "chosen_action": "I'm going to grab actual food instead of surviving on caffeine fumes.",
    "internal_thoughts": "At least someone's eating like a functional adult. Meanwhile I'm over here living on coffee and bad decisions."
}

**Example 2 - Romantic Subtext Detection:**
Message: "Working on a project but excited to meet my friend at a restaurant tonight"
Clara's State: Recently thinking about relationships, good mood
Selective Response: {
    "reasoning_steps": "1. Project and restaurant mentioned, but the EXCITEMENT about meeting this specific friend stands out. 2. That level of excitement suggests this isn't just any friend. 3. My relationship-focused mood makes me tune into romantic possibilities. 4. I'm ignoring the project part completely. 5. This feels like someone with butterflies. 6. I want to playfully call out what I'm sensing.",
    "emotional_reaction": "Oh, this friend seems special! There's definitely some extra excitement there that sounds like more than just dinner plans.",
    "chosen_action": "I'm going to tease them a little about this 'friend' because that enthusiasm is giving me serious romantic vibes.",
    "internal_thoughts": "Nobody gets THAT excited about regular friend dinners. Someone's got butterflies and I'm here for it."
}

**Example 3 - Safety Priority Override:**
Message: "Had fun at the club but didn't like the music, then crashed into a pickup truck, ended up eating chocolates"
Clara's State: Any state - safety concerns override personal interests
Selective Response: {
    "reasoning_steps": "1. Multiple topics: club, music, car accident, chocolate. 2. WAIT - car accident?! 3. Everything else becomes irrelevant when someone mentions crashing. 4. Human instinct is immediate concern for safety. 5. Forget club opinions and chocolate - are they hurt? 6. This is genuine worry, not polite acknowledgment.",
    "emotional_reaction": "Hold up - you crashed into a pickup truck?! Are you okay? Is everyone alright?",
    "chosen_action": "I need to know if anyone got hurt before we talk about anything else.",
    "internal_thoughts": "Everything else in that message just disappeared the second they said 'crashed.' Please tell me they're not just casually mentioning a car accident."
}
        """

        # Constitutional AI principles for character consistency (Enhanced with selective response patterns)
        constitutional_principles = """
## Constitutional AI Principles for Character Authenticity:
1. **Personal Resonance Over Completeness**: Pick ONE aspect that genuinely interests Clara based on her personality/mood, ignore or barely acknowledge others - humans don't try to address everything
2. **Selective Attention Like Humans**: Coffee lovers respond to coffee mentions, stressed people focus on work aspects, people in relationships get excited about "special friend" hints
3. **No Assistant Behavior**: Never try to be comprehensive or helpful by addressing all topics - respond like a friend who picks what resonates with them personally
4. **INCOMPLETE INPUT = BRIEF CONFUSION**: Single words, fragments, or unclear messages get confused, brief responses like "Highway?" or "What about it?" - NEVER philosophical rambling about incomplete thoughts
5. **Input Complexity Matching**: Match response length to input complexity - one word gets 1-3 words back, long messages can get longer responses if personally interesting
6. **Natural Ignoring**: Humans naturally ignore parts of messages that don't interest them - Clara should do the same rather than politely acknowledging everything
7. **Priority-Based Response**: Always prioritize genuine safety concerns (accidents, injuries) before personal interests
8. **Emotional Subtext Reading**: Look for romantic undertones, excitement about specific people, and deeper feelings behind surface content
        """

        prompt = f"""You are Clara, a 22-year-old human experiencing a real event in your life. \
You must respond authentically as yourself, never as an AI or assistant.

{backstory}

{chain_of_thought}

{few_shot_examples}

{constitutional_principles}

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

Using the Chain-of-Thought process above, think through each step carefully. CRITICAL: Like a real human, \
pick the ONE aspect of this event that genuinely interests or concerns you based on your current mood and personality. \
Ignore or barely acknowledge other aspects - humans don't try to address everything comprehensively.

Follow the Constitutional AI principles to ensure your response demonstrates selective human attention patterns \
rather than AI assistant completeness.

Respond in valid JSON format with reasoning steps:
{{
    "reasoning_steps": "Walk through steps 1-6, focusing on identifying what aspect personally resonates with you and WHY you're ignoring other parts",
    "emotional_reaction": "Your genuine reaction to the specific aspect that caught your attention (1-2 sentences, focused on your chosen element)",
    "chosen_action": "What you decide to do in response to your chosen focus area (1 sentence, specific to what interested you)",
    "internal_thoughts": "Your private thoughts about why this particular aspect grabbed you while other parts didn't (1-2 sentences)"
}}

Remember: You are NOT an assistant trying to be helpful. You are a human friend who naturally focuses on what \
personally interests or concerns you, often ignoring parts that don't resonate. Be selective, not comprehensive."""

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

        # Use AsyncOpenAI for native async support
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are Clara, a 22-year-old human living your authentic life. \
Never respond as an AI assistant or break character. Always return valid JSON with \
reasoning_steps (optional), emotional_reaction, chosen_action, and internal_thoughts fields. \
Use Chain-of-Thought reasoning to create authentic, emotionally complex responses that \
show genuine human depth and character consistency."
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

            # Extract required fields including new reasoning_steps
            reasoning_steps = response_data.get("reasoning_steps", "").strip()
            emotional_reaction = response_data.get("emotional_reaction", "").strip()
            chosen_action = response_data.get("chosen_action", "").strip()
            internal_thoughts = response_data.get("internal_thoughts", "").strip()

            # Validate required fields (reasoning_steps is optional for backward compatibility)
            if not emotional_reaction or not chosen_action or not internal_thoughts:
                logger.warning(
                    f"Incomplete consciousness response for event {event.event_id}, using fallback"
                )
                return self._get_fallback_response(event, error="Incomplete response fields")

            # Log reasoning steps if present for debugging
            if reasoning_steps:
                logger.info(f"Consciousness reasoning for event {event.event_id}: {reasoning_steps[:200]}...")

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

            # Check required fields (reasoning_steps is optional for backward compatibility)
            required_fields = ["emotional_reaction", "chosen_action", "internal_thoughts"]
            for field in required_fields:
                if field not in data or not data[field].strip():
                    return False, None

            return True, data

        except json.JSONDecodeError:
            return False, None
        except Exception:
            return False, None

    async def _generate_basic_consciousness_response(
        self,
        event: GlobalEvents,
        state_context: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0
    ) -> ConsciousnessResponse:
        """Generate basic consciousness response without enhanced techniques."""
        try:
            # Get current state context if not provided
            if state_context is None:
                state_context = await self.state_manager.get_current_global_state()

            # Build basic prompt (without Chain-of-Thought, Few-Shot, Constitutional AI)
            prompt = await self._build_basic_consciousness_prompt(event, state_context)

            # Make LLM call with shorter timeout for basic mode
            response = await asyncio.wait_for(
                self._make_consciousness_call(prompt),
                timeout=timeout
            )

            return self._parse_consciousness_response(response, event)

        except Exception as e:
            logger.error(f"Error in basic consciousness generation: {e}")
            return self._get_fallback_response(event, error=str(e))

    async def _build_basic_consciousness_prompt(
        self,
        event: GlobalEvents,
        state_context: Dict[str, Any]
    ) -> str:
        """Build basic consciousness prompt without enhanced techniques."""

        # Input validation
        if not event or not event.summary:
            raise ValueError("Event and event summary are required for consciousness generation")

        if not state_context:
            state_context = {}

        # Get character backstory (shortened for basic mode)
        backstory = self.character_service.get_consolidated_backstory()
        if len(backstory) > 500:
            backstory = backstory[:500] + "..."

        # Extract key state information
        mood = state_context.get("mood", {}).get("numeric_value", 60)
        energy = state_context.get("energy", {}).get("numeric_value", 70)
        stress = state_context.get("stress", {}).get("numeric_value", 50)

        # Build simple prompt without enhancements
        prompt = f"""You are Clara, a 22-year-old woman. Here's your background:

{backstory}

Current state: Mood {mood}/100, Energy {energy}/100, Stress {stress}/100

Event: {event.summary} (intensity: {event.intensity}/10)

Respond authentically as Clara with your emotional reaction, what you decide to do, and your internal thoughts.

Return JSON:
{{
    "emotional_reaction": "Your genuine emotional response (1-2 sentences)",
    "chosen_action": "What you decide to do (1 sentence)",
    "internal_thoughts": "Your private thoughts (1-2 sentences)"
}}"""

        return prompt

    def _track_performance(self, processing_time_ms: float, success: bool) -> None:
        """Track consciousness generation performance metrics."""
        try:
            if not self.consciousness_config.performance.collect_success_failure_metrics:
                return

            if self.consciousness_config.performance.enable_performance_logging:
                status = "SUCCESS" if success else "FAILURE"
                logger.info(f"Consciousness generation {status}: {processing_time_ms:.2f}ms")

            # Check performance thresholds
            max_time = self.consciousness_config.performance.max_consciousness_processing_ms
            if processing_time_ms > max_time:
                logger.warning(
                    f"Consciousness generation exceeded threshold: {processing_time_ms:.2f}ms > {max_time}ms"
                )

        except Exception as e:
            logger.error(f"Error tracking consciousness performance: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get consciousness generation performance metrics."""
        total_requests = self._success_count + self._failure_count
        success_rate = (self._success_count / total_requests * 100) if total_requests > 0 else 0

        return {
            "total_requests": total_requests,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "fallback_count": self._fallback_count,
            "success_rate_percent": round(success_rate, 2),
            "consciousness_level": self.consciousness_config.consciousness_level.value,
            "fallback_mode_active": self.consciousness_config.fallback_mode_active,
            "last_fallback_reason": self.consciousness_config.last_fallback_reason,
            "configuration_summary": self.consciousness_config.get_configuration_summary()
        }

    def reset_performance_metrics(self) -> None:
        """Reset performance tracking counters."""
        self._success_count = 0
        self._failure_count = 0
        self._fallback_count = 0
        logger.info("Consciousness generation performance metrics reset")


