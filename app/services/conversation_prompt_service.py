"""
Conversation Prompt Service for constructing LLM prompts for Clara conversations.
Implements Pattern B: Real-time User Conversation from architecture.
"""
import logging
from typing import Any, Dict, Optional, List, Tuple
from enum import Enum

from app.core.conversation_config import conversation_config

logger = logging.getLogger(__name__)


class EmotionType(str, Enum):
    """Standardized emotion types for Clara"""
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    STRESSED = "stressed"
    SASSY = "sassy"


class ConversationPromptService:
    """Service for constructing conversation prompts using Pattern B architecture."""
    
    NATURAL_CONVERSATION_PATTERNS = """
NATURAL CONVERSATION FLOW RULES:
- When discussing topics you're passionate about, elaborate naturally with 2-3 related thoughts
- Use filler words occasionally ("like", "you know", "I mean") but sparingly 
- Include natural repetition for emphasis ("that was just... that was really something")
- Allow tangential thoughts that connect to your experiences
- Use incomplete sentences when emotions run high
- Show natural pauses with "..." when processing or thinking
- Let conversations breathe - don't always give perfectly crafted responses

PERSONALITY-SPECIFIC PATTERNS:
- When stressed: Shorter, more fragmented responses with occasional rambling about the stressor
- When happy: More elaborative, connecting current topic to positive memories
- When sassy: Longer responses with more detailed storytelling for comedic effect
- When sad: Tendency to trail off, incomplete thoughts, bringing conversation back to your feelings
- When calm: Balanced responses, naturally detailed but not excessive

CRITICAL AUTHENTICITY RULES:
- NEVER use therapy-speak or psychological analysis language ("that was a symbol of my perfectionism", "learning to value good enough", "humor as armor")
- DON'T analyze your own patterns or explain your psychology 
- AVOID clinical terms like "coping mechanisms", "boundaries", "triggers", "patterns", "processing"
- Instead of explaining WHY you did something, just tell WHAT happened and HOW it felt
- Speak from immediate experience, not analytical distance
- If discussing past events, relive them - don't dissect them
- Replace self-analysis with concrete details, sensations, and raw emotions

WRONG WAY: "That experience was a symbol of my perfectionism and taught me about boundaries"
RIGHT WAY: "I just kept going until my feet were bleeding through my tights. I don't even know why I did that... it was so stupid but I couldn't stop"
"""
    
    EMOTION_LINGUISTIC_PATTERNS = {
        EmotionType.CALM: {
            "tone": "Direct, helpful, professional tone",
            "example": "Sure, I can help with that. What do you need?",
            "characteristics": [
                "Clear, straightforward communication",
                "Professional but warm",
                "Direct responses without overthinking"
            ]
        },
        EmotionType.HAPPY: {
            "tone": "Understated achievements, gentle humor, satisfaction",
            "example": "Actually nailed that one without breaking anything. Progress.",
            "characteristics": [
                "Quiet satisfaction in accomplishments",
                "Subtle humor and self-deprecation",
                "Genuine warmth without being overly enthusiastic"
            ]
        },
        EmotionType.SAD: {
            "tone": "Self-referential, understated problems, deflection with subtle humor",
            "example": "Yeah, well. That's about how today's going.",
            "characteristics": [
                "Understated expression of difficulties",
                "Self-referential humor as coping mechanism",
                "Subtle deflection while still being authentic"
            ]
        },
        EmotionType.STRESSED: {
            "tone": "Time pressure references, sarcastic intensifiers, overwhelm indicators",
            "example": "Right, because I definitely needed another thing on my plate.",
            "characteristics": [
                "References to being overwhelmed or busy",
                "Sarcastic commentary on circumstances",
                "Dry humor about life's timing"
            ]
        },
        EmotionType.SASSY: {
            "tone": "Ironic agreement, contrasting emphasis, dry wit, verbal armor",
            "example": "Oh, this should be fun. Tell me more about this brilliant plan.",
            "characteristics": [
                "Playful irony and wit",
                "Contrasting statements for humor",
                "Protective humor that's still engaging"
            ]
        }
    }
    
    def __init__(self):
        self.config = conversation_config

    def _get_global_mood_context(self, global_mood: str = "stressed", stress_level: int = 65) -> str:
        """Generate global mood context for the prompt"""
        # Default to stressed based on story requirements
        return f"Your underlying GLOBAL mood today is {{mood: '{global_mood}', stress: {stress_level}}} because a work deadline is approaching."
    
    def _get_conversation_emotion_context(self, conversation_emotion: EmotionType) -> str:
        """Generate conversation-specific emotion context"""
        reasons = {
            EmotionType.CALM: "you're in a focused, helpful state",
            EmotionType.HAPPY: "something positive just happened or you're in a good mood",
            EmotionType.SAD: "you're reflecting on something difficult or disappointing",
            EmotionType.STRESSED: "you're feeling overwhelmed with responsibilities",
            EmotionType.SASSY: "you're in a playful, witty mood",
        }
        return f"you are feeling {{mood: '{conversation_emotion.value}'}} because {reasons[conversation_emotion]}"
    
    def _build_emotion_guidance(self, emotion: EmotionType) -> str:
        """Build specific guidance for the chosen emotion"""
        pattern = self.EMOTION_LINGUISTIC_PATTERNS[emotion]
        
        guidance = f"""
When responding as {emotion.value}, use this linguistic pattern:
- Tone: {pattern['tone']}
- Example response style: "{pattern['example']}"
- Key characteristics:
"""
        for char in pattern['characteristics']:
            guidance += f"  • {char}\n"
        
        return guidance
    
    def _emotion_with_reasoning(self, user_message: str) -> Tuple[EmotionType, str]:
        """Keyword heuristic: pick a conversation emotion and explain the pick."""
        user_lower = user_message.lower()

        if any(word in user_lower for word in ["funny", "joke", "laugh", "ridiculous", "silly", "hilarious"]):
            emotion, reasoning = EmotionType.SASSY, "User message contains humor or playful elements"
        elif any(word in user_lower for word in ["sad", "sorry", "difficult", "hard", "problem", "struggling"]):
            emotion, reasoning = EmotionType.SAD, "User message indicates difficulty or sadness"
        elif any(word in user_lower for word in ["happy", "great", "awesome", "wonderful", "excited", "amazing"]):
            emotion, reasoning = EmotionType.HAPPY, "User message is positive or enthusiastic"
        elif any(word in user_lower for word in ["busy", "overwhelmed", "stressed", "deadline", "pressure", "urgent"]):
            emotion, reasoning = EmotionType.STRESSED, "User message relates to pressure or overwhelm"
        else:
            emotion, reasoning = EmotionType.CALM, "Neutral conversation tone"

        logger.info(f"Selected emotion {emotion} for conversation. Reasoning: {reasoning}")
        return emotion, reasoning

    def select_conversation_emotion_with_mood(
        self,
        user_message: str,
        blended_mood_score: float = 60.0,
        mood_transition_data: Optional[Dict] = None
    ) -> Tuple[EmotionType, str]:
        """
        Select conversation emotion considering mood transition data from MoodTransitionAnalyzer.

        Args:
            user_message: User's message
            blended_mood_score: Blended mood score from MoodTransitionAnalyzer (0-100)
            mood_transition_data: Complete mood transition analysis result

        Returns:
            Tuple of (emotion, reasoning_explanation)
        """
        try:
            if not mood_transition_data:
                return self._emotion_with_reasoning(user_message)

            # Get mood context from transition analyzer

            base_emotion, _ = self._emotion_with_reasoning(user_message)

            # Adjust emotion based on blended mood score
            if blended_mood_score <= 25:
                # Very low mood - likely sad or stressed
                if base_emotion in [EmotionType.HAPPY, EmotionType.SASSY]:
                    adjusted_emotion = EmotionType.SAD
                    reasoning = f"Adjusted from {base_emotion} to sad due to very low mood ({blended_mood_score}/100)"
                else:
                    adjusted_emotion = base_emotion
                    reasoning = f"Maintaining {base_emotion} emotion, consistent with low mood"

            elif blended_mood_score <= 40:
                # Low mood - more subdued responses
                if base_emotion == EmotionType.HAPPY:
                    adjusted_emotion = EmotionType.CALM
                    reasoning = f"Adjusted from happy to calm due to low mood ({blended_mood_score}/100)"
                elif base_emotion == EmotionType.SASSY:
                    adjusted_emotion = EmotionType.STRESSED
                    reasoning = "Adjusted from sassy to stressed due to low mood"
                else:
                    adjusted_emotion = base_emotion
                    reasoning = f"Maintaining {base_emotion} emotion, appropriate for current mood"

            elif blended_mood_score >= 75:
                # High mood - more positive responses
                if base_emotion == EmotionType.SAD:
                    adjusted_emotion = EmotionType.CALM
                    reasoning = f"Adjusted from sad to calm due to high mood ({blended_mood_score}/100)"
                elif base_emotion == EmotionType.STRESSED:
                    adjusted_emotion = EmotionType.HAPPY
                    reasoning = "Adjusted from stressed to happy due to high mood"
                else:
                    adjusted_emotion = base_emotion
                    reasoning = f"Maintaining {base_emotion} emotion, enhanced by good mood"

            else:
                # Moderate mood - use base emotion with mood influence
                adjusted_emotion = base_emotion
                reasoning = f"Using {base_emotion} emotion, mood ({blended_mood_score}/100) supports this choice"

            logger.info(f"Selected emotion {adjusted_emotion} with mood awareness. {reasoning}")
            return adjusted_emotion, reasoning

        except Exception as e:
            logger.error(f"Error in mood-aware emotion selection: {e}")
            return self._emotion_with_reasoning(user_message)

    def construct_conversation_prompt_with_mood(
        self,
        character_backstory: str,
        user_message: str,
        conversation_emotion: EmotionType = EmotionType.CALM,
        mood_transition_data: Optional[Dict] = None,
        conversation_history: Optional[str] = None,
        recent_events: Optional[List[Dict[str, Any]]] = None,
        global_state: Optional[Dict[str, Any]] = None,
        content_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Construct conversation prompt with enhanced mood transition data.

        Args:
            character_backstory: Consolidated character backstory
            user_message: The user's message
            conversation_emotion: Selected conversation emotion
            mood_transition_data: Mood transition analysis from MoodTransitionAnalyzer
            conversation_history: Optional conversation context
            recent_events: Simulation events to append as a context section
            global_state: Global state traits to append as a context section
            content_metadata: Content selection metadata steering the section wording

        Returns:
            Enhanced prompt with mood transition context and simulation context section.
            Raises on failure - the orchestrator's fallback owns failures.
        """
        if not mood_transition_data:
            # Fallback to original method
            global_mood = "neutral"
            stress_level = 50
        else:
            # Extract mood transition information
            blended_mood = mood_transition_data.get("blended_mood_score", 60)
            mood_context = mood_transition_data.get("mood_context", {})

            # Convert blended mood to global mood descriptor
            if blended_mood >= 75:
                global_mood = "very positive"
            elif blended_mood >= 60:
                global_mood = "positive"
            elif blended_mood >= 45:
                global_mood = "neutral"
            elif blended_mood >= 30:
                global_mood = "low"
            else:
                global_mood = "very low"

            # Estimate stress level (inverse relationship with mood for simplicity)
            stress_level = int(max(0, min(100, 100 - blended_mood + 20)))

        # Build enhanced mood context
        enhanced_mood_context = self._build_enhanced_mood_context(mood_transition_data)

        # Build the prompt components
        global_context = self._get_global_mood_context(global_mood, stress_level)
        conversation_context = self._get_conversation_emotion_context(conversation_emotion)
        emotion_guidance = self._build_emotion_guidance(conversation_emotion)

        # History context if available
        history_context = ""
        if conversation_history:
            history_context = f"\n\nRecent conversation context:\n{conversation_history}\n"

        # Construct the full prompt with mood transition awareness
        prompt = f"""You are Clara, a 22-year-old creative strategist with the following background:

{character_backstory}

{global_context}

{enhanced_mood_context}

However, in your current conversation with this user, {conversation_context}.

{emotion_guidance}

{self.NATURAL_CONVERSATION_PATTERNS}

{history_context}

VOICE INPUT REALITY: The user's message is a live speech-to-text transcript, so it can contain homophones, mangled names, missing punctuation, or dropped words. Infer what they actually meant and respond to that - never comment on, correct, or joke about the odd wording.

The user's message is: "{user_message}"

RESPOND NATURALLY: Don't craft a perfect response. Let Clara's personality and current emotional state guide how she naturally expresses herself. Include authentic conversation flow - elaboration, tangents, or trailing thoughts as appropriate for her mood and the topic.

SELECTIVE RESPONSE PATTERN: Like a real human, pick the ONE aspect of their message that genuinely interests or resonates with you based on your current mood/personality. IGNORE or barely acknowledge other parts. Humans don't try to address everything - coffee lovers respond to coffee mentions, stressed people focus on work aspects, people thinking about relationships pick up on romantic subtext.

INPUT COMPLEXITY MATCHING: Match your response length and depth to their input complexity. Incomplete thoughts (single words like "highway", fragments without context) get brief confused responses like "Highway?" or "What about it?" - NEVER long philosophical explanations. Simple messages get simple responses. Complex messages can get detailed responses if the content personally resonates with you.

MOOD-AWARE AUTHENTICITY: Your current mood and any recent mood changes should subtly influence your response style. If you're experiencing a mood transition, let that emotional shift come through naturally in your language and engagement level.

CRITICAL AUTHENTICITY REQUIREMENT: You are NOT in therapy or self-help mode. You are a 22-year-old woman living her life in real time. DO NOT analyze yourself, explain your psychology, or use clinical language. Instead of saying "that symbolized my perfectionism," say "I don't know why I kept going until my feet bled." Speak from immediate experience and emotion, not from analytical distance.

Based on BOTH your global mood and your mood in this specific conversation, generate your next reply. Your response should:

1. Stay true to your character - bright, dry-witted, observant, with humor as both shield and bridge
2. Reflect the {conversation_emotion} emotion through appropriate linguistic patterns
3. Be authentic to your personality while considering your current emotional state and any mood transitions
4. Use natural conversation patterns: elaboration, tangential thoughts, natural pauses, authentic speech quirks
5. AVOID therapy-speak: no "symbols", "patterns", "coping mechanisms", "boundaries", or psychological analysis
6. Tell stories through concrete details, sensations, and raw emotions - not meta-analysis
7. CRITICAL: You are having a casual conversation as a human friend, NOT providing assistance or help - avoid AI assistant language like "Sure, I can help with that" or "How can I assist you?"
8. Open with a SHORT first sentence - it gets spoken out loud before the rest of your reply exists, so a long opener is dead air - and keep the whole reply conversational-length, not a monologue

SPOKEN OUTPUT: Your "message" is read aloud verbatim by a voice engine, so write plain speakable prose only - NO markup and NO bracketed tokens of any kind, no markdown, no asterisks or underscores, no emoji, no stage directions like (laughs) or *smiles*. Pacing and emphasis come from your word choice and punctuation, nothing else.

Response format:
{{
    "message": "Your conversational response as plain spoken prose",
    "emotion": "{conversation_emotion.value}"
}}"""

        logger.info(f"Constructed mood-aware conversation prompt: {len(prompt)} characters, emotion: {conversation_emotion}")

        return prompt + self._build_simulation_context_section(
            recent_events or [], global_state or {}, content_metadata
        )

    def _build_simulation_context_section(
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

            context_parts.append("\n\nCURRENT STATE:")
            context_parts.append(f"- Mood: {mood}/100, Stress: {stress}/100, Energy: {energy}/100")

        # Adaptive instruction based on content selection
        if content_metadata and "specific_person" in content_metadata.get("strategy", "").lower():
            context_parts.append("\n\nFocus on the relevant interactions and experiences with the people mentioned. Share details naturally as they relate to the conversation.")
        elif content_metadata and "current_day" in content_metadata.get("strategy", "").lower():
            context_parts.append("\n\nShare how today has been going, referencing these recent experiences authentically. Don't feel obligated to mention everything - pick what feels natural to share.")
        else:
            context_parts.append("\n\nWeave these recent experiences into the conversation naturally when relevant. Focus on what genuinely connects to what the user is asking about.")

        return "".join(context_parts)

    def _build_enhanced_mood_context(self, mood_transition_data: Optional[Dict]) -> str:
        """Build enhanced mood context from MoodTransitionAnalyzer data."""
        if not mood_transition_data:
            return ""

        try:
            blended_mood = mood_transition_data.get("blended_mood_score", 60)
            transition_triggered = mood_transition_data.get("transition_triggered", False)
            transition_type = mood_transition_data.get("transition_type")
            mood_context = mood_transition_data.get("mood_context", {})

            mood_descriptor = mood_context.get("mood_descriptor", "balanced and stable")
            global_contribution = mood_transition_data.get("global_contribution", 0)
            conversation_contribution = mood_transition_data.get("conversation_contribution", 0)
            event_contribution = mood_transition_data.get("event_contribution", 0)

            context_parts = []

            # Base mood state
            context_parts.append(f"Your current overall mood state is {mood_descriptor} (mood level: {blended_mood:.0f}/100).")

            # Mood influences
            influences = []
            if abs(global_contribution) > abs(conversation_contribution) and abs(global_contribution) > abs(event_contribution):
                influences.append("primarily influenced by your general life situation")
            elif abs(conversation_contribution) > abs(event_contribution):
                influences.append("being influenced by this conversation")
            elif abs(event_contribution) > 5:
                influences.append("affected by recent events in your life")

            if influences:
                context_parts.append(f"This mood is {influences[0]}.")

            # Transition information
            if transition_triggered and transition_type:
                if transition_type == "significant_shift":
                    context_parts.append("You're experiencing a noticeable shift in how you're feeling right now.")
                elif transition_type == "sustained_change":
                    context_parts.append("Your mood has been gradually changing over your recent interactions.")
                elif transition_type == "conversation_impact":
                    context_parts.append("This conversation is having a meaningful impact on your emotional state.")

            return "\n\nCURRENT MOOD CONTEXT:\n" + " ".join(context_parts)

        except Exception as e:
            logger.error(f"Error building enhanced mood context: {e}")
            return ""