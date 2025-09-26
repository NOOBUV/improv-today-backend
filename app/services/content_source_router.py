"""
Content Source Router for intelligent content selection based on question type.
Determines which content sources (recent events, backstory, memories) to prioritize.
"""

import logging
import re
from typing import Dict, List, Any
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ContentSourceType(str, Enum):
    """Types of content sources available for conversations."""
    RECENT_EVENTS = "recent_events"          # 0-72 hours simulation events
    CHARACTER_GIST = "character_gist"        # Core personality traits
    BACKSTORY_MEMORIES = "backstory_memories" # Past experiences, childhood
    CURRENT_STATE = "current_state"          # Mood, stress, energy levels


class QuestionCategory(str, Enum):
    """Categories of user questions for content routing."""
    CURRENT_DAY = "current_day"              # "How is your day?"
    RECENT_LIFE = "recent_life"              # "How is your week/life?"
    SPECIFIC_PERSON = "specific_person"       # "Tell me about Chris"
    SPECIFIC_EVENT = "specific_event"         # "How was the meeting?"
    PERSONALITY = "personality"               # "What are you like?"
    PAST_MEMORIES = "past_memories"          # "Tell me about childhood"
    GENERAL_CHAT = "general_chat"            # Generic conversation


@dataclass
class ContentWeight:
    """Weight configuration for different content sources."""
    recent_events: float = 0.0
    character_gist: float = 0.0
    backstory_memories: float = 0.0
    current_state: float = 0.0

    def normalize(self) -> 'ContentWeight':
        """Ensure weights sum to 1.0"""
        total = self.recent_events + self.character_gist + self.backstory_memories + self.current_state
        if total == 0:
            return ContentWeight(0.25, 0.25, 0.25, 0.25)

        return ContentWeight(
            self.recent_events / total,
            self.character_gist / total,
            self.backstory_memories / total,
            self.current_state / total
        )


@dataclass
class ContentRoutingResult:
    """Result of content source routing analysis."""
    question_category: QuestionCategory
    content_weights: ContentWeight
    keywords_found: List[str]
    confidence: float
    reasoning: str


class ContentSourceRouter:
    """Routes user questions to appropriate content sources with intelligent weighting."""

    def __init__(self):
        # Question pattern definitions
        self.question_patterns = {
            QuestionCategory.CURRENT_DAY: [
                r'\b(how.*(day|today)|what.*today|today.*how)',
                r'\b(morning|afternoon|evening).*how',
                r'\bhow.*going.*(day|today)'
            ],

            QuestionCategory.RECENT_LIFE: [
                r'\b(how.*(life|week|lately|recently)|what.*been.*happening)',
                r'\b(anything.*(new|interesting|happened)|life.*going)',
                r'\bhow.*things.*(going|been)'
            ],

            QuestionCategory.SPECIFIC_PERSON: [
                r'\b(tell.*about|how.*was).*([A-Z][a-z]+)',
                r'\b(did.*meet|see.*with).*([A-Z][a-z]+)',
                r'\b([A-Z][a-z]+).*(how|what|when|tell|about)'
            ],

            QuestionCategory.SPECIFIC_EVENT: [
                r'\b(how.*was).*(meeting|dinner|coffee|call|workout|event)',
                r'\b(tell.*about).*(meeting|project|work|client)',
                r'\b(what.*happened).*(meeting|call|dinner)'
            ],

            QuestionCategory.PERSONALITY: [
                r'\b(what.*you.*like|describe.*yourself|who.*are.*you)',
                r'\b(your.*personality|what.*kind.*person)',
                r'\btell.*about.*yourself'
            ],

            QuestionCategory.PAST_MEMORIES: [
                r'\b(childhood|growing.*up|when.*were.*kid)',
                r'\b(remember.*when|tell.*about.*past)',
                r'\b(family|parents|mom|dad).*growing.*up'
            ]
        }

        # Content weight configurations for each question category
        self.content_weight_configs = {
            QuestionCategory.CURRENT_DAY: ContentWeight(
                recent_events=0.80,      # Heavy focus on today's events
                character_gist=0.15,     # Personality for authentic response
                current_state=0.05,      # Current mood influence
                backstory_memories=0.0
            ),

            QuestionCategory.RECENT_LIFE: ContentWeight(
                recent_events=0.70,      # Focus on recent 72h events
                character_gist=0.20,     # Personality lens
                current_state=0.10,      # Current state context
                backstory_memories=0.0
            ),

            QuestionCategory.SPECIFIC_PERSON: ContentWeight(
                recent_events=0.90,      # Search recent events first
                character_gist=0.10,     # How Clara relates to people
                backstory_memories=0.0,  # Use backstory only if no recent events
                current_state=0.0
            ),

            QuestionCategory.SPECIFIC_EVENT: ContentWeight(
                recent_events=0.95,      # Almost entirely from recent events
                character_gist=0.05,     # Clara's perspective on events
                backstory_memories=0.0,
                current_state=0.0
            ),

            QuestionCategory.PERSONALITY: ContentWeight(
                recent_events=0.10,      # Some behavioral examples
                character_gist=0.80,     # Primary source for personality
                backstory_memories=0.10, # Formative experiences
                current_state=0.0
            ),

            QuestionCategory.PAST_MEMORIES: ContentWeight(
                recent_events=0.0,
                character_gist=0.20,     # Personality context
                backstory_memories=0.80, # Primary source for memories
                current_state=0.0
            ),

            QuestionCategory.GENERAL_CHAT: ContentWeight(
                recent_events=0.40,      # Balanced mix
                character_gist=0.40,     # Personality-driven responses
                backstory_memories=0.10, # Occasional anecdotes
                current_state=0.10       # Current state influence
            )
        }

    def analyze_question(self, user_message: str, available_entities: Dict[str, List[str]] = None) -> ContentRoutingResult:
        """
        Analyze user question and determine optimal content source routing.

        Args:
            user_message: User's input message
            available_entities: Dictionary of entities from recent events/content
                                {"people": ["Chris", "Jessica"], "events": ["meeting", "dinner"]}

        Returns:
            ContentRoutingResult with category, weights, and reasoning
        """
        try:
            message_lower = user_message.lower().strip()

            # Extract potential person/event keywords dynamically
            keywords = self._extract_keywords(message_lower, available_entities or {})

            # Match question patterns
            best_category = QuestionCategory.GENERAL_CHAT
            best_confidence = 0.0
            best_match_details = ""

            for category, patterns in self.question_patterns.items():
                confidence = 0.0
                matched_patterns = []

                for pattern in patterns:
                    if re.search(pattern, message_lower):
                        confidence += 0.8  # High confidence for regex match
                        matched_patterns.append(pattern)

                # Boost confidence for keyword matches (using dynamic entities)
                if category == QuestionCategory.SPECIFIC_PERSON and available_entities:
                    available_people = [name.lower() for name in available_entities.get('people', [])]
                    person_keywords = [k for k in keywords if k in available_people]
                    confidence += len(person_keywords) * 0.3

                if category == QuestionCategory.SPECIFIC_EVENT and available_entities:
                    available_events = [event.lower() for event in available_entities.get('events', [])]
                    event_keywords = [k for k in keywords if k in available_events]
                    confidence += len(event_keywords) * 0.2

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_category = category
                    best_match_details = f"Matched patterns: {matched_patterns}, Keywords: {keywords}"

            # Get content weights for the determined category
            content_weights = self.content_weight_configs[best_category].normalize()

            reasoning = f"Categorized as {best_category.value} with {best_confidence:.2f} confidence. {best_match_details}"

            logger.debug(f"Question routing: '{user_message}' -> {best_category.value} (conf: {best_confidence:.2f})")

            return ContentRoutingResult(
                question_category=best_category,
                content_weights=content_weights,
                keywords_found=keywords,
                confidence=best_confidence,
                reasoning=reasoning
            )

        except Exception as e:
            logger.error(f"Error analyzing question: {e}")
            # Fallback to general chat
            return ContentRoutingResult(
                question_category=QuestionCategory.GENERAL_CHAT,
                content_weights=self.content_weight_configs[QuestionCategory.GENERAL_CHAT].normalize(),
                keywords_found=[],
                confidence=0.5,
                reasoning=f"Fallback due to error: {str(e)}"
            )

    def _extract_keywords(self, message_lower: str, available_entities: Dict[str, List[str]]) -> List[str]:
        """Extract relevant keywords using logical patterns and available entities."""
        keywords = []

        # Extract capitalized words (likely names) from the original message
        original_words = re.findall(r'\b[A-Z][a-z]+\b', message_lower.title())
        for word in original_words:
            # Check if it appears in available people entities
            if available_entities.get('people'):
                if word.lower() in [name.lower() for name in available_entities['people']]:
                    keywords.append(word.lower())

        # Extract potential event types using logical patterns
        event_patterns = [
            r'\b(meeting|call|session)s?\b',
            r'\b(dinner|lunch|coffee|drinks)s?\b',
            r'\b(workout|gym|exercise)s?\b',
            r'\b(project|work|client|report)s?\b',
            r'\b(errands|shopping|cleaning)s?\b',
            r'\b(event|party|gathering)s?\b'
        ]

        for pattern in event_patterns:
            matches = re.findall(pattern, message_lower)
            if matches and available_entities.get('events'):
                for match in matches:
                    # Check if this type appears in available events
                    if any(match in event.lower() for event in available_entities['events']):
                        keywords.append(match)

        # Extract time references using logical patterns
        time_patterns = [
            r'\b(today|yesterday|tomorrow)\b',
            r'\b(this|last|next)\s+(week|weekend|month)\b',
            r'\b(recently|lately|currently)\b',
            r'\b(morning|afternoon|evening)\b'
        ]

        for pattern in time_patterns:
            matches = re.findall(pattern, message_lower)
            keywords.extend([match if isinstance(match, str) else ' '.join(match) for match in matches])

        return list(set(keywords))  # Remove duplicates

    def get_content_guidance(self, routing_result: ContentRoutingResult) -> Dict[str, Any]:
        """
        Get specific content selection guidance based on routing result.

        Returns:
            Dictionary with specific instructions for content selection
        """
        guidance = {
            "primary_source": self._get_primary_source(routing_result.content_weights),
            "event_time_window": self._get_event_time_window(routing_result.question_category),
            "keyword_filters": routing_result.keywords_found,
            "content_strategy": self._get_content_strategy(routing_result.question_category),
            "avoid_repetition": routing_result.question_category in [QuestionCategory.CURRENT_DAY, QuestionCategory.RECENT_LIFE]
        }

        return guidance

    def _get_primary_source(self, weights: ContentWeight) -> ContentSourceType:
        """Determine primary content source from weights."""
        source_weights = {
            ContentSourceType.RECENT_EVENTS: weights.recent_events,
            ContentSourceType.CHARACTER_GIST: weights.character_gist,
            ContentSourceType.BACKSTORY_MEMORIES: weights.backstory_memories,
            ContentSourceType.CURRENT_STATE: weights.current_state
        }
        return max(source_weights, key=source_weights.get)

    def _get_event_time_window(self, category: QuestionCategory) -> int:
        """Get appropriate time window for event retrieval."""
        if category == QuestionCategory.CURRENT_DAY:
            return 24  # Last 24 hours
        elif category in [QuestionCategory.RECENT_LIFE, QuestionCategory.SPECIFIC_PERSON, QuestionCategory.SPECIFIC_EVENT]:
            return 72  # Last 72 hours
        else:
            return 0   # No events needed

    def _get_content_strategy(self, category: QuestionCategory) -> str:
        """Get content selection strategy description."""
        strategies = {
            QuestionCategory.CURRENT_DAY: "Prioritize today's events, use authentic personality voice",
            QuestionCategory.RECENT_LIFE: "Mix recent interesting events with personal reflection",
            QuestionCategory.SPECIFIC_PERSON: "Search events for person mentions, use relationship context",
            QuestionCategory.SPECIFIC_EVENT: "Find specific event details, add personal experience",
            QuestionCategory.PERSONALITY: "Focus on character traits, use behavioral examples",
            QuestionCategory.PAST_MEMORIES: "Use backstory content, connect to current identity",
            QuestionCategory.GENERAL_CHAT: "Balanced personality-driven response with recent context"
        }
        return strategies.get(category, "Default conversational approach")