"""
Mood Transition Analyzer service for intelligent mood analysis and transitions.
Combines global simulation state with conversation sentiment for contextual mood persistence.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum

from app.services.redis_service import RedisService
from app.services.simulation.state_manager import StateManagerService

logger = logging.getLogger(__name__)


class MoodTransitionTrigger(Enum):
    """Types of mood transitions that can be triggered."""
    SIGNIFICANT_SHIFT = "significant_shift"  # >20 point change
    CONVERSATION_IMPACT = "conversation_impact"  # Sentiment-driven change
    EVENT_REACTION = "event_reaction"  # Response to global events
    SUSTAINED_CHANGE = "sustained_change"  # Gradual change over time


@dataclass
class MoodTransitionResult:
    """Result of mood transition analysis."""
    blended_mood_score: float  # Final calculated mood (0-100)
    global_mood_contribution: float  # How much global state contributed
    conversation_contribution: float  # How much conversation contributed
    event_contribution: float  # How much recent events contributed

    transition_triggered: bool  # Whether a significant transition occurred
    transition_type: Optional[MoodTransitionTrigger] = None
    transition_magnitude: float = 0.0  # Size of the change

    session_mood_history: List[Dict[str, Any]] = None  # Recent mood changes
    mood_stability_index: float = 0.0  # How stable the mood has been (0-1)

    # Context for integration with conversation flow
    mood_context: Dict[str, Any] = None  # Additional context for conversation


class MoodTransitionAnalyzer:
    """
    Service for analyzing mood transitions by blending global state, recent events,
    and conversation sentiment with intelligent persistence and transition detection.
    """

    def __init__(self):
        self.redis_service = RedisService()
        self.state_manager = StateManagerService()

        # Mood blending weights as specified in AC: 2
        self.mood_weights = {
            "global_state": 0.60,  # 60% from global simulation state
            "recent_events": 0.25,  # 25% from recent events
            "conversation_sentiment": 0.15  # 15% from conversation
        }

        # Transition detection thresholds
        self.transition_thresholds = {
            "significant_shift": 20.0,  # >20 point change triggers transition
            "conversation_impact": 15.0,  # Strong conversation influence
            "event_reaction": 12.0,  # Strong event influence
            "stability_window": 5  # Number of measurements for stability
        }

        # Redis TTL for session mood data (2 hours as specified)
        self.session_ttl = 7200  # 2 hours in seconds

    async def analyze_mood_transition(
        self,
        global_state: Dict[str, Any],
        recent_events: List[Dict[str, Any]],
        conversation_sentiment: float,
        session_context: Dict[str, Any]
    ) -> MoodTransitionResult:
        """
        Analyze mood transition by combining global state, events, and conversation sentiment.

        Args:
            global_state: Current global simulation state from StateManager
            recent_events: List of recent GlobalEvents affecting mood
            conversation_sentiment: Sentiment analysis of current conversation (-1 to 1)
            session_context: Session-specific context including user_id, conversation_id

        Returns:
            MoodTransitionResult with blended mood and transition analysis
        """
        try:
            user_id = session_context.get("user_id")
            conversation_id = session_context.get("conversation_id")

            logger.info(f"Analyzing mood transition for user {user_id}, conversation {conversation_id}")

            # Get current session mood history
            session_mood_history = await self.get_session_mood_history(user_id, conversation_id)

            # Calculate individual mood components
            global_mood = self._extract_global_mood(global_state)
            events_mood_impact = self._calculate_events_mood_impact(recent_events)
            conversation_mood_impact = self._convert_sentiment_to_mood_impact(conversation_sentiment)

            # Blend mood components using weighted algorithm
            blended_mood = self._blend_mood_components(
                global_mood, events_mood_impact, conversation_mood_impact
            )

            # Detect mood transitions
            previous_mood = session_mood_history[-1]["mood_score"] if session_mood_history else global_mood
            transition_info = self._detect_mood_transition(
                previous_mood, blended_mood, session_mood_history
            )

            # Calculate mood stability
            stability_index = self._calculate_mood_stability(session_mood_history, blended_mood)

            # Build mood context for conversation integration
            mood_context = self._build_mood_context(
                blended_mood, global_mood, events_mood_impact, conversation_mood_impact
            )

            # Create result
            result = MoodTransitionResult(
                blended_mood_score=blended_mood,
                global_mood_contribution=global_mood * self.mood_weights["global_state"],
                conversation_contribution=conversation_mood_impact * self.mood_weights["conversation_sentiment"],
                event_contribution=events_mood_impact * self.mood_weights["recent_events"],

                transition_triggered=transition_info["triggered"],
                transition_type=transition_info["type"],
                transition_magnitude=transition_info["magnitude"],

                session_mood_history=session_mood_history,
                mood_stability_index=stability_index,
                mood_context=mood_context
            )

            # Update session mood state
            await self.update_session_mood_state(
                user_id, conversation_id, result, session_context
            )

            return result

        except Exception as e:
            logger.error(f"Error analyzing mood transition: {e}")
            return self._get_fallback_mood_result(global_state)

    def _extract_global_mood(self, global_state: Dict[str, Any]) -> float:
        """Extract and normalize mood from global simulation state."""
        try:
            # Extract primary mood value
            mood_data = global_state.get("mood", {})
            if isinstance(mood_data, dict):
                mood_value = mood_data.get("numeric_value", 60.0)
            else:
                mood_value = float(mood_data) if mood_data is not None else 60.0

            # Consider other state factors that influence mood
            energy = global_state.get("energy", {}).get("numeric_value", 70.0)
            stress = global_state.get("stress", {}).get("numeric_value", 50.0)

            # Adjust mood based on energy and stress (stress inversely affects mood)
            adjusted_mood = mood_value + (energy - 70) * 0.1 - (stress - 50) * 0.15

            # Clamp to valid range
            return max(0.0, min(100.0, adjusted_mood))

        except Exception as e:
            logger.warning(f"Error extracting global mood: {e}, using default")
            return 60.0

    def _calculate_events_mood_impact(self, recent_events: List[Dict[str, Any]]) -> float:
        """Calculate mood impact from recent global events."""
        if not recent_events:
            return 0.0  # Neutral impact

        total_impact = 0.0
        weight_sum = 0.0

        for event in recent_events[-5:]:  # Consider last 5 events
            try:
                # Calculate impact based on event intensity and type
                base_impact = event.get('intensity', 5.0)

                # Apply mood impact modifier
                impact_mood = event.get('impact_mood')
                if impact_mood:
                    impact_mood_name = impact_mood.get('name') if isinstance(impact_mood, dict) else str(impact_mood)
                    if impact_mood_name == 'POSITIVE':
                        mood_modifier = base_impact * 3.0
                    elif impact_mood_name == 'NEGATIVE':
                        mood_modifier = -base_impact * 3.0
                    elif impact_mood_name == 'MIXED':
                        mood_modifier = base_impact * 1.0
                    else:  # NEUTRAL
                        mood_modifier = 0.0
                else:
                    mood_modifier = 0.0

                # Weight recent events more heavily
                event_timestamp = event.get('timestamp')
                if isinstance(event_timestamp, str):
                    event_timestamp = datetime.fromisoformat(event_timestamp.replace('Z', '+00:00'))
                elif not isinstance(event_timestamp, datetime):
                    # Fallback: assume recent event if timestamp is invalid
                    event_timestamp = datetime.now(timezone.utc)

                hours_ago = (datetime.now(timezone.utc) - event_timestamp).total_seconds() / 3600
                time_weight = max(0.1, 1.0 - (hours_ago / 24.0))  # Decay over 24 hours

                total_impact += mood_modifier * time_weight
                weight_sum += time_weight

            except Exception as e:
                event_id = event.get('event_id', 'unknown')
                logger.warning(f"Error processing event {event_id}: {e}")
                continue

        # Normalize and convert to mood scale adjustment
        if weight_sum > 0:
            avg_impact = total_impact / weight_sum
            # Convert to mood adjustment (-20 to +20)
            mood_adjustment = max(-20.0, min(20.0, avg_impact))
            return mood_adjustment

        return 0.0

    def _convert_sentiment_to_mood_impact(self, sentiment: float) -> float:
        """Convert conversation sentiment (-1 to 1) to mood impact (-15 to +15)."""
        try:
            # Clamp sentiment to valid range
            sentiment = max(-1.0, min(1.0, sentiment))

            # Convert to mood adjustment scale
            # Positive sentiment can boost mood up to +15 points
            # Negative sentiment can lower mood up to -15 points
            mood_impact = sentiment * 15.0

            return mood_impact

        except Exception as e:
            logger.warning(f"Error converting sentiment to mood impact: {e}")
            return 0.0

    def _blend_mood_components(
        self,
        global_mood: float,
        events_impact: float,
        conversation_impact: float
    ) -> float:
        """Blend mood components using configured weights."""
        try:
            # Start with global mood as base
            blended_mood = global_mood * self.mood_weights["global_state"]

            # Add event impact (as adjustment to global mood)
            event_adjusted_mood = (global_mood + events_impact) * self.mood_weights["recent_events"]
            blended_mood += event_adjusted_mood

            # Add conversation impact (as adjustment to global mood)
            conversation_adjusted_mood = (global_mood + conversation_impact) * self.mood_weights["conversation_sentiment"]
            blended_mood += conversation_adjusted_mood

            # Clamp to valid range
            return max(0.0, min(100.0, blended_mood))

        except Exception as e:
            logger.error(f"Error blending mood components: {e}")
            return global_mood  # Fall back to global mood

    def _detect_mood_transition(
        self,
        previous_mood: float,
        current_mood: float,
        mood_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect if a significant mood transition has occurred."""
        try:
            mood_change = abs(current_mood - previous_mood)

            # Check for significant shift threshold
            if mood_change >= self.transition_thresholds["significant_shift"]:
                return {
                    "triggered": True,
                    "type": MoodTransitionTrigger.SIGNIFICANT_SHIFT,
                    "magnitude": mood_change,
                    "direction": "increase" if current_mood > previous_mood else "decrease"
                }

            # Check for sustained change pattern
            if len(mood_history) >= 3:
                recent_moods = [entry["mood_score"] for entry in mood_history[-3:]]
                recent_moods.append(current_mood)

                # Check if mood is consistently trending in one direction
                increases = sum(1 for i in range(1, len(recent_moods)) if recent_moods[i] > recent_moods[i-1])
                decreases = sum(1 for i in range(1, len(recent_moods)) if recent_moods[i] < recent_moods[i-1])

                if increases >= 3 or decreases >= 3:
                    total_change = abs(recent_moods[-1] - recent_moods[0])
                    if total_change >= 10.0:  # Sustained change threshold
                        return {
                            "triggered": True,
                            "type": MoodTransitionTrigger.SUSTAINED_CHANGE,
                            "magnitude": total_change,
                            "direction": "increase" if increases >= 3 else "decrease"
                        }

            # No significant transition detected
            return {
                "triggered": False,
                "type": None,
                "magnitude": mood_change,
                "direction": "stable"
            }

        except Exception as e:
            logger.error(f"Error detecting mood transition: {e}")
            return {"triggered": False, "type": None, "magnitude": 0.0, "direction": "stable"}

    def _calculate_mood_stability(
        self,
        mood_history: List[Dict[str, Any]],
        current_mood: float
    ) -> float:
        """Calculate mood stability index (0 = unstable, 1 = very stable)."""
        try:
            if len(mood_history) < 2:
                return 0.5  # Neutral stability for insufficient data

            # Get recent mood values
            recent_moods = [entry["mood_score"] for entry in mood_history[-5:]]
            recent_moods.append(current_mood)

            # Calculate variance
            mean_mood = sum(recent_moods) / len(recent_moods)
            variance = sum((mood - mean_mood) ** 2 for mood in recent_moods) / len(recent_moods)
            std_deviation = variance ** 0.5

            # Convert to stability index (lower variance = higher stability)
            # Max expected std_deviation is ~30 (for very unstable mood)
            stability_index = max(0.0, min(1.0, 1.0 - (std_deviation / 30.0)))

            return stability_index

        except Exception as e:
            logger.error(f"Error calculating mood stability: {e}")
            return 0.5

    def _build_mood_context(
        self,
        blended_mood: float,
        global_mood: float,
        events_impact: float,
        conversation_impact: float
    ) -> Dict[str, Any]:
        """Build mood context for conversation integration."""
        try:
            # Determine mood category
            if blended_mood >= 75:
                mood_category = "very_positive"
                mood_descriptor = "highly energized and optimistic"
            elif blended_mood >= 60:
                mood_category = "positive"
                mood_descriptor = "upbeat and positive"
            elif blended_mood >= 45:
                mood_category = "neutral"
                mood_descriptor = "balanced and stable"
            elif blended_mood >= 30:
                mood_category = "low"
                mood_descriptor = "subdued and contemplative"
            else:
                mood_category = "very_low"
                mood_descriptor = "struggling and low energy"

            # Identify primary influencer
            influences = [
                ("global_state", abs(global_mood - 50)),
                ("events", abs(events_impact)),
                ("conversation", abs(conversation_impact))
            ]
            primary_influence = max(influences, key=lambda x: x[1])[0]

            return {
                "mood_category": mood_category,
                "mood_descriptor": mood_descriptor,
                "mood_score": blended_mood,
                "primary_influence": primary_influence,
                "global_mood_baseline": global_mood,
                "event_influence_strength": abs(events_impact),
                "conversation_influence_strength": abs(conversation_impact),
                "context_timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error building mood context: {e}")
            return {"mood_category": "neutral", "mood_score": 50.0, "error": str(e)}

    async def get_session_mood_history(
        self,
        user_id: str,
        conversation_id: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve session-specific mood history from Redis.

        Args:
            user_id: User identifier
            conversation_id: Conversation identifier

        Returns:
            List of mood history entries for this session
        """
        try:
            client = self.redis_service._get_client()
            if not client:
                logger.warning("Redis unavailable, returning empty mood history")
                return []

            cache_key = f"mood_history:{user_id}:{conversation_id}"
            cached_data = client.get(cache_key)

            if cached_data:
                mood_history = json.loads(cached_data)
                logger.debug(f"Retrieved {len(mood_history)} mood history entries from cache")
                return mood_history

            return []

        except Exception as e:
            logger.error(f"Error getting session mood history: {e}")
            return []

    async def update_session_mood_state(
        self,
        user_id: str,
        conversation_id: str,
        mood_result: MoodTransitionResult,
        session_context: Dict[str, Any]
    ) -> bool:
        """
        Update session mood state in Redis with 2-hour TTL.

        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            mood_result: MoodTransitionResult to store
            session_context: Additional session context

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            client = self.redis_service._get_client()
            if not client:
                logger.warning("Redis unavailable, cannot update mood state")
                return False

            # Create mood history entry
            mood_entry = {
                "mood_score": mood_result.blended_mood_score,
                "global_contribution": mood_result.global_mood_contribution,
                "conversation_contribution": mood_result.conversation_contribution,
                "event_contribution": mood_result.event_contribution,
                "transition_triggered": mood_result.transition_triggered,
                "transition_type": mood_result.transition_type.value if mood_result.transition_type else None,
                "transition_magnitude": mood_result.transition_magnitude,
                "stability_index": mood_result.mood_stability_index,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Get existing history and append new entry
            cache_key = f"mood_history:{user_id}:{conversation_id}"
            existing_history = await self.get_session_mood_history(user_id, conversation_id)

            # Add new entry and keep only last 10 entries
            existing_history.append(mood_entry)
            if len(existing_history) > 10:
                existing_history = existing_history[-10:]

            # Store updated history with TTL
            client.setex(cache_key, self.session_ttl, json.dumps(existing_history))

            # Log transition if triggered
            if mood_result.transition_triggered:
                logger.info(
                    f"Mood transition detected for user {user_id}: "
                    f"{mood_result.transition_type.value} (magnitude: {mood_result.transition_magnitude:.2f})"
                )

            return True

        except Exception as e:
            logger.error(f"Error updating session mood state: {e}")
            return False

    def _get_fallback_mood_result(self, global_state: Dict[str, Any]) -> MoodTransitionResult:
        """Get fallback mood result when analysis fails."""
        global_mood = self._extract_global_mood(global_state)

        return MoodTransitionResult(
            blended_mood_score=global_mood,
            global_mood_contribution=global_mood,
            conversation_contribution=0.0,
            event_contribution=0.0,

            transition_triggered=False,
            mood_stability_index=0.5,
            mood_context={
                "mood_category": "neutral",
                "mood_score": global_mood,
                "fallback_mode": True
            }
        )