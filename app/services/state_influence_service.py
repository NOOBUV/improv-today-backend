"""
State influence system for conversation context.
Merges global and session state appropriately with configurable influence weights.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from enum import Enum

from app.services.session_state_service import SessionStateService
from app.services.simulation.state_manager import StateManagerService

logger = logging.getLogger(__name__)


class ConversationScenario(Enum):
    """Different conversation scenarios with different state influence patterns."""
    FIRST_MEETING = "first_meeting"
    CASUAL_CHAT = "casual_chat"
    DEEP_CONVERSATION = "deep_conversation"
    SUPPORT_SESSION = "support_session"
    CREATIVE_COLLABORATION = "creative_collaboration"
    PROBLEM_SOLVING = "problem_solving"


class StateInfluenceService:
    """
    Service for managing state influence on conversation context.

    This service handles:
    - Global state influence algorithms that affect conversation tone/mood
    - Session state blending that respects user personalization preferences
    - Conversation context building that merges global and session state appropriately
    - State influence weight configuration for different conversation scenarios
    """

    def __init__(self):
        self.session_service = SessionStateService()
        self.state_manager = StateManagerService()

        # Configure influence weights for different scenarios
        self.scenario_weights = {
            ConversationScenario.FIRST_MEETING: {
                "global_influence": 0.8,  # High global influence for new users
                "session_influence": 0.2,
                "mood_sensitivity": 0.7,
                "energy_impact": 0.6,
                "stress_awareness": 0.5
            },
            ConversationScenario.CASUAL_CHAT: {
                "global_influence": 0.6,
                "session_influence": 0.4,
                "mood_sensitivity": 0.8,
                "energy_impact": 0.7,
                "stress_awareness": 0.4
            },
            ConversationScenario.DEEP_CONVERSATION: {
                "global_influence": 0.7,
                "session_influence": 0.3,
                "mood_sensitivity": 0.9,
                "energy_impact": 0.5,
                "stress_awareness": 0.8
            },
            ConversationScenario.SUPPORT_SESSION: {
                "global_influence": 0.5,  # More session-focused for support
                "session_influence": 0.5,
                "mood_sensitivity": 0.9,
                "energy_impact": 0.4,
                "stress_awareness": 0.9
            },
            ConversationScenario.CREATIVE_COLLABORATION: {
                "global_influence": 0.6,
                "session_influence": 0.4,
                "mood_sensitivity": 0.6,
                "energy_impact": 0.9,  # High energy impact for creativity
                "stress_awareness": 0.3
            },
            ConversationScenario.PROBLEM_SOLVING: {
                "global_influence": 0.7,
                "session_influence": 0.3,
                "mood_sensitivity": 0.5,
                "energy_impact": 0.8,
                "stress_awareness": 0.7
            }
        }

    async def build_conversation_context(
        self,
        user_id: str,
        conversation_id: str,
        scenario: ConversationScenario = ConversationScenario.CASUAL_CHAT,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build comprehensive conversation context by merging global and session state.

        Args:
            user_id: Unique identifier for the user
            conversation_id: Conversation session identifier
            scenario: Type of conversation scenario
            user_preferences: User-specific preferences for state influence

        Returns:
            Complete conversation context with merged state influence
        """
        try:
            logger.info(f"Building conversation context for user {user_id}, scenario: {scenario.value}")

            # Get effective state (global + session adjustments)
            effective_state = await self.session_service.get_effective_state(user_id, conversation_id)

            # Get session state for context information
            session_state = await self.session_service.get_session_state(user_id, conversation_id)

            # Apply state influence algorithms
            conversation_context = await self._apply_state_influence(
                effective_state, session_state, scenario, user_preferences
            )

            # Add conversation tone and mood indicators
            conversation_context.update(
                await self._calculate_conversation_tone(effective_state, scenario)
            )

            # Add relationship context
            conversation_context.update(
                self._build_relationship_context(session_state, scenario)
            )

            logger.debug(f"Built conversation context with {len(conversation_context)} elements")
            return conversation_context

        except Exception as e:
            logger.error(f"Error building conversation context: {e}")
            return self._get_fallback_context(scenario)

    async def _apply_state_influence(
        self,
        effective_state: Dict[str, Any],
        session_state: Optional[Dict[str, Any]],
        scenario: ConversationScenario,
        user_preferences: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply state influence algorithms based on scenario and preferences."""

        weights = self.scenario_weights.get(scenario, self.scenario_weights[ConversationScenario.CASUAL_CHAT])

        # Override weights with user preferences if provided
        if user_preferences:
            weights.update(user_preferences.get("state_influence_overrides", {}))

        # Extract key state values
        mood = effective_state.get("mood", {}).get("numeric_value", 60)
        energy = effective_state.get("energy", {}).get("numeric_value", 70)
        stress = effective_state.get("stress", {}).get("numeric_value", 50)
        social_satisfaction = effective_state.get("social_satisfaction", {}).get("numeric_value", 60)

        # Calculate influence factors
        influence_context = {
            "mood_influence": self._calculate_mood_influence(mood, weights["mood_sensitivity"]),
            "energy_influence": self._calculate_energy_influence(energy, weights["energy_impact"]),
            "stress_influence": self._calculate_stress_influence(stress, weights["stress_awareness"]),
            "social_influence": self._calculate_social_influence(social_satisfaction),

            # Blending information
            "global_weight": weights["global_influence"],
            "session_weight": weights["session_influence"],
            "scenario": scenario.value,

            # State values for reference
            "current_mood": mood,
            "current_energy": energy,
            "current_stress": stress,
            "current_social_satisfaction": social_satisfaction
        }

        # Add session-specific influence if available
        if session_state:
            session_adjustments = session_state.get("session_adjustments", {})
            conversation_context = session_state.get("conversation_context", {})

            influence_context.update({
                "session_adjustments_count": len(session_adjustments),
                "relationship_level": conversation_context.get("relationship_level", "new"),
                "user_mood_indicators": conversation_context.get("user_mood_indicators", []),
                "preferred_communication_style": conversation_context.get("preferred_communication_style", "balanced")
            })

        return influence_context

    def _calculate_mood_influence(self, mood: float, sensitivity: float) -> Dict[str, Any]:
        """Calculate how mood should influence conversation tone."""

        # Normalize mood to influence scale
        mood_normalized = (mood - 50) / 50  # Range -1 to 1
        influence_strength = abs(mood_normalized) * sensitivity

        if mood >= 70:
            return {
                "tone": "upbeat",
                "energy_level": "high",
                "conversation_style": "enthusiastic",
                "influence_strength": influence_strength,
                "mood_descriptor": "very positive"
            }
        elif mood >= 55:
            return {
                "tone": "positive",
                "energy_level": "moderate",
                "conversation_style": "friendly",
                "influence_strength": influence_strength,
                "mood_descriptor": "positive"
            }
        elif mood >= 40:
            return {
                "tone": "neutral",
                "energy_level": "balanced",
                "conversation_style": "calm",
                "influence_strength": influence_strength,
                "mood_descriptor": "neutral"
            }
        elif mood >= 25:
            return {
                "tone": "subdued",
                "energy_level": "low",
                "conversation_style": "gentle",
                "influence_strength": influence_strength,
                "mood_descriptor": "somewhat low"
            }
        else:
            return {
                "tone": "careful",
                "energy_level": "very_low",
                "conversation_style": "supportive",
                "influence_strength": influence_strength,
                "mood_descriptor": "low"
            }

    def _calculate_energy_influence(self, energy: float, impact: float) -> Dict[str, Any]:
        """Calculate how energy should influence conversation dynamics."""

        energy_normalized = energy / 100
        influence_strength = energy_normalized * impact

        if energy >= 80:
            return {
                "responsiveness": "very_high",
                "conversation_pace": "energetic",
                "detail_level": "comprehensive",
                "influence_strength": influence_strength,
                "energy_descriptor": "very high"
            }
        elif energy >= 60:
            return {
                "responsiveness": "high",
                "conversation_pace": "lively",
                "detail_level": "detailed",
                "influence_strength": influence_strength,
                "energy_descriptor": "high"
            }
        elif energy >= 40:
            return {
                "responsiveness": "moderate",
                "conversation_pace": "steady",
                "detail_level": "balanced",
                "influence_strength": influence_strength,
                "energy_descriptor": "moderate"
            }
        elif energy >= 20:
            return {
                "responsiveness": "low",
                "conversation_pace": "relaxed",
                "detail_level": "concise",
                "influence_strength": influence_strength,
                "energy_descriptor": "low"
            }
        else:
            return {
                "responsiveness": "minimal",
                "conversation_pace": "slow",
                "detail_level": "brief",
                "influence_strength": influence_strength,
                "energy_descriptor": "very low"
            }

    def _calculate_stress_influence(self, stress: float, awareness: float) -> Dict[str, Any]:
        """Calculate how stress should influence conversation approach."""

        stress_normalized = stress / 100
        influence_strength = stress_normalized * awareness

        if stress >= 75:
            return {
                "communication_approach": "very_gentle",
                "topic_sensitivity": "high",
                "patience_level": "maximum",
                "support_focus": "high",
                "influence_strength": influence_strength,
                "stress_descriptor": "very high"
            }
        elif stress >= 60:
            return {
                "communication_approach": "gentle",
                "topic_sensitivity": "elevated",
                "patience_level": "high",
                "support_focus": "moderate",
                "influence_strength": influence_strength,
                "stress_descriptor": "high"
            }
        elif stress >= 40:
            return {
                "communication_approach": "balanced",
                "topic_sensitivity": "normal",
                "patience_level": "normal",
                "support_focus": "low",
                "influence_strength": influence_strength,
                "stress_descriptor": "moderate"
            }
        elif stress >= 25:
            return {
                "communication_approach": "relaxed",
                "topic_sensitivity": "low",
                "patience_level": "normal",
                "support_focus": "minimal",
                "influence_strength": influence_strength,
                "stress_descriptor": "low"
            }
        else:
            return {
                "communication_approach": "casual",
                "topic_sensitivity": "minimal",
                "patience_level": "normal",
                "support_focus": "none",
                "influence_strength": influence_strength,
                "stress_descriptor": "very low"
            }

    def _calculate_social_influence(self, social_satisfaction: float) -> Dict[str, Any]:
        """Calculate how social satisfaction influences interaction style."""

        if social_satisfaction >= 70:
            return {
                "social_openness": "high",
                "interaction_warmth": "warm",
                "conversation_depth": "open",
                "social_descriptor": "very satisfied"
            }
        elif social_satisfaction >= 50:
            return {
                "social_openness": "moderate",
                "interaction_warmth": "friendly",
                "conversation_depth": "balanced",
                "social_descriptor": "satisfied"
            }
        elif social_satisfaction >= 30:
            return {
                "social_openness": "cautious",
                "interaction_warmth": "polite",
                "conversation_depth": "surface",
                "social_descriptor": "somewhat dissatisfied"
            }
        else:
            return {
                "social_openness": "reserved",
                "interaction_warmth": "formal",
                "conversation_depth": "minimal",
                "social_descriptor": "dissatisfied"
            }

    async def _calculate_conversation_tone(
        self,
        effective_state: Dict[str, Any],
        scenario: ConversationScenario
    ) -> Dict[str, Any]:
        """Calculate overall conversation tone based on combined state factors."""

        mood = effective_state.get("mood", {}).get("numeric_value", 60)
        energy = effective_state.get("energy", {}).get("numeric_value", 70)
        stress = effective_state.get("stress", {}).get("numeric_value", 50)

        # Calculate composite scores
        positivity_score = (mood + (100 - stress)) / 2
        engagement_score = (energy + mood) / 2
        stability_score = 100 - stress

        # Determine overall tone
        if positivity_score >= 70 and engagement_score >= 65:
            overall_tone = "enthusiastic"
        elif positivity_score >= 60 and engagement_score >= 55:
            overall_tone = "positive"
        elif positivity_score >= 45 and stability_score >= 60:
            overall_tone = "balanced"
        elif stability_score >= 50:
            overall_tone = "calm"
        else:
            overall_tone = "gentle"

        return {
            "overall_tone": overall_tone,
            "positivity_score": positivity_score,
            "engagement_score": engagement_score,
            "stability_score": stability_score,
            "tone_confidence": min(100, abs(positivity_score - 50) * 2)  # How confident we are in the tone
        }

    def _build_relationship_context(
        self,
        session_state: Optional[Dict[str, Any]],
        scenario: ConversationScenario
    ) -> Dict[str, Any]:
        """Build relationship context from session state."""

        if not session_state:
            return {
                "relationship_level": "new",
                "interaction_history": "none",
                "personalization_available": False,
                "conversation_continuity": "fresh_start"
            }

        conversation_context = session_state.get("conversation_context", {})
        session_metadata = session_state.get("session_metadata", {})

        total_interactions = session_metadata.get("total_interactions", 0)

        # Determine relationship level based on interaction history
        if total_interactions == 0:
            relationship_level = "new"
        elif total_interactions < 5:
            relationship_level = "developing"
        elif total_interactions < 15:
            relationship_level = "familiar"
        else:
            relationship_level = "established"

        return {
            "relationship_level": relationship_level,
            "total_interactions": total_interactions,
            "preferred_communication_style": conversation_context.get("preferred_communication_style", "balanced"),
            "conversation_continuity": "continuing" if total_interactions > 0 else "fresh_start",
            "personalization_available": len(session_state.get("personalization", {})) > 0,
            "session_duration": session_metadata.get("session_duration_minutes", 0)
        }

    def _get_fallback_context(self, scenario: ConversationScenario) -> Dict[str, Any]:
        """Get fallback context when state data is unavailable."""

        return {
            "mood_influence": {
                "tone": "neutral",
                "energy_level": "balanced",
                "conversation_style": "friendly",
                "influence_strength": 0.5,
                "mood_descriptor": "neutral"
            },
            "energy_influence": {
                "responsiveness": "moderate",
                "conversation_pace": "steady",
                "detail_level": "balanced",
                "influence_strength": 0.5,
                "energy_descriptor": "moderate"
            },
            "stress_influence": {
                "communication_approach": "balanced",
                "topic_sensitivity": "normal",
                "patience_level": "normal",
                "support_focus": "low",
                "influence_strength": 0.5,
                "stress_descriptor": "moderate"
            },
            "overall_tone": "balanced",
            "scenario": scenario.value,
            "relationship_level": "new",
            "fallback_mode": True
        }

    async def get_state_influence_summary(
        self,
        user_id: str,
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Get summary of how state is influencing the current conversation.

        Args:
            user_id: Unique identifier for the user
            conversation_id: Conversation session identifier

        Returns:
            Summary of state influence factors
        """
        try:
            effective_state = await self.session_service.get_effective_state(user_id, conversation_id)
            session_state = await self.session_service.get_session_state(user_id, conversation_id)

            # Extract key values
            mood = effective_state.get("mood", {}).get("numeric_value", 60)
            energy = effective_state.get("energy", {}).get("numeric_value", 70)
            stress = effective_state.get("stress", {}).get("numeric_value", 50)

            # Determine primary influences
            primary_influences = []
            if mood >= 70 or mood <= 30:
                primary_influences.append(f"mood ({mood}/100)")
            if energy >= 80 or energy <= 25:
                primary_influences.append(f"energy ({energy}/100)")
            if stress >= 65:
                primary_influences.append(f"stress ({stress}/100)")

            # Session adjustments impact
            session_adjustments = 0
            if session_state:
                session_adjustments = len(session_state.get("session_adjustments", {}))

            return {
                "primary_influences": primary_influences,
                "session_adjustments_active": session_adjustments,
                "overall_state_impact": "significant" if len(primary_influences) > 1 else "moderate" if primary_influences else "minimal",
                "mood_level": mood,
                "energy_level": energy,
                "stress_level": stress,
                "personalization_active": session_adjustments > 0
            }

        except Exception as e:
            logger.error(f"Error getting state influence summary: {e}")
            return {
                "primary_influences": [],
                "session_adjustments_active": 0,
                "overall_state_impact": "unknown",
                "error": str(e)
            }