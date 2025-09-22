"""
Tests for StateInfluenceService with various state combinations.
Tests conversation context building and state influence algorithms.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from app.services.state_influence_service import StateInfluenceService, ConversationScenario


class TestStateInfluenceService:
    """Test suite for StateInfluenceService conversation context building."""

    @pytest.fixture
    def influence_service(self):
        """Create StateInfluenceService instance for testing."""
        service = StateInfluenceService()
        service.session_service = Mock()
        service.state_manager = Mock()
        return service

    @pytest.fixture
    def high_mood_state(self):
        """Mock effective state with high mood."""
        return {
            'mood': {'numeric_value': 85, 'trend': 'increasing'},
            'energy': {'numeric_value': 75, 'trend': 'stable'},
            'stress': {'numeric_value': 30, 'trend': 'decreasing'},
            'social_satisfaction': {'numeric_value': 80, 'trend': 'increasing'}
        }

    @pytest.fixture
    def low_mood_state(self):
        """Mock effective state with low mood."""
        return {
            'mood': {'numeric_value': 25, 'trend': 'decreasing'},
            'energy': {'numeric_value': 40, 'trend': 'decreasing'},
            'stress': {'numeric_value': 75, 'trend': 'increasing'},
            'social_satisfaction': {'numeric_value': 35, 'trend': 'decreasing'}
        }

    @pytest.fixture
    def balanced_state(self):
        """Mock effective state with balanced values."""
        return {
            'mood': {'numeric_value': 60, 'trend': 'stable'},
            'energy': {'numeric_value': 65, 'trend': 'stable'},
            'stress': {'numeric_value': 50, 'trend': 'stable'},
            'social_satisfaction': {'numeric_value': 60, 'trend': 'stable'}
        }

    @pytest.fixture
    def established_session_state(self):
        """Mock session state for established relationship."""
        return {
            'conversation_context': {
                'relationship_level': 'familiar',
                'preferred_communication_style': 'enthusiastic',
                'user_mood_indicators': ['happy', 'engaged']
            },
            'session_metadata': {
                'total_interactions': 12,
                'session_duration_minutes': 45
            },
            'personalization': {
                'topics_of_interest': ['technology', 'music']
            }
        }

    def test_calculate_mood_influence_very_positive(self, influence_service):
        """Test mood influence calculation with very positive mood."""
        mood_influence = influence_service._calculate_mood_influence(85, 0.8)

        assert mood_influence['tone'] == 'upbeat'
        assert mood_influence['energy_level'] == 'high'
        assert mood_influence['conversation_style'] == 'enthusiastic'
        assert mood_influence['mood_descriptor'] == 'very positive'
        assert mood_influence['influence_strength'] > 0.5

    def test_calculate_mood_influence_very_low(self, influence_service):
        """Test mood influence calculation with very low mood."""
        mood_influence = influence_service._calculate_mood_influence(20, 0.9)

        assert mood_influence['tone'] == 'careful'
        assert mood_influence['energy_level'] == 'very_low'
        assert mood_influence['conversation_style'] == 'supportive'
        assert mood_influence['mood_descriptor'] == 'low'
        assert mood_influence['influence_strength'] > 0.4

    def test_calculate_mood_influence_neutral(self, influence_service):
        """Test mood influence calculation with neutral mood."""
        mood_influence = influence_service._calculate_mood_influence(50, 0.7)

        assert mood_influence['tone'] == 'neutral'
        assert mood_influence['energy_level'] == 'balanced'
        assert mood_influence['conversation_style'] == 'calm'
        assert mood_influence['mood_descriptor'] == 'neutral'
        assert mood_influence['influence_strength'] == 0.0

    def test_calculate_energy_influence_very_high(self, influence_service):
        """Test energy influence calculation with very high energy."""
        energy_influence = influence_service._calculate_energy_influence(90, 0.8)

        assert energy_influence['responsiveness'] == 'very_high'
        assert energy_influence['conversation_pace'] == 'energetic'
        assert energy_influence['detail_level'] == 'comprehensive'
        assert energy_influence['energy_descriptor'] == 'very high'

    def test_calculate_energy_influence_very_low(self, influence_service):
        """Test energy influence calculation with very low energy."""
        energy_influence = influence_service._calculate_energy_influence(15, 0.6)

        assert energy_influence['responsiveness'] == 'minimal'
        assert energy_influence['conversation_pace'] == 'slow'
        assert energy_influence['detail_level'] == 'brief'
        assert energy_influence['energy_descriptor'] == 'very low'

    def test_calculate_stress_influence_very_high(self, influence_service):
        """Test stress influence calculation with very high stress."""
        stress_influence = influence_service._calculate_stress_influence(85, 0.9)

        assert stress_influence['communication_approach'] == 'very_gentle'
        assert stress_influence['topic_sensitivity'] == 'high'
        assert stress_influence['patience_level'] == 'maximum'
        assert stress_influence['support_focus'] == 'high'
        assert stress_influence['stress_descriptor'] == 'very high'

    def test_calculate_stress_influence_very_low(self, influence_service):
        """Test stress influence calculation with very low stress."""
        stress_influence = influence_service._calculate_stress_influence(15, 0.5)

        assert stress_influence['communication_approach'] == 'casual'
        assert stress_influence['topic_sensitivity'] == 'minimal'
        assert stress_influence['patience_level'] == 'normal'
        assert stress_influence['support_focus'] == 'none'
        assert stress_influence['stress_descriptor'] == 'very low'

    def test_calculate_social_influence_very_satisfied(self, influence_service):
        """Test social influence calculation with high social satisfaction."""
        social_influence = influence_service._calculate_social_influence(85)

        assert social_influence['social_openness'] == 'high'
        assert social_influence['interaction_warmth'] == 'warm'
        assert social_influence['conversation_depth'] == 'open'
        assert social_influence['social_descriptor'] == 'very satisfied'

    def test_calculate_social_influence_dissatisfied(self, influence_service):
        """Test social influence calculation with low social satisfaction."""
        social_influence = influence_service._calculate_social_influence(25)

        assert social_influence['social_openness'] == 'reserved'
        assert social_influence['interaction_warmth'] == 'formal'
        assert social_influence['conversation_depth'] == 'minimal'
        assert social_influence['social_descriptor'] == 'dissatisfied'

    @pytest.mark.asyncio
    async def test_calculate_conversation_tone_enthusiastic(self, influence_service, high_mood_state):
        """Test conversation tone calculation for enthusiastic tone."""
        tone_data = await influence_service._calculate_conversation_tone(
            high_mood_state, ConversationScenario.CASUAL_CHAT
        )

        assert tone_data['overall_tone'] == 'enthusiastic'
        assert tone_data['positivity_score'] >= 70
        assert tone_data['engagement_score'] >= 65
        assert tone_data['tone_confidence'] > 50

    @pytest.mark.asyncio
    async def test_calculate_conversation_tone_gentle(self, influence_service, low_mood_state):
        """Test conversation tone calculation for gentle tone."""
        tone_data = await influence_service._calculate_conversation_tone(
            low_mood_state, ConversationScenario.SUPPORT_SESSION
        )

        assert tone_data['overall_tone'] == 'gentle'
        assert tone_data['positivity_score'] < 50
        assert tone_data['stability_score'] < 50

    @pytest.mark.asyncio
    async def test_calculate_conversation_tone_balanced(self, influence_service, balanced_state):
        """Test conversation tone calculation for balanced tone."""
        tone_data = await influence_service._calculate_conversation_tone(
            balanced_state, ConversationScenario.CASUAL_CHAT
        )

        assert tone_data['overall_tone'] == 'balanced'
        assert 45 <= tone_data['positivity_score'] <= 65
        assert tone_data['stability_score'] >= 50

    def test_build_relationship_context_new_user(self, influence_service):
        """Test relationship context building for new user."""
        context = influence_service._build_relationship_context(None, ConversationScenario.FIRST_MEETING)

        assert context['relationship_level'] == 'new'
        assert context['interaction_history'] == 'none'
        assert context['personalization_available'] is False
        assert context['conversation_continuity'] == 'fresh_start'

    def test_build_relationship_context_established_user(self, influence_service, established_session_state):
        """Test relationship context building for established user."""
        context = influence_service._build_relationship_context(
            established_session_state, ConversationScenario.DEEP_CONVERSATION
        )

        assert context['relationship_level'] == 'familiar'  # 12 interactions
        assert context['total_interactions'] == 12
        assert context['preferred_communication_style'] == 'enthusiastic'
        assert context['conversation_continuity'] == 'continuing'
        assert context['personalization_available'] is True

    @pytest.mark.asyncio
    async def test_build_conversation_context_casual_chat(self, influence_service, balanced_state):
        """Test building conversation context for casual chat scenario."""
        influence_service.session_service.get_effective_state = AsyncMock(return_value=balanced_state)
        influence_service.session_service.get_session_state = AsyncMock(return_value=None)

        context = await influence_service.build_conversation_context(
            'user123', 'conv456', ConversationScenario.CASUAL_CHAT
        )

        assert context['scenario'] == 'casual_chat'
        assert 'mood_influence' in context
        assert 'energy_influence' in context
        assert 'stress_influence' in context
        assert 'social_influence' in context
        assert 'overall_tone' in context
        assert 'relationship_level' in context
        assert context['global_weight'] == 0.6  # Casual chat weights
        assert context['session_weight'] == 0.4

    @pytest.mark.asyncio
    async def test_build_conversation_context_support_session(self, influence_service, low_mood_state):
        """Test building conversation context for support session scenario."""
        influence_service.session_service.get_effective_state = AsyncMock(return_value=low_mood_state)
        influence_service.session_service.get_session_state = AsyncMock(return_value=None)

        context = await influence_service.build_conversation_context(
            'user123', 'conv456', ConversationScenario.SUPPORT_SESSION
        )

        assert context['scenario'] == 'support_session'
        assert context['global_weight'] == 0.5  # Support session weights
        assert context['session_weight'] == 0.5
        assert context['stress_influence']['communication_approach'] == 'very_gentle'
        assert context['mood_influence']['conversation_style'] == 'supportive'

    @pytest.mark.asyncio
    async def test_build_conversation_context_creative_collaboration(self, influence_service, high_mood_state):
        """Test building conversation context for creative collaboration."""
        influence_service.session_service.get_effective_state = AsyncMock(return_value=high_mood_state)
        influence_service.session_service.get_session_state = AsyncMock(return_value=None)

        context = await influence_service.build_conversation_context(
            'user123', 'conv456', ConversationScenario.CREATIVE_COLLABORATION
        )

        assert context['scenario'] == 'creative_collaboration'
        assert context['energy_influence']['influence_strength'] > 0.7  # High energy impact
        assert context['mood_influence']['conversation_style'] == 'enthusiastic'

    @pytest.mark.asyncio
    async def test_build_conversation_context_with_user_preferences(self, influence_service, balanced_state):
        """Test conversation context building with user preference overrides."""
        user_preferences = {
            'state_influence_overrides': {
                'mood_sensitivity': 1.0,  # Higher than default
                'stress_awareness': 0.2   # Lower than default
            }
        }

        influence_service.session_service.get_effective_state = AsyncMock(return_value=balanced_state)
        influence_service.session_service.get_session_state = AsyncMock(return_value=None)

        context = await influence_service.build_conversation_context(
            'user123', 'conv456', ConversationScenario.CASUAL_CHAT, user_preferences
        )

        # Preferences should override scenario defaults
        assert 'mood_influence' in context
        assert 'stress_influence' in context

    @pytest.mark.asyncio
    async def test_get_state_influence_summary_significant_impact(self, influence_service, high_mood_state, established_session_state):
        """Test state influence summary with significant state impact."""
        # High mood and low stress should be significant influences
        high_mood_state['stress']['numeric_value'] = 20  # Very low stress
        established_session_state['session_adjustments'] = {'mood': {'value': 5}}

        influence_service.session_service.get_effective_state = AsyncMock(return_value=high_mood_state)
        influence_service.session_service.get_session_state = AsyncMock(return_value=established_session_state)

        summary = await influence_service.get_state_influence_summary('user123', 'conv456')

        assert len(summary['primary_influences']) >= 1  # Should include mood
        assert summary['overall_state_impact'] == 'significant'
        assert summary['session_adjustments_active'] == 1
        assert summary['personalization_active'] is True

    @pytest.mark.asyncio
    async def test_get_state_influence_summary_minimal_impact(self, influence_service, balanced_state):
        """Test state influence summary with minimal state impact."""
        influence_service.session_service.get_effective_state = AsyncMock(return_value=balanced_state)
        influence_service.session_service.get_session_state = AsyncMock(return_value=None)

        summary = await influence_service.get_state_influence_summary('user123', 'conv456')

        assert len(summary['primary_influences']) == 0  # No extreme values
        assert summary['overall_state_impact'] == 'minimal'
        assert summary['session_adjustments_active'] == 0
        assert summary['personalization_active'] is False

    def test_scenario_weights_configuration(self, influence_service):
        """Test that all conversation scenarios have proper weight configuration."""
        for scenario in ConversationScenario:
            weights = influence_service.scenario_weights[scenario]

            # All scenarios should have required weight keys
            assert 'global_influence' in weights
            assert 'session_influence' in weights
            assert 'mood_sensitivity' in weights
            assert 'energy_impact' in weights
            assert 'stress_awareness' in weights

            # Weights should sum to 1.0 for global/session
            assert abs((weights['global_influence'] + weights['session_influence']) - 1.0) < 0.01

            # All weights should be between 0 and 1
            for weight in weights.values():
                assert 0 <= weight <= 1

    def test_get_fallback_context(self, influence_service):
        """Test fallback context when state data is unavailable."""
        fallback = influence_service._get_fallback_context(ConversationScenario.CASUAL_CHAT)

        assert fallback['fallback_mode'] is True
        assert fallback['scenario'] == 'casual_chat'
        assert 'mood_influence' in fallback
        assert 'energy_influence' in fallback
        assert 'stress_influence' in fallback
        assert fallback['relationship_level'] == 'new'

        # All influence values should be neutral/moderate
        assert fallback['mood_influence']['tone'] == 'neutral'
        assert fallback['energy_influence']['responsiveness'] == 'moderate'
        assert fallback['stress_influence']['communication_approach'] == 'balanced'

    @pytest.mark.asyncio
    async def test_build_conversation_context_error_handling(self, influence_service):
        """Test conversation context building with service errors."""
        influence_service.session_service.get_effective_state = AsyncMock(side_effect=Exception("Service error"))

        context = await influence_service.build_conversation_context(
            'user123', 'conv456', ConversationScenario.CASUAL_CHAT
        )

        # Should return fallback context
        assert context['fallback_mode'] is True
        assert context['scenario'] == 'casual_chat'

    def test_state_influence_edge_cases(self, influence_service):
        """Test state influence calculations with edge case values."""
        # Test with extreme values
        extreme_cases = [
            (0, 0.5),    # Minimum values
            (100, 1.0),  # Maximum values
            (50, 0.0),   # Zero sensitivity
        ]

        for value, sensitivity in extreme_cases:
            mood_influence = influence_service._calculate_mood_influence(value, sensitivity)
            energy_influence = influence_service._calculate_energy_influence(value, sensitivity)
            stress_influence = influence_service._calculate_stress_influence(value, sensitivity)

            # Should not crash and should return valid structures
            assert 'tone' in mood_influence
            assert 'responsiveness' in energy_influence
            assert 'communication_approach' in stress_influence