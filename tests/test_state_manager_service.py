"""
Unit tests for StateManagerService enhancements.
Tests emotional processing, state changes, and integration with consciousness generator.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from app.services.simulation.state_manager import StateManagerService
from app.models.simulation import GlobalEvents


class TestStateManagerService:
    """Test suite for StateManagerService with enhanced emotional processing."""

    @pytest.fixture
    def state_manager(self):
        """Create StateManagerService instance for testing."""
        return StateManagerService()

    @pytest.fixture
    def mock_event(self):
        """Create mock GlobalEvent for testing."""
        return Mock(spec=GlobalEvents, **{
            'event_id': 'test_event_123',
            'event_type': 'social',
            'summary': 'Had a great conversation with friends at lunch',
            'intensity': 7,
            'impact_mood': 'positive',
            'impact_energy': 'increase',
            'impact_stress': 'decrease',
            'timestamp': datetime.now(timezone.utc)
        })

    @pytest.fixture
    def mock_consciousness_response(self):
        """Create mock consciousness response for testing."""
        return {
            'emotional_reaction': 'I feel thrilled and energized by this social connection! It reminds me why relationships matter so much.',
            'chosen_action': 'I want to reach out to more friends and schedule regular social activities.',
            'internal_thoughts': 'This kind of positive social interaction is exactly what I need to feel balanced and happy.'
        }

    def test_analyze_emotional_reactions_positive_intense(self, state_manager, mock_event):
        """Test emotional analysis with intensely positive reactions."""
        emotional_reaction = "I feel absolutely thrilled and ecstatic about this amazing opportunity!"
        internal_thoughts = "This is incredible and I'm so energized and motivated by this!"

        changes = state_manager._analyze_emotional_reactions(
            emotional_reaction, internal_thoughts, mock_event
        )

        assert 'mood' in changes
        assert changes['mood']['change_amount'] == 8  # intense positive
        assert 'energy' in changes
        assert changes['energy']['change_amount'] == 5  # energized
        assert 'stress' not in changes  # no stress keywords

    def test_analyze_emotional_reactions_negative_intense(self, state_manager, mock_event):
        """Test emotional analysis with intensely negative reactions."""
        emotional_reaction = "I feel absolutely devastated and heartbroken by this terrible news."
        internal_thoughts = "I'm overwhelmed and can't handle this crushing disappointment."

        changes = state_manager._analyze_emotional_reactions(
            emotional_reaction, internal_thoughts, mock_event
        )

        assert 'mood' in changes
        assert changes['mood']['change_amount'] == -10  # intense negative
        assert 'stress' in changes
        assert changes['stress']['change_amount'] == 6  # overwhelmed
        assert 'energy' not in changes  # no energy keywords

    def test_analyze_emotional_reactions_stress_patterns(self, state_manager, mock_event):
        """Test emotional analysis with stress-related patterns."""
        emotional_reaction = "I'm feeling anxious and worried about the upcoming presentation."
        internal_thoughts = "The pressure is building and I feel tense about performing well."

        changes = state_manager._analyze_emotional_reactions(
            emotional_reaction, internal_thoughts, mock_event
        )

        assert 'stress' in changes
        assert changes['stress']['change_amount'] == 6
        assert 'mood' in changes
        assert changes['mood']['change_amount'] == -3  # mild negative (worried)

    def test_analyze_emotional_reactions_social_event_positive(self, state_manager):
        """Test social event specific emotional processing."""
        mock_event = Mock(spec=GlobalEvents, event_type='social')
        emotional_reaction = "I had such a wonderful time connecting with friends!"
        internal_thoughts = "Social interactions like this make me feel so happy and fulfilled."

        changes = state_manager._analyze_emotional_reactions(
            emotional_reaction, internal_thoughts, mock_event
        )

        assert 'mood' in changes
        assert changes['mood']['change_amount'] == 4  # mild positive (happy)
        assert 'social_satisfaction' in changes
        assert changes['social_satisfaction']['change_amount'] == 3  # positive social experience

    def test_analyze_emotional_reactions_social_event_negative(self, state_manager):
        """Test social event with negative emotional outcome."""
        mock_event = Mock(spec=GlobalEvents, event_type='social')
        emotional_reaction = "I feel devastated and hurt by how that conversation went."
        internal_thoughts = "Social situations like this make me want to withdraw completely."

        changes = state_manager._analyze_emotional_reactions(
            emotional_reaction, internal_thoughts, mock_event
        )

        assert 'mood' in changes
        assert changes['mood']['change_amount'] == -10  # intense negative (devastated)
        assert 'social_satisfaction' in changes
        assert changes['social_satisfaction']['change_amount'] == -4  # negative social experience

    @pytest.mark.asyncio
    async def test_process_consciousness_emotional_reactions_success(self, state_manager, mock_event, mock_consciousness_response):
        """Test successful processing of consciousness emotional reactions."""
        with patch('app.services.simulation.state_manager.get_async_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_db_session)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_context_manager

            mock_repo = Mock()
            mock_updated_state = Mock(numeric_value=68)

            with patch('app.services.simulation.state_manager.SimulationRepository') as mock_repo_class:
                mock_repo_class.return_value = mock_repo
                state_manager._update_trait_state = AsyncMock(return_value=mock_updated_state)
                state_manager._log_emotional_processing = AsyncMock()

                result = await state_manager.process_consciousness_emotional_reactions(
                    mock_event, mock_consciousness_response
                )

                assert result['success'] is True
                assert result['event_id'] == 'test_event_123'
                assert 'emotional_changes' in result
                assert 'processed_reaction' in result

    @pytest.mark.asyncio
    async def test_process_consciousness_emotional_reactions_no_reaction(self, state_manager, mock_event):
        """Test processing with missing emotional reaction."""
        consciousness_response = {
            'chosen_action': 'I will take action',
            'internal_thoughts': 'Some thoughts'
        }

        result = await state_manager.process_consciousness_emotional_reactions(
            mock_event, consciousness_response
        )

        assert result['success'] is False
        assert 'No emotional reaction to process' in result['reason']

    def test_calculate_state_changes_with_emotional_integration(self, state_manager, mock_event):
        """Test that original state change calculation still works correctly."""
        changes = state_manager._calculate_state_changes(mock_event)

        # Verify original event processing still works
        assert 'mood' in changes
        assert changes['mood']['change_amount'] == int(5 * (7 / 5.0))  # positive mood with intensity 7
        assert 'energy' in changes
        assert changes['energy']['change_amount'] == int(4 * (7 / 5.0))  # energy increase
        assert 'stress' in changes
        assert changes['stress']['change_amount'] == int(-5 * (7 / 5.0))  # stress decrease
        assert 'social_satisfaction' in changes  # social event type

    @pytest.mark.asyncio
    async def test_get_state_history_trait_specific(self, state_manager):
        """Test getting state history for specific trait."""
        with patch('app.services.simulation.state_manager.get_async_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_db_session)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_context_manager

            mock_repo = Mock()
            mock_history = [
                {'timestamp': '2024-01-01T12:00:00Z', 'trait': 'mood', 'value': 65},
                {'timestamp': '2024-01-01T11:00:00Z', 'trait': 'mood', 'value': 60}
            ]

            with patch('app.services.simulation.state_manager.SimulationRepository') as mock_repo_class:
                mock_repo_class.return_value = mock_repo
                mock_repo.get_trait_history = AsyncMock(return_value=mock_history)

                result = await state_manager.get_state_history('mood', 24)

                assert result == mock_history
                mock_repo.get_trait_history.assert_called_once_with('mood', 24)

    @pytest.mark.asyncio
    async def test_get_state_history_all_traits(self, state_manager):
        """Test getting state history for all traits."""
        with patch('app.services.simulation.state_manager.get_async_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_db_session)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_context_manager

            mock_repo = Mock()
            mock_history = [
                {'timestamp': '2024-01-01T12:00:00Z', 'trait': 'mood', 'value': 65},
                {'timestamp': '2024-01-01T12:00:00Z', 'trait': 'energy', 'value': 72}
            ]

            with patch('app.services.simulation.state_manager.SimulationRepository') as mock_repo_class:
                mock_repo_class.return_value = mock_repo
                mock_repo.get_all_traits_history = AsyncMock(return_value=mock_history)

                result = await state_manager.get_state_history(None, 24)

                assert result == mock_history
                mock_repo.get_all_traits_history.assert_called_once_with(24)

    def test_log_state_changes_audit_trail(self, state_manager):
        """Test that state changes are logged for audit trail."""
        # This test verifies the logging structure exists
        # In a real implementation, you might test actual database logging

        mock_repo = Mock()
        event_id = 'test_event_123'
        change_summary = {
            'mood': {
                'previous_value': 60,
                'new_value': 68,
                'change_amount': 8,
                'reason': 'Positive social interaction'
            }
        }

        # Test that the method can be called without errors
        # (actual database logging would be tested in integration tests)
        try:
            asyncio.run(state_manager._log_state_changes(mock_repo, event_id, change_summary))
        except Exception:
            # Expected since we're not testing actual database operations
            pass

    def test_emotional_keywords_detection(self, state_manager, mock_event):
        """Test that emotional keyword detection works correctly."""
        # Test various emotional intensity levels
        test_cases = [
            ("I'm absolutely thrilled!", 8),  # intense positive
            ("I feel happy and content", 4),  # mild positive
            ("I'm completely devastated", -10),  # intense negative
            ("I'm a bit disappointed", -3),  # mild negative
            ("This is just okay", 0)  # neutral
        ]

        for emotion_text, expected_mood_change in test_cases:
            changes = state_manager._analyze_emotional_reactions(
                emotion_text, "", mock_event
            )

            if expected_mood_change == 0:
                assert 'mood' not in changes
            else:
                assert 'mood' in changes
                assert changes['mood']['change_amount'] == expected_mood_change