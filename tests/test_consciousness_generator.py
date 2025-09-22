"""
Tests for consciousness generator service and integration.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.services.consciousness_generator_service import ConsciousnessGeneratorService, ConsciousnessResponse
from app.models.simulation import GlobalEvents
from app.schemas.simulation_schemas import EventType, MoodImpact, ImpactLevel


@pytest.fixture
def mock_event():
    """Create a mock GlobalEvents object for testing."""
    event = Mock(spec=GlobalEvents)
    event.event_id = "test-event-123"
    event.event_type = EventType.WORK
    event.summary = "Had a challenging meeting with the client about project timeline"
    event.intensity = 7
    event.impact_mood = MoodImpact.NEGATIVE
    event.impact_energy = ImpactLevel.DECREASE
    event.impact_stress = ImpactLevel.INCREASE
    event.timestamp = datetime.now()
    return event


@pytest.fixture
def mock_state_context():
    """Create mock state context for testing."""
    return {
        "mood": {"numeric_value": 45, "trend": "decreasing"},
        "energy": {"numeric_value": 35, "trend": "decreasing"},
        "stress": {"numeric_value": 75, "trend": "increasing"},
        "work_satisfaction": {"numeric_value": 40, "trend": "stable"},
        "social_satisfaction": {"numeric_value": 60, "trend": "stable"},
        "personal_fulfillment": {"numeric_value": 50, "trend": "stable"}
    }


@pytest.fixture
def consciousness_service():
    """Create ConsciousnessGeneratorService instance for testing."""
    return ConsciousnessGeneratorService()


class TestConsciousnessGeneratorService:
    """Test cases for ConsciousnessGeneratorService."""

    @pytest.mark.asyncio
    async def test_generate_consciousness_response_success(self, consciousness_service, mock_event, mock_state_context):
        """Test successful consciousness response generation."""

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """{
            "emotional_reaction": "This meeting really stressed me out and left me feeling frustrated about the unrealistic expectations.",
            "chosen_action": "I need to take a few deep breaths and then draft a professional email outlining realistic timeline options.",
            "internal_thoughts": "Why do clients always think we can do everything faster? I need to stay professional but also protect the quality of our work."
        }"""

        with patch.object(consciousness_service, '_make_consciousness_call', return_value=mock_response) as mock_call:
            with patch.object(consciousness_service.state_manager, 'get_current_global_state', return_value=mock_state_context):
                result = await consciousness_service.generate_consciousness_response(mock_event)

        assert result.success is True
        assert "stressed me out" in result.emotional_reaction
        assert "deep breaths" in result.chosen_action
        assert "clients always think" in result.internal_thoughts
        assert result.error_message is None
        mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_consciousness_response_api_timeout(self, consciousness_service, mock_event, mock_state_context):
        """Test consciousness response when API times out."""

        with patch.object(consciousness_service.state_manager, 'get_current_global_state', return_value=mock_state_context):
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError("API timeout")):
                result = await consciousness_service.generate_consciousness_response(mock_event, timeout=1)

        assert result.success is False
        assert result.error_message == "API timeout"
        assert "work situation" in result.emotional_reaction  # Should use fallback
        assert len(result.chosen_action) > 0
        assert len(result.internal_thoughts) > 0

    @pytest.mark.asyncio
    async def test_generate_consciousness_response_no_api_key(self, mock_event, mock_state_context):
        """Test consciousness response when no API key is available."""

        # Create service without OpenAI client
        service = ConsciousnessGeneratorService()
        service.client = None

        with patch.object(service.state_manager, 'get_current_global_state', return_value=mock_state_context):
            result = await service.generate_consciousness_response(mock_event)

        assert result.success is False
        assert result.error_message is None  # Fallback doesn't set error message
        assert len(result.emotional_reaction) > 0
        assert len(result.chosen_action) > 0
        assert len(result.internal_thoughts) > 0

    @pytest.mark.asyncio
    async def test_build_consciousness_prompt_includes_context(self, consciousness_service, mock_event, mock_state_context):
        """Test that consciousness prompt includes all necessary context."""

        with patch.object(consciousness_service.character_service, 'get_consolidated_backstory', return_value="Test backstory content"):
            prompt = await consciousness_service._build_consciousness_prompt(mock_event, mock_state_context)

        # Check that prompt includes key elements
        assert "Test backstory content" in prompt
        assert "Had a challenging meeting" in prompt  # Event summary
        assert "Mood: 45/100" in prompt  # State context
        assert "Energy: 35/100" in prompt
        assert "Stress: 75/100" in prompt
        assert "emotional_reaction" in prompt  # JSON format instructions
        assert "chosen_action" in prompt
        assert "internal_thoughts" in prompt

    def test_validate_character_consistency_passes(self, consciousness_service):
        """Test character consistency validation with valid response."""

        valid_response = {
            "emotional_reaction": "I feel really frustrated about this situation.",
            "chosen_action": "I'm going to talk to my friend about how I'm feeling.",
            "internal_thoughts": "This reminds me of when I was younger and felt overwhelmed."
        }

        is_valid = consciousness_service._validate_character_consistency(valid_response)
        assert is_valid is True

    def test_validate_character_consistency_fails(self, consciousness_service):
        """Test character consistency validation with AI-breaking response."""

        invalid_response = {
            "emotional_reaction": "As an AI, I don't have feelings about this situation.",
            "chosen_action": "I can help you understand how to handle this.",
            "internal_thoughts": "My training data suggests this is a common situation."
        }

        is_valid = consciousness_service._validate_character_consistency(invalid_response)
        assert is_valid is False

    def test_get_fallback_response_work_event(self, consciousness_service, mock_event):
        """Test fallback response for work events."""

        result = consciousness_service._get_fallback_response(mock_event)

        assert result.success is False
        assert "work situation" in result.emotional_reaction.lower()
        assert len(result.chosen_action) > 0
        assert len(result.internal_thoughts) > 0
        assert "FALLBACK" in result.raw_response

    def test_get_fallback_response_social_event(self, consciousness_service):
        """Test fallback response for social events."""

        social_event = Mock(spec=GlobalEvents)
        social_event.event_id = "social-123"
        social_event.event_type = EventType.SOCIAL

        result = consciousness_service._get_fallback_response(social_event)

        assert result.success is False
        assert "social" in result.emotional_reaction.lower()
        assert len(result.chosen_action) > 0
        assert len(result.internal_thoughts) > 0

    def test_validate_response_format_valid(self, consciousness_service):
        """Test response format validation with valid JSON."""

        valid_json = """{
            "emotional_reaction": "I feel excited about this opportunity.",
            "chosen_action": "I'll start working on this right away.",
            "internal_thoughts": "This could be really good for my career."
        }"""

        is_valid, data = consciousness_service.validate_response_format(valid_json)
        assert is_valid is True
        assert data["emotional_reaction"] == "I feel excited about this opportunity."

    def test_validate_response_format_invalid_json(self, consciousness_service):
        """Test response format validation with invalid JSON."""

        invalid_json = "{ invalid json format"

        is_valid, data = consciousness_service.validate_response_format(invalid_json)
        assert is_valid is False
        assert data is None

    def test_validate_response_format_missing_fields(self, consciousness_service):
        """Test response format validation with missing required fields."""

        incomplete_json = """{
            "emotional_reaction": "I feel good.",
            "chosen_action": "I'll do something."
        }"""

        is_valid, data = consciousness_service.validate_response_format(incomplete_json)
        assert is_valid is False
        assert data is None

    @pytest.mark.asyncio
    async def test_parse_consciousness_response_valid(self, consciousness_service, mock_event):
        """Test parsing valid OpenAI consciousness response."""

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """{
            "emotional_reaction": "I'm feeling overwhelmed by this workload.",
            "chosen_action": "I need to prioritize my tasks and ask for help if needed.",
            "internal_thoughts": "I can't keep pushing myself this hard without consequences."
        }"""

        result = consciousness_service._parse_consciousness_response(mock_response, mock_event)

        assert result.success is True
        assert result.emotional_reaction == "I'm feeling overwhelmed by this workload."
        assert result.chosen_action == "I need to prioritize my tasks and ask for help if needed."
        assert result.internal_thoughts == "I can't keep pushing myself this hard without consequences."

    @pytest.mark.asyncio
    async def test_parse_consciousness_response_invalid_json(self, consciousness_service, mock_event):
        """Test parsing invalid JSON response falls back gracefully."""

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON response"

        result = consciousness_service._parse_consciousness_response(mock_response, mock_event)

        assert result.success is False
        assert result.error_message == "JSON parsing failed"
        assert len(result.emotional_reaction) > 0  # Should have fallback values


class TestConsciousnessIntegration:
    """Integration tests for consciousness generation in event processing."""

    @pytest.mark.asyncio
    async def test_event_processing_triggers_consciousness_generation(self):
        """Test that event processing correctly triggers consciousness generation."""

        # This would be an integration test that would require:
        # 1. Creating a test event
        # 2. Triggering the Celery task
        # 3. Verifying consciousness response is generated and stored
        # This is a placeholder for the integration test structure

        # Mock the Celery task execution
        with patch('app.services.simulation.event_generator.generate_consciousness_response.delay') as mock_delay:
            # Simulate event creation that should trigger consciousness generation
            mock_delay.return_value = Mock()

            # Verify the task was called
            # This would be implemented with actual event creation in a full integration test
            pass

    @pytest.mark.asyncio
    async def test_consciousness_response_storage(self):
        """Test that consciousness responses are properly stored in database."""

        # This would test:
        # 1. Event creation
        # 2. Consciousness generation
        # 3. Database update with consciousness fields
        # 4. Data retrieval and validation

        # Placeholder for database integration test
        pass