"""
Tests for EnhancedConversationService - Story 2.6 Enhanced Conversational Context Integration
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.enhanced_conversation_service import EnhancedConversationService


class TestEnhancedConversationService:
    """Test suite for EnhancedConversationService functionality."""

    @pytest.fixture
    def mock_services(self):
        """Mock all dependent services."""
        mocks = {
            'contextual_backstory_service': Mock(),
            'conversation_prompt_service': Mock(),
            'state_influence_service': Mock(),
            'state_manager_service': Mock(),
            'simple_openai_service': Mock(),
            'openai_client': Mock()
        }

        # Setup async methods
        mocks['contextual_backstory_service'].select_relevant_content = AsyncMock()
        mocks['state_influence_service'].build_conversation_context = AsyncMock()
        mocks['state_manager_service'].get_current_global_state = AsyncMock()
        mocks['state_manager_service'].get_recent_events = AsyncMock()
        mocks['simple_openai_service'].generate_coaching_response = AsyncMock()

        return mocks

    @pytest.fixture
    def service(self, mock_services):
        """Create EnhancedConversationService with mocked dependencies."""
        with patch('app.services.enhanced_conversation_service.ContextualBackstoryService') as mock_backstory, \
             patch('app.services.enhanced_conversation_service.ConversationPromptService') as mock_prompt, \
             patch('app.services.enhanced_conversation_service.StateInfluenceService') as mock_influence, \
             patch('app.services.enhanced_conversation_service.StateManagerService') as mock_state, \
             patch('app.services.enhanced_conversation_service.SimpleOpenAIService') as mock_simple, \
             patch('app.services.enhanced_conversation_service.OpenAI') as mock_openai:

            mock_backstory.return_value = mock_services['contextual_backstory_service']
            mock_prompt.return_value = mock_services['conversation_prompt_service']
            mock_influence.return_value = mock_services['state_influence_service']
            mock_state.return_value = mock_services['state_manager_service']
            mock_simple.return_value = mock_services['simple_openai_service']
            mock_openai.return_value = mock_services['openai_client']

            service = EnhancedConversationService()
            return service, mock_services

    @pytest.mark.asyncio
    async def test_successful_enhanced_response(self, service):
        """Test successful enhanced response generation."""
        enhanced_service, mocks = service

        # Setup mock responses
        mocks['state_manager_service'].get_current_global_state.return_value = {
            "mood": {"numeric_value": 70},
            "stress": {"numeric_value": 40}
        }
        mocks['state_manager_service'].get_recent_events.return_value = [
            {
                "event_id": "event1",
                "summary": "Had a good meeting",
                "hours_ago": 2
            }
        ]
        mocks['contextual_backstory_service'].select_relevant_content.return_value = {
            "content": "Character backstory content",
            "content_types": ["character_gist"],
            "char_count": 500,
            "estimated_tokens": 125
        }
        mocks['state_influence_service'].build_conversation_context.return_value = {
            "mood_influence": {"tone": "positive"}
        }

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI response based on context"
        mocks['openai_client'].chat.completions.create.return_value = mock_response

        # Mock prompt construction
        mocks['conversation_prompt_service'].determine_emotion_from_context.return_value = (
            Mock(value="happy"), "User seems positive"
        )
        mocks['conversation_prompt_service'].construct_conversation_prompt.return_value = "Enhanced prompt"

        result = await enhanced_service.generate_enhanced_response(
            user_message="How are you feeling today?",
            user_id="user123",
            conversation_id="conv456"
        )

        assert result["ai_response"] == "AI response based on context"
        assert result["enhanced_mode"] == True
        assert result["fallback_mode"] == False
        assert "performance_metrics" in result
        assert "simulation_context" in result

    @pytest.mark.asyncio
    async def test_fallback_to_simple_service(self, service):
        """Test fallback to SimpleOpenAIService when enhanced context fails."""
        enhanced_service, mocks = service

        # Make context gathering fail
        mocks['state_manager_service'].get_current_global_state.side_effect = Exception("Database error")

        # Setup fallback response
        fallback_response = Mock()
        fallback_response.ai_response = "Fallback response"
        fallback_response.corrected_transcript = "How are you feeling today?"
        fallback_response.word_usage_status = "NOT_USED"
        fallback_response.usage_correctness_feedback = None
        mocks['simple_openai_service'].generate_coaching_response.return_value = fallback_response

        result = await enhanced_service.generate_enhanced_response(
            user_message="How are you feeling today?",
            user_id="user123",
            conversation_id="conv456"
        )

        assert result["ai_response"] == "Fallback response"
        assert result["enhanced_mode"] == False
        assert result["fallback_mode"] == True
        assert "performance_metrics" in result

    @pytest.mark.asyncio
    async def test_performance_timing_metrics(self, service):
        """Test that performance timing metrics are captured."""
        enhanced_service, mocks = service

        # Setup mock responses with slight delays
        async def delayed_global_state():
            import asyncio
            await asyncio.sleep(0.01)  # 10ms delay
            return {"mood": {"numeric_value": 60}}

        async def delayed_events(**kwargs):
            import asyncio
            await asyncio.sleep(0.01)  # 10ms delay
            return []

        mocks['state_manager_service'].get_current_global_state.side_effect = delayed_global_state
        mocks['state_manager_service'].get_recent_events.side_effect = delayed_events
        mocks['contextual_backstory_service'].select_relevant_content.return_value = {
            "content": "Content",
            "content_types": ["character_gist"],
            "char_count": 100,
            "estimated_tokens": 25
        }
        mocks['state_influence_service'].build_conversation_context.return_value = {}

        # Setup OpenAI mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mocks['openai_client'].chat.completions.create.return_value = mock_response

        mocks['conversation_prompt_service'].determine_emotion_from_context.return_value = (
            Mock(value="neutral"), "Neutral emotion"
        )
        mocks['conversation_prompt_service'].construct_conversation_prompt.return_value = "Prompt"

        result = await enhanced_service.generate_enhanced_response(
            user_message="Test message",
            user_id="user123",
            conversation_id="conv456"
        )

        metrics = result["performance_metrics"]
        assert "context_gathering_ms" in metrics
        assert "response_generation_ms" in metrics
        assert "total_response_time_ms" in metrics
        assert metrics["context_gathering_ms"] > 0
        assert metrics["total_response_time_ms"] > 0

    @pytest.mark.skip(reason="Timing-dependent test - hard to predict in CI")
    @pytest.mark.asyncio
    async def test_context_gathering_timeout_warning(self, service):
        """Test warning when context gathering exceeds threshold."""
        enhanced_service, mocks = service
        enhanced_service.config.MAX_CONTEXT_PROCESSING_MS = 1  # Very low threshold

        # Setup mock responses
        mocks['state_manager_service'].get_current_global_state.return_value = {}
        mocks['state_manager_service'].get_recent_events.return_value = []
        mocks['contextual_backstory_service'].select_relevant_content.return_value = {
            "content": "Content",
            "content_types": [],
            "char_count": 0,
            "estimated_tokens": 0
        }
        mocks['state_influence_service'].build_conversation_context.return_value = {}

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mocks['openai_client'].chat.completions.create.return_value = mock_response

        mocks['conversation_prompt_service'].determine_emotion_from_context.return_value = (
            Mock(value="neutral"), "Neutral"
        )
        mocks['conversation_prompt_service'].construct_conversation_prompt.return_value = "Prompt"

        # This should log a warning about exceeding threshold
        with patch('app.services.enhanced_conversation_service.logger') as mock_logger:
            result = await enhanced_service.generate_enhanced_response(
                user_message="Test",
                user_id="user123",
                conversation_id="conv456"
            )

            # Should have logged a warning about context gathering time
            mock_logger.warning.assert_called()
            warning_calls = [call for call in mock_logger.warning.call_args_list
                           if "Context gathering took" in str(call)]
            assert len(warning_calls) > 0

    @pytest.mark.asyncio
    async def test_simulation_context_integration(self, service):
        """Test that simulation context is properly integrated."""
        enhanced_service, mocks = service

        # Setup comprehensive simulation context
        mocks['state_manager_service'].get_current_global_state.return_value = {
            "mood": {"numeric_value": 85},
            "stress": {"numeric_value": 30},
            "energy": {"numeric_value": 75}
        }
        mocks['state_manager_service'].get_recent_events.return_value = [
            {
                "event_id": "event1",
                "summary": "Completed an important project",
                "hours_ago": 1,
                "impact_mood": "positive"
            },
            {
                "event_id": "event2",
                "summary": "Had lunch with a friend",
                "hours_ago": 3,
                "impact_mood": "positive"
            }
        ]
        mocks['contextual_backstory_service'].select_relevant_content.return_value = {
            "content": "Ava loves creative projects and collaboration",
            "content_types": ["positive_memories", "character_gist"],
            "char_count": 800,
            "estimated_tokens": 200
        }
        mocks['state_influence_service'].build_conversation_context.return_value = {
            "mood_influence": {"tone": "upbeat", "energy_level": "high"},
            "overall_tone": "enthusiastic"
        }

        # Setup OpenAI mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "I'm feeling great! Just finished a big project."
        mocks['openai_client'].chat.completions.create.return_value = mock_response

        mocks['conversation_prompt_service'].determine_emotion_from_context.return_value = (
            Mock(value="excited"), "High energy and positive mood"
        )
        mocks['conversation_prompt_service'].construct_conversation_prompt.return_value = "Enhanced prompt with context"

        result = await enhanced_service.generate_enhanced_response(
            user_message="How has your day been?",
            user_id="user123",
            conversation_id="conv456"
        )

        simulation_context = result["simulation_context"]
        assert simulation_context["recent_events_count"] == 2
        assert simulation_context["global_mood"] == 85
        assert simulation_context["stress_level"] == 30
        assert "positive_memories" in simulation_context["selected_content_types"]
        assert simulation_context["conversation_emotion"] == "excited"

    @pytest.mark.asyncio
    async def test_word_usage_evaluation(self, service):
        """Test word usage evaluation functionality."""
        enhanced_service, mocks = service

        # Setup basic context
        mocks['state_manager_service'].get_current_global_state.return_value = {}
        mocks['state_manager_service'].get_recent_events.return_value = []
        mocks['contextual_backstory_service'].select_relevant_content.return_value = {
            "content": "Content",
            "content_types": [],
            "char_count": 100,
            "estimated_tokens": 25
        }
        mocks['state_influence_service'].build_conversation_context.return_value = {}

        # Setup OpenAI mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "That's a sophisticated approach!"
        mocks['openai_client'].chat.completions.create.return_value = mock_response

        mocks['conversation_prompt_service'].determine_emotion_from_context.return_value = (
            Mock(value="neutral"), "Neutral"
        )
        mocks['conversation_prompt_service'].construct_conversation_prompt.return_value = "Prompt"

        result = await enhanced_service.generate_enhanced_response(
            user_message="I used sophisticated methods to solve the problem",
            user_id="user123",
            conversation_id="conv456",
            suggested_word="sophisticated"
        )

        assert result["word_usage_status"].value == "used_correctly"
        assert result["usage_correctness_feedback"] is None

    @pytest.mark.asyncio
    async def test_context_summary_functionality(self, service):
        """Test get_context_summary functionality."""
        enhanced_service, mocks = service

        # Setup mocks for context summary
        mocks['state_influence_service'].get_state_influence_summary = AsyncMock(return_value={
            "primary_influences": ["mood (85/100)"],
            "overall_state_impact": "significant"
        })
        mocks['state_manager_service'].get_current_global_state.return_value = {
            "mood": {"numeric_value": 85}
        }
        mocks['contextual_backstory_service'].get_cache_status.return_value = {
            "cached_content_types": ["character_gist"],
            "cache_size": 1
        }

        result = await enhanced_service.get_context_summary("user123", "conv456")

        assert "state_influence" in result
        assert "global_state_available" in result
        assert "backstory_cache" in result
        assert "config" in result
        assert result["global_state_available"] == True

    @pytest.mark.asyncio
    async def test_error_recovery(self, service):
        """Test error recovery and graceful degradation."""
        enhanced_service, mocks = service

        # Make everything fail except fallback
        mocks['state_manager_service'].get_current_global_state.side_effect = Exception("DB error")
        mocks['state_manager_service'].get_recent_events.side_effect = Exception("DB error")
        mocks['contextual_backstory_service'].select_relevant_content.side_effect = Exception("File error")
        mocks['state_influence_service'].build_conversation_context.side_effect = Exception("Context error")

        # Setup working fallback
        fallback_response = Mock()
        fallback_response.ai_response = "Fallback response"
        fallback_response.corrected_transcript = "Test message"
        fallback_response.word_usage_status = "NOT_USED"
        fallback_response.usage_correctness_feedback = None
        mocks['simple_openai_service'].generate_coaching_response.return_value = fallback_response

        result = await enhanced_service.generate_enhanced_response(
            user_message="Test message",
            user_id="user123",
            conversation_id="conv456"
        )

        # Should still return a valid response via fallback
        assert result["ai_response"] == "Fallback response"
        assert result["fallback_mode"] == True
        assert result["enhanced_mode"] == False

    @pytest.mark.asyncio
    async def test_user_preferences_integration(self, service):
        """Test integration of user preferences."""
        enhanced_service, mocks = service

        user_preferences = {
            "communication_style": "casual",
            "topic_preferences": ["work", "creativity"],
            "state_influence_overrides": {
                "mood_sensitivity": 0.9
            }
        }

        # Setup mocks
        mocks['state_manager_service'].get_current_global_state.return_value = {}
        mocks['state_manager_service'].get_recent_events.return_value = []
        mocks['contextual_backstory_service'].select_relevant_content.return_value = {
            "content": "Content",
            "content_types": [],
            "char_count": 100,
            "estimated_tokens": 25
        }

        # Mock that should receive user preferences
        mocks['state_influence_service'].build_conversation_context.return_value = {}

        # Setup OpenAI mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mocks['openai_client'].chat.completions.create.return_value = mock_response

        mocks['conversation_prompt_service'].determine_emotion_from_context.return_value = (
            Mock(value="neutral"), "Neutral"
        )
        mocks['conversation_prompt_service'].construct_conversation_prompt.return_value = "Prompt"

        result = await enhanced_service.generate_enhanced_response(
            user_message="Test message",
            user_id="user123",
            conversation_id="conv456",
            user_preferences=user_preferences
        )

        # Verify user preferences were passed to state influence service
        mocks['state_influence_service'].build_conversation_context.assert_called_with(
            user_id="user123",
            conversation_id="conv456",
            scenario=mocks['state_influence_service'].build_conversation_context.call_args[1]["scenario"],
            user_preferences=user_preferences
        )