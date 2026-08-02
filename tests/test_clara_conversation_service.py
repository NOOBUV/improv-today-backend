"""
Tests for ClaraConversationService - Story 2.6 Enhanced Conversational Context Integration
"""
import json

import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.clara_conversation_service import CLARA_MAX_TOKENS, ClaraConversationService
from app.services.state_influence_service import ConversationScenario


def _completion(content):
    """Stand-in for a non-streaming OpenAI/Gemini completion."""
    c = Mock()
    c.choices = [Mock()]
    c.choices[0].message.content = content
    return c


class TestClaraConversationService:
    """Test suite for ClaraConversationService functionality."""

    @pytest.fixture
    def mock_services(self):
        """Mock every constructor dependency - no DB, no network."""
        mocks = {
            'character_content_service': Mock(select_relevant_content=AsyncMock()),
            'conversation_prompt_service': Mock(),
            'state_influence_service': Mock(build_conversation_context=AsyncMock(return_value={})),
            'state_manager_service': Mock(get_current_global_state=AsyncMock(return_value={})),
            'session_state_service': Mock(
                add_conversation_message=AsyncMock(),
                get_conversation_history=AsyncMock(return_value=""),
            ),
            'event_selection_service': Mock(
                get_contextual_events=AsyncMock(return_value=[]),
                track_events_mentioned_in_response=AsyncMock(),
            ),
            'openai_client': Mock(),
        }

        # The service awaits client.chat.completions.create(...) - plain Mock isn't awaitable
        mocks['openai_client'].chat.completions.create = AsyncMock(return_value=_completion("Response"))

        # Prompt service is sync; both calls unpack/format their return value
        mocks['conversation_prompt_service'].select_conversation_emotion_with_mood.return_value = (
            Mock(value="neutral"), "neutral baseline"
        )
        mocks['conversation_prompt_service'].construct_conversation_prompt_with_mood.return_value = "PROMPT"

        return mocks

    @pytest.fixture
    def service(self, mock_services):
        """Create ClaraConversationService with mocked dependencies."""
        with patch.multiple(
            'app.services.clara_conversation_service',
            CharacterContentService=Mock(return_value=mock_services['character_content_service']),
            ConversationPromptService=Mock(return_value=mock_services['conversation_prompt_service']),
            StateInfluenceService=Mock(return_value=mock_services['state_influence_service']),
            StateManagerService=Mock(return_value=mock_services['state_manager_service']),
            SessionStateService=Mock(return_value=mock_services['session_state_service']),
            EventSelectionService=Mock(return_value=mock_services['event_selection_service']),
            AsyncOpenAI=Mock(return_value=mock_services['openai_client']),
        ):
            service = ClaraConversationService()
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
        mocks['event_selection_service'].get_contextual_events.return_value = [
            {
                "event_id": "event1",
                "summary": "Had a good meeting",
                "hours_ago": 2
            }
        ]
        mocks['character_content_service'].select_relevant_content.return_value = {
            "content": "Character backstory content",
            "content_types": ["character_gist"],
            "char_count": 500,
            "estimated_tokens": 125
        }
        mocks['state_influence_service'].build_conversation_context.return_value = {
            "mood_transition": {
                "blended_mood_score": 72,
                "mood_context": {"current_mood": 70, "stress_level": 40},
            }
        }

        # The model replies with the {"message", "emotion"} JSON contract
        mocks['openai_client'].chat.completions.create.return_value = _completion(
            json.dumps({"message": "AI response based on context", "emotion": "happy"})
        )
        mocks['conversation_prompt_service'].select_conversation_emotion_with_mood.return_value = (
            Mock(value="happy"), "User seems positive"
        )

        # If the enhanced path silently degrades, this would be awaited instead
        enhanced_service._fallback_response = AsyncMock(return_value="Fallback response")

        result = await enhanced_service.generate_enhanced_response(
            user_message="How are you feeling today?",
            user_id="user123",
            conversation_id="conv456"
        )

        assert result["ai_response"] == "AI response based on context"  # unwrapped from the JSON
        assert result["enhanced_mode"] is True
        assert result["fallback_mode"] is False
        assert result["simulation_context"]["conversation_emotion"] == "happy"
        assert result["simulation_context"]["recent_events_count"] == 1
        assert result["correlation_id"] == result["performance_metrics"]["correlation_id"]

        enhanced_service._fallback_response.assert_not_awaited()
        assert "fallback_response" not in result["performance_metrics"]["sub_operations"]

        # The enhanced turn is persisted as enhanced, not as a degraded one
        stored = mocks['session_state_service'].add_conversation_message.await_args_list[-1].kwargs
        assert stored["message_type"] == "assistant"
        assert stored["message_content"] == "AI response based on context"
        assert stored["metadata"]["enhanced_mode"] is True

    @pytest.mark.asyncio
    async def test_fallback_to_inline_response(self, service):
        """Test fallback to the inline _fallback_response when enhanced context fails."""
        enhanced_service, mocks = service

        # Make context gathering fail
        mocks['state_manager_service'].get_current_global_state.side_effect = Exception("Database error")

        # Setup fallback response
        enhanced_service._fallback_response = AsyncMock(return_value="Fallback response")

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
        """The monitor must record the real pipeline breakdown, not just a total.

        Pins what conversation_performance.py emits: correlation id, operation
        name, and a sub_operations map whose durations track actual elapsed time.
        """
        enhanced_service, mocks = service

        async def delayed_global_state():
            import asyncio
            await asyncio.sleep(0.02)  # 20ms delay - must show up in the breakdown
            return {"mood": {"numeric_value": 60}}

        mocks['state_manager_service'].get_current_global_state.side_effect = delayed_global_state
        mocks['character_content_service'].select_relevant_content.return_value = {
            "content": "Content",
            "content_types": ["character_gist"],
            "char_count": 100,
            "estimated_tokens": 25
        }

        result = await enhanced_service.generate_enhanced_response(
            user_message="Test message",
            user_id="user123",
            conversation_id="conv456"
        )

        metrics = result["performance_metrics"]
        assert metrics["operation"] == "enhanced_conversation_response"
        assert metrics["correlation_id"].startswith("conv_user123_conv456_")
        assert metrics["start_timestamp"] < metrics["end_timestamp"]

        steps = metrics["sub_operations"]
        # Every stage of the enhanced path, including the nested prompt/API steps
        assert {
            "request_parsing", "context_gathering", "global_state_retrieval",
            "event_selection", "backstory_selection", "state_influence_calculation",
            "context_extraction", "emotion_selection", "prompt_construction",
            "openai_api_call", "response_parsing", "consciousness_generation",
            "response_formatting",
        } <= steps.keys(), sorted(steps)

        # Durations are measured, not stubbed: the injected 20ms sleep is visible
        # and rolls up into the parent step and the total.
        assert steps["global_state_retrieval"]["duration_ms"] >= 20
        assert steps["context_gathering"]["duration_ms"] >= steps["global_state_retrieval"]["duration_ms"]
        assert metrics["total_duration_ms"] >= steps["context_gathering"]["duration_ms"] > 0

        # Step metadata the service stashes via s.update(...)
        assert steps["backstory_selection"]["chars_selected"] == 100
        assert steps["openai_api_call"]["max_tokens"] == CLARA_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_simulation_context_integration(self, service):
        """The gathered context must actually reach the prompt and the response.

        Events are unwrapped from the selection service's envelope, mood/stress
        come from the state influence service's mood_transition block.
        """
        enhanced_service, mocks = service

        global_state = {
            "mood": {"numeric_value": 85},
            "stress": {"numeric_value": 30},
            "energy": {"numeric_value": 75}
        }
        mocks['state_manager_service'].get_current_global_state.return_value = global_state
        mocks['event_selection_service'].get_contextual_events.return_value = [
            {
                "event_id": "event1",
                "summary": "Completed an important project",
                "hours_ago": 1,
                "impact_mood": "positive"
            },
            # Wrapped form: the service must unwrap "original_event"
            {
                "id": "event2",
                "original_event": {
                    "event_id": "event2",
                    "summary": "Had lunch with a friend",
                    "hours_ago": 3,
                    "impact_mood": "positive"
                }
            }
        ]
        mocks['character_content_service'].select_relevant_content.return_value = {
            "content": "Clara loves creative projects and collaboration",
            "content_types": ["positive_memories", "character_gist"],
            "char_count": 800,
            "estimated_tokens": 200
        }
        mocks['state_influence_service'].build_conversation_context.return_value = {
            "mood_transition": {
                "blended_mood_score": 82,
                "mood_context": {"current_mood": 85, "stress_level": 30},
            },
            "overall_tone": "enthusiastic",
        }

        # Plain (non-JSON) reply: emotion falls back to the selected one
        mocks['openai_client'].chat.completions.create.return_value = _completion(
            "I'm feeling great! Just finished a big project."
        )
        mocks['conversation_prompt_service'].select_conversation_emotion_with_mood.return_value = (
            Mock(value="excited"), "High energy and positive mood"
        )

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

        # The context is not just reported back - it is what the prompt was built from
        prompt_kwargs = mocks['conversation_prompt_service'].construct_conversation_prompt_with_mood.call_args.kwargs
        assert [e["summary"] for e in prompt_kwargs["recent_events"]] == [
            "Completed an important project", "Had lunch with a friend"
        ]
        assert prompt_kwargs["global_state"] == global_state
        assert prompt_kwargs["character_backstory"] == "Clara loves creative projects and collaboration"
        assert prompt_kwargs["mood_transition_data"]["blended_mood_score"] == 82

        emotion_kwargs = mocks['conversation_prompt_service'].select_conversation_emotion_with_mood.call_args.kwargs
        assert emotion_kwargs["blended_mood_score"] == 82

    @pytest.mark.asyncio
    async def test_error_recovery(self, service):
        """Test error recovery and graceful degradation."""
        enhanced_service, mocks = service

        # Make everything fail except fallback
        mocks['state_manager_service'].get_current_global_state.side_effect = Exception("DB error")
        mocks['state_manager_service'].get_recent_events.side_effect = Exception("DB error")
        mocks['character_content_service'].select_relevant_content.side_effect = Exception("File error")
        mocks['state_influence_service'].build_conversation_context.side_effect = Exception("Context error")

        # Setup working fallback
        enhanced_service._fallback_response = AsyncMock(return_value="Fallback response")

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
        events = [{"event_id": "event1", "summary": "Shipped the redesign", "hours_ago": 2}]
        mocks['event_selection_service'].get_contextual_events.return_value = events
        mocks['character_content_service'].select_relevant_content.return_value = {
            "content": "Content",
            "content_types": [],
            "char_count": 100,
            "estimated_tokens": 25
        }

        await enhanced_service.generate_enhanced_response(
            user_message="Work has been great and fun lately",
            user_id="user123",
            conversation_id="conv456",
            user_preferences=user_preferences
        )

        # Everything the state influence service needs must reach it
        kwargs = mocks['state_influence_service'].build_conversation_context.call_args.kwargs
        assert kwargs["user_id"] == "user123"
        assert kwargs["conversation_id"] == "conv456"
        assert kwargs["scenario"] is ConversationScenario.CASUAL_CHAT
        assert kwargs["user_preferences"] == user_preferences
        assert kwargs["recent_events"] == events
        # Sentiment is computed from the message, not hardcoded (this one is positive)
        assert kwargs["conversation_sentiment"] > 0


class TestAwaitRegression:
    """Regression lock for the missing-await bug: the non-streaming path must
    return enhanced output, not silently degrade to the fallback service.

    Hermetic: every constructor dependency is patched."""

    @pytest.fixture
    def hermetic_service(self):
        deps = {
            'character_content_service': Mock(
                select_relevant_content=AsyncMock(return_value={
                    "content": "backstory", "content_types": ["character_gist"],
                    "char_count": 9, "estimated_tokens": 3,
                })
            ),
            'conversation_prompt_service': Mock(),
            'state_influence_service': Mock(
                build_conversation_context=AsyncMock(return_value={"mood_influence": {"tone": "warm"}})
            ),
            'state_manager_service': Mock(
                get_current_global_state=AsyncMock(return_value={
                    "mood": {"numeric_value": 70}, "stress": {"numeric_value": 30},
                    "energy": {"numeric_value": 60},
                })
            ),
            'session_state_service': Mock(
                add_conversation_message=AsyncMock(),
                get_conversation_history=AsyncMock(return_value=[]),
            ),
            'event_selection_service': Mock(
                get_contextual_events=AsyncMock(return_value=[
                    {"id": "ev1", "summary": "Had coffee with Mel", "hours_ago": 3}
                ]),
                mark_events_used=AsyncMock(),
            ),
        }
        emotion = Mock()
        emotion.value = "happy"
        deps['conversation_prompt_service'].select_conversation_emotion_with_mood.return_value = (
            emotion, "user is upbeat"
        )
        deps['conversation_prompt_service'].construct_conversation_prompt_with_mood.return_value = "BASE PROMPT"

        openai_client = Mock()
        completion = Mock()
        completion.choices = [Mock()]
        completion.choices[0].message.content = '{"message": "hi", "emotion": "happy"}'
        openai_client.chat.completions.create = AsyncMock(return_value=completion)

        with patch.multiple(
            'app.services.clara_conversation_service',
            CharacterContentService=Mock(return_value=deps['character_content_service']),
            ConversationPromptService=Mock(return_value=deps['conversation_prompt_service']),
            StateInfluenceService=Mock(return_value=deps['state_influence_service']),
            StateManagerService=Mock(return_value=deps['state_manager_service']),
            SessionStateService=Mock(return_value=deps['session_state_service']),
            EventSelectionService=Mock(return_value=deps['event_selection_service']),
            AsyncOpenAI=Mock(return_value=openai_client),
        ):
            service = ClaraConversationService()
        # Fallback must stay untouched on the enhanced path
        service._fallback_response = AsyncMock(return_value="FALLBACK")
        return service, deps, openai_client

    @pytest.mark.asyncio
    async def test_non_streaming_returns_enhanced_not_fallback(self, hermetic_service):
        service, deps, openai_client = hermetic_service

        result = await service.generate_enhanced_response(
            user_message="How was your day?",
            user_id="user123",
            conversation_id="conv789",
        )

        assert result["enhanced_mode"] is True
        assert result["fallback_mode"] is False
        assert result["ai_response"] == "hi"
        openai_client.chat.completions.create.assert_awaited_once()
        service._fallback_response.assert_not_called()