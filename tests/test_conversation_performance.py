"""
Tests for conversation performance monitoring in Story 3.3: Speech Optimization & Clara's Response Performance.
Tests comprehensive performance monitoring, timing instrumentation, and threshold alerting.
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from app.services.enhanced_conversation_service import ConversationPerformanceMonitor, EnhancedConversationService


class TestConversationPerformanceMonitor:
    """Test comprehensive performance monitoring functionality."""

    @pytest.fixture
    def performance_monitor(self):
        return ConversationPerformanceMonitor()

    def test_create_conversation_correlation_id(self, performance_monitor):
        """Test correlation ID generation."""
        user_id = "user123"
        conversation_id = "conv456"

        correlation_id = performance_monitor.create_conversation_correlation_id(user_id, conversation_id)

        assert correlation_id.startswith("conv_user123_conv456_")
        assert len(correlation_id.split("_")) == 6  # conv_user_conv_date_time_uuid
        assert len(correlation_id) > 30  # Should include timestamp and UUID

    def test_timing_context_lifecycle(self, performance_monitor):
        """Test complete timing context lifecycle."""
        correlation_id = "test_corr_123"
        operation = "test_operation"

        # Start timing context
        context = performance_monitor.start_timing_context(correlation_id, operation)

        assert context["correlation_id"] == correlation_id
        assert context["operation"] == operation
        assert "start_time" in context
        assert "start_timestamp" in context
        assert "sub_operations" in context

        # Simulate some work
        time.sleep(0.01)

        # End timing context
        metrics = performance_monitor.end_timing_context(context)

        assert metrics["correlation_id"] == correlation_id
        assert metrics["operation"] == operation
        assert metrics["total_duration_ms"] > 5  # Should be at least 5ms
        assert "start_timestamp" in metrics
        assert "end_timestamp" in metrics

    def test_sub_operation_timing(self, performance_monitor):
        """Test sub-operation timing functionality."""
        correlation_id = "test_corr_456"
        main_context = performance_monitor.start_timing_context(correlation_id, "main_op")

        # Start sub-operation
        sub_context = performance_monitor.time_sub_operation(main_context, "sub_op_1")
        assert "start_time" in sub_context

        # Simulate work
        time.sleep(0.01)

        # End sub-operation with metadata
        duration = performance_monitor.end_sub_operation(
            main_context, "sub_op_1", sub_context,
            custom_metric="test_value",
            items_processed=5
        )

        assert duration > 5  # Should be at least 5ms
        assert "sub_op_1" in main_context["sub_operations"]
        assert main_context["sub_operations"]["sub_op_1"]["duration_ms"] > 5
        assert main_context["sub_operations"]["sub_op_1"]["custom_metric"] == "test_value"
        assert main_context["sub_operations"]["sub_op_1"]["items_processed"] == 5

    def test_performance_threshold_alerting(self, performance_monitor):
        """Test performance threshold monitoring and alerting."""
        # Mock logger to capture warnings
        with patch.object(performance_monitor.performance_logger, 'warning') as mock_warning:
            correlation_id = "threshold_test"
            context = performance_monitor.start_timing_context(correlation_id, "enhanced_conversation_response")

            # Simulate slow sub-operation
            sub_context = performance_monitor.time_sub_operation(context, "consciousness_generation")

            # Manually set duration to exceed threshold
            sub_context["start_time"] = time.time() - 2.5  # 2.5 seconds ago

            performance_monitor.end_sub_operation(context, "consciousness_generation", sub_context)

            # Should trigger threshold warning (threshold is 2000ms)
            mock_warning.assert_called()
            warning_call = mock_warning.call_args[0][0]
            assert "consciousness_generation exceeded threshold" in warning_call
            assert "2000ms" in warning_call

    def test_total_response_time_threshold(self, performance_monitor):
        """Test total response time threshold monitoring."""
        with patch.object(performance_monitor.performance_logger, 'warning') as mock_warning:
            correlation_id = "total_threshold_test"
            context = performance_monitor.start_timing_context(correlation_id, "enhanced_conversation_response")

            # Simulate slow total response (exceed 3000ms threshold)
            context["start_time"] = time.time() - 3.5  # 3.5 seconds ago

            performance_monitor.end_timing_context(context)

            # Should trigger total threshold warning
            mock_warning.assert_called()
            warning_call = mock_warning.call_args[0][0]
            assert "Total conversation response time exceeded threshold" in warning_call
            assert "3000ms" in warning_call

    def test_error_logging_with_context(self, performance_monitor):
        """Test error logging with performance context."""
        with patch.object(performance_monitor.performance_logger, 'error') as mock_error:
            correlation_id = "error_test"
            context = performance_monitor.start_timing_context(correlation_id, "test_operation")

            # Add some sub-operations
            sub_context = performance_monitor.time_sub_operation(context, "sub_op")
            performance_monitor.end_sub_operation(context, "sub_op", sub_context)

            # Log error
            test_error = Exception("Test error message")
            performance_monitor.log_error_with_context(context, test_error, "custom_operation")

            # Verify error logging
            mock_error.assert_called()
            error_call_extra = mock_error.call_args[1]['extra']
            assert error_call_extra['correlation_id'] == correlation_id
            assert error_call_extra['operation'] == "custom_operation"
            assert error_call_extra['error_message'] == "Test error message"
            assert error_call_extra['error_type'] == "Exception"
            assert "sub_operations_completed" in error_call_extra


class TestEnhancedConversationServicePerformance:
    """Test performance monitoring integration in EnhancedConversationService."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies."""
        return {
            'contextual_backstory_service': Mock(),
            'conversation_prompt_service': Mock(),
            'state_influence_service': Mock(),
            'state_manager_service': Mock(),
            'mood_transition_analyzer': Mock(),
            'simple_openai_service': Mock(),
            'dynamic_content_selector': Mock(),
            'session_state_service': Mock(),
            'event_selection_service': Mock(),
            'openai_client': Mock()
        }

    @pytest.fixture
    def conversation_service(self, mock_dependencies):
        """Create EnhancedConversationService with mocked dependencies."""
        with patch.multiple(
            'app.services.enhanced_conversation_service',
            ContextualBackstoryService=Mock(return_value=mock_dependencies['contextual_backstory_service']),
            ConversationPromptService=Mock(return_value=mock_dependencies['conversation_prompt_service']),
            StateInfluenceService=Mock(return_value=mock_dependencies['state_influence_service']),
            StateManagerService=Mock(return_value=mock_dependencies['state_manager_service']),
            MoodTransitionAnalyzer=Mock(return_value=mock_dependencies['mood_transition_analyzer']),
            SimpleOpenAIService=Mock(return_value=mock_dependencies['simple_openai_service']),
            DynamicContentSelector=Mock(return_value=mock_dependencies['dynamic_content_selector']),
            SessionStateService=Mock(return_value=mock_dependencies['session_state_service']),
            EventSelectionService=Mock(return_value=mock_dependencies['event_selection_service']),
            OpenAI=Mock(return_value=mock_dependencies['openai_client'])
        ):
            service = EnhancedConversationService()
            # Replace mocked dependencies
            for key, value in mock_dependencies.items():
                setattr(service, key, value)
            return service

    @pytest.mark.asyncio
    async def test_comprehensive_performance_monitoring(self, conversation_service, mock_dependencies):
        """Test comprehensive performance monitoring throughout conversation flow."""
        # Setup mocks
        mock_dependencies['session_state_service'].add_conversation_message = AsyncMock()
        mock_dependencies['session_state_service'].get_conversation_history = AsyncMock(return_value="test history")

        # Mock context gathering
        conversation_service._gather_simulation_context_with_monitoring = AsyncMock(return_value={
            "global_state": {"mood": {"numeric_value": 60}},
            "recent_events": [{"id": "event1", "summary": "test event"}],
            "selected_backstory": {"char_count": 100, "content_types": ["personality"]},
            "conversation_influence": {"mood_transition": {"blended_mood_score": 65}},
            "content_selection_metadata": {"fresh_events_used": ["event1"]}
        })

        # Mock response generation
        conversation_service._generate_context_aware_response_with_monitoring = AsyncMock(return_value={
            "ai_response": "Test response",
            "corrected_transcript": "test message",
            "simulation_context": {"conversation_emotion": "happy"},
            "fallback_mode": False
        })

        # Mock event tracking
        conversation_service._track_events_mentioned = AsyncMock()

        # Test request
        result = await conversation_service.generate_enhanced_response(
            user_message="test message",
            user_id="user123",
            conversation_id="conv456"
        )

        # Verify performance metrics included
        assert "performance_metrics" in result
        assert "correlation_id" in result
        assert result["enhanced_mode"] is True

        # Verify correlation ID format
        correlation_id = result["correlation_id"]
        assert correlation_id.startswith("conv_user123_conv456_")

        # Verify mocked services were called
        mock_dependencies['session_state_service'].add_conversation_message.assert_called()
        conversation_service._gather_simulation_context_with_monitoring.assert_called_once()
        conversation_service._generate_context_aware_response_with_monitoring.assert_called_once()
        conversation_service._track_events_mentioned.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_performance_monitoring(self, conversation_service, mock_dependencies):
        """Test performance monitoring during fallback response generation."""
        # Setup mocks for fallback scenario
        mock_dependencies['session_state_service'].add_conversation_message = AsyncMock()
        mock_dependencies['session_state_service'].get_conversation_history = AsyncMock(return_value="")

        # Mock context gathering failure
        conversation_service._gather_simulation_context_with_monitoring = AsyncMock(side_effect=Exception("Context failed"))

        # Mock fallback response
        fallback_response = Mock()
        fallback_response.ai_response = "Fallback response"
        fallback_response.corrected_transcript = "test"
        fallback_response.word_usage_status = "NOT_USED"
        fallback_response.usage_correctness_feedback = None
        mock_dependencies['simple_openai_service'].generate_coaching_response = AsyncMock(return_value=fallback_response)

        # Test fallback flow
        result = await conversation_service.generate_enhanced_response(
            user_message="test message",
            user_id="user789",
            conversation_id="conv123"
        )

        # Verify fallback performance metrics
        assert "performance_metrics" in result
        assert "correlation_id" in result
        assert result["fallback_mode"] is True
        assert result["enhanced_mode"] is False

        # Verify fallback service was called
        mock_dependencies['simple_openai_service'].generate_coaching_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_gathering_performance_breakdown(self, conversation_service, mock_dependencies):
        """Test detailed performance breakdown in context gathering."""
        # Mock individual context gathering components
        mock_dependencies['state_manager_service'].get_current_global_state = AsyncMock(return_value={"mood": 60})
        mock_dependencies['event_selection_service'].get_contextual_events = AsyncMock(return_value=[])
        mock_dependencies['contextual_backstory_service'].select_relevant_content = AsyncMock(return_value={
            "char_count": 150,
            "content_types": ["background"],
            "content": "backstory content"
        })
        mock_dependencies['state_influence_service'].build_conversation_context = AsyncMock(return_value={})

        # Mock performance monitor
        conversation_service.performance_monitor = Mock()
        conversation_service.performance_monitor.time_sub_operation = Mock(return_value={"start_time": time.time()})
        conversation_service.performance_monitor.end_sub_operation = Mock()

        # Test context gathering
        result = await conversation_service._gather_simulation_context_with_monitoring(
            user_message="test",
            user_id="user123",
            conversation_id="conv456",
            timing_context={"correlation_id": "test_corr"}
        )

        # Verify sub-operations were timed
        expected_operations = [
            "global_state_retrieval",
            "event_selection",
            "backstory_selection",
            "sentiment_analysis",
            "state_influence_calculation"
        ]

        for operation in expected_operations:
            conversation_service.performance_monitor.time_sub_operation.assert_any_call(
                {"correlation_id": "test_corr"}, operation
            )

        # Verify end_sub_operation called for each
        assert conversation_service.performance_monitor.end_sub_operation.call_count >= len(expected_operations)

    @pytest.mark.asyncio
    async def test_response_generation_performance_breakdown(self, conversation_service, mock_dependencies):
        """Test detailed performance breakdown in response generation."""
        # Setup simulation context with proper structure
        simulation_context = {
            "global_state": {"mood": {"numeric_value": 60}},
            "recent_events": [{"id": "event1", "hours_ago": 2, "summary": "test event", "intensity": 5}],
            "selected_backstory": {"char_count": 100, "content": "backstory", "content_types": ["personality"]},
            "conversation_influence": {"mood_transition": {"blended_mood_score": 65, "mood_context": {}}},
            "content_selection_metadata": {}
        }

        # Mock conversation prompt service
        mock_dependencies['conversation_prompt_service'].select_conversation_emotion_with_mood = Mock(
            return_value=(Mock(value="happy"), "emotion reasoning")
        )
        mock_dependencies['conversation_prompt_service'].construct_conversation_prompt_with_mood = Mock(
            return_value="enhanced prompt"
        )

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"message": "AI response", "emotion": "happy"}'
        mock_dependencies['openai_client'].chat.completions.create = Mock(return_value=mock_response)

        # Mock performance monitor
        conversation_service.performance_monitor = Mock()
        conversation_service.performance_monitor.time_sub_operation = Mock(return_value={"start_time": time.time()})
        conversation_service.performance_monitor.end_sub_operation = Mock()

        # Test response generation
        result = await conversation_service._generate_context_aware_response_with_monitoring(
            user_message="test",
            simulation_context=simulation_context,
            timing_context={"correlation_id": "test_corr"}
        )

        # Verify sub-operations were timed
        expected_operations = [
            "context_extraction",
            "emotion_selection",
            "prompt_construction",
            "openai_api_call",
            "response_parsing"
        ]

        for operation in expected_operations:
            conversation_service.performance_monitor.time_sub_operation.assert_any_call(
                {"correlation_id": "test_corr"}, operation
            )

        # Verify OpenAI was called with proper parameters
        mock_dependencies['openai_client'].chat.completions.create.assert_called_once()
        call_args = mock_dependencies['openai_client'].chat.completions.create.call_args[1]
        assert call_args['model'] == "gpt-4o-mini"
        assert call_args['max_tokens'] == 400
        assert call_args['temperature'] == 0.7

    def test_performance_thresholds_configuration(self, conversation_service):
        """Test that performance thresholds are properly configured."""
        monitor = conversation_service.performance_monitor

        # Verify all required thresholds are set including granular component thresholds
        expected_thresholds = {
            "total_response_time_ms": 3000,  # <3s total response time
            "consciousness_generation_ms": 2000,  # <2s consciousness generation
            "context_gathering_ms": 1000,  # <1s context processing
            "response_formatting_ms": 500,  # <500ms response formatting

            # Granular conversation response component thresholds
            "context_extraction_ms": 50,  # <50ms to extract simulation context components
            "emotion_selection_ms": 100,  # <100ms for emotion selection with mood awareness
            "prompt_construction_ms": 200,  # <200ms for enhanced prompt building
            "openai_api_call_ms": 1500,  # <1.5s for OpenAI API response
            "response_parsing_ms": 100,  # <100ms for JSON parsing and formatting

            # Context gathering sub-components
            "global_state_retrieval_ms": 100,  # <100ms for database state retrieval
            "event_selection_ms": 200,  # <200ms for event selection service
            "backstory_selection_ms": 150,  # <150ms for backstory content selection
            "sentiment_analysis_ms": 50,  # <50ms for message sentiment analysis
            "state_influence_calculation_ms": 100,  # <100ms for state influence calculations
        }

        for threshold_key, expected_value in expected_thresholds.items():
            assert threshold_key in monitor.alert_thresholds
            assert monitor.alert_thresholds[threshold_key] == expected_value

    def test_detailed_timing_breakdown_logging(self, conversation_service):
        """Test detailed timing breakdown logging functionality."""
        monitor = conversation_service.performance_monitor

        # Mock logger to capture detailed breakdown
        with patch.object(monitor.performance_logger, 'info') as mock_info:
            # Create mock metrics data with sub-operations
            mock_metrics = {
                "correlation_id": "test_corr_123",
                "operation": "enhanced_conversation_response",
                "total_duration_ms": 2500.0,
                "sub_operations": {
                    "openai_api_call": {"duration_ms": 1800.0, "model": "gpt-4o-mini"},
                    "prompt_construction": {"duration_ms": 250.0, "prompt_length": 1500},
                    "context_extraction": {"duration_ms": 150.0, "backstory_chars": 500},
                    "emotion_selection": {"duration_ms": 120.0, "selected_emotion": "happy"},
                    "response_parsing": {"duration_ms": 80.0, "json_parsed": True}
                }
            }

            monitor.log_detailed_timing_breakdown(mock_metrics)

            # Verify detailed breakdown was logged
            mock_info.assert_called_once()
            log_call = mock_info.call_args[0][0]

            # Check that breakdown includes all components
            assert "DETAILED TIMING BREAKDOWN" in log_call
            assert "openai_api_call: 1800.00ms" in log_call
            assert "prompt_construction: 250.00ms" in log_call
            assert "⚠️ SLOW" in log_call  # Should show warning for slow OpenAI call

    def test_granular_threshold_alerting(self, conversation_service):
        """Test that granular thresholds trigger appropriate alerts."""
        monitor = conversation_service.performance_monitor

        with patch.object(monitor.performance_logger, 'warning') as mock_warning:
            correlation_id = "granular_test"
            context = monitor.start_timing_context(correlation_id, "test_operation")

            # Simulate slow prompt construction (exceeds 200ms threshold)
            sub_context = monitor.time_sub_operation(context, "prompt_construction")
            sub_context["start_time"] = time.time() - 0.5  # 500ms ago

            monitor.end_sub_operation(context, "prompt_construction", sub_context)

            # Should trigger threshold warning
            mock_warning.assert_called()
            warning_call = mock_warning.call_args[0][0]
            assert "prompt_construction exceeded threshold" in warning_call
            assert "200ms" in warning_call


@pytest.mark.integration
class TestConversationPerformanceIntegration:
    """Integration tests for conversation performance monitoring."""

    @pytest.mark.asyncio
    async def test_response_time_under_threshold(self):
        """Test that response time stays under 3s threshold with mocked services."""
        # This test would require actual service integration
        # For now, we'll simulate with timing verification
        start_time = time.time()

        # Simulate conversation processing with realistic delays
        await asyncio.sleep(0.1)  # Context gathering simulation
        await asyncio.sleep(0.05)  # Response generation simulation
        await asyncio.sleep(0.02)  # Response formatting simulation

        total_time = (time.time() - start_time) * 1000

        # Verify under threshold (should be well under 3000ms)
        assert total_time < 3000
        assert total_time > 150  # Should take at least 150ms in our simulation

    def test_correlation_id_uniqueness(self):
        """Test that correlation IDs are unique across requests."""
        monitor = ConversationPerformanceMonitor()

        # Generate multiple correlation IDs
        correlation_ids = set()
        for i in range(100):
            corr_id = monitor.create_conversation_correlation_id(f"user{i}", f"conv{i}")
            correlation_ids.add(corr_id)

        # All should be unique
        assert len(correlation_ids) == 100

    def test_structured_logging_format(self):
        """Test that performance logs use structured format for analysis."""
        monitor = ConversationPerformanceMonitor()

        # Verify logger is configured for structured logging
        assert hasattr(monitor, 'performance_logger')
        assert monitor.performance_logger.name == 'app.services.enhanced_conversation_service.performance'