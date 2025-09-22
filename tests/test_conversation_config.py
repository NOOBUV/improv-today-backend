"""
Tests for ConversationContextConfig - Story 2.6 Enhanced Conversational Context Integration
"""
import pytest
import os
from unittest.mock import patch
from app.core.conversation_config import ConversationContextConfig, conversation_config


class TestConversationContextConfig:
    """Test suite for ConversationContextConfig functionality."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        config = ConversationContextConfig()

        assert config.RECENT_EVENTS_HOURS_BACK == 72
        assert config.MAX_EVENTS_COUNT == 5
        assert config.MAX_BACKSTORY_CHARS == 8000
        assert config.MAX_BACKSTORY_TOKENS == 2000
        assert config.GLOBAL_STATE_CACHE_TTL_MINUTES == 5
        assert config.SESSION_STATE_CACHE_TTL_MINUTES == 30
        assert config.SIMULATION_EVENTS_WEIGHT == 0.6
        assert config.BACKSTORY_CONTENT_WEIGHT == 0.4
        assert config.MAX_CONTEXT_PROCESSING_MS == 100
        assert config.CIRCUIT_BREAKER_FAILURE_THRESHOLD == 3
        assert config.CIRCUIT_BREAKER_TIMEOUT_SECONDS == 30

    def test_content_type_priorities(self):
        """Test that content type priorities are set correctly."""
        config = ConversationContextConfig()

        expected_priorities = {
            "character_gist": 1,
            "childhood_memories": 3,
            "positive_memories": 2,
            "connecting_memories": 4,
            "friend_character": 2
        }

        assert config.CONTENT_TYPE_PRIORITIES == expected_priorities

    def test_from_env_with_no_env_vars(self):
        """Test loading from environment with no environment variables set."""
        with patch.dict(os.environ, {}, clear=True):
            config = ConversationContextConfig.from_env()

            # Should use default values
            assert config.RECENT_EVENTS_HOURS_BACK == 24
            assert config.MAX_EVENTS_COUNT == 5
            assert config.MAX_BACKSTORY_CHARS == 8000

    def test_from_env_with_custom_values(self):
        """Test loading from environment with custom environment variables."""
        env_vars = {
            "CONV_RECENT_EVENTS_HOURS": "48",
            "CONV_MAX_EVENTS_COUNT": "10",
            "CONV_MAX_BACKSTORY_CHARS": "12000",
            "CONV_MAX_BACKSTORY_TOKENS": "3000",
            "CONV_GLOBAL_STATE_CACHE_TTL": "10",
            "CONV_SESSION_STATE_CACHE_TTL": "60",
            "CONV_SIMULATION_EVENTS_WEIGHT": "0.8",
            "CONV_BACKSTORY_CONTENT_WEIGHT": "0.2",
            "CONV_MAX_CONTEXT_PROCESSING_MS": "200",
            "CONV_CIRCUIT_BREAKER_FAILURE_THRESHOLD": "5",
            "CONV_CIRCUIT_BREAKER_TIMEOUT_SECONDS": "60"
        }

        with patch.dict(os.environ, env_vars):
            config = ConversationContextConfig.from_env()

            assert config.RECENT_EVENTS_HOURS_BACK == 48
            assert config.MAX_EVENTS_COUNT == 10
            assert config.MAX_BACKSTORY_CHARS == 12000
            assert config.MAX_BACKSTORY_TOKENS == 3000
            assert config.GLOBAL_STATE_CACHE_TTL_MINUTES == 10
            assert config.SESSION_STATE_CACHE_TTL_MINUTES == 60
            assert config.SIMULATION_EVENTS_WEIGHT == 0.8
            assert config.BACKSTORY_CONTENT_WEIGHT == 0.2
            assert config.MAX_CONTEXT_PROCESSING_MS == 200
            assert config.CIRCUIT_BREAKER_FAILURE_THRESHOLD == 5
            assert config.CIRCUIT_BREAKER_TIMEOUT_SECONDS == 60

    def test_from_env_with_invalid_values(self):
        """Test loading from environment with invalid values (should raise exceptions)."""
        with patch.dict(os.environ, {"CONV_RECENT_EVENTS_HOURS": "invalid_number"}):
            with pytest.raises(ValueError):
                ConversationContextConfig.from_env()

    def test_from_env_with_partial_values(self):
        """Test loading from environment with only some values set."""
        env_vars = {
            "CONV_RECENT_EVENTS_HOURS": "72",
            "CONV_MAX_BACKSTORY_CHARS": "15000"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = ConversationContextConfig.from_env()

            # Should use custom values for set variables
            assert config.RECENT_EVENTS_HOURS_BACK == 72
            assert config.MAX_BACKSTORY_CHARS == 15000

            # Should use defaults for unset variables
            assert config.MAX_EVENTS_COUNT == 5
            assert config.MAX_BACKSTORY_TOKENS == 2000

    def test_weight_values_sum_to_one(self):
        """Test that simulation and backstory weights sum to 1.0."""
        config = ConversationContextConfig()

        total_weight = config.SIMULATION_EVENTS_WEIGHT + config.BACKSTORY_CONTENT_WEIGHT
        assert abs(total_weight - 1.0) < 0.01  # Allow for floating point precision

    def test_reasonable_default_values(self):
        """Test that default values are reasonable for production use."""
        config = ConversationContextConfig()

        # Time-based values should be reasonable
        assert 1 <= config.RECENT_EVENTS_HOURS_BACK <= 168  # 1 hour to 1 week
        assert 1 <= config.MAX_EVENTS_COUNT <= 50

        # Character/token limits should be reasonable for LLM context
        assert 1000 <= config.MAX_BACKSTORY_CHARS <= 50000
        assert 250 <= config.MAX_BACKSTORY_TOKENS <= 12500

        # Cache TTLs should be reasonable
        assert 1 <= config.GLOBAL_STATE_CACHE_TTL_MINUTES <= 60
        assert 1 <= config.SESSION_STATE_CACHE_TTL_MINUTES <= 120

        # Performance thresholds should be reasonable
        assert 50 <= config.MAX_CONTEXT_PROCESSING_MS <= 1000
        assert 1 <= config.CIRCUIT_BREAKER_FAILURE_THRESHOLD <= 10
        assert 10 <= config.CIRCUIT_BREAKER_TIMEOUT_SECONDS <= 300

    def test_global_config_instance(self):
        """Test that the global conversation_config instance is properly initialized."""
        assert conversation_config is not None
        assert isinstance(conversation_config, ConversationContextConfig)
        assert hasattr(conversation_config, 'MAX_BACKSTORY_CHARS')

    def test_config_immutability_intent(self):
        """Test that configuration values are intended to be immutable (dataclass behavior)."""
        config = ConversationContextConfig()

        # These should work (assignment)
        config.MAX_BACKSTORY_CHARS = 10000
        assert config.MAX_BACKSTORY_CHARS == 10000

        # But for production, we rely on the pattern of creating new instances
        # rather than modifying existing ones

    def test_content_type_priorities_completeness(self):
        """Test that all expected content types have priorities defined."""
        config = ConversationContextConfig()

        required_content_types = [
            "character_gist",
            "childhood_memories",
            "positive_memories",
            "connecting_memories",
            "friend_character"
        ]

        for content_type in required_content_types:
            assert content_type in config.CONTENT_TYPE_PRIORITIES
            assert isinstance(config.CONTENT_TYPE_PRIORITIES[content_type], int)
            assert config.CONTENT_TYPE_PRIORITIES[content_type] > 0

    def test_priority_ordering_logic(self):
        """Test that priority values make logical sense."""
        config = ConversationContextConfig()
        priorities = config.CONTENT_TYPE_PRIORITIES

        # connecting_memories (trauma) should have highest priority when relevant
        assert priorities["connecting_memories"] == max(priorities.values())

        # character_gist should have lowest priority (most general)
        assert priorities["character_gist"] == min(priorities.values())

        # childhood_memories should have higher priority than positive_memories
        assert priorities["childhood_memories"] > priorities["positive_memories"]

    def test_float_environment_variables(self):
        """Test that float environment variables are properly handled."""
        env_vars = {
            "CONV_SIMULATION_EVENTS_WEIGHT": "0.7",
            "CONV_BACKSTORY_CONTENT_WEIGHT": "0.3"
        }

        with patch.dict(os.environ, env_vars):
            config = ConversationContextConfig.from_env()

            assert config.SIMULATION_EVENTS_WEIGHT == 0.7
            assert config.BACKSTORY_CONTENT_WEIGHT == 0.3
            assert isinstance(config.SIMULATION_EVENTS_WEIGHT, float)
            assert isinstance(config.BACKSTORY_CONTENT_WEIGHT, float)


class TestConfigurationValidation:
    """Test configuration validation and edge cases."""

    def test_zero_values_handling(self):
        """Test handling of zero values in configuration."""
        env_vars = {
            "CONV_MAX_EVENTS_COUNT": "0",
            "CONV_MAX_CONTEXT_PROCESSING_MS": "0"
        }

        with patch.dict(os.environ, env_vars):
            config = ConversationContextConfig.from_env()

            assert config.MAX_EVENTS_COUNT == 0
            assert config.MAX_CONTEXT_PROCESSING_MS == 0

    def test_negative_values_handling(self):
        """Test that negative values can be set (though they may not make logical sense)."""
        env_vars = {
            "CONV_RECENT_EVENTS_HOURS": "-1"
        }

        with patch.dict(os.environ, env_vars):
            config = ConversationContextConfig.from_env()
            assert config.RECENT_EVENTS_HOURS_BACK == -1

    def test_very_large_values_handling(self):
        """Test handling of very large configuration values."""
        env_vars = {
            "CONV_MAX_BACKSTORY_CHARS": "1000000",  # 1 million characters
            "CONV_CIRCUIT_BREAKER_TIMEOUT_SECONDS": "86400"  # 1 day
        }

        with patch.dict(os.environ, env_vars):
            config = ConversationContextConfig.from_env()

            assert config.MAX_BACKSTORY_CHARS == 1000000
            assert config.CIRCUIT_BREAKER_TIMEOUT_SECONDS == 86400

    def test_performance_configuration_consistency(self):
        """Test that performance-related configurations are consistent."""
        config = ConversationContextConfig()

        # Cache TTLs should be longer than context processing time
        assert config.GLOBAL_STATE_CACHE_TTL_MINUTES * 60 * 1000 > config.MAX_CONTEXT_PROCESSING_MS
        assert config.SESSION_STATE_CACHE_TTL_MINUTES * 60 * 1000 > config.MAX_CONTEXT_PROCESSING_MS

        # Circuit breaker timeout should be longer than context processing time
        assert config.CIRCUIT_BREAKER_TIMEOUT_SECONDS * 1000 > config.MAX_CONTEXT_PROCESSING_MS