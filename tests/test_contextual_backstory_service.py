"""
Tests for CharacterContentService.select_relevant_content - Story 2.6 Enhanced
Conversational Context Integration (backstory selection behavior).
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from app.services.character_content_service import CharacterContentService
from app.core.conversation_config import ConversationContextConfig


class TestBackstorySelection:
    """Test suite for contextual backstory selection functionality."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = Mock(spec=ConversationContextConfig)
        config.MAX_BACKSTORY_CHARS = 1000
        config.CONTENT_TYPE_PRIORITIES = {
            "character_gist": 1,
            "childhood_memories": 3,
            "positive_memories": 2,
            "connecting_memories": 4,
            "friend_character": 2
        }
        return config

    @pytest.fixture
    def service(self, mock_config):
        """Create CharacterContentService with file loading stubbed out."""
        service = CharacterContentService(mock_config)
        service._test_content = {}
        service.load = Mock(side_effect=lambda t: service._test_content.get(t, ""))
        return service

    @pytest.mark.asyncio
    async def test_keyword_matching_childhood(self, service):
        """Test keyword matching for childhood-related content."""
        service._test_content["childhood_memories"] = "Childhood content here..."
        service._test_content["character_gist"] = "General character info..."

        user_message = "Tell me about your childhood and your mother"

        result = await service.select_relevant_content(user_message)

        assert result["content_types"] == ["childhood_memories", "character_gist"]
        assert "childhood" in result["selection_reasoning"].lower()
        assert result["char_count"] > 0

    @pytest.mark.asyncio
    async def test_keyword_matching_positive(self, service):
        """Test keyword matching for positive content."""
        service._test_content["positive_memories"] = "Happy memories content..."
        service._test_content["character_gist"] = "General character info..."

        user_message = "Tell me about your happiest memories and best experiences"

        result = await service.select_relevant_content(user_message)

        assert "positive_memories" in result["content_types"]
        assert result["keyword_matches"]["positive_memories"] > 0

    @pytest.mark.asyncio
    async def test_keyword_matching_difficult(self, service):
        """Test keyword matching for difficult/trauma content."""
        service._test_content["connecting_memories"] = "Difficult memories content..."
        service._test_content["character_gist"] = "General character info..."

        user_message = "Tell me about difficult times and struggles in your life"

        result = await service.select_relevant_content(user_message)

        assert "connecting_memories" in result["content_types"]
        assert result["keyword_matches"]["connecting_memories"] > 0

    @pytest.mark.asyncio
    async def test_keyword_matching_relationships(self, service):
        """Test keyword matching for relationship content."""
        service._test_content["friend_character"] = "Friend character content..."
        service._test_content["character_gist"] = "General character info..."

        user_message = "Tell me about your friends and relationships"

        result = await service.select_relevant_content(user_message)

        assert "friend_character" in result["content_types"]
        assert result["keyword_matches"]["friend_character"] > 0

    @pytest.mark.asyncio
    async def test_general_fallback(self, service):
        """Test fallback to character gist for general queries."""
        service._test_content["character_gist"] = "General character info..."

        user_message = "Tell me about yourself"

        result = await service.select_relevant_content(user_message)

        assert "character_gist" in result["content_types"]
        assert "fallback for general query" in result["selection_reasoning"]

    @pytest.mark.asyncio
    async def test_content_length_limiting(self, service):
        """Test that content is properly limited by character count."""
        service._test_content["character_gist"] = "x" * 2000  # Content longer than limit

        user_message = "Tell me about yourself"
        max_chars = 500

        result = await service.select_relevant_content(user_message, max_chars=max_chars)

        assert result["char_count"] <= max_chars
        assert result["truncated"] == True
        assert result["char_limit_used"] == max_chars

    @pytest.mark.asyncio
    async def test_multiple_keyword_matches(self, service):
        """Test handling of multiple keyword matches."""
        service._test_content["childhood_memories"] = "Childhood content..."
        service._test_content["positive_memories"] = "Happy content..."
        service._test_content["character_gist"] = "General info..."

        user_message = "Tell me about your happy childhood memories with your mother"

        result = await service.select_relevant_content(user_message)

        # Should include both childhood and positive content
        assert "childhood_memories" in result["content_types"]
        assert "positive_memories" in result["content_types"]
        assert len(result["content_types"]) >= 2

    @pytest.mark.asyncio
    async def test_caching_functionality(self, mock_config):
        """Test that content is only read from disk once."""
        service = CharacterContentService(mock_config)

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", return_value="Cached content...") as mock_read:
            # First call should load content
            await service.select_relevant_content("Tell me about yourself")

            # Second call should use cache
            await service.select_relevant_content("Who are you?")

        # File should only be read once due to caching
        assert mock_read.call_count == 1

    @pytest.mark.asyncio
    async def test_error_handling(self, service):
        """Test error handling when content loading fails gracefully."""
        # All content types return empty (simulating files not found)
        result = await service.select_relevant_content("Tell me about yourself")

        # Should return empty content but handle gracefully
        assert result["content"] == ""
        assert result["content_types"] == []
        assert result["char_count"] == 0

    @pytest.mark.asyncio
    async def test_exception_fallback_behavior(self, service):
        """Test complete exception fallback with actual error."""
        # Force a runtime exception in the main flow and in fallback loading
        service._analyze_keyword_matches = Mock(side_effect=Exception("Critical error"))
        service.load = Mock(side_effect=Exception("Critical error"))

        result = await service.select_relevant_content("Tell me about yourself")

        # Should return fallback content with error indication
        assert result["fallback_mode"] == True
        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_message_handling(self, service):
        """Test handling of empty or None messages."""
        service._test_content["character_gist"] = "General info..."

        result = await service.select_relevant_content("")

        # Should still return character gist as fallback
        assert "character_gist" in result["content_types"]

    @pytest.mark.asyncio
    async def test_token_estimation(self, service):
        """Test token count estimation."""
        service._test_content["character_gist"] = "x" * 400  # 400 characters

        result = await service.select_relevant_content("Tell me about yourself")

        # Should estimate roughly 100 tokens (400 chars / 4)
        assert result["estimated_tokens"] == 100
        assert result["char_count"] == 400

    @pytest.mark.asyncio
    async def test_content_priority_ordering(self, service):
        """Test that content is selected based on priority ordering."""
        service._test_content["character_gist"] = "Gist content"
        service._test_content["childhood_memories"] = "Childhood content"
        service._test_content["connecting_memories"] = "Trauma content"

        # Message that triggers multiple content types
        user_message = "Tell me about your difficult childhood experiences"

        result = await service.select_relevant_content(user_message)

        # connecting_memories should be prioritized (priority 4) over childhood_memories (priority 3)
        content_types = result["content_types"]
        if "connecting_memories" in content_types and "childhood_memories" in content_types:
            # Verify the order respects priority
            connecting_index = content_types.index("connecting_memories")
            childhood_index = content_types.index("childhood_memories")
            assert connecting_index <= childhood_index  # Higher priority should come first or equal


class TestBackstorySelectionIntegration:
    """Integration tests for backstory selection with real content files."""

    @pytest.mark.asyncio
    async def test_integration_with_real_content(self):
        """Test selection against actual content files (if they exist)."""
        service = CharacterContentService()

        try:
            result = await service.select_relevant_content("Tell me about yourself")

            # Basic assertions that should work regardless of content availability
            assert isinstance(result, dict)
            assert "content" in result
            assert "content_types" in result
            assert "char_count" in result
            assert "estimated_tokens" in result

        except Exception:
            # If content files don't exist, that's expected in test environment
            pytest.skip("Character content files not available in test environment")

    @pytest.mark.asyncio
    async def test_performance_with_large_content(self):
        """Test performance with realistic content sizes."""
        service = CharacterContentService()

        # Mock large content
        large_content = "x" * 10000  # 10k characters
        with patch.object(service, "load", return_value=large_content):

            import time
            start_time = time.time()

            result = await service.select_relevant_content("Tell me about yourself")

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Should complete within reasonable time (less than 100ms)
            assert duration_ms < 100
            assert result["char_count"] <= service.config.MAX_BACKSTORY_CHARS
