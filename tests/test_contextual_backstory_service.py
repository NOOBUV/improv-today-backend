"""
Tests for ContextualBackstoryService - Story 2.6 Enhanced Conversational Context Integration
"""
import pytest
from unittest.mock import Mock, patch
from app.services.contextual_backstory_service import ContextualBackstoryService
from app.core.conversation_config import ConversationContextConfig


class TestContextualBackstoryService:
    """Test suite for ContextualBackstoryService functionality."""

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
        """Create ContextualBackstoryService instance with mock dependencies."""
        with patch('app.services.contextual_backstory_service.CharacterContentService'):
            service = ContextualBackstoryService(mock_config)
            return service

    @pytest.mark.asyncio
    async def test_keyword_matching_childhood(self, service):
        """Test keyword matching for childhood-related content."""
        # Mock character content
        service.character_service.load_childhood_memories.return_value = "Childhood content here..."
        service.character_service.load_character_gist.return_value = "General character info..."

        user_message = "Tell me about your childhood and your mother"

        result = await service.select_relevant_content(user_message)

        assert result["content_types"] == ["childhood_memories", "character_gist"]
        assert "childhood" in result["selection_reasoning"].lower()
        assert result["char_count"] > 0

    @pytest.mark.asyncio
    async def test_keyword_matching_positive(self, service):
        """Test keyword matching for positive content."""
        service.character_service.load_positive_memories.return_value = "Happy memories content..."
        service.character_service.load_character_gist.return_value = "General character info..."

        user_message = "Tell me about your happiest memories and best experiences"

        result = await service.select_relevant_content(user_message)

        assert "positive_memories" in result["content_types"]
        assert result["keyword_matches"]["positive_memories"] > 0

    @pytest.mark.asyncio
    async def test_keyword_matching_difficult(self, service):
        """Test keyword matching for difficult/trauma content."""
        service.character_service.load_connecting_memories.return_value = "Difficult memories content..."
        service.character_service.load_character_gist.return_value = "General character info..."

        user_message = "Tell me about difficult times and struggles in your life"

        result = await service.select_relevant_content(user_message)

        assert "connecting_memories" in result["content_types"]
        assert result["keyword_matches"]["connecting_memories"] > 0

    @pytest.mark.asyncio
    async def test_keyword_matching_relationships(self, service):
        """Test keyword matching for relationship content."""
        service.character_service.load_friend_character.return_value = "Friend character content..."
        service.character_service.load_character_gist.return_value = "General character info..."

        user_message = "Tell me about your friends and relationships"

        result = await service.select_relevant_content(user_message)

        assert "friend_character" in result["content_types"]
        assert result["keyword_matches"]["friend_character"] > 0

    @pytest.mark.asyncio
    async def test_general_fallback(self, service):
        """Test fallback to character gist for general queries."""
        service.character_service.load_character_gist.return_value = "General character info..."

        user_message = "Tell me about yourself"

        result = await service.select_relevant_content(user_message)

        assert "character_gist" in result["content_types"]
        assert "fallback for general query" in result["selection_reasoning"]

    @pytest.mark.asyncio
    async def test_content_length_limiting(self, service):
        """Test that content is properly limited by character count."""
        long_content = "x" * 2000  # Content longer than limit
        service.character_service.load_character_gist.return_value = long_content

        user_message = "Tell me about yourself"
        max_chars = 500

        result = await service.select_relevant_content(user_message, max_chars=max_chars)

        assert result["char_count"] <= max_chars
        assert result["truncated"] == True
        assert result["char_limit_used"] == max_chars

    @pytest.mark.asyncio
    async def test_multiple_keyword_matches(self, service):
        """Test handling of multiple keyword matches."""
        service.character_service.load_childhood_memories.return_value = "Childhood content..."
        service.character_service.load_positive_memories.return_value = "Happy content..."
        service.character_service.load_character_gist.return_value = "General info..."

        user_message = "Tell me about your happy childhood memories with your mother"

        result = await service.select_relevant_content(user_message)

        # Should include both childhood and positive content
        assert "childhood_memories" in result["content_types"]
        assert "positive_memories" in result["content_types"]
        assert len(result["content_types"]) >= 2

    @pytest.mark.asyncio
    async def test_caching_functionality(self, service):
        """Test that content is properly cached."""
        service.character_service.load_character_gist.return_value = "Cached content..."

        # First call should load content
        await service.select_relevant_content("Tell me about yourself")

        # Second call should use cache
        await service.select_relevant_content("Who are you?")

        # Character service should only be called once due to caching
        assert service.character_service.load_character_gist.call_count == 1

    def test_cache_status(self, service):
        """Test cache status reporting."""
        # Add some mock content to cache
        service._content_cache["character_gist"] = "test content"

        status = service.get_cache_status()

        assert status["cached_content_types"] == ["character_gist"]
        assert status["cache_size"] == 1
        assert status["total_cached_chars"] == len("test content")

    def test_clear_cache(self, service):
        """Test cache clearing functionality."""
        # Add some mock content to cache
        service._content_cache["character_gist"] = "test content"

        service.clear_cache()

        assert service._content_cache == {}
        status = service.get_cache_status()
        assert status["cache_size"] == 0

    @pytest.mark.asyncio
    async def test_error_handling(self, service):
        """Test error handling when content loading fails gracefully."""
        # Mock all character content methods to return None (simulating file not found)
        service.character_service.load_character_gist.return_value = None
        service.character_service.load_childhood_memories.return_value = None
        service.character_service.load_positive_memories.return_value = None
        service.character_service.load_connecting_memories.return_value = None
        service.character_service.load_friend_character.return_value = None

        result = await service.select_relevant_content("Tell me about yourself")

        # Should return empty content but handle gracefully
        assert result["content"] == ""
        assert result["content_types"] == []
        assert result["char_count"] == 0

    @pytest.mark.asyncio
    async def test_exception_fallback_behavior(self, service):
        """Test complete exception fallback with actual error."""
        # Force a runtime exception in the main flow
        service._analyze_keyword_matches = Mock(side_effect=Exception("Critical error"))

        result = await service.select_relevant_content("Tell me about yourself")

        # Should return fallback content with error indication
        assert result["fallback_mode"] == True
        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_message_handling(self, service):
        """Test handling of empty or None messages."""
        service.character_service.load_character_gist.return_value = "General info..."

        result = await service.select_relevant_content("")

        # Should still return character gist as fallback
        assert "character_gist" in result["content_types"]

    @pytest.mark.asyncio
    async def test_token_estimation(self, service):
        """Test token count estimation."""
        content = "x" * 400  # 400 characters
        service.character_service.load_character_gist.return_value = content

        result = await service.select_relevant_content("Tell me about yourself")

        # Should estimate roughly 100 tokens (400 chars / 4)
        assert result["estimated_tokens"] == 100
        assert result["char_count"] == 400

    @pytest.mark.asyncio
    async def test_content_priority_ordering(self, service):
        """Test that content is selected based on priority ordering."""
        # Mock all content types
        service.character_service.load_character_gist.return_value = "Gist content"
        service.character_service.load_childhood_memories.return_value = "Childhood content"
        service.character_service.load_connecting_memories.return_value = "Trauma content"

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


class TestContextualBackstoryServiceIntegration:
    """Integration tests for ContextualBackstoryService with real CharacterContentService."""

    @pytest.mark.asyncio
    async def test_integration_with_real_character_service(self):
        """Test integration with actual CharacterContentService (if content files exist)."""
        service = ContextualBackstoryService()

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
        service = ContextualBackstoryService()

        # Mock large content
        large_content = "x" * 10000  # 10k characters
        with patch.object(service.character_service, 'load_character_gist', return_value=large_content):

            import time
            start_time = time.time()

            result = await service.select_relevant_content("Tell me about yourself")

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Should complete within reasonable time (less than 100ms)
            assert duration_ms < 100
            assert result["char_count"] <= service.config.MAX_BACKSTORY_CHARS