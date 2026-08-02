"""
Tests for CharacterContentService content loading.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from app.core.conversation_config import ConversationContextConfig

from app.services.character_content_service import CharacterContentService


@pytest.fixture
def content_service():
    """Create CharacterContentService instance for testing"""
    return CharacterContentService()

from app.core.conversation_config import ConversationContextConfig

class TestCharacterContentService:
    """Test suite for CharacterContentService"""

    def test_init_sets_correct_base_path(self, content_service):
        """Test that initialization sets correct content base path"""
        assert content_service.content_base_path.name == "clara"
        assert content_service.content_base_path.parent.name == "content"

    def test_content_files_map(self):
        """CONTENT covers all six content types"""
        assert set(CharacterContentService.CONTENT) == {
            "character_gist",
            "connecting_memories",
            "childhood_memories",
            "positive_memories",
            "friend_character",
            "romantic_relationship",
        }

    def test_load_success(self, content_service):
        """Test successful loading of a content file"""
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", return_value="# Character Gist\nTest content"):
            result = content_service.load("character_gist")

        assert result == "# Character Gist\nTest content"

    def test_load_file_not_found(self, content_service):
        """Test behavior when content file doesn't exist"""
        with patch.object(Path, "exists", return_value=False):
            result = content_service.load("character_gist")

        assert result == ""

    def test_load_file_error(self, content_service):
        """Test behavior when file access fails"""
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", side_effect=IOError("File access error")):
            result = content_service.load("character_gist")

        assert result == ""

    def test_load_unknown_content_type(self, content_service):
        """Test behavior for a content type not in CONTENT"""
        assert content_service.load("nonexistent_type") == ""

    def test_load_caches_content(self, content_service):
        """Test that repeated loads only read the file once"""
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", return_value="cached content") as mock_read:
            first = content_service.load("character_gist")
            second = content_service.load("character_gist")

        assert first == second == "cached content"
        assert mock_read.call_count == 1

    def test_get_consolidated_backstory(self, content_service):
        """Test consolidated backstory construction"""
        content = {
            "character_gist": "# Gist\nGist content",
            "connecting_memories": "# Memories\nMemory content",
            "childhood_memories": "# Childhood\nChildhood content",
            "positive_memories": "",  # Empty content
            "friend_character": "# Friend\nFriend content",
            "romantic_relationship": "",  # Empty content
        }
        with patch.object(content_service, "load", side_effect=content.get):
            result = content_service.get_consolidated_backstory()

        # Check that sections are properly formatted and empty content is skipped
        assert "# Character Overview\n# Gist\nGist content" in result
        assert "# Key Life Experiences\n# Memories\nMemory content" in result
        assert "# Childhood Context\n# Childhood\nChildhood content" in result
        assert "# Important Relationships\n# Friend\nFriend content" in result
        assert "# Positive Memories" not in result  # Empty content should be skipped
        assert "# Romantic Relationship History" not in result

        # Check sections are separated by double newlines
        sections = result.split("\n\n")
        assert len(sections) == 4  # 4 non-empty sections

    def test_get_consolidated_backstory_all_empty(self, content_service):
        """Test consolidated backstory when all content is empty"""
        with patch.object(content_service, "load", return_value=""):
            result = content_service.get_consolidated_backstory()

        assert result == ""


class TestBackstorySelection:
    """Backstory selection behavior (merged from the old contextual-backstory suite)."""

    @pytest.fixture
    def mock_config(self):
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
        service = CharacterContentService(mock_config)
        service._test_content = {}
        service.load = Mock(side_effect=lambda t: service._test_content.get(t, ""))
        return service

    @pytest.mark.asyncio
    async def test_keyword_matching_childhood(self, service):
        service._test_content["childhood_memories"] = "Childhood content here..."
        service._test_content["character_gist"] = "General character info..."

        result = await service.select_relevant_content("Tell me about your childhood and your mother")

        assert result["content_types"] == ["childhood_memories", "character_gist"]
        assert result["char_count"] > 0

    @pytest.mark.asyncio
    async def test_keyword_matching_positive(self, service):
        service._test_content["positive_memories"] = "Happy memories content..."
        service._test_content["character_gist"] = "General character info..."

        result = await service.select_relevant_content("Tell me about your happiest memories and best experiences")

        assert "positive_memories" in result["content_types"]

    @pytest.mark.asyncio
    async def test_keyword_matching_difficult(self, service):
        service._test_content["connecting_memories"] = "Difficult memories content..."
        service._test_content["character_gist"] = "General character info..."

        result = await service.select_relevant_content("Tell me about difficult times and struggles in your life")

        assert "connecting_memories" in result["content_types"]

    @pytest.mark.asyncio
    async def test_keyword_matching_relationships(self, service):
        service._test_content["friend_character"] = "Friend character content..."
        service._test_content["character_gist"] = "General character info..."

        result = await service.select_relevant_content("Tell me about your friends and relationships")

        assert "friend_character" in result["content_types"]

    @pytest.mark.asyncio
    async def test_general_fallback(self, service):
        service._test_content["character_gist"] = "General character info..."

        result = await service.select_relevant_content("Tell me about yourself")

        assert "character_gist" in result["content_types"]

    @pytest.mark.asyncio
    async def test_content_length_limiting(self, service):
        service._test_content["character_gist"] = "x" * 2000

        result = await service.select_relevant_content("Tell me about yourself", max_chars=500)

        assert result["char_count"] <= 500

    @pytest.mark.asyncio
    async def test_multiple_keyword_matches(self, service):
        service._test_content["childhood_memories"] = "Childhood content..."
        service._test_content["positive_memories"] = "Happy content..."
        service._test_content["character_gist"] = "General info..."

        result = await service.select_relevant_content("Tell me about your happy childhood memories with your mother")

        assert "childhood_memories" in result["content_types"]
        assert "positive_memories" in result["content_types"]
        assert len(result["content_types"]) >= 2

    @pytest.mark.asyncio
    async def test_caching_functionality(self, mock_config):
        service = CharacterContentService(mock_config)

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", return_value="Cached content...") as mock_read:
            await service.select_relevant_content("Tell me about yourself")
            await service.select_relevant_content("Who are you?")

        assert mock_read.call_count == 1

    @pytest.mark.asyncio
    async def test_error_handling(self, service):
        result = await service.select_relevant_content("Tell me about yourself")

        assert result["content"] == ""
        assert result["content_types"] == []
        assert result["char_count"] == 0

    @pytest.mark.asyncio
    async def test_exception_fallback_behavior(self, service):
        """A failure mid-selection falls back to the gist (real load() swallows I/O errors)."""
        service._analyze_keyword_matches = Mock(side_effect=Exception("Critical error"))
        service._test_content["character_gist"] = "General character info..."

        result = await service.select_relevant_content("Tell me about yourself")

        assert result["content_types"] == ["character_gist"]
        assert result["content"] == "General character info..."

    @pytest.mark.asyncio
    async def test_empty_message_handling(self, service):
        service._test_content["character_gist"] = "General info..."

        result = await service.select_relevant_content("")

        assert "character_gist" in result["content_types"]

    @pytest.mark.asyncio
    async def test_content_priority_ordering(self, service):
        service._test_content["character_gist"] = "Gist content"
        service._test_content["childhood_memories"] = "Childhood content"
        service._test_content["connecting_memories"] = "Trauma content"

        result = await service.select_relevant_content("Tell me about your difficult childhood experiences")

        content_types = result["content_types"]
        if "connecting_memories" in content_types and "childhood_memories" in content_types:
            assert content_types.index("connecting_memories") <= content_types.index("childhood_memories")
