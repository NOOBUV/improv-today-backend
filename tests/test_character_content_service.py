"""
Tests for CharacterContentService content loading.
"""
import pytest
from pathlib import Path
from unittest.mock import patch

from app.services.character_content_service import CharacterContentService


@pytest.fixture
def content_service():
    """Create CharacterContentService instance for testing"""
    return CharacterContentService()


class TestCharacterContentService:
    """Test suite for CharacterContentService"""

    def test_init_sets_correct_base_path(self, content_service):
        """Test that initialization sets correct content base path"""
        assert content_service.content_base_path.name == "clara"
        assert content_service.content_base_path.parent.name == "content"

    def test_content_files_map(self):
        """CONTENT_FILES covers all six content types"""
        assert set(CharacterContentService.CONTENT_FILES) == {
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
        """Test behavior for a content type not in CONTENT_FILES"""
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
