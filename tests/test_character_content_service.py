"""
Tests for CharacterContentService
"""
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

from app.services.character_content_service import CharacterContentService


@pytest.fixture
def content_service():
    """Create CharacterContentService instance for testing"""
    return CharacterContentService()


class TestCharacterContentService:
    """Test suite for CharacterContentService"""
    
    def test_init_sets_correct_base_path(self, content_service):
        """Test that initialization sets correct content base path"""
        expected_path = Path(__file__).parent.parent / "content" / "clara"
        # The path might be different in test environment, just check structure
        assert content_service.content_base_path.name == "clara"
        assert content_service.content_base_path.parent.name == "content"
    
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="# Character Gist\nTest content")
    def test_load_character_gist_success(self, mock_file, mock_exists, content_service):
        """Test successful loading of character gist"""
        mock_exists.return_value = True
        
        result = content_service.load_character_gist()
        
        assert result == "# Character Gist\nTest content"
        mock_exists.assert_called_once()
        mock_file.assert_called_once()
    
    @patch("pathlib.Path.exists")
    def test_load_character_gist_file_not_found(self, mock_exists, content_service):
        """Test behavior when character gist file doesn't exist"""
        mock_exists.return_value = False
        
        result = content_service.load_character_gist()
        
        assert result == ""
        mock_exists.assert_called_once()
    
    @patch("pathlib.Path.exists")
    @patch("builtins.open", side_effect=IOError("File access error"))
    def test_load_character_gist_file_error(self, mock_file, mock_exists, content_service):
        """Test behavior when file access fails"""
        mock_exists.return_value = True
        
        result = content_service.load_character_gist()
        
        assert result == ""
    
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="# Memories\nDetailed memories content")
    def test_load_connecting_memories_success(self, mock_file, mock_exists, content_service):
        """Test successful loading of connecting memories"""
        mock_exists.return_value = True
        
        result = content_service.load_connecting_memories()
        
        assert result == "# Memories\nDetailed memories content"
        mock_exists.assert_called_once()
        mock_file.assert_called_once()
    
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="# Childhood\nChildhood content")
    def test_load_childhood_memories_success(self, mock_file, mock_exists, content_service):
        """Test successful loading of childhood memories"""
        mock_exists.return_value = True
        
        result = content_service.load_childhood_memories()
        
        assert result == "# Childhood\nChildhood content"
    
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="# Positive\nPositive content")
    def test_load_positive_memories_success(self, mock_file, mock_exists, content_service):
        """Test successful loading of positive memories"""
        mock_exists.return_value = True
        
        result = content_service.load_positive_memories()
        
        assert result == "# Positive\nPositive content"
    
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="# Friend\nFriend content")
    def test_load_friend_character_success(self, mock_file, mock_exists, content_service):
        """Test successful loading of friend character"""
        mock_exists.return_value = True
        
        result = content_service.load_friend_character()
        
        assert result == "# Friend\nFriend content"
    
    @patch.object(CharacterContentService, 'load_character_gist')
    @patch.object(CharacterContentService, 'load_connecting_memories')
    @patch.object(CharacterContentService, 'load_childhood_memories')
    @patch.object(CharacterContentService, 'load_positive_memories')
    @patch.object(CharacterContentService, 'load_friend_character')
    def test_load_all_character_content(self, mock_friend, mock_positive, mock_childhood, 
                                      mock_connecting, mock_gist, content_service):
        """Test loading all character content"""
        # Setup mocks
        mock_gist.return_value = "gist content"
        mock_connecting.return_value = "connecting content"
        mock_childhood.return_value = "childhood content"
        mock_positive.return_value = "positive content"
        mock_friend.return_value = "friend content"
        
        result = content_service.load_all_character_content()
        
        expected = {
            "character_gist": "gist content",
            "connecting_memories": "connecting content",
            "childhood_memories": "childhood content",
            "positive_memories": "positive content",
            "friend_character": "friend content"
        }
        
        assert result == expected
        
        # Verify all methods were called
        mock_gist.assert_called_once()
        mock_connecting.assert_called_once()
        mock_childhood.assert_called_once()
        mock_positive.assert_called_once()
        mock_friend.assert_called_once()
    
    @patch.object(CharacterContentService, 'load_all_character_content')
    def test_get_consolidated_backstory(self, mock_load_all, content_service):
        """Test consolidated backstory construction"""
        mock_load_all.return_value = {
            "character_gist": "# Gist\nGist content",
            "connecting_memories": "# Memories\nMemory content",
            "childhood_memories": "# Childhood\nChildhood content",
            "positive_memories": "",  # Empty content
            "friend_character": "# Friend\nFriend content"
        }
        
        result = content_service.get_consolidated_backstory()
        
        # Check that sections are properly formatted and empty content is skipped
        assert "# Character Overview\n# Gist\nGist content" in result
        assert "# Key Life Experiences\n# Memories\nMemory content" in result
        assert "# Childhood Context\n# Childhood\nChildhood content" in result
        assert "# Important Relationships\n# Friend\nFriend content" in result
        assert "# Positive Memories" not in result  # Empty content should be skipped
        
        # Check sections are separated by double newlines
        sections = result.split("\n\n")
        assert len(sections) == 4  # 4 non-empty sections
    
    @patch.object(CharacterContentService, 'load_all_character_content')
    def test_get_consolidated_backstory_all_empty(self, mock_load_all, content_service):
        """Test consolidated backstory when all content is empty"""
        mock_load_all.return_value = {
            "character_gist": "",
            "connecting_memories": "",
            "childhood_memories": "",
            "positive_memories": "",
            "friend_character": ""
        }
        
        result = content_service.get_consolidated_backstory()
        
        assert result == ""