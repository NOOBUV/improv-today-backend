import pytest
from app.services.suggestion_service import SuggestionService


class TestSuggestionService:
    """Test cases for the SuggestionService"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.service = SuggestionService()
    
    def test_generate_suggestion_with_matching_keyword(self):
        """Test suggestion generation when a keyword matches"""
        result = self.service.generate_suggestion("I feel very happy today!")
        
        assert result is not None
        assert result["word"] == "elated"
        assert result["definition"] == "feeling or expressing great happiness and triumph"
        assert result["triggered_by"] == "happy"
    
    def test_generate_suggestion_multiple_keywords(self):
        """Test that first matching keyword is used"""
        result = self.service.generate_suggestion("I'm happy and excited about this good news!")
        
        assert result is not None
        # Should match "happy" first (comes first in text)
        assert result["word"] == "elated"
        assert result["triggered_by"] == "happy"
    
    def test_generate_suggestion_no_match(self):
        """Test suggestion generation when no keywords match"""
        result = self.service.generate_suggestion("The quantum mechanics principles are fascinating!")
        
        assert result is None
    
    def test_generate_suggestion_empty_input(self):
        """Test suggestion generation with empty input"""
        result = self.service.generate_suggestion("")
        assert result is None
        
        result = self.service.generate_suggestion(None)
        assert result is None
        
        result = self.service.generate_suggestion("   ")
        assert result is None
    
    def test_generate_suggestion_case_insensitive(self):
        """Test that keyword matching is case insensitive"""
        result = self.service.generate_suggestion("I'm HAPPY today!")
        
        assert result is not None
        assert result["word"] == "elated"
        assert result["triggered_by"] == "happy"
    
    def test_generate_suggestion_with_punctuation(self):
        """Test suggestion generation with punctuation"""
        result = self.service.generate_suggestion("It's so good, really good!")
        
        assert result is not None
        assert result["word"] == "excellent"
        assert result["triggered_by"] == "good"
    
    def test_normalize_text(self):
        """Test text normalization"""
        normalized = self.service._normalize_text("Hello, World! It's Good.")
        assert normalized == "hello world it s good"
        
        normalized = self.service._normalize_text("   Multiple   Spaces   ")
        assert normalized == "multiple spaces"
    
    def test_suggestion_service_error_handling(self):
        """Test error handling in suggestion service"""
        # Create a service with corrupted data to trigger errors
        service = SuggestionService()
        service.keyword_suggestions = None  # This should cause an error
        
        result = service.generate_suggestion("happy")
        assert result is None  # Should handle error gracefully
    
    def test_all_predefined_keywords_work(self):
        """Test that all predefined keywords can generate suggestions"""
        service = SuggestionService()
        
        # Test a few key examples from each category
        test_cases = [
            ("happy", "elated"),
            ("good", "excellent"),
            ("walk", "stroll"),
            ("thing", "object"),
            ("work", "profession"),
            ("learn", "acquire")
        ]
        
        for keyword, expected_word in test_cases:
            result = service.generate_suggestion(f"I {keyword} every day")
            assert result is not None, f"Failed to generate suggestion for '{keyword}'"
            assert result["word"] == expected_word, f"Wrong suggestion for '{keyword}'"
            assert result["triggered_by"] == keyword