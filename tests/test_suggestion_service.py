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
        assert result["exampleSentence"] == "She was elated when she received the promotion she had been working toward."
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
            assert "exampleSentence" in result, f"Missing exampleSentence for '{keyword}'"
            assert isinstance(result["exampleSentence"], str), f"exampleSentence should be string for '{keyword}'"
            assert len(result["exampleSentence"]) > 0, f"exampleSentence should not be empty for '{keyword}'"
    
    def test_enhanced_suggestion_structure(self):
        """Test that enhanced suggestion structure includes all required fields"""
        result = self.service.generate_suggestion("This is good work")
        
        assert result is not None
        # Verify all required fields are present
        assert "word" in result
        assert "definition" in result  
        assert "exampleSentence" in result
        assert "triggered_by" in result
        
        # Verify field types
        assert isinstance(result["word"], str)
        assert isinstance(result["definition"], str)
        assert isinstance(result["exampleSentence"], str)
        assert isinstance(result["triggered_by"], str)
        
        # Verify field content is meaningful
        assert len(result["word"]) > 0
        assert len(result["definition"]) > 0
        assert len(result["exampleSentence"]) > 0
        assert len(result["triggered_by"]) > 0