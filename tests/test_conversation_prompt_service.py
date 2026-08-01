"""
Tests for ConversationPromptService
"""
import pytest
from app.services.conversation_prompt_service import ConversationPromptService, EmotionType


@pytest.fixture
def prompt_service():
    """Create ConversationPromptService instance for testing"""
    return ConversationPromptService()


class TestConversationPromptService:
    """Test suite for ConversationPromptService"""
    
    def test_emotion_types_are_valid(self):
        """Test that all emotion types are properly defined"""
        expected_emotions = {"calm", "happy", "sad", "stressed", "sassy"}
        actual_emotions = {e.value for e in EmotionType}
        assert actual_emotions == expected_emotions
    
    def test_linguistic_patterns_exist_for_all_emotions(self, prompt_service):
        """Test that linguistic patterns are defined for all emotions"""
        for emotion in EmotionType:
            assert emotion in prompt_service.EMOTION_LINGUISTIC_PATTERNS
            pattern = prompt_service.EMOTION_LINGUISTIC_PATTERNS[emotion]
            assert "tone" in pattern
            assert "example" in pattern
            assert "characteristics" in pattern
            assert isinstance(pattern["characteristics"], list)
    
    def test_get_global_mood_context_default(self, prompt_service):
        """Test global mood context with default values"""
        result = prompt_service._get_global_mood_context()
        
        assert "stressed" in result
        assert "65" in result
        assert "work deadline" in result
    
    def test_get_global_mood_context_custom(self, prompt_service):
        """Test global mood context with custom values"""
        result = prompt_service._get_global_mood_context("happy", 20)
        
        assert "happy" in result
        assert "20" in result
    
    def test_get_conversation_emotion_context_all_emotions(self, prompt_service):
        """Test conversation emotion context for all emotion types"""
        for emotion in EmotionType:
            result = prompt_service._get_conversation_emotion_context(emotion, "test message")
            
            assert emotion.value in result
            assert "you are feeling" in result
    
    def test_build_emotion_guidance(self, prompt_service):
        """Test emotion guidance construction"""
        result = prompt_service._build_emotion_guidance(EmotionType.SASSY)
        
        assert "sassy" in result.lower()
        assert "Tone:" in result
        assert "Example response style:" in result
        assert "Key characteristics:" in result
        assert "Ironic agreement" in result  # From sassy characteristics
    
    @pytest.mark.parametrize("message,expected", [
        ("That's so funny!", EmotionType.SASSY),
        ("What a ridiculous joke", EmotionType.SASSY),
        ("I'm feeling sad today", EmotionType.SAD),
        ("This is really difficult", EmotionType.SAD),
        ("I'm so happy about this!", EmotionType.HAPPY),
        ("That's great news!", EmotionType.HAPPY),
        ("I'm so busy with this deadline", EmotionType.STRESSED),
        ("Under so much pressure", EmotionType.STRESSED),
        ("How are you doing?", EmotionType.CALM),
        ("What's the weather like?", EmotionType.CALM),
    ])
    def test_emotion_keyword_selection(self, prompt_service, message, expected):
        """Keyword heuristic still drives emotion selection through the public API."""
        emotion, reasoning = prompt_service.select_conversation_emotion_with_mood(message)

        assert emotion == expected
        assert isinstance(reasoning, str) and reasoning

    def test_mood_score_overrides_keyword_emotion(self, prompt_service):
        """Low blended mood pulls a happy keyword match down to calm."""
        emotion, reasoning = prompt_service.select_conversation_emotion_with_mood(
            "That's great news!",
            blended_mood_score=35.0,
            mood_transition_data={"mood_context": {"mood_category": "low"}},
        )

        assert emotion == EmotionType.CALM
        assert "low mood" in reasoning
