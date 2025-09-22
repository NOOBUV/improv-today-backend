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
    
    def test_select_conversation_emotion_funny_keywords(self, prompt_service):
        """Test emotion selection for funny/humorous content"""
        test_messages = [
            "That's so funny!",
            "What a ridiculous joke",
            "You make me laugh",
            "That's silly"
        ]
        
        for message in test_messages:
            emotion = prompt_service.select_conversation_emotion(message)
            assert emotion == EmotionType.SASSY
    
    def test_select_conversation_emotion_sad_keywords(self, prompt_service):
        """Test emotion selection for sad content"""
        test_messages = [
            "I'm feeling sad today",
            "Sorry to hear about your problem",
            "This is really difficult",
            "I'm struggling with this"
        ]
        
        for message in test_messages:
            emotion = prompt_service.select_conversation_emotion(message)
            assert emotion == EmotionType.SAD
    
    def test_select_conversation_emotion_happy_keywords(self, prompt_service):
        """Test emotion selection for happy content"""
        test_messages = [
            "I'm so happy about this!",
            "That's great news!",
            "How wonderful!",
            "I'm excited about this amazing opportunity"
        ]
        
        for message in test_messages:
            emotion = prompt_service.select_conversation_emotion(message)
            assert emotion == EmotionType.HAPPY
    
    def test_select_conversation_emotion_stressed_keywords(self, prompt_service):
        """Test emotion selection for stressed content"""
        test_messages = [
            "I'm so busy with this deadline",
            "Feeling overwhelmed by work",
            "Under so much pressure",
            "This is urgent and stressful"
        ]
        
        for message in test_messages:
            emotion = prompt_service.select_conversation_emotion(message)
            assert emotion == EmotionType.STRESSED
    
    def test_select_conversation_emotion_default_calm(self, prompt_service):
        """Test emotion selection defaults to calm for neutral content"""
        test_messages = [
            "How are you doing?",
            "What's the weather like?",
            "Can you help me with this task?",
            "Let's discuss this topic"
        ]
        
        for message in test_messages:
            emotion = prompt_service.select_conversation_emotion(message)
            assert emotion == EmotionType.CALM
    
    def test_determine_emotion_from_context(self, prompt_service):
        """Test emotion determination with reasoning"""
        emotion, reasoning = prompt_service.determine_emotion_from_context("That's hilarious!")
        
        assert emotion == EmotionType.SASSY
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
    
    def test_construct_conversation_prompt_structure(self, prompt_service):
        """Test conversation prompt construction includes all required elements"""
        backstory = "Test backstory content"
        user_message = "Hello, how are you?"
        
        result = prompt_service.construct_conversation_prompt(
            character_backstory=backstory,
            user_message=user_message,
            conversation_emotion=EmotionType.CALM
        )
        
        # Check required components are present
        assert "You are Ava" in result
        assert backstory in result
        assert user_message in result
        assert "stressed" in result  # Global mood
        assert "65" in result  # Stress level
        assert "calm" in result  # Conversation emotion
        assert "Response format:" in result
        assert "message" in result
        assert "emotion" in result
    
    def test_construct_conversation_prompt_with_history(self, prompt_service):
        """Test conversation prompt construction with conversation history"""
        backstory = "Test backstory"
        user_message = "What do you think?"
        history = "Previous conversation context"
        
        result = prompt_service.construct_conversation_prompt(
            character_backstory=backstory,
            user_message=user_message,
            conversation_emotion=EmotionType.HAPPY,
            conversation_history=history
        )
        
        assert history in result
        assert "Recent conversation context:" in result
    
    def test_construct_conversation_prompt_custom_values(self, prompt_service):
        """Test conversation prompt with custom global mood and stress"""
        result = prompt_service.construct_conversation_prompt(
            character_backstory="Test backstory",
            user_message="Test message",
            conversation_emotion=EmotionType.STRESSED,
            global_mood="happy",
            stress_level=30
        )
        
        assert "happy" in result
        assert "30" in result
        assert "stressed" in result  # Conversation emotion
    
    def test_construct_conversation_prompt_different_emotions(self, prompt_service):
        """Test prompt construction for different conversation emotions"""
        backstory = "Test backstory"
        user_message = "Test message"
        
        for emotion in EmotionType:
            result = prompt_service.construct_conversation_prompt(
                character_backstory=backstory,
                user_message=user_message,
                conversation_emotion=emotion
            )
            
            # Each emotion should appear in the prompt
            assert emotion.value in result
            
            # Should contain emotion-specific guidance
            pattern = prompt_service.EMOTION_LINGUISTIC_PATTERNS[emotion]
            assert pattern["tone"] in result