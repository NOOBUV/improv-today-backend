"""
Integration tests for Story 2.4: End-to-End Correctness Feedback Loop

This test suite verifies that the complete feedback loop functionality works:
- Suggestion shown → word attempted → correctness evaluated → UI updated
- Correct usage removes suggestion
- Incorrect usage keeps suggestion active with remediation feedback
- Graceful timeout after 3-4 turns without usage
"""

import pytest
import uuid
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone, timedelta

from app.api.conversation import handle_conversation, ConversationRequest
from app.services.simple_openai import OpenAICoachingResponse, WordUsageStatus
from app.models.vocabulary import VocabularySuggestion


class TestStory24Integration:
    """Integration tests for Story 2.4 End-to-End Correctness Feedback Loop"""

    @pytest.mark.asyncio
    @patch('app.api.conversation.get_or_create_user')
    @patch('app.api.conversation.SimpleOpenAIService')
    @patch('app.api.conversation.VocabularyTierService')
    @patch('app.api.conversation.SuggestionService')
    @patch('app.api.conversation.RedisService')
    async def test_correct_word_usage_removes_suggestion(self,
                                                       mock_redis_service,
                                                       mock_suggestion_service,
                                                       mock_vocabulary_service,
                                                       mock_openai_service,
                                                       mock_get_user):
        """Test IV1: Correct word usage removes suggestion from UI"""
        
        # Setup mocks
        mock_get_user.return_value = 123
        user_id = 123
        conversation_id = uuid.uuid4()
        suggestion_id = 456
        
        # Mock database session
        mock_db = Mock()
        
        # Create existing suggestion
        existing_suggestion = Mock()
        existing_suggestion.id = suggestion_id
        existing_suggestion.user_id = str(user_id)
        existing_suggestion.suggested_word = "elaborate"
        existing_suggestion.status = "shown"
        existing_suggestion.created_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        
        # Mock database queries
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = existing_suggestion
        
        # Mock conversation query
        mock_conversation = Mock()
        mock_conversation.id = conversation_id
        mock_db.query.return_value.filter.return_value.first.return_value = mock_conversation
        
        # Mock Redis service
        mock_redis_instance = Mock()
        mock_redis_instance.get_conversation_history.return_value = []
        mock_redis_instance.build_conversation_context.return_value = ""
        mock_redis_instance.cache_message.return_value = True
        mock_redis_service.return_value = mock_redis_instance
        
        # Mock OpenAI service - word used correctly
        mock_openai_instance = Mock()
        mock_coaching_response = OpenAICoachingResponse(
            corrected_transcript="I want to elaborate on this topic.",
            ai_response="Great use of 'elaborate'! That shows sophisticated vocabulary.",
            word_usage_status=WordUsageStatus.USED_CORRECTLY,
            usage_correctness_feedback=None
        )
        mock_openai_instance.generate_coaching_response = AsyncMock(return_value=mock_coaching_response)
        mock_openai_service.return_value = mock_openai_instance
        
        # Mock vocabulary service
        mock_vocabulary_instance = Mock()
        mock_tier_analysis = Mock()
        mock_tier_analysis.tier = "mid"
        mock_tier_analysis.score = 75
        mock_tier_analysis.word_count = 8
        mock_tier_analysis.complex_word_count = 1
        mock_tier_analysis.average_word_length = 5.2
        mock_tier_analysis.analysis_details = {}
        mock_vocabulary_instance.analyze_vocabulary_tier.return_value = mock_tier_analysis
        mock_vocabulary_instance.get_vocabulary_recommendations.return_value = []
        mock_vocabulary_service.return_value = mock_vocabulary_instance
        
        # Mock suggestion service
        mock_suggestion_instance = Mock()
        mock_suggestion_instance.generate_suggestion.return_value = {
            "word": "fascinating",
            "definition": "extremely interesting",
            "exampleSentence": "The museum had a fascinating exhibit."
        }
        mock_suggestion_service.return_value = mock_suggestion_instance
        
        # Create request with correct word usage
        request = ConversationRequest(
            message="I want to elaborate on this topic",
            session_id=None,
            personality="friendly_neutral"
        )
        
        # Execute conversation
        mock_current_user = {"sub": "auth0|test", "email": "test@example.com"}
        response = await handle_conversation(request, mock_db, mock_current_user)
        
        # Verify correct usage behavior
        assert response.success is True
        assert response.used_suggestion_id == str(suggestion_id)  # AC: 3 - Signal frontend to remove
        assert existing_suggestion.status == "used"  # AC: 3 - Update status to "used"
        assert response.remediation_feedback is None  # AC: 4 - No feedback for correct usage
        
        # Verify new suggestion is generated (since word was used correctly)
        assert response.suggestion is not None
        assert response.suggestion["word"] == "fascinating"
        
        print("✅ Test passed: Correct word usage removes suggestion")

    @pytest.mark.asyncio
    @patch('app.api.conversation.get_or_create_user')
    @patch('app.api.conversation.SimpleOpenAIService')
    @patch('app.api.conversation.VocabularyTierService')
    @patch('app.api.conversation.SuggestionService')
    @patch('app.api.conversation.RedisService')
    async def test_incorrect_word_usage_shows_feedback(self,
                                                     mock_redis_service,
                                                     mock_suggestion_service,
                                                     mock_vocabulary_service,
                                                     mock_openai_service,
                                                     mock_get_user):
        """Test IV1: Incorrect word usage shows remediation feedback and keeps suggestion active"""
        
        # Setup mocks
        mock_get_user.return_value = 123
        user_id = 123
        conversation_id = uuid.uuid4()
        suggestion_id = 456
        
        # Mock database session
        mock_db = Mock()
        
        # Create existing suggestion
        existing_suggestion = Mock()
        existing_suggestion.id = suggestion_id
        existing_suggestion.user_id = str(user_id)
        existing_suggestion.suggested_word = "elaborate"
        existing_suggestion.status = "shown"
        existing_suggestion.created_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        
        # Mock database queries
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = existing_suggestion
        
        # Mock conversation query
        mock_conversation = Mock()
        mock_conversation.id = conversation_id
        mock_db.query.return_value.filter.return_value.first.return_value = mock_conversation
        
        # Mock Redis service
        mock_redis_instance = Mock()
        mock_redis_instance.get_conversation_history.return_value = []
        mock_redis_instance.build_conversation_context.return_value = ""
        mock_redis_instance.cache_message.return_value = True
        mock_redis_service.return_value = mock_redis_instance
        
        # Mock OpenAI service - word used incorrectly
        mock_openai_instance = Mock()
        mock_coaching_response = OpenAICoachingResponse(
            corrected_transcript="I want to elaborate this car.",
            ai_response="I see you're trying to use 'elaborate'! Let me help with that.",
            word_usage_status=WordUsageStatus.USED_INCORRECTLY,
            usage_correctness_feedback="Good try! 'Elaborate' means to add more detail. Try 'elaborate on' instead of 'elaborate this'."
        )
        mock_openai_instance.generate_coaching_response = AsyncMock(return_value=mock_coaching_response)
        mock_openai_service.return_value = mock_openai_instance
        
        # Mock vocabulary service
        mock_vocabulary_instance = Mock()
        mock_tier_analysis = Mock()
        mock_tier_analysis.tier = "mid"
        mock_tier_analysis.score = 70
        mock_tier_analysis.word_count = 7
        mock_tier_analysis.complex_word_count = 1
        mock_tier_analysis.average_word_length = 4.8
        mock_tier_analysis.analysis_details = {}
        mock_vocabulary_instance.analyze_vocabulary_tier.return_value = mock_tier_analysis
        mock_vocabulary_instance.get_vocabulary_recommendations.return_value = []
        mock_vocabulary_service.return_value = mock_vocabulary_instance
        
        # Mock suggestion service - should not generate new suggestion
        mock_suggestion_instance = Mock()
        mock_suggestion_instance.generate_suggestion.return_value = None
        mock_suggestion_service.return_value = mock_suggestion_instance
        
        # Create request with incorrect word usage
        request = ConversationRequest(
            message="I want to elaborate this car",
            session_id=None,
            personality="friendly_neutral"
        )
        
        # Execute conversation
        mock_current_user = {"sub": "auth0|test", "email": "test@example.com"}
        response = await handle_conversation(request, mock_db, mock_current_user)
        
        # Verify incorrect usage behavior
        assert response.success is True
        assert response.used_suggestion_id is None  # AC: 5 - Suggestion remains active
        assert existing_suggestion.status == "used_incorrectly"  # AC: 4 - Update status
        assert response.remediation_feedback is not None  # AC: 4 - Include feedback in response
        assert "Good try!" in response.remediation_feedback
        assert "elaborate on" in response.remediation_feedback
        
        # Verify no new suggestion is generated (keep existing active)
        assert response.suggestion is None
        
        print("✅ Test passed: Incorrect word usage shows remediation feedback")

    @pytest.mark.asyncio
    @patch('app.api.conversation.get_or_create_user')
    @patch('app.api.conversation.SimpleOpenAIService')
    @patch('app.api.conversation.VocabularyTierService')
    @patch('app.api.conversation.SuggestionService')
    @patch('app.api.conversation.RedisService')
    async def test_graceful_suggestion_replacement_after_timeout(self,
                                                               mock_redis_service,
                                                               mock_suggestion_service,
                                                               mock_vocabulary_service,
                                                               mock_openai_service,
                                                               mock_get_user):
        """Test IV2: Graceful suggestion replacement after 3-4 turns without usage"""
        
        # Setup mocks
        mock_get_user.return_value = 123
        user_id = 123
        conversation_id = uuid.uuid4()
        suggestion_id = 456
        
        # Mock database session
        mock_db = Mock()
        
        # Create old suggestion (more than 4 turns ago)
        old_suggestion = Mock()
        old_suggestion.id = suggestion_id
        old_suggestion.user_id = str(user_id)
        old_suggestion.suggested_word = "elaborate"
        old_suggestion.status = "shown"
        # Created 10 minutes ago to simulate old suggestion
        old_suggestion.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        
        # Mock database queries
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = old_suggestion
        
        # Mock conversation query
        mock_conversation = Mock()
        mock_conversation.id = conversation_id
        mock_db.query.return_value.filter.return_value.first.return_value = mock_conversation
        
        # Mock Redis service with conversation history (5 messages = more than 4 turns)
        mock_redis_instance = Mock()
        # Simulate 5 messages since suggestion was created
        recent_timestamp = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        mock_redis_instance.get_conversation_history.return_value = [
            {"role": "user", "content": "Hello", "timestamp": recent_timestamp},
            {"role": "assistant", "content": "Hi there", "timestamp": recent_timestamp},
            {"role": "user", "content": "How are you", "timestamp": recent_timestamp},
            {"role": "assistant", "content": "Good thanks", "timestamp": recent_timestamp},
            {"role": "user", "content": "What's new", "timestamp": recent_timestamp},
        ]
        mock_redis_instance.build_conversation_context.return_value = "Previous conversation context"
        mock_redis_instance.cache_message.return_value = True
        mock_redis_service.return_value = mock_redis_instance
        
        # Mock OpenAI service - word not used
        mock_openai_instance = Mock()
        mock_coaching_response = OpenAICoachingResponse(
            corrected_transcript="What's the weather like today?",
            ai_response="It's sunny and warm! Perfect for outdoor activities.",
            word_usage_status=WordUsageStatus.NOT_USED,
            usage_correctness_feedback=None
        )
        mock_openai_instance.generate_coaching_response = AsyncMock(return_value=mock_coaching_response)
        mock_openai_service.return_value = mock_openai_instance
        
        # Mock vocabulary service
        mock_vocabulary_instance = Mock()
        mock_tier_analysis = Mock()
        mock_tier_analysis.tier = "basic"
        mock_tier_analysis.score = 60
        mock_tier_analysis.word_count = 6
        mock_tier_analysis.complex_word_count = 0
        mock_tier_analysis.average_word_length = 4.5
        mock_tier_analysis.analysis_details = {}
        mock_vocabulary_instance.analyze_vocabulary_tier.return_value = mock_tier_analysis
        mock_vocabulary_instance.get_vocabulary_recommendations.return_value = []
        mock_vocabulary_service.return_value = mock_vocabulary_instance
        
        # Mock suggestion service - generate replacement
        mock_suggestion_instance = Mock()
        mock_suggestion_instance.generate_suggestion.return_value = {
            "word": "magnificent",
            "definition": "extremely beautiful or impressive",
            "exampleSentence": "The sunset was truly magnificent."
        }
        mock_suggestion_service.return_value = mock_suggestion_instance
        
        # Create request unrelated to suggestion
        request = ConversationRequest(
            message="What's the weather like today?",
            session_id=None,
            personality="friendly_neutral"
        )
        
        # Execute conversation
        mock_current_user = {"sub": "auth0|test", "email": "test@example.com"}
        response = await handle_conversation(request, mock_db, mock_current_user)
        
        # Verify graceful replacement behavior
        assert response.success is True
        assert old_suggestion.status == "ignored"  # IV2 - Gracefully replaced without negative feedback
        assert response.used_suggestion_id is None  # No usage to report
        assert response.remediation_feedback is None  # IV2 - No negative feedback for ignored suggestions
        
        # Verify new suggestion is generated
        assert response.suggestion is not None
        assert response.suggestion["word"] == "magnificent"
        
        print("✅ Test passed: Graceful suggestion replacement after timeout")

    @pytest.mark.asyncio
    @patch('app.api.conversation.get_or_create_user')
    @patch('app.api.conversation.SimpleOpenAIService')
    @patch('app.api.conversation.VocabularyTierService')
    @patch('app.api.conversation.SuggestionService')
    @patch('app.api.conversation.RedisService')
    async def test_suggestion_persists_after_incorrect_usage(self,
                                                           mock_redis_service,
                                                           mock_suggestion_service,
                                                           mock_vocabulary_service,
                                                           mock_openai_service,
                                                           mock_get_user):
        """Test AC: 5 - Suggestion remains active after incorrect usage until correct usage"""
        
        # Setup mocks
        mock_get_user.return_value = 123
        user_id = 123
        conversation_id = uuid.uuid4()
        suggestion_id = 456
        
        # Mock database session
        mock_db = Mock()
        
        # Create suggestion with incorrect usage status
        existing_suggestion = Mock()
        existing_suggestion.id = suggestion_id
        existing_suggestion.user_id = str(user_id)
        existing_suggestion.suggested_word = "elaborate"
        existing_suggestion.status = "used_incorrectly"  # Previously used incorrectly
        existing_suggestion.created_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        
        # Mock database queries
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = existing_suggestion
        
        # Mock conversation query
        mock_conversation = Mock()
        mock_conversation.id = conversation_id
        mock_db.query.return_value.filter.return_value.first.return_value = mock_conversation
        
        # Mock Redis service
        mock_redis_instance = Mock()
        mock_redis_instance.get_conversation_history.return_value = []
        mock_redis_instance.build_conversation_context.return_value = ""
        mock_redis_instance.cache_message.return_value = True
        mock_redis_service.return_value = mock_redis_instance
        
        # Mock OpenAI service - word not used in this turn
        mock_openai_instance = Mock()
        mock_coaching_response = OpenAICoachingResponse(
            corrected_transcript="I love reading books.",
            ai_response="That's wonderful! Reading is such a great hobby.",
            word_usage_status=WordUsageStatus.NOT_USED,
            usage_correctness_feedback=None
        )
        mock_openai_instance.generate_coaching_response = AsyncMock(return_value=mock_coaching_response)
        mock_openai_service.return_value = mock_openai_instance
        
        # Mock vocabulary service
        mock_vocabulary_instance = Mock()
        mock_tier_analysis = Mock()
        mock_tier_analysis.tier = "basic"
        mock_tier_analysis.score = 55
        mock_tier_analysis.word_count = 4
        mock_tier_analysis.complex_word_count = 0
        mock_tier_analysis.average_word_length = 4.0
        mock_tier_analysis.analysis_details = {}
        mock_vocabulary_instance.analyze_vocabulary_tier.return_value = mock_tier_analysis
        mock_vocabulary_instance.get_vocabulary_recommendations.return_value = []
        mock_vocabulary_service.return_value = mock_vocabulary_instance
        
        # Mock suggestion service - should not generate new suggestion
        mock_suggestion_instance = Mock()
        mock_suggestion_instance.generate_suggestion.return_value = None
        mock_suggestion_service.return_value = mock_suggestion_instance
        
        # Create request without using the suggested word
        request = ConversationRequest(
            message="I love reading books",
            session_id=None,
            personality="friendly_neutral"
        )
        
        # Execute conversation
        mock_current_user = {"sub": "auth0|test", "email": "test@example.com"}
        response = await handle_conversation(request, mock_db, mock_current_user)
        
        # Verify suggestion persistence behavior
        assert response.success is True
        assert existing_suggestion.status == "used_incorrectly"  # AC: 5 - Status unchanged, suggestion persists
        assert response.used_suggestion_id is None  # No usage this turn
        assert response.remediation_feedback is None  # No feedback for not using word
        
        # Verify no new suggestion is generated (existing one remains active)
        assert response.suggestion is None
        
        print("✅ Test passed: Suggestion persists after incorrect usage until correct usage")

    def test_word_detection_accuracy(self):
        """Test word detection logic handles various forms correctly"""
        from app.api.conversation import detect_word_usage
        
        # Test exact match
        assert detect_word_usage("I want to elaborate on this", "elaborate") is True
        
        # Test case insensitivity
        assert detect_word_usage("I want to Elaborate on this", "elaborate") is True
        
        # Test plural form
        assert detect_word_usage("Multiple books on shelves", "book") is True
        
        # Test word boundaries (avoid partial matches)
        assert detect_word_usage("I want to elaborate on this", "labor") is False
        
        # Test with punctuation
        assert detect_word_usage("Can you elaborate? Thanks!", "elaborate") is True
        
        # Test word not present
        assert detect_word_usage("I like reading books", "elaborate") is False
        
        # Test empty inputs
        assert detect_word_usage("", "elaborate") is False
        assert detect_word_usage("Hello world", "") is False
        
        print("✅ Test passed: Word detection accuracy")
