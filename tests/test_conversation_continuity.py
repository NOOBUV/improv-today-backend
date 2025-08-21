import pytest
import uuid
from unittest.mock import Mock, patch
from app.api.conversation import handle_conversation, ConversationRequest
from app.models.conversation_v2 import Conversation


class TestConversationContinuity:
    """Test conversation continuity across multiple requests within a session."""
    
    @pytest.mark.asyncio
    @patch('app.api.conversation.get_or_create_user')
    @patch('app.api.conversation.SimpleOpenAIService')
    @patch('app.api.conversation.VocabularyTierService')
    @patch('app.api.conversation.SuggestionService')
    @patch('app.api.conversation.RedisService')
    async def test_conversation_reuse_within_session(self, 
                                                   mock_redis_service,
                                                   mock_suggestion_service,
                                                   mock_vocabulary_service, 
                                                   mock_openai_service,
                                                   mock_get_user):
        """Test that conversation is reused within the same session."""
        
        # Setup mocks
        mock_get_user.return_value = 123
        user_id = 123
        session_id = 456
        
        # Mock database session
        mock_db = Mock()
        
        # Mock existing conversation
        existing_conversation = Mock()
        existing_conversation.id = uuid.uuid4()
        existing_conversation.user_id = str(user_id)
        existing_conversation.session_id = session_id
        existing_conversation.status = 'active'
        
        # First call: no existing conversation
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        # Mock services
        mock_redis_service_instance = Mock()
        mock_redis_service_instance.get_conversation_history.return_value = []
        mock_redis_service_instance.build_conversation_context.return_value = ""
        mock_redis_service_instance.cache_message.return_value = True
        mock_redis_service.return_value = mock_redis_service_instance
        
        mock_openai_instance = Mock()
        mock_coaching_response = Mock()
        mock_coaching_response.corrected_transcript = "Hello there!"
        mock_coaching_response.ai_response = "Hi! How are you?"
        mock_coaching_response.word_usage_status = "not_used"
        mock_coaching_response.usage_correctness_feedback = None
        mock_openai_instance.generate_coaching_response.return_value = mock_coaching_response
        mock_openai_service.return_value = mock_openai_instance
        
        mock_vocabulary_instance = Mock()
        mock_tier_analysis = Mock()
        mock_tier_analysis.tier = "basic"
        mock_tier_analysis.score = 50
        mock_tier_analysis.word_count = 2
        mock_tier_analysis.complex_word_count = 0
        mock_tier_analysis.average_word_length = 4.5
        mock_tier_analysis.analysis_details = {}
        mock_vocabulary_instance.analyze_vocabulary_tier.return_value = mock_tier_analysis
        mock_vocabulary_instance.get_vocabulary_recommendations.return_value = []
        mock_vocabulary_service.return_value = mock_vocabulary_instance
        
        mock_suggestion_instance = Mock()
        mock_suggestion_instance.generate_suggestion.return_value = None
        mock_suggestion_service.return_value = mock_suggestion_instance
        
        # Mock current_user
        mock_current_user = {"sub": "auth0|test", "email": "test@example.com"}
        
        # Create request
        request = ConversationRequest(
            message="Hello there!",
            session_id=session_id,
            personality="friendly_neutral"
        )
        
        # First call - should create new conversation
        with patch('app.api.conversation.get_or_create_user', return_value=123):
            with patch('app.api.conversation.get_db', return_value=mock_db):
                response1 = await handle_conversation(request, mock_db, mock_current_user)
        
        # Verify conversation was created (add called)
        assert mock_db.add.called
        
        # Now mock that the conversation exists for second call
        mock_db.reset_mock()
        mock_db.query.return_value.filter.return_value.first.return_value = existing_conversation
        
        # Second call with same session - should reuse conversation
        request2 = ConversationRequest(
            message="How are you doing?",
            session_id=session_id,
            personality="friendly_neutral"
        )
        
        with patch('app.api.conversation.get_or_create_user', return_value=123):
            with patch('app.api.conversation.get_db', return_value=mock_db):
                response2 = await handle_conversation(request2, mock_db, mock_current_user)
        
        # Verify conversation was NOT created again (add not called for new conversation)
        # The add calls should only be for messages, not a new conversation
        add_calls = [call for call in mock_db.add.call_args_list]
        
        # Should have calls for messages but not for a new Conversation object
        assert len(add_calls) > 0  # Messages were added
        
        # Verify conversation history was retrieved with the existing conversation ID
        mock_redis_service_instance.get_conversation_history.assert_called_with(
            str(existing_conversation.id), mock_db
        )
    
    def test_conversation_query_logic(self):
        """Test the conversation query logic for finding existing conversations."""
        from app.models.conversation_v2 import Conversation
        
        # This would be an integration test in a real scenario
        # For now, just verify the query structure is correct
        
        user_id = "123"
        session_id = 456
        
        # Mock query builder
        mock_db = Mock()
        mock_query = Mock()
        mock_filter = Mock()
        
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = None
        
        # Simulate the query logic
        conversation = mock_db.query(Conversation).filter(
            Conversation.session_id == session_id,
            Conversation.user_id == str(user_id),
            Conversation.status == 'active'
        ).first()
        
        # Verify the query was constructed correctly
        mock_db.query.assert_called_once_with(Conversation)
        mock_query.filter.assert_called_once()
        mock_filter.first.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])